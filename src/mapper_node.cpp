#include "semantic_mapper_cpp/mapper_node.hpp"
#include <cv_bridge/cv_bridge.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <pcl_ros/transforms.hpp> // Required for transforming Point Clouds
#include <chrono>

MapperNode::MapperNode() : Node("semantic_mapper_node") {
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    depth_sub_.subscribe(this, "/camera/depth/image_rect_raw", rmw_qos_profile_sensor_data);
    det_sub_.subscribe(this, "/vision/detections", rmw_qos_profile_sensor_data);

    sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), depth_sub_, det_sub_);
    sync_->registerCallback(std::bind(&MapperNode::syncedCallback, this, std::placeholders::_1, std::placeholders::_2));

    maintenance_timer_ = this->create_wall_timer(
        std::chrono::seconds(2),
        std::bind(&MapperNode::maintenanceCallback, this)
    );

    RCLCPP_INFO(this->get_logger(), "Mapper Node started successfully.");
}

void MapperNode::syncedCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
    const yolo11_seg_interfaces::msg::DetectedObjectV3Array::ConstSharedPtr& det_msg) 
{
    // 1. Lookup TF Transform from Camera to Map
    geometry_msgs::msg::TransformStamped transform;
    try {
        transform = tf_buffer_->lookupTransform(
            "map", depth_msg->header.frame_id, 
            depth_msg->header.stamp, rclcpp::Duration::from_seconds(0.1)
        );
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "TF Error: %s", ex.what());
        return; 
    }

    // 2. Camera Intrinsics 
    // IMPORTANT: Replace these with the actual values from your camera's CameraInfo topic
    const float fx = 600.0f; // Focal length X
    const float fy = 600.0f; // Focal length Y
    const float cx = 320.0f; // Optical center X
    const float cy = 240.0f; // Optical center Y

    std::vector<std::string> names;
    std::vector<std::string> track_ids;
    std::vector<Eigen::VectorXf> embeddings;
    std::vector<float> confidences;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> points_cam_list;

    // Convert ROS depth image to OpenCV Matrix
    cv_bridge::CvImageConstPtr cv_depth;
    try {
        cv_depth = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // 3. Process Detections
    for (const auto& det : det_msg->detections) {
        names.push_back(det.class_name);
        track_ids.push_back(std::to_string(det.instance_id));
        confidences.push_back(det.confidence);
        
        Eigen::VectorXf emb = Eigen::Map<const Eigen::VectorXf>(det.embedding.data(), det.embedding.size());
        embeddings.push_back(emb);

        // Convert the YOLO mask to OpenCV
        cv_bridge::CvImagePtr cv_mask;
        try {
            cv_mask = cv_bridge::toCvCopy(det.mask, sensor_msgs::image_encodings::MONO8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception on mask: %s", e.what());
            continue;
        }
        // Create an empty cloud for the camera frame
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cam(new pcl::PointCloud<pcl::PointXYZ>());
        
        // --- THE BACK-PROJECTION MATH ---
        // Loop through every pixel in the 2D image
        for (int v = 0; v < cv_mask->image.rows; ++v) {
            for (int u = 0; u < cv_mask->image.cols; ++u) {
                
                // If the pixel belongs to the YOLO object mask (value > 0)
                if (cv_mask->image.at<uint8_t>(v, u) > 0) {
                    
                    // Get the depth in meters
                    float z = cv_depth->image.at<float>(v, u); 
                    
                    // Ignore dead pixels and extreme noise
                    if (std::isfinite(z) && z > 0.1f) {
                        pcl::PointXYZ pt;
                        pt.z = z;
                        pt.x = (u - cx) * z / fx;
                        pt.y = (v - cy) * z / fy;
                        cloud_cam->points.push_back(pt);
                    }
                }
            }
        }

        // Finalize the cloud
        cloud_cam->width = cloud_cam->points.size();
        cloud_cam->height = 1;
        cloud_cam->is_dense = true;

        // Skip if the object was just empty pixels
        if (cloud_cam->points.size() < 5) continue;

        // --- THE TF TRANSFORM ---
        // Automatically move the points from the camera frame to the map frame
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZ>());
        pcl_ros::transformPointCloud(*cloud_cam, *cloud_map, transform.transform);
        
        points_cam_list.push_back(cloud_map);
    }

    // 4. Send the 3D data into the Mapper Matrix
    uint64_t stamp_ns = depth_msg->header.stamp.sec * 1000000000ULL + depth_msg->header.stamp.nanosec;
    mapper_.addDetectionsBatch(names, track_ids, stamp_ns, "map", embeddings, confidences, points_cam_list);
}

void MapperNode::maintenanceCallback() {
    mapper_.resolveDuplicates();
    mapper_.ExportJson("/tmp", "map_v5.json"); 
}