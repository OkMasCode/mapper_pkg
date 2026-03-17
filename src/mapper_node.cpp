#include "mapper_pkg/mapper_node.hpp"
#include "mapper_pkg/semantic_object_map.hpp"

// PCL Headers for 3D filtering
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>

// Standard C++ and System Headers
#include <zlib.h> // For CRC32 deterministic coloring
#include <chrono>
#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <tf2/exceptions.h>

using std::placeholders::_1;
using std::placeholders::_2;

PointCloudMapperNodeV5::PointCloudMapperNodeV5() : Node("pointcloud_mapper_node_v5") {
    RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:init] starting initialization");

    // 1. Declare and load core I/O parameters
    dm_topic_ = this->declare_parameter("detection_message", "/vision/detections");
    map_frame_ = this->declare_parameter("map_frame", "map");
    camera_frame_ = this->declare_parameter("camera_frame", "camera_color_optical_frame");
    output_dir_ = this->declare_parameter("output_dir", "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/");
    output_map_file_ = this->declare_parameter("output_map_file", "map_v5.json");
    export_interval_ = this->declare_parameter("export_interval", 3.0);
    stable_pointcloud_topic_ = this->declare_parameter("stable_pointcloud_topic", "/vision/semantic_map_v5/points");
    publish_stable_pointcloud_enabled_ = this->declare_parameter("publish_stable_pointcloud", true);

    // Initialize the core Semantic Mapper logic (The Brain)
    semantic_map_ = std::make_unique<SemanticObjectMapV5>();

    // 2. Load and map tuning parameters directly to the mapper instance
    // (Assuming SemanticObjectMapV5 has these as public members matching the Python logic)
    /* semantic_map_->min_input_confidence = this->declare_parameter("min_input_confidence", 0.55);
    semantic_map_->confirmation_min_hits = this->declare_parameter("confirmation_min_hits", 5);
    semantic_map_->confirmation_min_age_sec = this->declare_parameter("confirmation_min_age_sec", 1.0);
    semantic_map_->min_detection_depth_m = this->declare_parameter("min_detection_depth_m", 0.25);
    semantic_map_->max_detection_depth_m = this->declare_parameter("max_detection_depth_m", 4.0);
    // ... [Map the rest of the tuning parameters here] ...
    */

    // 3. Initialize Camera Intrinsics state
    fx_ = fy_ = cx_ = cy_ = 0.0;
    intrinsics_ready_ = false;

    // 4. Initialize TF2 Buffer and Listener for spatial transforms
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 5. Setup QoS (Quality of Service) for sensor data (Best Effort / Volatile)
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);

    // 6. Setup Subscribers
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera/depth/camera_info", qos, 
        std::bind(&PointCloudMapperNodeV5::camera_info_cb, this, _1)
    );

    mask_sub_.subscribe(this, dm_topic_, qos.get_rmw_qos_profile());
    depth_sub_.subscribe(this, "/camera/depth", qos.get_rmw_qos_profile());

    // 7. Setup Approximate Time Synchronizer (Queue size = 10)
    sync_ = std::make_shared<Sync>(SyncPolicy(10), mask_sub_, depth_sub_);
    sync_->registerCallback(std::bind(&PointCloudMapperNodeV5::synced_detection_callback, this, _1, _2));

    // 8. Setup Publishers & Timers
    map_pub_ = this->create_publisher<yolo11_seg_interfaces::msg::SemanticObjectArray>("/vision/semantic_map_v5", 10);
    stable_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(stable_pointcloud_topic_, 10);
    
    export_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(export_interval_),
        std::bind(&PointCloudMapperNodeV5::export_callback, this)
    );

    RCLCPP_INFO(this->get_logger(), "[mapper_node_v5] ready. input=%s", dm_topic_.c_str());
}

void PointCloudMapperNodeV5::camera_info_cb(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if (!intrinsics_ready_) {
        fx_ = msg->k[0];
        cx_ = msg->k[2];
        fy_ = msg->k[4];
        cy_ = msg->k[5];
        intrinsics_ready_ = true;
        RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:camera_info] received intrinsics fx=%.3f fy=%.3f cx=%.3f cy=%.3f", fx_, fy_, cx_, cy_);
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudMapperNodeV5::get_points_in_mask(
    const cv::Mat& depth_m, 
    const cv::Mat& binary_mask) 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    // Iterate through the image pixels to project 2D mask to 3D Camera Coordinates
    for (int v = 0; v < depth_m.rows; ++v) {
        for (int u = 0; u < depth_m.cols; ++u) {
            if (binary_mask.at<uint8_t>(v, u) > 0) {
                float z = depth_m.at<float>(v, u);
                
                // Depth gating (e.g., 0.25m to 4.0m)
                if (std::isfinite(z) && z >= 0.25f && z <= 4.0f) {
                    pcl::PointXYZ pt;
                    pt.x = (u - cx_) * z / fx_;
                    pt.y = (v - cy_) * z / fy_;
                    pt.z = z;
                    cloud->points.push_back(pt);
                }
            }
        }
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

void PointCloudMapperNodeV5::synced_detection_callback(
    const yolo11_seg_interfaces::msg::DetectedObjectV3Array::ConstSharedPtr mask_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr depth_msg) 
{
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "[mapper_node_v5:sync] callback active. detections=%zu depth_encoding=%s",
        mask_msg->detections.size(), depth_msg->encoding.c_str());

    if (!intrinsics_ready_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
            "[mapper_node_v5:sync] skipping frame because camera intrinsics are not ready");
        return;
    }

    // Lock the mutex to ensure thread safety during map updates
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Convert ROS Depth Image to OpenCV float32 matrix (meters)
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat depth_m;
    if (cv_ptr->image.type() == CV_16UC1) {
        cv_ptr->image.convertTo(depth_m, CV_32FC1, 0.001); // Convert mm to meters
    } else {
        depth_m = cv_ptr->image; 
    }

    // Transform all generated clouds into map frame before semantic fusion.
    const std::string source_frame = depth_msg->header.frame_id.empty() ? camera_frame_ : depth_msg->header.frame_id;
    const rclcpp::Time source_stamp(depth_msg->header.stamp);
    bool use_identity_tf = (source_frame == map_frame_);
    Eigen::Affine3f tf_source_to_map = Eigen::Affine3f::Identity();

    if (!use_identity_tf) {
        try {
            const auto tf_msg = tf_buffer_->lookupTransform(
                map_frame_, source_frame, source_stamp, rclcpp::Duration::from_seconds(0.10));

            tf_source_to_map.translation() <<
                static_cast<float>(tf_msg.transform.translation.x),
                static_cast<float>(tf_msg.transform.translation.y),
                static_cast<float>(tf_msg.transform.translation.z);

            const Eigen::Quaternionf q(
                static_cast<float>(tf_msg.transform.rotation.w),
                static_cast<float>(tf_msg.transform.rotation.x),
                static_cast<float>(tf_msg.transform.rotation.y),
                static_cast<float>(tf_msg.transform.rotation.z));
            tf_source_to_map.linear() = q.normalized().toRotationMatrix();
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "[mapper_node_v5:tf] cannot transform %s -> %s at t=%.3f: %s",
                source_frame.c_str(), map_frame_.c_str(), source_stamp.seconds(), ex.what());
            return;
        }
    }

    std::vector<std::string> object_names;
    std::vector<std::string> tracker_ids;
    std::vector<float> confidences;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> points_cam_list;

    int accepted_detections = 0;

    // 2. Iterate through batched 2D detections
    for (const auto& det : mask_msg->detections) {
        if (det.mask.width == 0 || det.mask.height == 0) continue;

        cv_bridge::CvImagePtr mask_ptr = cv_bridge::toCvCopy(det.mask, "mono8");
        cv::Mat cv_mask = mask_ptr->image;

        // Ensure mask dimensions match depth dimensions
        if (cv_mask.size() != depth_m.size()) {
            cv::resize(cv_mask, cv_mask, depth_m.size(), 0, 0, cv::INTER_NEAREST);
        }

        // Project 2D pixels to 3D point cloud
        auto raw_cloud = get_points_in_mask(depth_m, cv_mask);
        if (raw_cloud->points.size() < 4) {
            RCLCPP_DEBUG(this->get_logger(),
                "[mapper_node_v5:filter] reject det class=%s id=%d reason=raw_cloud_too_small n=%zu",
                det.class_name.c_str(), det.instance_id, raw_cloud->points.size());
            continue;
        }

        // PCL 3D Filtering: Voxel downsample (0.05m = 5cm)
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(raw_cloud);
        voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f); 
        voxel_filter.filter(*downsampled_cloud);

        // PCL 3D Filtering: Statistical Outlier Removal
        pcl::PointCloud<pcl::PointXYZ>::Ptr clean_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (downsampled_cloud->points.size() > 7) {
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(downsampled_cloud);
            sor.setMeanK(10);
            sor.setStddevMulThresh(2.0);
            sor.filter(*clean_cloud);
        } else {
            clean_cloud = downsampled_cloud;
        }

        if (clean_cloud->points.size() < 4) {
            RCLCPP_DEBUG(this->get_logger(),
                "[mapper_node_v5:filter] reject det class=%s id=%d reason=clean_cloud_too_small n=%zu",
                det.class_name.c_str(), det.instance_id, clean_cloud->points.size());
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZ>());
        if (use_identity_tf) {
            cloud_map = clean_cloud;
        } else {
            pcl::transformPointCloud(*clean_cloud, *cloud_map, tf_source_to_map);
        }

        // Append to batch lists (all points are map-frame now)
        object_names.push_back(det.class_name);
        tracker_ids.push_back(std::to_string(det.instance_id));
        confidences.push_back(det.confidence);
        points_cam_list.push_back(cloud_map);
        accepted_detections++;
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "[mapper_node_v5:sync] accepted_detections=%d raw_detections=%zu src_frame=%s map_frame=%s",
        accepted_detections, mask_msg->detections.size(), source_frame.c_str(), map_frame_.c_str());

    // 3. Pass the batch to the Semantic Mapper Logic (The Bipartite Matrix)
    if (!points_cam_list.empty()) {
        semantic_map_->add_detections_batch(
            object_names, tracker_ids, confidences, points_cam_list,
            mask_msg->header.stamp, camera_frame_, map_frame_
        );
    } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[mapper_node_v5:sync] no detections survived filtering in this frame");
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "[mapper_node_v5:state] objects=%zu tentative=%zu",
        semantic_map_->objects.size(), semantic_map_->tentative_tracks.size());

    publish_semantic_map();
    
    if (publish_stable_pointcloud_enabled_) {
        publish_stable_pointcloud();
    }
}

void PointCloudMapperNodeV5::publish_semantic_map() {
    if (semantic_map_->objects.empty()) {
        RCLCPP_DEBUG(this->get_logger(), "[mapper_node_v5:publish_map] skip: no objects");
        return;
    }

    yolo11_seg_interfaces::msg::SemanticObjectArray msg;

    for (const auto& [map_id, entry] : semantic_map_->objects) {
        yolo11_seg_interfaces::msg::SemanticObject obj_msg;
        
        obj_msg.object_id = map_id;
        obj_msg.name = entry.current_name;
        obj_msg.frame = entry.frame;
        obj_msg.timestamp = entry.timestamp;
        
        // Populate standard geometry vectors
        obj_msg.pose_cam.x = entry.pose_cam[0];
        obj_msg.pose_cam.y = entry.pose_cam[1];
        obj_msg.pose_cam.z = entry.pose_cam[2];
        
        obj_msg.pose_map.x = entry.pose_map[0];
        obj_msg.pose_map.y = entry.pose_map[1];
        obj_msg.pose_map.z = entry.pose_map[2];
        
        obj_msg.occurrences = entry.occurrences;
        obj_msg.similarity = entry.similarity;
        obj_msg.confidence = entry.confidence_ema;
        obj_msg.image_embedding = entry.image_embedding;

        msg.objects.push_back(obj_msg);
    }

    map_pub_->publish(msg);
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "[mapper_node_v5:publish_map] published objects=%zu", msg.objects.size());
}

void PointCloudMapperNodeV5::publish_stable_pointcloud() {
    if (semantic_map_->objects.empty()) return;

    // Use built-in PCL type to handle RGB byte packing automatically
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    for (const auto& [map_id, entry] : semantic_map_->objects) {
        if (!entry.accumulated_points || entry.accumulated_points->empty()) continue;

        // Get RGB colors based on class name/map_id
        auto [r, g, b] = class_to_color_rgb(map_id);

        for (const auto& pt : entry.accumulated_points->points) {
            pcl::PointXYZRGB colored_pt;
            colored_pt.x = pt.x;
            colored_pt.y = pt.y;
            colored_pt.z = pt.z;
            colored_pt.r = r;
            colored_pt.g = g;
            colored_pt.b = b;
            colored_cloud->points.push_back(colored_pt);
        }
    }

    if (colored_cloud->points.empty()) return;

    colored_cloud->width = colored_cloud->points.size();
    colored_cloud->height = 1;
    colored_cloud->is_dense = true;

    // Convert PCL cloud to ROS PointCloud2 message
    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(*colored_cloud, cloud_msg);
    
    cloud_msg.header.stamp = this->get_clock()->now();
    cloud_msg.header.frame_id = map_frame_;

    stable_cloud_pub_->publish(cloud_msg);
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "[mapper_node_v5:publish_cloud] published_points=%zu", colored_cloud->points.size());
}

std::tuple<uint8_t, uint8_t, uint8_t> PointCloudMapperNodeV5::class_to_color_rgb(const std::string& class_name) {
    // Generates deterministic color matching the Python CRC32 implementation
    unsigned long crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, reinterpret_cast<const Bytef*>(class_name.c_str()), class_name.length());

    uint8_t r = 60 + (crc & 0x7F);
    uint8_t g = 60 + ((crc >> 8) & 0x7F);
    uint8_t b = 60 + ((crc >> 16) & 0x7F);
    return {r, g, b};
}

void PointCloudMapperNodeV5::export_callback() {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        // 1. Clean the point clouds for all confirmed objects
        for (const auto& [map_id, entry] : semantic_map_->objects) {
            semantic_map_->refine_object_geometry(map_id);
        }
        // Run the geometric duplicate cleanup routine
        semantic_map_->resolve_overlapping_duplicates();
        
        // Export to JSON (Handled internally by the Mapper or bypassed if separated to Python)
        semantic_map_->export_to_json(output_dir_, output_map_file_);
        
        publish_semantic_map();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Export/Merge error: %s", e.what());
    }
}

void PointCloudMapperNodeV5::shutdown_callback() {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:shutdown] exporting final map");
        semantic_map_->export_to_json(output_dir_, "map_v5_final.json");
        RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:shutdown] final export complete");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "[mapper_node_v5] final export error: %s", e.what());
    }
}

// ==========================================
// MAIN ENTRY POINT
// ==========================================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<PointCloudMapperNodeV5>();

    // Use MultiThreadedExecutor to allow sync callback and timer to run concurrently
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    try {
        executor.spin();
    } catch (const std::exception& e) {
        // Catch interrupts cleanly
    }

    // Ensure state is safely saved on Ctrl+C shutdown
    node->shutdown_callback();
    rclcpp::shutdown();
    return 0;
}