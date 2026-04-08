#include "mapper_pkg/mapper_node.hpp"
#include "mapper_pkg/semantic_object_map.hpp"

// PCL Headers for 3D filtering
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>

// Standard C++ and System Headers
#include <zlib.h> // For CRC32 deterministic coloring
#include <chrono>
#include <optional>
#include <array>
#include <iomanip>
#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <tf2/exceptions.h>
#include <std_msgs/msg/float32_multi_array.hpp>

using std::placeholders::_1;
using std::placeholders::_2;

PointCloudMapperNodeV5::PointCloudMapperNodeV5() : Node("pointcloud_mapper_node_v5") {
    RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:init] starting initialization");

    // 1. Declare and load core I/O parameters
    dm_topic_ = this->declare_parameter("detection_message", "/vision/detections");
    map_frame_ = this->declare_parameter("map_frame", "camera_color_optical_frame");
    camera_frame_ = this->declare_parameter("camera_frame", "camera_color_optical_frame");
    output_dir_ = this->declare_parameter("output_dir", "/home/workspace/ros2_ws/src/yolo11_seg_bringup/config/");
    output_map_file_ = this->declare_parameter("output_map_file", "map_v6.json");
    export_interval_ = this->declare_parameter("export_interval", 3.0);
    stable_pointcloud_topic_ = this->declare_parameter("stable_pointcloud_topic", "/vision/semantic_map_v5/points");
    publish_stable_pointcloud_enabled_ = this->declare_parameter("publish_stable_pointcloud", true);
    viewer_enabled_ = this->declare_parameter("viewer_enabled", true);

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

    // Initialize timing
    last_timing_print_ = std::chrono::steady_clock::now();

    // 4. Initialize TF2 Buffer and Listener for spatial transforms
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 5. Setup QoS (Quality of Service) for sensor data (Best Effort / Volatile)
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);

    // 6. Setup Subscribers
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera/camera/aligned_depth_to_color/camera_info", qos, // /camera/camera/aligned_depth_to_color/camera_info
        std::bind(&PointCloudMapperNodeV5::camera_info_cb, this, _1)
    ); 

    text_emb_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/vision/text_embedding", 10,
        std::bind(&PointCloudMapperNodeV5::text_embedding_cb, this, _1)
    );

    mask_sub_.subscribe(this, dm_topic_, qos.get_rmw_qos_profile());
    depth_sub_.subscribe(this, "/camera/camera/aligned_depth_to_color/image_raw", qos.get_rmw_qos_profile()); // /camera/camera/aligned_depth_to_color/image_raw

    // 7. Setup Approximate Time Synchronizer (Queue size = 10)
    sync_ = std::make_shared<Sync>(SyncPolicy(10), mask_sub_, depth_sub_);
    sync_->registerCallback(std::bind(&PointCloudMapperNodeV5::synced_detection_callback, this, _1, _2));

    // 8. Setup Publishers & Timers
    map_pub_ = this->create_publisher<yolo11_seg_interfaces::msg::SemanticObjectArray>("/vision/semantic_map_v5", 10);
    if (viewer_enabled_) {
        stable_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(stable_pointcloud_topic_, 10);
    }
    export_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(export_interval_),
        std::bind(&PointCloudMapperNodeV5::export_callback, this)
    );

    RCLCPP_INFO(this->get_logger(), "[mapper_node_v5] ready. input=%s", dm_topic_.c_str());
}

void PointCloudMapperNodeV5::camera_info_cb(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (!intrinsics_ready_) {
        fx_ = msg->k[0];
        cx_ = msg->k[2];
        fy_ = msg->k[4];
        cy_ = msg->k[5];
        intrinsics_ready_ = true;
        RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:camera_info] received intrinsics fx=%.3f fy=%.3f cx=%.3f cy=%.3f", fx_, fy_, cx_, cy_);
    }
}

void PointCloudMapperNodeV5::text_embedding_cb(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    // Lock the mutex to ensure thread safety with the map publisher
    std::lock_guard<std::mutex> lock(mutex_);
    
    // We expect the embedding + 2 extra values (scale and bias) at the end
    if (msg->data.size() > 2) {
        float bias = msg->data.back();
        float scale = msg->data[msg->data.size() - 2];
        
        // Extract the actual embedding vector
        std::vector<float> emb(msg->data.begin(), msg->data.end() - 2);
        
        // Pass the updated global goal to the map logic
        semantic_map_->set_text_embedding(emb, scale, bias);
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudMapperNodeV5::get_points_in_mask(
    const cv::Mat& depth_m, 
    const cv::Mat& binary_mask) 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    // SOTA Optimization: Compute the bounding box of the non-zero mask pixels.
    // This restricts the nested loops to a tiny fraction of the image.
    cv::Rect bbox = cv::boundingRect(binary_mask);

    // If the mask is completely empty, return early
    if (bbox.width == 0 || bbox.height == 0) return cloud;

    // Iterate ONLY within the bounding box bounds
    for (int v = bbox.y; v < bbox.y + bbox.height; ++v) {
        for (int u = bbox.x; u < bbox.x + bbox.width; ++u) {
            if (binary_mask.at<uint8_t>(v, u) > 0) {
                float z = depth_m.at<float>(v, u);
                
                // Depth gating
                if (std::isfinite(z) && z >= 0.1f && z <= 4.0f) {
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
    auto t_total_start = std::chrono::steady_clock::now();
    
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "[mapper_node_v5:sync] callback active. detections=%zu depth_encoding=%s",
        mask_msg->detections.size(), depth_msg->encoding.c_str());

    std::lock_guard<std::mutex> lock(mutex_);

    if (!intrinsics_ready_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
            "[mapper_node_v5:sync] skipping frame because camera intrinsics are not ready");
        return;
    }

    // 1. Convert ROS Depth Image to OpenCV float32 matrix (meters)
    auto t_depth_start = std::chrono::steady_clock::now();
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
    auto t_depth_end = std::chrono::steady_clock::now();
    timing_depth_conversion_.total_time_ms += std::chrono::duration<double, std::milli>(t_depth_end - t_depth_start).count();
    timing_depth_conversion_.count++;

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
    std::vector<std::optional<std::vector<float>>> embeddings_list_masked;
    std::vector<std::optional<std::vector<float>>> embeddings_list_unmasked;

    int accepted_detections = 0;

    // 2. Iterate through batched 2D detections
    for (const auto& det : mask_msg->detections) {
        if (det.mask.width == 0 || det.mask.height == 0) continue;

        // Point extraction timing
        auto t_extract_start = std::chrono::steady_clock::now();
        cv_bridge::CvImagePtr mask_ptr = cv_bridge::toCvCopy(det.mask, "mono8");
        cv::Mat cv_mask = mask_ptr->image;

        // Ensure mask dimensions match depth dimensions
        if (cv_mask.size() != depth_m.size()) {
            cv::resize(cv_mask, cv_mask, depth_m.size(), 0, 0, cv::INTER_NEAREST);
        }

        // Project 2D pixels to 3D point cloud
        auto raw_cloud = get_points_in_mask(depth_m, cv_mask);
        auto t_extract_end = std::chrono::steady_clock::now();
        timing_point_extraction_.total_time_ms += std::chrono::duration<double, std::milli>(t_extract_end - t_extract_start).count();
        timing_point_extraction_.count++;
        
        if (raw_cloud->points.size() < 4) {
            RCLCPP_DEBUG(this->get_logger(),
                "[mapper_node_v5:filter] reject det class=%s id=%d reason=raw_cloud_too_small n=%zu",
                det.class_name.c_str(), det.instance_id, raw_cloud->points.size());
            continue;
        }

        // Filtering timing
        auto t_filter_start = std::chrono::steady_clock::now();

        // ==========================================
        // SOTA: ADAPTIVE VOXELIZATION
        // ==========================================
        float voxel_size = 0.02f; // Default to 5cm
        int num_raw_points = raw_cloud->points.size();

        if (num_raw_points < 10000) voxel_size = 0.005f;
        else if (num_raw_points > 30000) voxel_size = 0.03f;
        else voxel_size = 0.005f + (0.02f * ((num_raw_points - 10000.0f) / 25000.0f));
        // PCL 3D Filtering: Voxel downsample (0.006m = 6mm)
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(raw_cloud);
        voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size); 
        voxel_filter.filter(*downsampled_cloud);

        // PCL 3D Filtering: Euclidean Clustering (keeps only largest contiguous cluster)
        // This removes ALL isolated/distant points even if they form clusters
        pcl::PointCloud<pcl::PointXYZ>::Ptr clean_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (downsampled_cloud->points.size() > 10) {
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            tree->setInputCloud(downsampled_cloud);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(0.06);    // 6cm distance to stay in same cluster
            ec.setMinClusterSize(5);         // Minimum 5 points per cluster
            ec.setMaxClusterSize(50000);     // Max cluster size (keep large objects)
            ec.setSearchMethod(tree);
            ec.setInputCloud(downsampled_cloud);
            ec.extract(cluster_indices);

            if (!cluster_indices.empty()) {
                // Keep only the LARGEST cluster (main object, not outliers)
                const auto& largest_cluster = cluster_indices[0];
                for (const auto& idx : largest_cluster.indices) {
                    clean_cloud->points.push_back(downsampled_cloud->points[idx]);
                }
            } else {
                *clean_cloud = *downsampled_cloud;
            }
        } else {
            *clean_cloud = *downsampled_cloud;
        }
        clean_cloud->width = clean_cloud->points.size();
        clean_cloud->height = 1;
        clean_cloud->is_dense = true;
        
        auto t_filter_end = std::chrono::steady_clock::now();
        timing_filtering_.total_time_ms += std::chrono::duration<double, std::milli>(t_filter_end - t_filter_start).count();
        timing_filtering_.count++;

        if (clean_cloud->points.size() < 4) {
            RCLCPP_DEBUG(this->get_logger(),
                "[mapper_node_v5:filter] reject det class=%s id=%d reason=clean_cloud_too_small n=%zu",
                det.class_name.c_str(), det.instance_id, clean_cloud->points.size());
            continue;
        }

        // Transformation timing
        auto t_tf_start = std::chrono::steady_clock::now();
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_map(new pcl::PointCloud<pcl::PointXYZ>());
        if (use_identity_tf) {
            cloud_map = clean_cloud;
        } else {
            pcl::transformPointCloud(*clean_cloud, *cloud_map, tf_source_to_map);
        }
        auto t_tf_end = std::chrono::steady_clock::now();
        timing_transformation_.total_time_ms += std::chrono::duration<double, std::milli>(t_tf_end - t_tf_start).count();
        timing_transformation_.count++;

        // Append to batch lists (all points are map-frame now)
        object_names.push_back(det.class_name);
        tracker_ids.push_back(std::to_string(det.instance_id));
        confidences.push_back(det.confidence);
        points_cam_list.push_back(cloud_map);
        if (!det.image_embedding_masked.empty()) {
            embeddings_list_masked.emplace_back(det.image_embedding_masked);
        } else {
            embeddings_list_masked.emplace_back(std::nullopt);
        }
        if (!det.image_embedding_unmasked.empty()) {
            embeddings_list_unmasked.emplace_back(det.image_embedding_unmasked);
        } else {
            embeddings_list_unmasked.emplace_back(std::nullopt);
        }
        accepted_detections++;
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "[mapper_node_v5:sync] accepted_detections=%d raw_detections=%zu src_frame=%s map_frame=%s",
        accepted_detections, mask_msg->detections.size(), source_frame.c_str(), map_frame_.c_str());

    // 3. Pass the batch to the Semantic Mapper Logic (The Bipartite Matrix)
    if (!points_cam_list.empty()) {
        auto t_batch_start = std::chrono::steady_clock::now();
        semantic_map_->add_detections_batch(
            object_names, tracker_ids, confidences, points_cam_list, embeddings_list_masked, embeddings_list_unmasked,
            mask_msg->header.stamp, camera_frame_, map_frame_
        );
        auto t_batch_end = std::chrono::steady_clock::now();
        timing_batch_addition_.total_time_ms += std::chrono::duration<double, std::milli>(t_batch_end - t_batch_start).count();
        timing_batch_addition_.count++;
    } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "[mapper_node_v5:sync] no detections survived filtering in this frame");
    }

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "[mapper_node_v5:state] objects=%zu tentative=%zu",
        semantic_map_->objects.size(), semantic_map_->tentative_tracks.size());

    auto t_pub_start = std::chrono::steady_clock::now();
    publish_semantic_map();
    
    if (publish_stable_pointcloud_enabled_) {
        publish_stable_pointcloud();
    }
    auto t_pub_end = std::chrono::steady_clock::now();
    timing_publishing_.total_time_ms += std::chrono::duration<double, std::milli>(t_pub_end - t_pub_start).count();
    timing_publishing_.count++;
    
    // Total timing and periodic reporting
    auto t_total_end = std::chrono::steady_clock::now();
    timing_total_.total_time_ms += std::chrono::duration<double, std::milli>(t_total_end - t_total_start).count();
    timing_total_.count++;
    frame_count_++;
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_timing_print_).count();
    if (elapsed >= 30.0) {
        print_timing_stats();
        last_timing_print_ = now;
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

        // Publish true OBB metadata (extents + orientation + oriented corners).
        const float sx = std::max(0.0f, entry.obb.extents[0]);
        const float sy = std::max(0.0f, entry.obb.extents[1]);
        const float sz = std::max(0.0f, entry.obb.extents[2]);

        obj_msg.bbox_type = "obb";
        obj_msg.box_size.x = sx;
        obj_msg.box_size.y = sy;
        obj_msg.box_size.z = sz;

        if (entry.obb.rotation.size() >= 9) {
            Eigen::Matrix3f R;
            R(0, 0) = entry.obb.rotation[0]; R(0, 1) = entry.obb.rotation[1]; R(0, 2) = entry.obb.rotation[2];
            R(1, 0) = entry.obb.rotation[3]; R(1, 1) = entry.obb.rotation[4]; R(1, 2) = entry.obb.rotation[5];
            R(2, 0) = entry.obb.rotation[6]; R(2, 1) = entry.obb.rotation[7]; R(2, 2) = entry.obb.rotation[8];
            Eigen::Quaternionf q(R);
            q.normalize();
            obj_msg.bbox_orientation.x = static_cast<double>(q.x());
            obj_msg.bbox_orientation.y = static_cast<double>(q.y());
            obj_msg.bbox_orientation.z = static_cast<double>(q.z());
            obj_msg.bbox_orientation.w = static_cast<double>(q.w());
        } else {
            obj_msg.bbox_orientation.x = 0.0;
            obj_msg.bbox_orientation.y = 0.0;
            obj_msg.bbox_orientation.z = 0.0;
            obj_msg.bbox_orientation.w = 1.0;
        }

        const auto corners = semantic_map_->compute_obb_corners(entry.obb);

        obj_msg.bbox_corners.clear();
        obj_msg.bbox_corners.reserve(corners.size());
        for (const auto& c : corners) {
            geometry_msgs::msg::Point p;
            p.x = static_cast<double>(c[0]);
            p.y = static_cast<double>(c[1]);
            p.z = static_cast<double>(c[2]);
            obj_msg.bbox_corners.push_back(p);
        }
        
        obj_msg.occurrences = entry.occurrences;
        obj_msg.confidence = entry.confidence_ema;
        obj_msg.image_embedding_masked = entry.image_embedding_masked;
        obj_msg.image_embedding_unmasked = entry.image_embedding_unmasked;
        obj_msg.similarity = semantic_map_->get_goal_similarity(map_id);
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

    
    if (viewer_enabled_) {
        stable_cloud_pub_->publish(cloud_msg);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "[mapper_node_v5:publish_cloud] published_points=%zu", colored_cloud->points.size());
    }
    
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
        semantic_map_->remove_wrong_detections();
    
        publish_semantic_map();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Export/Merge error: %s", e.what());
    }
}

void PointCloudMapperNodeV5::shutdown_callback() {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:shutdown] exporting final map");
        RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:shutdown] final export complete");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "[mapper_node_v5] final export error: %s", e.what());
    }
}

void PointCloudMapperNodeV5::print_timing_stats() {
    if (frame_count_ == 0) return;
    
    // Print header
    std::cout << "\n" << std::string(75, '=') << "\n";
    std::cout << "TIMING STATISTICS (30-second window, " << frame_count_ << " frames)\n";
    std::cout << std::string(75, '=') << "\n";
    std::cout << std::left 
              << std::setw(30) << "Step" 
              << std::setw(20) << "Avg Time (ms)" 
              << std::setw(15) << "Frames" 
              << "\n";
    std::cout << std::string(75, '-') << "\n";
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left 
              << std::setw(30) << "Depth Conversion"
              << std::setw(20) << timing_depth_conversion_.average()
              << std::setw(15) << timing_depth_conversion_.count << "\n";
    
    std::cout << std::left 
              << std::setw(30) << "Point Extraction"
              << std::setw(20) << timing_point_extraction_.average()
              << std::setw(15) << timing_point_extraction_.count << "\n";
    
    std::cout << std::left 
              << std::setw(30) << "Filtering (Voxel+SOR)"
              << std::setw(20) << timing_filtering_.average()
              << std::setw(15) << timing_filtering_.count << "\n";
    
    std::cout << std::left 
              << std::setw(30) << "Transform to Map"
              << std::setw(20) << timing_transformation_.average()
              << std::setw(15) << timing_transformation_.count << "\n";
    
    std::cout << std::left 
              << std::setw(30) << "Batch Addition (Matching)"
              << std::setw(20) << timing_batch_addition_.average()
              << std::setw(15) << timing_batch_addition_.count << "\n";
    
    std::cout << std::left 
              << std::setw(30) << "Publishing"
              << std::setw(20) << timing_publishing_.average()
              << std::setw(15) << timing_publishing_.count << "\n";
    
    std::cout << std::string(75, '-') << "\n";
    
    std::cout << std::left 
              << std::setw(30) << "TOTAL"
              << std::setw(20) << timing_total_.average()
              << std::setw(15) << timing_total_.count << "\n";
    
    std::cout << std::string(75, '=') << "\n\n";
    
    // Reset counters
    timing_depth_conversion_.reset();
    timing_point_extraction_.reset();
    timing_filtering_.reset();
    timing_transformation_.reset();
    timing_batch_addition_.reset();
    timing_publishing_.reset();
    timing_total_.reset();
    frame_count_ = 0;
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