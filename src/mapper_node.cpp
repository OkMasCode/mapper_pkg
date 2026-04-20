#include "mapper_pkg/mapper_node.hpp"
#include "mapper_pkg/semantic_object_map.hpp"
// PCL headers for 3D filtering.
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
// Standard C++ and system headers.
#include <zlib.h>
#include <chrono>
#include <optional>
#include <array>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <Eigen/Geometry>
#include <pcl/common/transforms.h>
#include <tf2/exceptions.h>
#include <std_msgs/msg/float32_multi_array.hpp>

using std::placeholders::_1;
using std::placeholders::_2;

namespace {

std::string format_detection_trace(
    size_t index,
    const yolo11_seg_interfaces::msg::DetectedObjectV3& det,
    const char* stage)
{
    std::ostringstream oss;
    oss << "[mapper_node_v5:det:" << stage << "]"
        << " idx=" << index
        << " id=" << det.instance_id
        << " class='" << det.class_name << "'"
        << " conf=" << std::fixed << std::setprecision(3) << det.confidence;
    return oss.str();
}

} // namespace

PointCloudMapperNodeV5::PointCloudMapperNodeV5() : Node("pointcloud_mapper_node_v5") {
    RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:init] starting initialization");
    // Declare node parameters.
    dm_topic_ = this->declare_parameter("detection_message", "/vision/detections");
    map_frame_ = this->declare_parameter("map_frame", "map");
    camera_frame_ = this->declare_parameter("camera_frame", "camera_0_color_optical_frame");
    std::string depth_topic = this->declare_parameter("depth_topic", "/jackal/sensors/camera_0/aligned_depth_to_color/image");
    std::string cm_info_topic = this->declare_parameter("camera_info_topic", "/jackal/sensors/camera_0/aligned_depth_to_color/camera_info");
    std::string text_emb_topic = this->declare_parameter("vision_topic", "/vision/text_embedding");
    output_dir_ = this->declare_parameter("output_dir", "/workspaces/ros2_ws/src/yolo11_seg_bringup/config/");
    output_map_file_ = this->declare_parameter("output_map_file", "map_v6.json");
    stable_pointcloud_topic_ = this->declare_parameter("stable_pointcloud_topic", "/vision/semantic_map_v5/points");
    publish_stable_pointcloud_enabled_ = this->declare_parameter("publish_stable_pointcloud", true);
    viewer_enabled_ = this->declare_parameter("viewer_enabled", true);
    export_interval_ = this->declare_parameter("export_interval", 5.0);
    // Distance filtering
    min_range_ = this->declare_parameter("min_range", 0.1f);
    max_range_ = this->declare_parameter("max_range", 3.0f);
    // Voxel filtering
    do_voxel_filtering_ = this->declare_parameter("do_voxel_filtering", true);
    voxel_size_ = this->declare_parameter("voxel_size", 0.05f);
    min_point_count_ = this->declare_parameter("min_point_count", 5000.0f);
    max_point_count_ = this->declare_parameter("max_point_count", 12000.0f);
    min_voxel_size_ = this->declare_parameter("min_voxel_size", 0.01f);
    max_voxel_size_ = this->declare_parameter("max_voxel_size", 0.3f);
    // Statistical outlier removal
    do_sor_ = this->declare_parameter("do_sor_", true);
    sor_mean_k_ = this->declare_parameter("sor_mean_k", 100);
    min_sor_k_ = this->declare_parameter("min_sor_k", 10);
    max_sor_k_ = this->declare_parameter("max_sor_k", 80);
    sor_stddev_mul_thresh_ = this->declare_parameter("sor_stddev_mul_thresh", 1.0f);
    min_sor_stddev_mul_thresh_ = this->declare_parameter("min_sor_stddev_mul_thresh", 0.6f);
    max_sor_stddev_mul_thresh_ = this->declare_parameter("max_sor_stddev_mul_thresh", 0.6f);
    // Initialize semantic mapper logic.
    semantic_map_ = std::make_unique<SemanticObjectMapV5>();
    // Initialize camera intrinsic state.
    fx_ = fy_ = cx_ = cy_ = 0.0;
    intrinsics_ready_ = false;
    // Initialize timing window state.
    last_timing_print_ = std::chrono::steady_clock::now();
    // Initialize TF interfaces.
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    // Use sensor-data QoS profile.
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);
    // Setup subscriptions.
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        cm_info_topic, qos, // /camera/camera/aligned_depth_to_color/camera_info
        std::bind(&PointCloudMapperNodeV5::camera_info_cb, this, _1)
    ); 
    text_emb_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        text_emb_topic, 10,
        std::bind(&PointCloudMapperNodeV5::text_embedding_cb, this, _1)
    );
    mask_sub_.subscribe(this, dm_topic_, qos.get_rmw_qos_profile());
    depth_sub_.subscribe(this, depth_topic, qos.get_rmw_qos_profile());
    // Setup approximate-time synchronizer.
    sync_ = std::make_shared<Sync>(SyncPolicy(10), mask_sub_, depth_sub_);
    sync_->registerCallback(std::bind(&PointCloudMapperNodeV5::synced_detection_callback, this, _1, _2));

    // Setup publishers and periodic maintenance timer.
    map_pub_ = this->create_publisher<yolo11_seg_interfaces::msg::SemanticObjectArray>("/vision/semantic_map_v5", 10);
    if (viewer_enabled_) {
        stable_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(stable_pointcloud_topic_, 10);
    }
    export_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(export_interval_),
        std::bind(&PointCloudMapperNodeV5::export_callback, this));
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
    std::lock_guard<std::mutex> lock(mutex_);
    RCLCPP_INFO(this->get_logger(),
        "[mapper_node_v5:text_embedding] received vector size=%zu", msg->data.size());
    // Expect embedding values followed by scale and bias (last two elements of the vector).
    if (msg->data.size() > 2) {
        float bias = msg->data.back();
        float scale = msg->data[msg->data.size() - 2];
        std::vector<float> emb(msg->data.begin(), msg->data.end() - 2);
        semantic_map_->set_text_embedding(emb, scale, bias);
        RCLCPP_INFO(this->get_logger(),
            "[mapper_node_v5:text_embedding] applied embedding_len=%zu scale=%.3f bias=%.3f",
            emb.size(), scale, bias);
    } else {
        RCLCPP_WARN(this->get_logger(),
            "[mapper_node_v5:text_embedding] ignored short message size=%zu", msg->data.size());
    }
}

void PointCloudMapperNodeV5::synced_detection_callback(
    const yolo11_seg_interfaces::msg::DetectedObjectV3Array::ConstSharedPtr mask_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr depth_msg) 
{
    auto t_total_start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex_);
    double depth_secs = rclcpp::Time(depth_msg->header.stamp).seconds();
    double mask_secs = rclcpp::Time(mask_msg->header.stamp).seconds();
    double delta_secs = std::abs(mask_secs - depth_secs);
    RCLCPP_INFO(this->get_logger(),
        "[mapper_node_v5:sync] enter detections=%zu depth_stamp=%.3f mask_stamp=%.3f delta=%.3f depth_frame=%s mask_frame=%s",
        mask_msg->detections.size(),
        depth_secs,
        mask_secs,
        delta_secs,
        depth_msg->header.frame_id.c_str(),
        mask_msg->header.frame_id.c_str());
    detections_seen_total_ += static_cast<int>(mask_msg->detections.size());
    if (!intrinsics_ready_) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
            "[mapper_node_v5:sync] skipping frame because camera intrinsics are not ready");
        return;
    }

    /*--------------PROCESS DEPTH---------------*/

    // Convert ROS depth image to a float matrix in meters.
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
        cv_ptr->image.convertTo(depth_m, CV_32FC1, 0.001); // Convert millimeters to meters if necessary
    } else {
        depth_m = cv_ptr->image; 
    }
    // --- 2D Bilateral Filtering on the Depth Map ---
    cv::Mat smoothed_depth;
    // Parameters for cv::bilateralFilter:
    // 1. Input image (depth_m)
    // 2. Output image (smoothed_depth)
    // 3. d: Diameter of pixel neighborhood (5 is a good balance of speed and smoothing)
    // 4. sigmaColor: Depth threshold in meters. 0.05 = 5cm. It won't blur across gaps larger than 5cm.
    // 5. sigmaSpace: Spatial threshold in pixels. How far across the 2D image to blur.
    cv::bilateralFilter(depth_m, smoothed_depth, 5, 0.05, 5.0);
    // Replace the original noisy depth map with the smoothed version
    depth_m = smoothed_depth;
    auto t_depth_end = std::chrono::steady_clock::now();
    timing_depth_conversion_.total_time_ms += std::chrono::duration<double, std::milli>(t_depth_end - t_depth_start).count();
    timing_depth_conversion_.count++;
    // Generate transform matrix for mapping depth points into the global map frame.
    const std::string source_frame = depth_msg->header.frame_id.empty() ? camera_frame_ : depth_msg->header.frame_id;
    const rclcpp::Time source_stamp(depth_msg->header.stamp);
    bool use_identity_tf = (source_frame == map_frame_); // necessary for testing with camera only
    Eigen::Affine3f tf_source_to_map = Eigen::Affine3f::Identity();
    if (!use_identity_tf) {
        try {
            const auto tf_msg = tf_buffer_->lookupTransform(
                map_frame_, source_frame, source_stamp, rclcpp::Duration::from_seconds(0.10));
            // Populate the matrix
            tf_source_to_map.translation() <<
                static_cast<float>(tf_msg.transform.translation.x),
                static_cast<float>(tf_msg.transform.translation.y),
                static_cast<float>(tf_msg.transform.translation.z);
            const Eigen::Quaternionf q(
                static_cast<float>(tf_msg.transform.rotation.w),
                static_cast<float>(tf_msg.transform.rotation.x),
                static_cast<float>(tf_msg.transform.rotation.y),
                static_cast<float>(tf_msg.transform.rotation.z));
            tf_source_to_map.linear() = q.normalized().toRotationMatrix(); // Update of the rotation compoment of the transform
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "[mapper_node_v5:tf] cannot transform %s -> %s at t=%.3f: %s",
                source_frame.c_str(), map_frame_.c_str(), source_stamp.seconds(), ex.what());
            return;
        }
    }

    /*--------------PROCESS DETECTIONS---------------*/

    std::vector<std::string> object_names;
    std::vector<std::string> tracker_ids;
    std::vector<float> confidences;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> points_cam_list;
    std::vector<std::optional<std::vector<float>>> embeddings_list_masked;
    std::vector<std::optional<std::vector<float>>> embeddings_list_unmasked;
    int accepted_detections = 0;
    // Process each detection and build a map-frame batch.
    for (size_t det_index = 0; det_index < mask_msg->detections.size(); ++det_index) {
        const auto& det = mask_msg->detections[det_index];
        const std::string det_tag = format_detection_trace(det_index, det, "enter");
        if (det.mask.width == 0 || det.mask.height == 0) {
            RCLCPP_INFO(this->get_logger(), "%s drop=no_mask_image", det_tag.c_str());
            continue;
        }
        auto t_extract_start = std::chrono::steady_clock::now();
        cv_bridge::CvImagePtr mask_ptr = cv_bridge::toCvCopy(det.mask, "mono8");
        cv::Mat cv_mask = mask_ptr->image;
        // Keep mask and depth dimensions aligned.
        if (cv_mask.size() != depth_m.size()) {
            cv::resize(cv_mask, cv_mask, depth_m.size(), 0, 0, cv::INTER_NEAREST);
        }
        auto raw_cloud = get_points_in_mask(depth_m, cv_mask);
        auto t_extract_end = std::chrono::steady_clock::now();
        timing_point_extraction_.total_time_ms += std::chrono::duration<double, std::milli>(t_extract_end - t_extract_start).count();
        timing_point_extraction_.count++;
        if (raw_cloud->points.size() < 4) {
            RCLCPP_INFO(this->get_logger(),
                "%s drop=raw_cloud_too_small points=%zu",
                det_tag.c_str(), raw_cloud->points.size());
            continue;
        }
        auto t_filter_start = std::chrono::steady_clock::now();

        /*--------------FILTERING--------------*/

        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr sor_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr clean_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        // Adapt voxel size and statistical outlier removal parameters to raw cloud density.
        int num_raw_points = raw_cloud->points.size();
        if (num_raw_points < min_point_count_) {
            voxel_size_ = min_voxel_size_;
            sor_mean_k_ = min_sor_k_;
            sor_stddev_mul_thresh_ = min_sor_stddev_mul_thresh_;
        }
        else if (num_raw_points > max_point_count_){
            voxel_size_ = max_voxel_size_;
            sor_mean_k_ = max_sor_k_;
            sor_stddev_mul_thresh_ = max_sor_stddev_mul_thresh_;
        }
        else {
            float scale = (static_cast<float>(num_raw_points) - min_point_count_) / (max_point_count_ - min_point_count_); // Linear interpolation for in between values
            voxel_size_ = min_voxel_size_ + scale * (max_voxel_size_ - min_voxel_size_);
            sor_mean_k_ = static_cast<int>(min_sor_k_ + scale * (max_sor_k_ - min_sor_k_));
            sor_stddev_mul_thresh_ = min_sor_stddev_mul_thresh_ + scale * (max_sor_stddev_mul_thresh_ - min_sor_stddev_mul_thresh_);
        }
        // --- Voxel Grid Filtering ---
        // Perform distance-based filtering only if flag is set
        if (do_voxel_filtering_) {   
            // Downsample cloud before clustering.
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setInputCloud(raw_cloud);
            voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_); 
            voxel_filter.filter(*downsampled_cloud);
        } else {
            downsampled_cloud = raw_cloud;
        }
        // --- Statistical Outlier Removal ---
        if (do_sor_) {
            // Statistical outlier removal.
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(downsampled_cloud);
            sor.setMeanK(sor_mean_k_);
            sor.setStddevMulThresh(sor_stddev_mul_thresh_);
            sor.filter(*sor_cloud);
        } else {
            sor_cloud = downsampled_cloud;
        }
        *clean_cloud = *sor_cloud;
        clean_cloud->width = clean_cloud->points.size();
        clean_cloud->height = 1;
        clean_cloud->is_dense = true;
        auto t_filter_end = std::chrono::steady_clock::now();
        timing_filtering_.total_time_ms += std::chrono::duration<double, std::milli>(t_filter_end - t_filter_start).count();
        timing_filtering_.count++;
        if (clean_cloud->points.size() < 4) {
            RCLCPP_INFO(this->get_logger(),
                "%s drop=clean_cloud_too_small points=%zu",
                det_tag.c_str(), clean_cloud->points.size());
            continue;
        }
        // Transform the cloud into map frame for downstream association.
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
        // Append detection data for batch association.
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
    detections_accepted_total_ += accepted_detections;
    RCLCPP_INFO(this->get_logger(),
        "[mapper_node_v5:sync] accepted=%d seen_total=%d accepted_total=%d batch_size=%zu",
        accepted_detections, detections_seen_total_, detections_accepted_total_, points_cam_list.size());
    // Pass the prepared batch into semantic association logic.
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
    auto t_pub_start = std::chrono::steady_clock::now();
    try {
        // Refine each object cloud before duplicate cleanup.
        RCLCPP_INFO(this->get_logger(),
            "[mapper_node_v5:publish] semantic_map objects=%zu tentative=%zu before refine",
            semantic_map_->objects.size(), semantic_map_->tentative_tracks.size());
        for (const auto& [map_id, entry] : semantic_map_->objects) {
            semantic_map_->refine_object_geometry(map_id);
        }
        // Run duplicate and stale-object cleanup.
        semantic_map_->resolve_overlapping_duplicates();
        semantic_map_->remove_wrong_detections();
        publish_semantic_map();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Export/Merge error: %s", e.what());
    }
    if (publish_stable_pointcloud_enabled_) {
        publish_stable_pointcloud();
    }
    auto t_pub_end = std::chrono::steady_clock::now();
    timing_publishing_.total_time_ms += std::chrono::duration<double, std::milli>(t_pub_end - t_pub_start).count();
    timing_publishing_.count++;
    // Update total timing and print periodic stats.
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

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudMapperNodeV5::get_points_in_mask(
    const cv::Mat& depth_m, 
    const cv::Mat& binary_mask) 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    // Project mask pixels into camera-space 3D points.
    for (int v = 0; v < depth_m.rows; ++v) {
        for (int u = 0; u < depth_m.cols; ++u) {
            if (binary_mask.at<uint8_t>(v, u) > 0) {
                float z = depth_m.at<float>(v, u);
                // Keep only finite depth in a practical range.
                if (std::isfinite(z) && z >= min_range_ && z <= max_range_) {
                    pcl::PointXYZ pt;
                    pt.x = (u - cx_) * z / fx_; // Pinhole projection to get X and Y in camera space
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

void PointCloudMapperNodeV5::publish_semantic_map() {
    if (semantic_map_->objects.empty()) {
        RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:publish_map] skip: no objects");
        return;
    }
    yolo11_seg_interfaces::msg::SemanticObjectArray msg;
    for (const auto& [map_id, entry] : semantic_map_->objects) {
        yolo11_seg_interfaces::msg::SemanticObject obj_msg;
        obj_msg.object_id = map_id;
        obj_msg.name = entry.current_name;
        obj_msg.frame = entry.frame;
        obj_msg.timestamp = entry.timestamp;
        obj_msg.pose_cam.x = entry.pose_cam[0];
        obj_msg.pose_cam.y = entry.pose_cam[1];
        obj_msg.pose_cam.z = entry.pose_cam[2];
        obj_msg.pose_map.x = entry.pose_map[0];
        obj_msg.pose_map.y = entry.pose_map[1];
        obj_msg.pose_map.z = entry.pose_map[2];
        // Populate oriented bounding-box metadata.
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
}

void PointCloudMapperNodeV5::publish_stable_pointcloud() {
    if (semantic_map_->objects.empty()) {
        RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:publish_cloud] skip: no objects");
        return;
    }
    // Use XYZRGB cloud to publish per-object coloring.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    for (const auto& [map_id, entry] : semantic_map_->objects) {
        if (!entry.accumulated_points || entry.accumulated_points->empty()) continue;
        // Derive deterministic color from object key.
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
    if (colored_cloud->points.empty()) {
        RCLCPP_INFO(this->get_logger(), "[mapper_node_v5:publish_cloud] skip: empty colored cloud");
        return;
    }
    colored_cloud->width = colored_cloud->points.size();
    colored_cloud->height = 1;
    colored_cloud->is_dense = true;
    // Convert to ROS PointCloud2 and publish.
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
    // Match deterministic CRC32 coloring used in the Python pipeline.
    unsigned long crc = crc32(0L, Z_NULL, 0);
    crc = crc32(crc, reinterpret_cast<const Bytef*>(class_name.c_str()), class_name.length());
    uint8_t r = 60 + (crc & 0x7F);
    uint8_t g = 60 + ((crc >> 8) & 0x7F);
    uint8_t b = 60 + ((crc >> 16) & 0x7F);
    return {r, g, b};
}

void PointCloudMapperNodeV5::export_callback() {
    std::lock_guard<std::mutex> lock(mutex_);
    RCLCPP_INFO(this->get_logger(),
        "[mapper_node_v5:heartbeat] intrinsics=%s frames=%d seen=%d accepted=%d objects=%zu tentative=%zu map_pub=%s cloud_pub=%s",
        intrinsics_ready_ ? "ready" : "missing",
        frame_count_,
        detections_seen_total_,
        detections_accepted_total_,
        semantic_map_ ? semantic_map_->objects.size() : 0,
        semantic_map_ ? semantic_map_->tentative_tracks.size() : 0,
        map_pub_ ? "ready" : "missing",
        (viewer_enabled_ && stable_cloud_pub_) ? "ready" : "disabled");
    if (frame_count_ == 0) {
        RCLCPP_WARN(this->get_logger(),
            "[mapper_node_v5:heartbeat] no completed synced callbacks yet; check topic names, sync timing, depth/frame ids, and TF availability");
    }
}

void PointCloudMapperNodeV5::print_timing_stats() {
    if (frame_count_ == 0) return;
    struct StepRow {
        const char* name;
        const TimingStats& stats;
    };
    const std::array<StepRow, 6> rows = {{
        {"Depth Conversion", timing_depth_conversion_},
        {"Point Extraction", timing_point_extraction_},
        {"Filtering (Voxel+Cluster)", timing_filtering_},
        {"Transform to Map", timing_transformation_},
        {"Batch Addition (Matching)", timing_batch_addition_},
        {"Publishing", timing_publishing_}
    }};
    const double total_window_ms = timing_total_.total_time_ms;
    const double avg_total_frame_ms = timing_total_.average();
    const double fps = avg_total_frame_ms > 1e-6 ? (1000.0 / avg_total_frame_ms) : 0.0;
    // Print header
    std::cout << "\n" << std::string(118, '=') << "\n";
    std::cout << "TIMING STATISTICS (30-second window, " << frame_count_ << " frames)\n";
    std::cout << "Total avg frame time: " << std::fixed << std::setprecision(3) << avg_total_frame_ms
              << " ms  |  Effective FPS: " << fps << "\n";
    std::cout << "Detections seen: " << detections_seen_total_
              << "  accepted: " << detections_accepted_total_;
    if (detections_seen_total_ > 0) {
        const double accept_rate = 100.0 * static_cast<double>(detections_accepted_total_) /
                                   static_cast<double>(detections_seen_total_);
        std::cout << "  (accept rate: " << accept_rate << "%)";
    }
    std::cout << "\n";
    std::cout << std::string(118, '=') << "\n";
    std::cout << std::left
              << std::setw(30) << "Step"
              << std::setw(14) << "Calls"
              << std::setw(13) << "Calls/Frame"
              << std::setw(16) << "Avg/Call (ms)"
              << std::setw(16) << "Total (ms)"
              << std::setw(16) << "Avg/Frame (ms)"
              << std::setw(11) << "%Total"
              << "\n";
    std::cout << std::string(118, '-') << "\n";
    std::cout << std::fixed << std::setprecision(3);
    double step_total_sum_ms = 0.0;
    for (const auto& row : rows) {
        const double calls_per_frame = static_cast<double>(row.stats.count) / static_cast<double>(frame_count_);
        const double avg_call_ms = row.stats.average();
        const double avg_frame_ms = row.stats.total_time_ms / static_cast<double>(frame_count_);
        const double pct_total = total_window_ms > 1e-6 ? (100.0 * row.stats.total_time_ms / total_window_ms) : 0.0;
        step_total_sum_ms += row.stats.total_time_ms;
        std::cout << std::left
                  << std::setw(30) << row.name
                  << std::setw(14) << row.stats.count
                  << std::setw(13) << calls_per_frame
                  << std::setw(16) << avg_call_ms
                  << std::setw(16) << row.stats.total_time_ms
                  << std::setw(16) << avg_frame_ms
                  << std::setw(11) << pct_total
                  << "\n";
    }
    const double unaccounted_ms = std::max(0.0, total_window_ms - step_total_sum_ms);
    const double unaccounted_avg_frame_ms = unaccounted_ms / static_cast<double>(frame_count_);
    const double unaccounted_pct = total_window_ms > 1e-6 ? (100.0 * unaccounted_ms / total_window_ms) : 0.0;
    std::cout << std::string(118, '-') << "\n";
    std::cout << std::left
              << std::setw(30) << "Unaccounted (other work)"
              << std::setw(14) << "-"
              << std::setw(13) << "-"
              << std::setw(16) << "-"
              << std::setw(16) << unaccounted_ms
              << std::setw(16) << unaccounted_avg_frame_ms
              << std::setw(11) << unaccounted_pct
              << "\n";
    std::cout << std::string(118, '-') << "\n";
    std::cout << std::left
              << std::setw(30) << "TOTAL callback"
              << std::setw(14) << timing_total_.count
              << std::setw(13) << 1.0
              << std::setw(16) << avg_total_frame_ms
              << std::setw(16) << total_window_ms
              << std::setw(16) << avg_total_frame_ms
              << std::setw(11) << 100.0
              << "\n";
    std::cout << std::string(118, '=') << "\n\n";
    // Reset counters for the next reporting window.
    timing_depth_conversion_.reset();
    timing_point_extraction_.reset();
    timing_filtering_.reset();
    timing_transformation_.reset();
    timing_batch_addition_.reset();
    timing_publishing_.reset();
    timing_total_.reset();
    frame_count_ = 0;
    detections_seen_total_ = 0;
    detections_accepted_total_ = 0;
}

// Node entry point.
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudMapperNodeV5>();
    // Multi-threaded executor lets timer and sync callback run concurrently.
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    try {
        executor.spin();
    } catch (const std::exception& e) {
        // Exit cleanly on executor exceptions or interrupts.
    }
    rclcpp::shutdown();
    return 0;
}