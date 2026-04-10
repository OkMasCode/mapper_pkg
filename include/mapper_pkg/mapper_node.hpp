#ifndef POINTCLOUD_MAPPER_NODE_V5_HPP_
#define POINTCLOUD_MAPPER_NODE_V5_HPP_

// Standard C++ libraries.
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <chrono>

// ROS 2 core.
#include <rclcpp/rclcpp.hpp>

// ROS 2 standard messages.
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

// TF2 transforms between sensor and map frames.
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// OpenCV and cv_bridge for image conversion.
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// Message filters for synchronized streams.
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// PCL for 3D point cloud processing.
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Package message types.
#include "yolo11_seg_interfaces/msg/detected_object_v3_array.hpp"
#include "yolo11_seg_interfaces/msg/semantic_object_array.hpp"
#include "yolo11_seg_interfaces/msg/semantic_object.hpp"

// Forward declaration to avoid including implementation details here.
class SemanticObjectMapV5;

// Rolling timing accumulator for periodic performance reporting.
struct TimingStats {
    double total_time_ms = 0.0;
    int count = 0;
    
    double average() const {
        return count > 0 ? total_time_ms / count : 0.0;
    }
    
    void reset() {
        total_time_ms = 0.0;
        count = 0;
    }
};

class PointCloudMapperNodeV5 : public rclcpp::Node {
public:
    // Initializes parameters, subscriptions, publishers, and mapper state.
    PointCloudMapperNodeV5();
    
    // Uses default destruction behavior.
    ~PointCloudMapperNodeV5() override = default;

    // Invoked on shutdown to finalize map export-related work.
    void shutdown_callback();

private:
    // Receives camera intrinsics used for depth projection.
    void camera_info_cb(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    // Receives global text embedding used for similarity scoring.
    void text_embedding_cb(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

    // Main synchronized callback for detections and depth.
    void synced_detection_callback(
        const yolo11_seg_interfaces::msg::DetectedObjectV3Array::ConstSharedPtr mask_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr depth_msg);

    // Periodic maintenance callback for cleanup and republishing.
    void export_callback();

    // Projects a binary mask and depth image to a 3D point cloud.
    pcl::PointCloud<pcl::PointXYZ>::Ptr get_points_in_mask(
        const cv::Mat& depth_m, 
        const cv::Mat& binary_mask);

    // Publishes the semantic object map message.
    void publish_semantic_map();

    // Publishes accumulated map geometry as a colored point cloud.
    void publish_stable_pointcloud();

    // Generates deterministic RGB values from a stable string key.
    std::tuple<uint8_t, uint8_t, uint8_t> class_to_color_rgb(const std::string& class_name);

    // Configurable node parameters.
    std::string dm_topic_;
    std::string map_frame_;
    std::string camera_frame_;
    std::string output_dir_;
    std::string output_map_file_;
    std::string stable_pointcloud_topic_;
    bool viewer_enabled_;

    double export_interval_;
    bool publish_stable_pointcloud_enabled_;

    // Depth range filtering parameters.
    float min_range_;
    float max_range_;

    // Voxel filtering parameters.
    bool do_voxel_filtering_;
    float voxel_size_;
    float min_point_count_;
    float max_point_count_;
    float min_voxel_size_;
    float max_voxel_size_;

    // Euclidean clustering parameters.
    float cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    
    // Camera intrinsics from CameraInfo.
    double fx_, fy_, cx_, cy_;
    bool intrinsics_ready_;

    // Guards shared mapper state across callbacks and timer threads.
    std::mutex mutex_;

    // Core semantic mapping engine.
    std::unique_ptr<SemanticObjectMapV5> semantic_map_;

    // TF2 transform interfaces.
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Input subscriptions.
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    message_filters::Subscriber<yolo11_seg_interfaces::msg::DetectedObjectV3Array> mask_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;

    // Text embedding input channel.
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr text_emb_sub_;
    
    // Synchronizer that pairs detections with depth by timestamp.
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        yolo11_seg_interfaces::msg::DetectedObjectV3Array, 
        sensor_msgs::msg::Image>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;
    std::shared_ptr<Sync> sync_;

    // Output publishers and maintenance timer.
    rclcpp::Publisher<yolo11_seg_interfaces::msg::SemanticObjectArray>::SharedPtr map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr stable_cloud_pub_;
    rclcpp::TimerBase::SharedPtr export_timer_;

    // Per-stage timing accumulators.
    TimingStats timing_depth_conversion_;
    TimingStats timing_point_extraction_;
    TimingStats timing_filtering_;
    TimingStats timing_transformation_;
    TimingStats timing_batch_addition_;
    TimingStats timing_publishing_;
    TimingStats timing_total_;
    
    int frame_count_ = 0;
    int detections_seen_total_ = 0;
    int detections_accepted_total_ = 0;
    std::chrono::steady_clock::time_point last_timing_print_;
    
    void print_timing_stats();
};

#endif // POINTCLOUD_MAPPER_NODE_V5_HPP_