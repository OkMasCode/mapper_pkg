#ifndef POINTCLOUD_MAPPER_NODE_V5_HPP_
#define POINTCLOUD_MAPPER_NODE_V5_HPP_

// Standard C++ Libraries
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <chrono>

// ROS 2 Core
#include <rclcpp/rclcpp.hpp>

// ROS 2 Standard Messages
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/vector3.hpp>

// TF2 for Coordinate Transformations (Camera -> Map)
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// OpenCV & CV Bridge for Image manipulation
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

// Message Filters for Exact/Approximate Time Synchronization
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// PCL (Point Cloud Library) for built-in 3D processing
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Custom Messages (These must match your package generation)
#include "yolo11_seg_interfaces/msg/detected_object_v3_array.hpp"
#include "yolo11_seg_interfaces/msg/semantic_object_array.hpp"
#include "yolo11_seg_interfaces/msg/semantic_object.hpp"

// Forward declaration of the mapper logic class to keep this header clean
class SemanticObjectMapV5;

// Timing statistics structure for tracking step performance
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
    // Constructor: Sets up parameters, topics, and memory
    PointCloudMapperNodeV5();
    
    // Destructor
    ~PointCloudMapperNodeV5() override = default;

    // Called during Ctrl+C to ensure the final map is exported safely
    void shutdown_callback();

private:
    // ==========================================
    // ROS 2 CALLBACKS
    // ==========================================
    
    // Grabs the fx, fy, cx, cy lens parameters required for 3D projection
    void camera_info_cb(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

    // The main loop: Fires when YOLO masks and Depth images arrive together
    void synced_detection_callback(
        const yolo11_seg_interfaces::msg::DetectedObjectV3Array::ConstSharedPtr mask_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr depth_msg);

    // Periodic timer to trigger map cleanup and JSON export routines
    void export_callback();

    // ==========================================
    // HELPER FUNCTIONS
    // ==========================================
    
    // Projects a 2D binary mask and Depth image into a 3D PCL Point Cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr get_points_in_mask(
        const cv::Mat& depth_m, 
        const cv::Mat& binary_mask);

    // Translates the C++ mapper state into custom ROS 2 messages
    void publish_semantic_map();

    // Publishes all accumulated object points as a colored cloud for RViz
    void publish_stable_pointcloud();

    // Generates a deterministic, bright RGB color based on the class name
    std::tuple<uint8_t, uint8_t, uint8_t> class_to_color_rgb(const std::string& class_name);

    // ==========================================
    // MEMBER VARIABLES
    // ==========================================

    // Node Parameters
    std::string dm_topic_;
    std::string map_frame_;
    std::string camera_frame_;
    std::string output_dir_;
    std::string output_map_file_;
    std::string stable_pointcloud_topic_;
    bool viewer_enabled_;

    double export_interval_;
    bool publish_stable_pointcloud_enabled_;
    
    // Camera Intrinsics
    double fx_, fy_, cx_, cy_;
    bool intrinsics_ready_;

    // Thread Safety: Prevents the export timer and camera callback from crashing each other
    std::mutex mutex_;

    // Pointer to the core logic class that tracks and merges objects
    std::unique_ptr<SemanticObjectMapV5> semantic_map_;

    // TF2 Transforms
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ROS 2 Subscribers
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    message_filters::Subscriber<yolo11_seg_interfaces::msg::DetectedObjectV3Array> mask_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    
    // ROS 2 Synchronizer Definition (Matches masks with depth based on timestamps)
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        yolo11_seg_interfaces::msg::DetectedObjectV3Array, 
        sensor_msgs::msg::Image>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;
    std::shared_ptr<Sync> sync_;

    // ROS 2 Publishers & Timers
    rclcpp::Publisher<yolo11_seg_interfaces::msg::SemanticObjectArray>::SharedPtr map_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr stable_cloud_pub_;
    rclcpp::TimerBase::SharedPtr export_timer_;

    // Timing statistics
    TimingStats timing_depth_conversion_;
    TimingStats timing_point_extraction_;
    TimingStats timing_filtering_;
    TimingStats timing_transformation_;
    TimingStats timing_batch_addition_;
    TimingStats timing_publishing_;
    TimingStats timing_total_;
    
    int frame_count_ = 0;
    std::chrono::steady_clock::time_point last_timing_print_;
    
    void print_timing_stats();
};

#endif // POINTCLOUD_MAPPER_NODE_V5_HPP_