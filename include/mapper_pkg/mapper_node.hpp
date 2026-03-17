#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <yolo11_seg_interfaces/msg/detected_object_v3_array.hpp>

#include "semantic_mapper_cpp/semantic_object_map.hpp"

class MapperNode : public rclcpp::Node {
public:
    MapperNode();
    ~MapperNode() = default;

private:
    // This alias is standard in ROS 2. It prevents the Synchronizer definition from 
    // becoming completely unreadable.
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        yolo11_seg_interfaces::msg::DetectedObjectV3Array
    >;

    // ---------------------------------------------------------
    // ROS 2 Comms
    // ---------------------------------------------------------
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    message_filters::Subscriber<yolo11_seg_interfaces::msg::DetectedObjectV3Array> det_sub_;
    std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    // ---------------------------------------------------------
    // TF2 (Odometry / Transforms)
    // ---------------------------------------------------------
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ---------------------------------------------------------
    // Background Timers & Core Math
    // ---------------------------------------------------------
    rclcpp::TimerBase::SharedPtr maintenance_timer_;
    SemanticObjectMap mapper_; 

    // ---------------------------------------------------------
    // Callbacks
    // ---------------------------------------------------------
    void syncedCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
        const yolo11_seg_interfaces::msg::DetectedObjectV3Array::ConstSharedPtr& det_msg
    );
    
    void maintenanceCallback();
};