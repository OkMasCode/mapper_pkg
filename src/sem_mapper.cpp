#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

// Import your custom message
#include "yolo11_seg_interfaces/msg/similarity.hpp"

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <mutex>
#include <cmath>

using std::placeholders::_1;

class SemanticMapper : public rclcpp::Node {
public:
    SemanticMapper() : Node("semantic_mapper"), map_initialized_(false), fov_rad_(1.4) // Default ~80 deg
    {
        // 1. Initialize TF2
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);

        // 2. Setup Subscribers
        sub_map_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/map", qos, 
            std::bind(&SemanticMapper::mapCallback, this, _1));

        sub_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/jackal/sensors/camera_0/aligned_depth_to_color/camera_info", qos, 
            std::bind(&SemanticMapper::infoCallback, this, _1));

        // Subscribe using your custom message type
        sub_sim_ = this->create_subscription<yolo11_seg_interfaces::msg::Similarity>(
            "/vision/scene_similarity_raw", 10, 
            std::bind(&SemanticMapper::similarityCallback, this, _1));

        // 3. Setup Publisher & Timer (Publishing the map at a steady 5 Hz)
        pub_grid_ = this->create_publisher<grid_map_msgs::msg::GridMap>("/semantic_grid_map", 1);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(200), std::bind(&SemanticMapper::publishMap, this));

        RCLCPP_INFO(this->get_logger(), "Semantic Mapper Node Initialized with Custom Similarity Message");
    }

private:
    grid_map::GridMap semantic_map_;
    std::mutex map_mutex_;
    bool map_initialized_;
    
    double fov_rad_;
    double max_depth_ = 3.0; // Stop projecting scores after 5 meters
    double alpha_ = 0.6;     // Exponential Moving Average weight

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
    rclcpp::Subscription<yolo11_seg_interfaces::msg::Similarity>::SharedPtr sub_sim_;
    rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr pub_grid_;
    rclcpp::TimerBase::SharedPtr timer_;

    // --- CALLBACKS ---

    void infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        // Dynamically calculate horizontal FOV from the intrinsic matrix
        if (msg->k[0] > 0.0) {
            fov_rad_ = 2.0 * atan(msg->width / (2.0 * msg->k[0]));
        }
    }

    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(map_mutex_);

        grid_map::GridMap temp_map;
        grid_map::GridMapRosConverter::fromOccupancyGrid(*msg, "occupancy", temp_map);

        if (!map_initialized_) {
            semantic_map_ = temp_map;
            semantic_map_.add("similarity", NAN); 
            
            // FIX: Remove "similarity" from this list! 
            semantic_map_.setBasicLayers({"occupancy"}); 
            
            map_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "Base Occupancy Map ingested successfully.");
        } else {
            // Update physical obstacles layer, leaving similarities untouched
            semantic_map_.add("occupancy", temp_map.get("occupancy"));
        }
    }

    void similarityCallback(const yolo11_seg_interfaces::msg::Similarity::SharedPtr msg) {
        if (!map_initialized_) return;

        std::lock_guard<std::mutex> lock(map_mutex_);
        double score = msg->similarity; 

        geometry_msgs::msg::TransformStamped transform;
        try {
            transform = tf_buffer_->lookupTransform(
                "map", msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.1)
            );
        } catch (const tf2::TransformException & ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "TF dropped frame! %s", ex.what());
            return;
        }

        // 1. Get the Camera Lens (Origin) in the global map
        geometry_msgs::msg::Pose p_lens, g_lens;
        p_lens.position.x = 0; p_lens.position.y = 0; p_lens.position.z = 0;
        tf2::doTransform(p_lens, g_lens, transform);
        grid_map::Position start_pos(g_lens.position.x, g_lens.position.y);

        // 2. Create a "Visited" mask to prevent double-applying EMA to overlapping rays
        int rows = semantic_map_.getSize()(0);
        int cols = semantic_map_.getSize()(1);
        std::vector<bool> visited(rows * cols, false);

        // 3. RAYCASTING: Sweep from -half_fov to +half_fov
        double half_fov = fov_rad_ / 2.0;
        
        // Calculate how many rays we need so we don't miss any cells at max_depth
        double map_resolution = semantic_map_.getResolution();
        double angular_resolution = map_resolution / max_depth_; // Roughly 0.01 radians
        
        for (double angle = -half_fov; angle <= half_fov; angle += angular_resolution) {
            
            // Define the end of this specific ray in the camera's optical frame
            // Z is forward depth, X is horizontal sweep
            geometry_msgs::msg::Pose p_end, g_end;
            p_end.position.x = max_depth_ * tan(angle); 
            p_end.position.y = 0.0; 
            p_end.position.z = max_depth_;

            // Transform the end of the ray to the global map
            tf2::doTransform(p_end, g_end, transform);
            grid_map::Position end_pos(g_end.position.x, g_end.position.y);

            // Shoot the ray!
            for (grid_map::LineIterator iterator(semantic_map_, start_pos, end_pos); !iterator.isPastEnd(); ++iterator) {
                grid_map::Index index = *iterator;

                // Stop the ray if it goes off the map
                if (!semantic_map_.isValid(index)) break; 

                // Check the SLAM map: Stop the ray if it hits a wall or unknown space!
                float occ_val = semantic_map_.at("occupancy", index);
                if (occ_val > 50.0 || occ_val == -1.0) {
                    break; // THIS IS THE MAGIC! The ray hits a wall and dies instantly.
                }

                // Calculate the 1D index for our visited mask
                int flat_index = index(0) * cols + index(1);
                
                // Only apply math if we haven't touched this cell during this specific frame
                if (!visited[flat_index]) {
                    visited[flat_index] = true; // Mark as visited

                    // Progressive Averaging (Exponential Moving Average)
                    float current_score = semantic_map_.at("similarity", index);
                    if (std::isnan(current_score)) {
                        semantic_map_.at("similarity", index) = score;
                    } else {
                        semantic_map_.at("similarity", index) = (alpha_ * score) + ((1.0 - alpha_) * current_score);
                    }
                }
            }
        }
    }

    void publishMap() {
        if (!map_initialized_) return;

        std::lock_guard<std::mutex> lock(map_mutex_);
        
        // FIX: In ROS 2 Humble, toMessage() directly returns a unique_ptr
        auto message = grid_map::GridMapRosConverter::toMessage(semantic_map_);
        
        // Stamp the published map with the current time
        message->header.stamp = this->now();
        message->header.frame_id = "map";
        
        pub_grid_->publish(std::move(message));
    }
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SemanticMapper>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}