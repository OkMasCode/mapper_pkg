#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

// Import your custom message
#include "yolo11_seg_interfaces/msg/similarity.hpp"
#include "yolo11_seg_interfaces/msg/similarity_centroid_array.hpp"
#include "yolo11_seg_interfaces/msg/similarity_centroid.hpp"

#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <mutex>
#include <cmath>
#include <cmath>
#include <vector>

using std::placeholders::_1;

struct FrontierCluster {
    grid_map::Position center;
    int point_count;
    float avg_similarity;
};

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
        pub_frontiers_ = this->create_publisher<visualization_msgs::msg::Marker>("/frontiers", 1);
        pub_clusters_ = this->create_publisher<visualization_msgs::msg::Marker>("/clusters", 1);
        pub_cluster_data_ = this->create_publisher<yolo11_seg_interfaces::msg::SimilarityCentroidArray>("/similarity_centroids_data", 1);
        RCLCPP_INFO(this->get_logger(), "Semantic Mapper Node Initialized with Custom Similarity Message");
    }

private:
    grid_map::GridMap semantic_map_;
    std::mutex map_mutex_;
    bool map_initialized_;
    
    double fov_rad_;
    double max_depth_ = 2.5; // Stop projecting scores after 5 meters
    double alpha_ = 0.6;     // Exponential Moving Average weight

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
    rclcpp::Subscription<yolo11_seg_interfaces::msg::Similarity>::SharedPtr sub_sim_;
    rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr pub_grid_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_frontiers_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_clusters_;
    rclcpp::Publisher<yolo11_seg_interfaces::msg::SimilarityCentroidArray>::SharedPtr pub_cluster_data_;
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

    std::vector<grid_map::Position> extractSemanticFrontiers(
        double safe_distance_m = 0.25,      // TUNING 2: Kept small so we don't clip corners
        int min_unknown_neighbors = 2)      // TUNING 1: Require at least 2 unknown cells to ignore "holes"
    {
        std::vector<grid_map::Position> raw_frontiers;
        std::vector<grid_map::Position> safe_frontiers;

        std::lock_guard<std::mutex> lock(map_mutex_);

        // Check if map and similarity layer exist
        if (!map_initialized_ || !semantic_map_.exists("similarity")) {
            return safe_frontiers; 
        }

        // ==========================================
        // STEP 1: Find Valid Semantic Frontiers (Filtering out noise)
        // ==========================================
        for (grid_map::GridMapIterator it(semantic_map_); !it.isPastEnd(); ++it) {
            grid_map::Index index = *it;

            // 1. The cell itself must be explored (have a similarity score)
            if (!semantic_map_.isValid(index, "similarity")) continue; 

            // 2. The cell itself must NOT be a wall
            if (semantic_map_.isValid(index, "occupancy") && 
                semantic_map_.at("occupancy", index) >= 50.0) {
                continue; 
            }

            int valid_unknown_count = 0;

            // Check the 8 surrounding cells
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    if (dx == 0 && dy == 0) continue; 

                    grid_map::Index neighbor_idx(index(0) + dx, index(1) + dy);

                    if (semantic_map_.isValid(neighbor_idx)) {
                        // Check if neighbor lacks a similarity score
                        if (!semantic_map_.isValid(neighbor_idx, "similarity")) {
                            
                            // Ensure this unknown neighbor isn't just the inside of a solid wall
                            bool is_wall = semantic_map_.isValid(neighbor_idx, "occupancy") && 
                                        semantic_map_.at("occupancy", neighbor_idx) >= 50.0;

                            if (!is_wall) {
                                valid_unknown_count++; // We found a valid piece of the true frontier!
                            }
                        }
                    } else {
                        // If the neighbor is completely off the map grid, count it as unknown space
                        valid_unknown_count++;
                    }
                }
            }

            // TUNING 1 APPLIED: Only keep it if it touches enough unknown space
            if (valid_unknown_count >= min_unknown_neighbors) {
                grid_map::Position pos;
                semantic_map_.getPosition(index, pos);
                raw_frontiers.push_back(pos);
            }
        }

        // ==========================================
        // STEP 2: Filter with a relaxed Safety Radius
        // ==========================================
        for (const auto& pos : raw_frontiers) {
            bool too_close_to_wall = false;

            // Check a smaller circle to prevent deleting frontiers near walls
            for (grid_map::CircleIterator circle_it(semantic_map_, pos, safe_distance_m);
                !circle_it.isPastEnd(); ++circle_it) {

                if (semantic_map_.isValid(*circle_it, "occupancy")) {
                    if (semantic_map_.at("occupancy", *circle_it) >= 50.0) {
                        too_close_to_wall = true;
                        break;
                    }
                }
            }

            if (!too_close_to_wall) {
                safe_frontiers.push_back(pos);
            }
        }

        return safe_frontiers;
    }

    std::vector<FrontierCluster> clusterFrontiers(
    const std::vector<grid_map::Position>& raw_frontiers, 
    double clustering_radius_m = 0.5)  // Group points within 0.5 meters
    {
        std::vector<FrontierCluster> clusters;

        // Lock the map so we can safely read the similarity scores
        std::lock_guard<std::mutex> lock(map_mutex_);

        for (const auto& pos : raw_frontiers) {
            
            // Read the similarity for this specific point
            grid_map::Index index;
            semantic_map_.getIndex(pos, index);
            float sim_val = semantic_map_.at("similarity", index);
            
            // Safety check in case a NaN slipped through
            if (std::isnan(sim_val)) sim_val = 0.0;

            bool added_to_existing = false;

            // Check if this point belongs to any existing cluster
            for (auto& cluster : clusters) {
                // Calculate straight-line distance using Eigen's norm()
                double distance = (pos - cluster.center).norm();

                if (distance <= clustering_radius_m) {
                    // Update the cluster's physical center (Moving Average)
                    cluster.center = (cluster.center * cluster.point_count + pos) / (cluster.point_count + 1.0);
                    
                    // Update the average similarity score
                    cluster.avg_similarity = (cluster.avg_similarity * cluster.point_count + sim_val) / (cluster.point_count + 1.0);
                    
                    cluster.point_count++;
                    added_to_existing = true;
                    break;
                }
            }

            // If it's too far from all existing clusters, create a brand new one
            if (!added_to_existing) {
                clusters.push_back({pos, 1, sim_val});
            }
        }

        // Filter out tiny clusters (e.g., 1 or 2 stray points)
        std::vector<FrontierCluster> valid_clusters;
        for (const auto& cluster : clusters) {
            if (cluster.point_count >= 3) { 
                valid_clusters.push_back(cluster);
            }
        }

        return valid_clusters;
    }

    void publishClusterMarkers(const std::vector<FrontierCluster>& clusters) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.ns = "clusters";
        marker.id = 1; // Different ID from the raw frontiers
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Make these spheres larger so they stand out as target destinations
        marker.scale.x = 0.4; 
        marker.scale.y = 0.4;
        marker.scale.z = 0.4;

        for (const auto& cluster : clusters) {
            geometry_msgs::msg::Point p;
            p.x = cluster.center.x();
            p.y = cluster.center.y();
            p.z = 0.2; // Slightly higher than raw frontiers
            marker.points.push_back(p);

            // Color the cluster based on its AVERAGE similarity score
            std_msgs::msg::ColorRGBA color;
            color.a = 1.0; 
            color.r = 1.0 - cluster.avg_similarity; 
            color.g = cluster.avg_similarity;       
            color.b = 1.0; // Adding a bit of blue to make them distinct from raw points
            
            marker.colors.push_back(color);
        }

        pub_clusters_->publish(marker); // Ensure you initialized this publisher in the constructor!
    }

    void publishFrontierMarkers(const std::vector<grid_map::Position>& frontiers) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.ns = "frontiers";
        marker.id = 0;
        
        // SPHERE_LIST is highly optimized for drawing many points at once
        marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Set the size of the spheres (e.g., 15cm wide)
        marker.scale.x = 0.15;
        marker.scale.y = 0.15;
        marker.scale.z = 0.15;

        // We must lock the map to safely read the similarity values
        std::lock_guard<std::mutex> lock(map_mutex_);

        for (const auto& pos : frontiers) {
            // 1. Set the physical position
            geometry_msgs::msg::Point p;
            p.x = pos.x();
            p.y = pos.y();
            p.z = 0.1; // Lift slightly off the floor so it doesn't clip into the map
            marker.points.push_back(p);

            // 2. Read the similarity value from the map
            grid_map::Index index;
            semantic_map_.getIndex(pos, index);
            float similarity = semantic_map_.at("similarity", index);

            // 3. Map the similarity score to a color (Example: 0.0 to 1.0)
            // Adjust these math bounds based on whatever scale your YOLO model outputs!
            std_msgs::msg::ColorRGBA color;
            color.a = 1.0; // Fully opaque
            
            // Simple color mapping: High similarity -> Green, Low -> Red
            if (std::isnan(similarity)) {
                // Fallback for safety, though your extract function should prevent this
                color.r = 0.5; color.g = 0.5; color.b = 0.5; 
            } else {
                color.r = 1.0 - similarity; // Low similarity gives high red
                color.g = similarity;       // High similarity gives high green
                color.b = 0.0;
            }
            
            marker.colors.push_back(color);
        }

        pub_frontiers_->publish(marker);
    }

    void publishClusterData(const std::vector<FrontierCluster>& clusters) {
        yolo11_seg_interfaces::msg::SimilarityCentroidArray cluster_array;
        cluster_array.header.frame_id = "map";
        cluster_array.header.stamp = this->now();

        for (const auto& cluster : clusters) {
            yolo11_seg_interfaces::msg::SimilarityCentroid custom_cluster;
            
            // Populate standard X and Y
            custom_cluster.position.x = cluster.center.x();
            custom_cluster.position.y = cluster.center.y();
            custom_cluster.position.z = 0.0; 
            
            // Populate your dedicated similarity field cleanly!
            custom_cluster.similarity = cluster.avg_similarity; 
            
            cluster_array.clusters.push_back(custom_cluster);
        }
        
        pub_cluster_data_->publish(cluster_array);
    }

    void publishMap() {
        if (!map_initialized_) return;

        std::vector<grid_map::Position> my_frontiers = extractSemanticFrontiers(0.25, 3);
        std::vector<FrontierCluster> my_clusters = clusterFrontiers(my_frontiers, 0.6);
        publishFrontierMarkers(my_frontiers);
        publishClusterMarkers(my_clusters);
        publishClusterData(my_clusters);
        RCLCPP_INFO(this->get_logger(), "Found %zu safe frontier points!", my_frontiers.size());

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