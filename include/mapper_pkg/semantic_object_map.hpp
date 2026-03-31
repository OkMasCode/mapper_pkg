#ifndef SEMANTIC_OBJECT_MAP_HPP_
#define SEMANTIC_OBJECT_MAP_HPP_

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <tuple>
#include <optional>
#include <array>

// ROS 2 Time and Transforms
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

// PCL (Point Cloud Library)
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Forward declaration of the OBB struct that we will implement in the .cpp file
// based on the robinvista/pointcloud-OBB logic.
struct OrientedBoundingBox {
    std::vector<float> center{0.0f, 0.0f, 0.0f};
    std::vector<float> extents{0.0f, 0.0f, 0.0f}; // length, width, height
    // Row-major 3x3 rotation matrix for local->world transform.
    std::vector<float> rotation{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
};

// ==========================================
// 1. DATA STRUCTURES (Replaces Python @dataclass)
// ==========================================

struct MapObject {
    std::string map_id;
    std::string frame;
    builtin_interfaces::msg::Time timestamp;
    
    // Geometry
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_points;
    OrientedBoundingBox obb; 
    
    // Tracking & Lifecycle
    int occurrences = 0;
    long long first_seen_ns = 0;
    long long last_seen_ns = 0;
    
    // Semantic Voting
    std::string current_name;
    std::unordered_map<std::string, float> class_votes;
    std::unordered_map<std::string, int> class_counts;
    std::unordered_map<std::string, float> class_conf_sums;
    
    // Embeddings & Confidence
    float similarity = 0.0f;
    float confidence_ema = 0.0f;
    std::vector<float> image_embedding_masked;
    std::vector<float> image_embedding_unmasked;
    float embedding_confidence_max = -1.0f;
    std::string source_track_id;

    // Cached state to prevent recomputing for ROS messages
    std::vector<float> pose_cam{0.0f, 0.0f, 0.0f};
    std::vector<float> pose_map{0.0f, 0.0f, 0.0f};
};

struct TentativeTrack {
    std::string track_id;
    std::string frame;
    builtin_interfaces::msg::Time timestamp;
    
    // Geometry
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_points;
    
    // Tracking
    int hits = 0;
    long long first_seen_ns = 0;
    long long last_seen_ns = 0;
    
    // Semantics
    std::string class_name;
    float confidence_max = 0.0f;
    float confidence_sum = 0.0f;
    std::vector<float> image_embedding_masked;
    std::vector<float> image_embedding_unmasked;
    float embedding_confidence_max = -1.0f;
};

// ==========================================
// 2. THE MAPPER ENGINE
// ==========================================

class SemanticObjectMapV5 {
public:
    SemanticObjectMapV5();
    ~SemanticObjectMapV5() = default;

    // The primary memory banks
    std::unordered_map<std::string, MapObject> objects;
    std::unordered_map<std::string, TentativeTrack> tentative_tracks;
    
    // Tracker relationships
    std::unordered_map<std::string, std::string> track_to_map;
    std::unordered_map<std::string, long long> track_last_seen_ns;

    // Tuning Parameters (Mapped directly from Python)
    float min_input_confidence = 0.55f;
    float min_detection_depth_m = 0.25f;
    float max_detection_depth_m = 6.0f;
    int confirmation_min_hits = 6;
    float confirmation_time_window_sec = 2.5f;
    float confirmation_min_age_sec = 0.8f;
    float min_confidence_for_promotion = 0.50f;
    float min_avg_confidence_for_promotion = 0.55f;
    float tentative_max_stale_sec = 2.0f;
    float binding_ttl_sec = 4.0f;
    float confidence_ema_alpha = 0.20f;
    
    float class_count_weight = 1.0f;
    float class_confidence_weight = 2.0f;
    float class_switch_margin = 0.75f;
    int min_class_votes_to_lock = 4;

    // --- Core Public Interface ---

    void add_detections_batch(
        const std::vector<std::string>& object_names,
        const std::vector<std::string>& tracker_ids,
        const std::vector<float>& confidences,
        const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& points_cam_list,
        const std::vector<std::optional<std::vector<float>>>& embeddings_list_masked,
        const std::vector<std::optional<std::vector<float>>>& embeddings_list_unmasked,
        const builtin_interfaces::msg::Time& stamp,
        const std::string& camera_frame,
        const std::string& map_frame);

    void resolve_overlapping_duplicates();
    
    void export_to_json(const std::string& directory_path, const std::string& file);

    void refine_object_geometry(const std::string& map_id);

    std::array<std::array<float, 3>, 8> compute_obb_corners(const OrientedBoundingBox& obb) const;
    
    void set_text_embedding(const std::vector<float>& emb, float scale, float bias);

    float get_goal_similarity(const std::string& map_id) const;

private:
    int next_map_id_ = 1;

    std::vector<float> goal_text_embedding_;
    float logit_scale_ = 1.0f;
    float logit_bias_ = 0.0f;

    // --- State Management Helpers ---
    
    void prune_stale_state(long long current_ns);
    std::string new_map_id();
    long long stamp_to_ns(const builtin_interfaces::msg::Time& stamp);

    // --- Geometry & Math Helpers ---
    
    std::vector<float> normalize_embedding(const std::vector<float>& embedding) const;
    
    std::vector<float> fuse_embeddings_running_avg(
        const std::vector<float>& current_embedding, int current_count,
        const std::vector<float>& new_embedding, int new_count);
    
    // Image-to-image cosine similarity for object correspondence
    float compute_embedding_similarity(
        const std::vector<float>& emb1,
        const std::vector<float>& emb2);

    pcl::PointCloud<pcl::PointXYZ>::Ptr fuse_geometry(
        pcl::PointCloud<pcl::PointXYZ>::Ptr old_points, 
        pcl::PointCloud<pcl::PointXYZ>::Ptr new_points);
        
    
    OrientedBoundingBox compute_obb(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    
    float compute_obb_iou(
        pcl::PointCloud<pcl::PointXYZ>::Ptr points1, const OrientedBoundingBox& obb1,
        pcl::PointCloud<pcl::PointXYZ>::Ptr points2, const OrientedBoundingBox& obb2);

    float oriented_overlap_ratio(
        pcl::PointCloud<pcl::PointXYZ>::Ptr points1, const OrientedBoundingBox& obb1,
        pcl::PointCloud<pcl::PointXYZ>::Ptr points2, const OrientedBoundingBox& obb2);

    // --- Update Logic Helpers ---

    bool update_tentative(
        const std::string& object_name,
        const std::string& tracker_id,
        pcl::PointCloud<pcl::PointXYZ>::Ptr points_map,
        const builtin_interfaces::msg::Time& detection_stamp,
        float confidence,
        const std::vector<float>& image_embedding_masked,
        const std::vector<float>& image_embedding_unmasked,
        long long current_ns,
        const std::string& frame);

    void update_object(
        const std::string& map_id,
        const std::string& object_name,
        const builtin_interfaces::msg::Time& detection_stamp,
        pcl::PointCloud<pcl::PointXYZ>::Ptr points_map,
        float similarity,
        float confidence,
        const std::vector<float>& image_embedding_masked,
        const std::vector<float>& image_embedding_unmasked,
        long long current_ns,
        const std::string& source_track_id);

    void fuse_objects(const std::string& keep_id, const std::string& drop_id);

    std::string choose_consensus_class(
        const std::unordered_map<std::string, int>& class_counts,
        const std::unordered_map<std::string, float>& class_conf_sums,
        const std::string& current_name);
};

#endif // SEMANTIC_OBJECT_MAP_HPP_