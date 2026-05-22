#ifndef SEMANTIC_OBJECT_MAP_HPP_
#define SEMANTIC_OBJECT_MAP_HPP_

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <tuple>
#include <optional>
#include <array>
#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

struct OrientedBoundingBox {
    std::vector<float> center{0.0f, 0.0f, 0.0f};
    std::vector<float> extents{0.0f, 0.0f, 0.0f}; // length, width, height
    // Row-major 3x3 rotation matrix mapping local coordinates to world coordinates.
    std::vector<float> rotation{
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
};

// Persistent object entry stored in the semantic map.
struct MapObject {
    std::string map_id;
    std::string frame;
    builtin_interfaces::msg::Time timestamp;    
    // Geometry accumulated over multiple observations.
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_points;
    OrientedBoundingBox obb; 
    // Lifecycle statistics used for stability and pruning.
    int occurrences = 0;
    long long first_seen_ns = 0;
    long long last_seen_ns = 0;
    // Running class evidence for consensus naming.
    std::string current_name;
    std::unordered_map<std::string, float> class_votes;
    std::unordered_map<std::string, int> class_counts;
    std::unordered_map<std::string, float> class_conf_sums;
    // Similarity and embedding history used for association and goal scoring.
    float similarity = 0.0f;
    float confidence_ema = 0.0f;
    std::vector<float> image_embedding_masked;
    std::vector<float> image_embedding_unmasked;
    float embedding_confidence_max = -1.0f;
    std::string source_track_id;
    // Cached poses reused by message publishing paths.
    std::vector<float> pose_cam{0.0f, 0.0f, 0.0f};
    std::vector<float> pose_map{0.0f, 0.0f, 0.0f};
};

// Temporary track that must mature before promotion into the persistent map.
struct TentativeTrack {
    std::string track_id;
    std::string frame;
    builtin_interfaces::msg::Time timestamp;
    // Geometry collected while the track is still tentative.
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_points;
    // Promotion counters and timestamps.
    int hits = 0;
    long long first_seen_ns = 0;
    long long last_seen_ns = 0;
    // Class and embedding evidence collected before promotion.
    std::string class_name;
    float confidence_max = 0.0f;
    float confidence_sum = 0.0f;
    std::vector<float> image_embedding_masked;
    std::vector<float> image_embedding_unmasked;
    float embedding_confidence_max = -1.0f;
};

// Core mapper that associates detections, tracks object state, and maintains geometry.
class SemanticObjectMapV5 {
    public:
    SemanticObjectMapV5();
    ~SemanticObjectMapV5() = default;
    // Main object stores.
    std::unordered_map<std::string, MapObject> objects;
    std::unordered_map<std::string, TentativeTrack> tentative_tracks;
    // Tracker binding tables.
    std::unordered_map<std::string, std::string> track_to_map;
    std::unordered_map<std::string, long long> track_last_seen_ns;
    // Update tentative parameters
    float min_input_confidence = 0.55f;
    int confirmation_min_hits = 5;
    float confirmation_min_age_sec = 1.0f;
    float min_confidence_for_promotion = 0.6f;
    float min_avg_confidence_for_promotion = 0.55f;
    // stale pruning parameters
    float tentative_max_stale_sec = 5.0f;
    float binding_ttl_sec = 10.0f;
    float confidence_ema_alpha = 0.20f;
    // class consensus parameters
    float class_count_weight = 1.0f;
    float class_confidence_weight = 2.0f;
    float class_switch_margin = 0.75f;
    int min_class_votes_to_lock = 4;
    // voxel filtering
    float voxel_size = 0.04f;
    float min_point_count = 800.0f;
    float max_point_count = 6000.0f;
    float min_voxel_size = 0.05f;
    float max_voxel_size = 0.2f;
    int sor_mean_k = 50;
    int min_sor_k = 20;
    int max_sor_k = 120;
    float sor_stddev_mul_thresh = 1.0f;
    float min_sor_stddev_mul_thresh = 1.0f;
    float max_sor_stddev_mul_thresh = 1.0f;
    bool enable_voxel_filtering = true;
    bool enable_icp_fusion_ = false;
    bool enable_statistical_outlier_removal = true;
    // geometry refinement parameters
    int refine_min_point_count = 20;
    double refine_cluster_tolerance = 0.06;
    int refine_min_cluster_size = 70;
    int refine_max_cluster_size = 25000;
    bool enable_clustering_refinement = false;
    bool enable_sor_refinement = false;
    // scoring parameters
    double w_sem = 3.5;
    double MAX_COST = 4.5;
    double kBlockedCost = 999.0;
    int kTopKPerDetection = 3;
    double max_class_penalty = 3.0;
    // Dynamic association weights based on object size.
    float association_small_object_max_size = 0.05f;
    float association_large_object_min_size = 2.0f;
    double association_small_object_dist_weight = 5.0;
    double association_small_object_iou_weight = 4.0;
    double association_large_object_dist_weight = 0.2;
    double association_large_object_iou_weight = 0.3;
    // Remove wrong detections parameters
    double kMaxAgeSec = 20.0;
    int kMinOccurrences = 50;


    // Adds a synchronized batch of detections and updates object memory.
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

    // Merges map objects that overlap strongly and represent the same class.
    void resolve_overlapping_duplicates();

    // Removes stale low-support objects that are likely false positives.
    void remove_wrong_detections();
    
    // Keeps only the dominant geometric cluster for a map object.
    void refine_object_geometry(const std::string& map_id);

    // Returns the eight world-space corners of an oriented box.
    std::array<std::array<float, 3>, 8> compute_obb_corners(const OrientedBoundingBox& obb) const;
    
    // Stores the text embedding used for global goal matching.
    void set_text_embedding(const std::vector<float>& emb, float scale, float bias);

    // Computes object-to-goal similarity using the stored goal embedding.
    float get_goal_similarity(const std::string& map_id) const;

    private:
    int next_map_id_ = 1;

    std::vector<float> goal_text_embedding_;
    float logit_scale_ = 1.0f;
    float logit_bias_ = 0.0f;

    // State cleanup and timestamp helpers.
    void prune_stale_state(long long current_ns);
    std::string new_map_id();
    long long stamp_to_ns(const builtin_interfaces::msg::Time& stamp);

    // Embedding and geometry utilities.
    std::vector<float> normalize_embedding(const std::vector<float>& embedding) const;
    
    std::vector<float> fuse_embeddings_running_avg(
        const std::vector<float>& current_embedding, int current_count,
        const std::vector<float>& new_embedding, int new_count);
    
    // Image-to-image cosine similarity used for correspondence.
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

    // Internal update routines for tentative and confirmed objects.
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