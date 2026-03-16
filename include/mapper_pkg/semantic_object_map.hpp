#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <shared_mutex>

#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

struct TentativeTrack {
    std::string track_id;
    std::string frame;
    uint64_t first_seen_ns;
    uint64_t last_seen_ns;
    int hits;

    std::string class_name;
    float confidence_max;
    float confidence_sum;

    Eigen::VectorXf image_embedding;
    float embedding_confidence_max;
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_points;

    Eigen::Vector3f pose_map;
    Eigen::Vector3f obb_extents;
};

struct MapObject {
    std::string map_id;
    std::string frame;
    uint64_t first_seen_ns;
    uint64_t last_seen_ns;
    int occurrences;

    std::string current_name;
    std::unordered_map<std::string, float> class_votes;
    std::unordered_map<std::string, float> class_counts;
    std::unordered_map<std::string, float> class_conf_sums;

    float similarity;
    float confidence_ema;

    Eigen::VectorXf image_embedding;
    float embedding_confidence_max;
    pcl::PointCloud<pcl::PointXYZ>::Ptr accumulated_points;

    Eigen::Vector3f pose_map;
    Eigen::Vector3f obb_extents;
};

class SemanticObjectMap {
    public:

        SemanticObjectMap();
        ~SemanticObjectMap() = default;

        void addDetectionsBatch(
            const std::vector<std::string>& object_names,
            const std::vector<std::string>& tracker_ids,
            uint64_t detection_stamp_ns,
            const std::string& frame_id,
            const std::vector<Eigen::VectorXf>& embeddings_list,
            const std::vector<float>& confidences,
            const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& points_cam_list
        );

        void resolveDuplicates();
        void ExportJson(const std::string& directory_path, const std::string& file_name);
        
        std::unordered_map<std::string, MapObject> getObjectCopy() const;
        
        float w_dist = 1.0f;
        float w_iou = 1.0f;
        float w_sem = 2.5f;
        float max_cost = 3.5f;

        int confirmation_hits = 6;
        uint64_t confirmation_age = 800000000;
        int max_points = 5000;

    private:

        std::unordered_map<std::string, TentativeTrack> tentative_tracks_;
        std::unordered_map<std::string, MapObject> objects_;
        std::unordered_map<std::string, std::string> track_to_map_;
        std::unordered_map<std::string, uint64_t> track_last_seen_;

        int next_map_id_ = 1;

        mutable std::shared_mutex map_mutex_;

        void pruneStaleState(uint64_t current_ns);
        std::string generateNewMapId();

        pcl::PointCloud<pcl::PointXYZ>::Ptr fuseGeometry(
            const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& old_points,
            const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& new_points
        );

        void updateChachedGeometry(MapObject& obj);
        void updateChachedGeometry(TentativeTrack& track);

        bool updateTentative(
            const std::string& object_name, 
            const std::string& tracker_id,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& points_map, 
            uint64_t current_ns, 
            float confidence,
            const Eigen::VectorXf& image_embedding, 
            const std::string& frame
        );
    
        void updateObject(
            const std::string& map_id, 
            const std::string& object_name,
            pcl::PointCloud<pcl::PointXYZ>::Ptr& points_map, 
            float confidence,
            const Eigen::VectorXf& image_embedding, 
            uint64_t current_ns,
            const std::string& source_track_id
        );
    
        void fuseObjects(const std::string& keep_id, const std::string& drop_id);

        // Math and AI
        Eigen::VectorXf fuseEmbeddingsRunningAvg(
            const Eigen::VectorXf& cur_emb, int cur_count, 
            const Eigen::VectorXf& new_emb, int new_count
        );
        
        float computeSemanticDistance(const Eigen::VectorXf& emb1, const Eigen::VectorXf& emb2);
        float computeObbIou(const MapObject& obj1, const MapObject& obj2);
        
        std::string chooseConsensusClass(
            const std::unordered_map<std::string, int>& class_counts,
            const std::unordered_map<std::string, float>& class_conf_sums,
            const std::string& current_name
        );
};
