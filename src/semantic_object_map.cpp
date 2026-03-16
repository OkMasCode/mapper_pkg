#include "mapper_pkg/semantic_object_map.hpp" // Adjust path as needed
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <iostream>
#include <algorithm>

// ==============================================================================
// CONSTRUCTOR
// ==============================================================================
SemanticObjectMap::SemanticObjectMap() {
    // Initialization if needed
}

// ==============================================================================
// GEOMETRY & MATH (PCL & EIGEN)
// ==============================================================================

// Fuses two point clouds and applies a Voxel Grid downsample to prevent memory explosion
pcl::PointCloud<pcl::PointXYZ>::Ptr SemanticObjectMap::fuseGeometry(
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& old_points,
    const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& new_points) 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr combined(new pcl::PointCloud<pcl::PointXYZ>());
    
    // Add existing points safely
    if (old_points && !old_points->empty()) {
        *combined += *old_points;
    }
    
    // Add new points safely
    if (new_points && !new_points->empty()) {
        *combined += *new_points;
    }

    // Downsample the combined cloud to 5cm voxels
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(combined);
    sor.setLeafSize(0.05f, 0.05f, 0.05f);
    sor.filter(*filtered);

    // Enforce hard memory cap for edge devices
    if (filtered->points.size() > static_cast<size_t>(max_points)) {
        filtered->points.resize(max_points);
        filtered->width = max_points;
        filtered->height = 1;
    }

    return filtered;
}

// Uses PCL's Moment of Inertia to extract the bounding box and centroid
void SemanticObjectMap::updateCachedGeometry(MapObject& obj) {
    if (!obj.accumulated_points || obj.accumulated_points->empty()) return;

    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(obj.accumulated_points);
    feature_extractor.compute();

    pcl::PointXYZ min_point_OBB, max_point_OBB, position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;

    // Extract Oriented Bounding Box
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

    // Cache the geometry to avoid recalculating in loops
    obj.pose_map = position_OBB.getVector3fMap();
    obj.obb_extents = (max_point_OBB.getVector3fMap() - min_point_OBB.getVector3fMap());
}

// Same extraction logic for TentativeTracks
void SemanticObjectMap::updateCachedGeometry(TentativeTrack& track) {
    if (!track.accumulated_points || track.accumulated_points->empty()) return;

    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(track.accumulated_points);
    feature_extractor.compute();

    pcl::PointXYZ min_point_OBB, max_point_OBB, position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;

    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

    track.pose_map = position_OBB.getVector3fMap();
    track.obb_extents = (max_point_OBB.getVector3fMap() - min_point_OBB.getVector3fMap());
}

// Computes Cosine Distance between two CLIP embeddings
float SemanticObjectMap::computeSemanticDistance(const Eigen::VectorXf& emb1, const Eigen::VectorXf& emb2) {
    if (emb1.size() == 0 || emb2.size() == 0) return 1.0f;
    
    // Dot product of normalized vectors gives cosine similarity
    float sim = emb1.normalized().dot(emb2.normalized());
    
    // Clamp to prevent floating point errors
    sim = std::max(-1.0f, std::min(1.0f, sim)); 
    return 1.0f - sim; // Convert similarity to distance
}

// Fuses AI embeddings using a running average
Eigen::VectorXf SemanticObjectMap::fuseEmbeddingsRunningAvg(
    const Eigen::VectorXf& cur_emb, int cur_count, 
    const Eigen::VectorXf& new_emb, int new_count) 
{
    if (cur_emb.size() == 0) return new_emb.normalized();
    if (new_emb.size() == 0) return cur_emb.normalized();

    int w1 = std::max(cur_count, 1);
    int w2 = std::max(new_count, 1);

    // Weighted average of the embeddings
    Eigen::VectorXf fused = (cur_emb * w1 + new_emb * w2) / (w1 + w2);
    return fused.normalized();
}

// Logic for choosing consensus class based on counts and confidences
std::string SemanticObjectMap::chooseConsensusClass(
    const std::unordered_map<std::string, int>& class_counts,
    const std::unordered_map<std::string, float>& class_conf_sums,
    const std::string& current_name) 
{
    if (class_counts.empty()) return current_name;

    std::string best_name = current_name;
    float best_score = -1.0f;

    for (const auto& pair : class_counts) {
        const std::string& name = pair.first;
        // The value in class_counts is now correctly an int
        float count = static_cast<float>(pair.second); 
        
        float conf_sum = 0.0f;
        auto it = class_conf_sums.find(name);
        if (it != class_conf_sums.end()) conf_sum = it->second;

        float avg_conf = conf_sum / std::max(count, 1.0f);
        float score = (w_dist * count) + (w_sem * avg_conf); // Using placeholder weights for scoring

        if (score > best_score) {
            best_score = score;
            best_name = name;
        }
    }
    return best_name;
}

// Utility to generate unique map IDs
std::string SemanticObjectMap::generateNewMapId() {
    std::string id = "map_obj_" + std::to_string(next_map_id_);
    next_map_id_++;
    return id;
}

// Thread-safe map retrieval for ROS publishing
std::unordered_map<std::string, MapObject> SemanticObjectMap::getObjectCopy() const {
    std::shared_lock<std::shared_mutex> lock(map_mutex_);
    return objects_;
}

// ==============================================================================
// STATE MANAGEMENT & LIFECYCLE (Skeleton implementations)
// ==============================================================================

void SemanticObjectMap::pruneStaleState(uint64_t current_ns) {
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    // Implement pruning logic using timestamps...
}

void SemanticObjectMap::addDetectionsBatch(
    const std::vector<std::string>& object_names,
    const std::vector<std::string>& tracker_ids,
    uint64_t detection_stamp_ns,
    const std::string& frame_id,
    const std::vector<Eigen::VectorXf>& embeddings_list,
    const std::vector<float>& confidences,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& points_cam_list)
{
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    // Matching logic will go here
}

bool SemanticObjectMap::updateTentative(
    const std::string& object_name, 
    const std::string& tracker_id,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& points_map, 
    uint64_t current_ns, 
    float confidence,
    const Eigen::VectorXf& image_embedding, 
    const std::string& frame)
{
    // Tentative track logic...
    return false;
}

void SemanticObjectMap::updateObject(
    const std::string& map_id, 
    const std::string& object_name,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& points_map, 
    float confidence,
    const Eigen::VectorXf& image_embedding, 
    uint64_t current_ns,
    const std::string& source_track_id)
{
    // Object update logic...
}

void SemanticObjectMap::fuseObjects(const std::string& keep_id, const std::string& drop_id) {
    // Fusion logic...
}

float SemanticObjectMap::computeObbIou(const MapObject& obj1, const MapObject& obj2) {
    return 0.0f; // Placeholder
}

void SemanticObjectMap::resolveDuplicates() {
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    // Loop through objects_ and fuseObjects if IOU > threshold
}

void SemanticObjectMap::ExportJson(const std::string& directory_path, const std::string& file_name) {
    std::shared_lock<std::shared_mutex> lock(map_mutex_);
    // Export objects_ to JSON
}