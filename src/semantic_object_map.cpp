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
    // 1. Lock the map for writing so no other threads can access it while we delete things
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    
    // 2. Declare your local counter variables
    int removed_tentative = 0;
    int removed_bindings = 0;

    // Define thresholds (In nanoseconds. 2 seconds and 4 seconds respectively)
    // Note: You can move these to the .hpp file as class variables later
    uint64_t tentative_max_stale_ns = 2000000000; 
    uint64_t binding_ttl_ns = 4000000000;

    // ---------------------------------------------------------
    // 3. PRUNE TENTATIVE TRACKS
    // ---------------------------------------------------------
    // Notice there is no "it++" inside the for-loop declaration. 
    // We control the increment manually inside the loop to avoid Segmentation Faults.
    for (auto it = tentative_tracks_.begin(); it != tentative_tracks_.end(); ) {
        
        // Access the struct variables using it->second
        if (current_ns - it->second.last_seen_ns > tentative_max_stale_ns) {
            
            // erase() safely deletes the item and returns a valid iterator to the NEXT item
            it = tentative_tracks_.erase(it);
            removed_tentative++;
            
        } else {
            // Only increment if we DID NOT erase anything
            it++;
        }
    }

    // ---------------------------------------------------------
    // 4. PRUNE OLD BINDINGS
    // ---------------------------------------------------------
    for (auto it = track_last_seen_.begin(); it != track_last_seen_.end(); ) {
        
        // it->second is the uint64_t timestamp for this map
        if (current_ns - it->second > binding_ttl_ns) {
            
            // it->first is the string track_id. We must erase it from the other map too.
            track_to_map_.erase(it->first);
            
            // Safely erase from this map and catch the next iterator
            it = track_last_seen_.erase(it);
            removed_bindings++;
            
        } else {
            it++;
        }
    } 
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
    pruneStaleState(detection_stamp_ns);

    std::vector<ValidDetection> valid_detections;

    // A. Filter and package valid detections
    for (size_t i = 0; i < points_cam_list.size(); ++i) {
        if (!points_cam_list[i] || points_cam_list[i]->points.size() < 5) continue;

        // Calculate median depth to reject flying pixels
        std::vector<float> z_values;
        z_values.reserve(points_cam_list[i]->points.size());
        for (const auto& pt : points_cam_list[i]->points) {
            if (std::isfinite(pt.z)) z_values.push_back(pt.z);
        }
        
        float depth_m = 0.0f;
        if (!z_values.empty()) {
            size_t mid = z_values.size() / 2;
            std::nth_element(z_values.begin(), z_values.begin() + mid, z_values.end());
            depth_m = z_values[mid];
        }

        // Apply depth gate (hardcoded for example, use your class variables)
        if (depth_m < 0.25f || depth_m > 6.0f) continue;

        // Extract Geometry
        pcl::MomentOfInertiaEstimation<pcl::PointXYZ> extractor;
        extractor.setInputCloud(points_cam_list[i]);
        extractor.compute();
        pcl::PointXYZ min_OBB, max_OBB, pos_OBB;
        Eigen::Matrix3f rot_OBB;
        extractor.getOBB(min_OBB, max_OBB, pos_OBB, rot_OBB);

        ValidDetection det;
        det.name = object_names[i];
        det.track_id = tracker_ids[i];
        det.points_map = points_cam_list[i];
        det.centroid = pos_OBB.getVector3fMap();
        det.obb_extents = max_OBB.getVector3fMap() - min_OBB.getVector3fMap();
        det.embedding = embeddings_list[i];
        det.confidence = confidences[i];
        
        valid_detections.push_back(det);
    }

    std::vector<ValidDetection> unmatched_detections;

    // B. Phase 1: Direct ID Routing (Trust the Tracker)
    for (const auto& det : valid_detections) {
        bool matched = false;
        
        // Check confirmed objects
        if (track_to_map_.count(det.track_id) && objects_.count(track_to_map_[det.track_id])) {
            std::string map_id = track_to_map_[det.track_id];
            if (objects_[map_id].current_name == det.name) { // Class safety gate
                updateObject(map_id, det.name, det.points_map, det.confidence, det.embedding, detection_stamp_ns, det.track_id);
                track_last_seen_[det.track_id] = detection_stamp_ns;
                matched = true;
            }
        } 
        // Check tentative tracks
        else if (tentative_tracks_.count(det.track_id)) {
            if (tentative_tracks_[det.track_id].class_name == det.name) {
                updateTentative(det.name, det.track_id, det.points_map, detection_stamp_ns, det.confidence, det.embedding, frame_id);
                matched = true;
            }
        }

        if (!matched) unmatched_detections.push_back(det);
    }

    // C. Phase 2: Greedy Bipartite Matching for unmatched
    std::set<std::string> available_map_ids;
    for (const auto& pair : objects_) available_map_ids.insert(pair.first);

    for (const auto& det : unmatched_detections) {
        std::string best_map_id = "";
        float best_cost = max_cost;

        for (const std::string& m_id : available_map_ids) {
            const MapObject& obj = objects_[m_id];
            
            float dist = (det.centroid - obj.pose_map).norm();
            if (dist > 1.5f) continue;

            float cost_sem = computeSemanticDistance(det.embedding, obj.image_embedding);
            float class_penalty = (det.name != obj.current_name) ? 5.0f : 0.0f;
            
            // Total cost (simplified without IoU for speed, add computeObbIou if needed)
            float total_cost = (w_dist * dist) + (w_sem * cost_sem) + class_penalty;

            if (total_cost < best_cost) {
                best_cost = total_cost;
                best_map_id = m_id;
            }
        }

        if (!best_map_id.empty()) {
            updateObject(best_map_id, det.name, det.points_map, det.confidence, det.embedding, detection_stamp_ns, det.track_id);
            track_to_map_[det.track_id] = best_map_id;
            track_last_seen_[det.track_id] = detection_stamp_ns;
            available_map_ids.erase(best_map_id); // Prevent multiple detections claiming one object
        } else {
            // D. Phase 3: Spawn New Tentative Track
            updateTentative(det.name, det.track_id, det.points_map, detection_stamp_ns, det.confidence, det.embedding, frame_id);
        }
    }
}

// ==============================================================================
// 2. LIFECYCLE UPDATES
// ==============================================================================
bool SemanticObjectMap::updateTentative(
    const std::string& object_name, const std::string& tracker_id,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& points_map, uint64_t current_ns, 
    float confidence, const Eigen::VectorXf& image_embedding, const std::string& frame)
{
    if (tentative_tracks_.find(tracker_id) == tentative_tracks_.end()) {
        TentativeTrack track;
        track.track_id = tracker_id;
        track.frame = frame;
        track.first_seen_ns = current_ns;
        track.last_seen_ns = current_ns;
        track.hits = 1;
        track.class_name = object_name;
        track.confidence_max = confidence;
        track.confidence_sum = confidence;
        track.image_embedding = image_embedding.normalized();
        track.accumulated_points = points_map;
        updateCachedGeometry(track);
        tentative_tracks_[tracker_id] = track;
        return false;
    }

    TentativeTrack& track = tentative_tracks_[tracker_id];
    
    // Reset if class flips
    if (track.class_name != object_name) {
        track.class_name = object_name;
        track.hits = 1;
        track.confidence_sum = confidence;
        track.first_seen_ns = current_ns;
    } else {
        track.hits += 1;
        track.confidence_sum += confidence;
    }

    track.last_seen_ns = current_ns;
    track.confidence_max = std::max(track.confidence_max, confidence);
    track.accumulated_points = fuseGeometry(track.accumulated_points, points_map);
    track.image_embedding = fuseEmbeddingsRunningAvg(track.image_embedding, track.hits - 1, image_embedding, 1);
    updateCachedGeometry(track);

    // Check Promotion
    uint64_t age_ns = current_ns - track.first_seen_ns;
    float avg_conf = track.confidence_sum / static_cast<float>(track.hits);

    if (track.hits >= confirmation_hits && age_ns >= confirmation_age && avg_conf >= 0.55f) {
        std::string map_id = generateNewMapId();
        MapObject obj;
        obj.map_id = map_id;
        obj.frame = track.frame;
        obj.first_seen_ns = track.first_seen_ns;
        obj.last_seen_ns = current_ns;
        obj.occurrences = track.hits;
        obj.current_name = track.class_name;
        obj.class_votes[track.class_name] = track.confidence_sum;
        obj.class_counts[track.class_name] = track.hits;
        obj.class_conf_sums[track.class_name] = track.confidence_sum;
        obj.confidence_ema = track.confidence_max;
        obj.accumulated_points = track.accumulated_points;
        obj.image_embedding = track.image_embedding;
        obj.pose_map = track.pose_map;
        obj.obb_extents = track.obb_extents;

        objects_[map_id] = obj;
        track_to_map_[tracker_id] = map_id;
        track_last_seen_[tracker_id] = current_ns;
        
        tentative_tracks_.erase(tracker_id);
        return true;
    }
    return false;
}

void SemanticObjectMap::updateObject(
    const std::string& map_id, const std::string& object_name,
    pcl::PointCloud<pcl::PointXYZ>::Ptr& points_map, float confidence,
    const Eigen::VectorXf& image_embedding, uint64_t current_ns, const std::string& source_track_id)
{
    MapObject& obj = objects_[map_id];
    
    obj.accumulated_points = fuseGeometry(obj.accumulated_points, points_map);
    obj.occurrences += 1;
    obj.last_seen_ns = current_ns;
    
    obj.class_counts[object_name] += 1;
    obj.class_conf_sums[object_name] += confidence;
    obj.current_name = chooseConsensusClass(obj.class_counts, obj.class_conf_sums, obj.current_name);
    
    obj.image_embedding = fuseEmbeddingsRunningAvg(obj.image_embedding, obj.occurrences - 1, image_embedding, 1);
    
    updateCachedGeometry(obj);
}

void SemanticObjectMap::fuseObjects(const std::string& keep_id, const std::string& drop_id) {
    MapObject& keep = objects_[keep_id];
    MapObject& drop = objects_[drop_id];

    keep.accumulated_points = fuseGeometry(keep.accumulated_points, drop.accumulated_points);
    keep.occurrences += drop.occurrences;
    keep.first_seen_ns = std::min(keep.first_seen_ns, drop.first_seen_ns);
    keep.last_seen_ns = std::max(keep.last_seen_ns, drop.last_seen_ns);

    for (const auto& pair : drop.class_counts) {
        keep.class_counts[pair.first] += pair.second;
        keep.class_conf_sums[pair.first] += drop.class_conf_sums[pair.first];
    }
    keep.current_name = chooseConsensusClass(keep.class_counts, keep.class_conf_sums, keep.current_name);
    
    keep.image_embedding = fuseEmbeddingsRunningAvg(keep.image_embedding, keep.occurrences, drop.image_embedding, drop.occurrences);
    updateCachedGeometry(keep);

    // Reroute active trackers
    for (auto& pair : track_to_map_) {
        if (pair.second == drop_id) pair.second = keep_id;
    }
    objects_.erase(drop_id);
}

// ==============================================================================
// 3. IOU AND RESOLUTION
// ==============================================================================
float SemanticObjectMap::computeObbIou(const MapObject& obj1, const MapObject& obj2) {
    // Exact 3D OBB Intersection requires external collision libraries. 
    // This is a fast, built-in Axis-Aligned Bounding Box (AABB) approximation.
    Eigen::Vector3f min1 = obj1.pose_map - (obj1.obb_extents / 2.0f);
    Eigen::Vector3f max1 = obj1.pose_map + (obj1.obb_extents / 2.0f);
    Eigen::Vector3f min2 = obj2.pose_map - (obj2.obb_extents / 2.0f);
    Eigen::Vector3f max2 = obj2.pose_map + (obj2.obb_extents / 2.0f);

    Eigen::Vector3f inter_min = min1.cwiseMax(min2);
    Eigen::Vector3f inter_max = max1.cwiseMin(max2);

    if (inter_min.x() >= inter_max.x() || inter_min.y() >= inter_max.y() || inter_min.z() >= inter_max.z()) {
        return 0.0f; // No overlap
    }

    float intersection_vol = (inter_max - inter_min).prod();
    float vol1 = obj1.obb_extents.prod();
    float vol2 = obj2.obb_extents.prod();
    float union_vol = vol1 + vol2 - intersection_vol;

    return (union_vol > 0.0f) ? (intersection_vol / union_vol) : 0.0f;
}

void SemanticObjectMap::resolveDuplicates() {
    std::unique_lock<std::shared_mutex> lock(map_mutex_);
    std::vector<std::string> ids;
    for (const auto& pair : objects_) ids.push_back(pair.first);

    for (size_t i = 0; i < ids.size(); ++i) {
        for (size_t j = i + 1; j < ids.size(); ++j) {
            std::string id1 = ids[i];
            std::string id2 = ids[j];
            
            if (objects_.count(id1) == 0 || objects_.count(id2) == 0) continue;
            
            MapObject& obj1 = objects_[id1];
            MapObject& obj2 = objects_[id2];

            if (obj1.current_name != obj2.current_name) continue;

            float sem_dist = computeSemanticDistance(obj1.image_embedding, obj2.image_embedding);
            if (sem_dist > 0.40f) continue;

            float iou = computeObbIou(obj1, obj2);
            if (iou > 0.15f) {
                fuseObjects(id1, id2);
            }
        }
    }
}

// ==============================================================================
// 4. EXPORT
// ==============================================================================
void SemanticObjectMap::ExportJson(const std::string& directory_path, const std::string& file_name) {
    std::shared_lock<std::shared_mutex> lock(map_mutex_);
    std::string full_path = directory_path + "/" + file_name;
    std::ofstream file(full_path);
    
    if (!file.is_open()) return;

    file << "{\n";
    bool first = true;
    for (const auto& pair : objects_) {
        if (!first) file << ",\n";
        first = false;
        const MapObject& obj = pair.second;
        
        file << "  \"" << pair.first << "\": {\n"
             << "    \"name\": \"" << obj.current_name << "\",\n"
             << "    \"frame\": \"" << obj.frame << "\",\n"
             << "    \"pose_map\": {\"x\": " << obj.pose_map.x() << ", \"y\": " << obj.pose_map.y() << ", \"z\": " << obj.pose_map.z() << "},\n"
             << "    \"box_size\": {\"x\": " << obj.obb_extents.x() << ", \"y\": " << obj.obb_extents.y() << ", \"z\": " << obj.obb_extents.z() << "},\n"
             << "    \"occurrences\": " << obj.occurrences << ",\n"
             << "    \"confidence\": " << obj.confidence_ema << "\n"
             << "  }";
    }
    file << "\n}\n";
    file.close();
}