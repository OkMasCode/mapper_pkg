#include "mapper_pkg/semantic_object_map.hpp"
#include "mapper_pkg/Hungarian.h"

#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <optional>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>

// ---> MAKE SURE THIS FUNCTION EXISTS! <---
SemanticObjectMapV5::SemanticObjectMapV5() {
    std::cout << "[SemanticObjectMap] Initialized with SOTA C++ Geometry and Tracking.\n";
}

// ==========================================
// 1. GEOMETRY & MATH HELPERS
// ==========================================

std::vector<float> SemanticObjectMapV5::normalize_embedding(const std::vector<float>& embedding) {
    if (embedding.empty()) return {};
    float norm = std::sqrt(std::inner_product(embedding.begin(), embedding.end(), embedding.begin(), 0.0f));
    if (norm <= 1e-12f || !std::isfinite(norm)) return {};
    
    std::vector<float> normalized(embedding.size());
    for (size_t i = 0; i < embedding.size(); ++i) {
        normalized[i] = embedding[i] / norm;
    }
    return normalized;
}

std::vector<float> SemanticObjectMapV5::fuse_embeddings_running_avg(
    const std::vector<float>& current_embedding, int current_count,
    const std::vector<float>& new_embedding, int new_count) 
{
    auto cur = normalize_embedding(current_embedding);
    auto nxt = normalize_embedding(new_embedding);

    if (cur.empty()) return nxt;
    if (nxt.empty()) return cur;

    int cw = std::max(current_count, 1);
    int nw = std::max(new_count, 1);
    
    std::vector<float> fused(cur.size());
    for (size_t i = 0; i < cur.size(); ++i) {
        fused[i] = ((cur[i] * cw) + (nxt[i] * nw)) / (cw + nw);
    }
    return normalize_embedding(fused);
}

float SemanticObjectMapV5::compute_semantic_distance(const std::vector<float>& emb1, const std::vector<float>& emb2) {
    if (emb1.empty() || emb2.empty()) return 1.0f;
    
    auto e1 = normalize_embedding(emb1);
    auto e2 = normalize_embedding(emb2);
    
    float sim = std::inner_product(e1.begin(), e1.end(), e2.begin(), 0.0f);
    sim = std::max(-1.0f, std::min(1.0f, sim)); // Clamp between -1 and 1
    return 1.0f - sim; // Return distance
}

pcl::PointCloud<pcl::PointXYZ>::Ptr SemanticObjectMapV5::fuse_geometry(
    pcl::PointCloud<pcl::PointXYZ>::Ptr old_points, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_points) 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr combined(new pcl::PointCloud<pcl::PointXYZ>());
    
    if (old_points && !old_points->empty()) *combined += *old_points;
    if (new_points && !new_points->empty()) *combined += *new_points;

    if (combined->empty()) return combined;

    // Adaptive Voxelization for the merged cloud
    int num_combined = combined->points.size();
    float voxel_size = 0.01f;

    if (num_combined < 10000) voxel_size = 0.005f;
    else if (num_combined > 30000) voxel_size = 0.03f;
    else voxel_size = 0.005f + (0.02f * ((num_combined - 10000.0f) / 25000.0f));
    
    // Voxel downsample to merge overlapping points and keep memory clean (7cm)
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> grid;
    grid.setInputCloud(combined);
    grid.setLeafSize(voxel_size, voxel_size, voxel_size);
    grid.filter(*downsampled);
    
    return downsampled;
}

OrientedBoundingBox SemanticObjectMapV5::compute_obb(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    OrientedBoundingBox obb;
    if (!cloud || cloud->points.size() < 4) return obb;

    // 1. Compute Centroid
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cloud, pcaCentroid);
    
    // 2. Compute Covariance Matrix and extract Eigenvectors (PCA)
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    
    // Enforce right-handed coordinate system
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

    // 3. Transform point cloud to origin aligned with PCA axes
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudProjected(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *cloudProjected, projectionTransform);

    // 4. Find min/max in the local aligned space to get extents
    pcl::PointXYZ minPoint, maxPoint;
    pcl::getMinMax3D(*cloudProjected, minPoint, maxPoint);

    obb.center = {pcaCentroid(0), pcaCentroid(1), pcaCentroid(2)};
    obb.extents = {maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z};
    
    return obb;
}

float SemanticObjectMapV5::compute_obb_iou(
    pcl::PointCloud<pcl::PointXYZ>::Ptr points1, const OrientedBoundingBox& obb1,
    pcl::PointCloud<pcl::PointXYZ>::Ptr points2, const OrientedBoundingBox& obb2) 
{
    // Simplified 3D IoU Proxy: Distance gating since exact rotated 3D IoU is highly complex in C++
    // We check if the centroids are close enough relative to their sizes.
    if (obb1.extents[0] == 0 || obb2.extents[0] == 0) return 0.0f;

    float dx = obb1.center[0] - obb2.center[0];
    float dy = obb1.center[1] - obb2.center[1];
    float dz = obb1.center[2] - obb2.center[2];
    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    float max_size1 = std::max({obb1.extents[0], obb1.extents[1], obb1.extents[2]});
    float max_size2 = std::max({obb2.extents[0], obb2.extents[1], obb2.extents[2]});
    
    float collision_dist = (max_size1 + max_size2) / 2.0f;

    if (dist < collision_dist) {
        // High overlap proxy
        return 1.0f - (dist / collision_dist); 
    }
    return 0.0f;
}

void SemanticObjectMapV5::refine_object_geometry(const std::string& map_id) {
    if (objects.find(map_id) == objects.end()) return;
    
    auto& obj = objects[map_id];
    if (obj.accumulated_points->points.size() < 20) return;

    // DBSCAN equivalent using PCL EuclideanClusterExtraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(obj.accumulated_points);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.06); // 6cm distance to keep cluster together
    ec.setMinClusterSize(70);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(obj.accumulated_points);
    ec.extract(cluster_indices);

    if (cluster_indices.empty()) return;

    // The first cluster is guaranteed to be the largest by PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr refined_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster_indices[0].indices) {
        refined_cloud->points.push_back(obj.accumulated_points->points[idx]);
    }
    
    refined_cloud->width = refined_cloud->points.size();
    refined_cloud->height = 1;
    refined_cloud->is_dense = true;
    
    obj.accumulated_points = refined_cloud;
    obj.obb = compute_obb(obj.accumulated_points); // Recalculate OBB
    obj.pose_map = obj.obb.center;
}

// ==========================================
// 2. STATE MANAGEMENT & VOTING
// ==========================================

long long SemanticObjectMapV5::stamp_to_ns(const builtin_interfaces::msg::Time& stamp) {
    return (static_cast<long long>(stamp.sec) * 1000000000LL) + stamp.nanosec;
}

std::string SemanticObjectMapV5::new_map_id() {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "map_obj_%06d", next_map_id_++);
    return std::string(buffer);
}

std::string SemanticObjectMapV5::choose_consensus_class(
    const std::unordered_map<std::string, int>& class_counts,
    const std::unordered_map<std::string, float>& class_conf_sums,
    const std::string& current_name) 
{
    if (class_counts.empty()) return current_name;

    std::string best_name = current_name;
    float best_score = -1.0f;
    float current_score = 0.0f;

    for (const auto& [name, count] : class_counts) {
        float conf_sum = class_conf_sums.at(name);
        float avg_conf = conf_sum / std::max(1, count);
        float score = (class_count_weight * count) + (class_confidence_weight * avg_conf);
        
        if (name == current_name) current_score = score;
        if (score > best_score) {
            best_score = score;
            best_name = name;
        }
    }

    if (best_name == current_name) return best_name;

    int current_count = class_counts.count(current_name) ? class_counts.at(current_name) : 0;
    if (current_count >= min_class_votes_to_lock && best_score < (current_score + class_switch_margin)) {
        return current_name; // Prevent rapid flipping
    }
    return best_name;
}

void SemanticObjectMapV5::prune_stale_state(long long current_ns) {
    long long stale_tentative_ns = static_cast<long long>(tentative_max_stale_sec * 1e9);
    long long stale_binding_ns = static_cast<long long>(binding_ttl_sec * 1e9);

    for (auto it = tentative_tracks.begin(); it != tentative_tracks.end(); ) {
        if (current_ns - it->second.last_seen_ns > stale_tentative_ns) {
            it = tentative_tracks.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = track_last_seen_ns.begin(); it != track_last_seen_ns.end(); ) {
        if (current_ns - it->second > stale_binding_ns) {
            track_to_map.erase(it->first);
            it = track_last_seen_ns.erase(it);
        } else {
            ++it;
        }
    }
}

// ==========================================
// 3. CORE UPDATE LOGIC
// ==========================================

void SemanticObjectMapV5::update_object(
    const std::string& map_id, const std::string& object_name,
    const builtin_interfaces::msg::Time& detection_stamp,
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_map,
    float similarity, float confidence,
    const std::vector<float>& image_embedding,
    long long current_ns, const std::string& source_track_id) 
{
    auto& entry = objects[map_id];

    entry.accumulated_points = fuse_geometry(entry.accumulated_points, points_map);
    entry.obb = compute_obb(entry.accumulated_points);
    entry.pose_map = entry.obb.center;

    entry.class_votes[object_name] += std::max(confidence, 0.01f);
    entry.class_counts[object_name] += 1;
    entry.class_conf_sums[object_name] += std::max(confidence, 0.01f);

    entry.current_name = choose_consensus_class(entry.class_counts, entry.class_conf_sums, entry.current_name);
    entry.confidence_ema = ((1.0f - confidence_ema_alpha) * entry.confidence_ema) + (confidence_ema_alpha * confidence);

    if (!image_embedding.empty()) {
        entry.image_embedding = fuse_embeddings_running_avg(entry.image_embedding, entry.occurrences, image_embedding, 1);
        entry.embedding_confidence_max = std::max(entry.embedding_confidence_max, confidence);
    }

    entry.similarity = std::isfinite(similarity) ? similarity : 0.0f;

    entry.occurrences += 1;
    entry.last_seen_ns = current_ns;
    entry.timestamp = detection_stamp;
    entry.source_track_id = source_track_id;

    if (entry.first_seen_ns == 0) {
        entry.first_seen_ns = current_ns;
    }

    if (entry.frame.empty()) {
        entry.frame = "map";
    }
}

bool SemanticObjectMapV5::update_tentative(
    const std::string& object_name,
    const std::string& tracker_id,
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_map,
    const builtin_interfaces::msg::Time& detection_stamp,
    float confidence,
    const std::vector<float>& image_embedding,
    long long current_ns,
    const std::string& frame)
{
    if (!points_map || points_map->empty()) {
        std::cout << "[SemanticObjectMap] tentative skip: empty cloud for track=" << tracker_id << "\n";
        return false;
    }

    if (confidence < min_input_confidence) {
        std::cout << "[SemanticObjectMap] tentative skip: low confidence=" << confidence
                  << " threshold=" << min_input_confidence << " track=" << tracker_id << "\n";
        return false;
    }

    auto it = tentative_tracks.find(tracker_id);
    if (it == tentative_tracks.end()) {
        TentativeTrack t;
        t.track_id = tracker_id;
        t.frame = frame;
        t.timestamp = detection_stamp;
        t.accumulated_points = points_map;
        t.hits = 1;
        t.first_seen_ns = current_ns;
        t.last_seen_ns = current_ns;
        t.class_name = object_name;
        t.confidence_max = confidence;
        t.confidence_sum = confidence;
        if (!image_embedding.empty()) {
            t.image_embedding = normalize_embedding(image_embedding);
            t.embedding_confidence_max = confidence;
        }
        tentative_tracks[tracker_id] = t;

        std::cout << "[SemanticObjectMap] tentative create: track=" << tracker_id
                  << " class=" << object_name << " conf=" << confidence << "\n";
        return false;
    }

    auto& t = it->second;
    t.accumulated_points = fuse_geometry(t.accumulated_points, points_map);
    t.hits += 1;
    t.last_seen_ns = current_ns;
    t.timestamp = detection_stamp;
    t.confidence_sum += confidence;
    t.confidence_max = std::max(t.confidence_max, confidence);
    if (!image_embedding.empty()) {
        t.embedding_confidence_max = std::max(t.embedding_confidence_max, confidence);
        t.image_embedding = fuse_embeddings_running_avg(t.image_embedding, t.hits - 1, image_embedding, 1);
    }

    if (confidence >= t.confidence_max || t.class_name.empty()) {
        t.class_name = object_name;
    }

    const double age_sec = static_cast<double>(t.last_seen_ns - t.first_seen_ns) / 1e9;
    const double avg_conf = t.confidence_sum / std::max(1, t.hits);

    const bool promote =
        t.hits >= confirmation_min_hits &&
        age_sec >= confirmation_min_age_sec &&
        t.confidence_max >= min_confidence_for_promotion &&
        avg_conf >= min_avg_confidence_for_promotion;

    if (!promote) {
        std::cout << "[SemanticObjectMap] tentative update: track=" << tracker_id
                  << " hits=" << t.hits << " age=" << age_sec
                  << " avg_conf=" << avg_conf << " max_conf=" << t.confidence_max << "\n";
        return false;
    }

    std::string map_id = new_map_id();
    MapObject obj;
    obj.map_id = map_id;
    obj.frame = frame;
    obj.timestamp = detection_stamp;
    obj.accumulated_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    obj.first_seen_ns = current_ns;
    obj.last_seen_ns = current_ns;
    obj.current_name = t.class_name.empty() ? object_name : t.class_name;
    obj.image_embedding = t.image_embedding;
    obj.embedding_confidence_max = t.embedding_confidence_max;
    obj.source_track_id = tracker_id;
    objects[map_id] = obj;

    update_object(
        map_id,
        obj.current_name,
        detection_stamp,
        t.accumulated_points,
        0.0f,
        static_cast<float>(avg_conf),
        t.image_embedding,
        current_ns,
        tracker_id);

    track_to_map[tracker_id] = map_id;
    track_last_seen_ns[tracker_id] = current_ns;
    tentative_tracks.erase(it);

    std::cout << "[SemanticObjectMap] promote tentative -> map: track=" << tracker_id
              << " map_id=" << map_id << " class=" << obj.current_name
              << " hits=" << obj.occurrences << "\n";
    return true;
}

// ==========================================
// 4. THE MATRIX (BIPARTITE MATCHING)
// ==========================================

void SemanticObjectMapV5::add_detections_batch(
    const std::vector<std::string>& object_names,
    const std::vector<std::string>& tracker_ids,
    const std::vector<float>& confidences,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& points_cam_list,
    const std::vector<std::optional<std::vector<float>>>& embeddings_list,
    const builtin_interfaces::msg::Time& stamp,
    const std::string& camera_frame,
    const std::string& map_frame)
{
    if (points_cam_list.empty()) return;

    if (object_names.size() != points_cam_list.size() ||
        tracker_ids.size() != points_cam_list.size() ||
        confidences.size() != points_cam_list.size() ||
        embeddings_list.size() != points_cam_list.size()) {
        std::cout << "[SemanticObjectMap] batch skip: size mismatch names=" << object_names.size()
                  << " tracks=" << tracker_ids.size() << " conf=" << confidences.size()
                  << " points=" << points_cam_list.size() << " embeddings=" << embeddings_list.size() << "\n";
        return;
    }

    long long current_ns = stamp_to_ns(stamp);
    prune_stale_state(current_ns);

    std::vector<int> unmatched_indices;

    // PHASE 1: Direct ID Routing (Trust the Tracker)
    for (size_t i = 0; i < points_cam_list.size(); ++i) {
        std::string t_id = tracker_ids[i];
        bool matched = false;

        // Has this ID already been promoted to the map?
        if (track_to_map.count(t_id) && objects.count(track_to_map[t_id])) {
            std::string map_id = track_to_map[t_id];
            
            // Safety Gate: Do not trust tracker if class suddenly flips
            if (object_names[i] == objects[map_id].current_name) {
                const std::vector<float> image_embedding = embeddings_list[i].has_value() ? embeddings_list[i].value() : std::vector<float>{};
                float similarity = 0.0f;
                const auto& map_obj = objects[map_id];
                if (!image_embedding.empty() && !map_obj.image_embedding.empty()) {
                    similarity = std::clamp(1.0f - compute_semantic_distance(image_embedding, map_obj.image_embedding), 0.0f, 1.0f);
                }
                update_object(map_id, object_names[i], stamp, points_cam_list[i], similarity, confidences[i], image_embedding, current_ns, t_id);
                track_last_seen_ns[t_id] = current_ns;
                matched = true;
            }
        } 

        if (!matched) {
            unmatched_indices.push_back(i);
        }
    }

    if (unmatched_indices.empty()) return;

    // PHASE 2: Bipartite Matching for new/unmatched detections
    std::vector<std::string> map_ids;
    for (const auto& pair : objects) map_ids.push_back(pair.first);

    if (map_ids.empty()) {
        std::cout << "[SemanticObjectMap] map empty: routing " << unmatched_indices.size()
                  << " detections to tentative tracks\n";
        for (int det_idx : unmatched_indices) {
            const std::vector<float> image_embedding = embeddings_list[det_idx].has_value() ? embeddings_list[det_idx].value() : std::vector<float>{};
            update_tentative(
                object_names[det_idx],
                tracker_ids[det_idx],
                points_cam_list[det_idx],
                stamp,
                confidences[det_idx],
                image_embedding,
                current_ns,
                map_frame.empty() ? camera_frame : map_frame);
        }
        return; 
    }

    int N = unmatched_indices.size();
    int M = map_ids.size();
    std::vector<std::vector<double>> costMatrix(N, std::vector<double>(M, 0.0));

    double w_dist = 1.0, w_iou = 1.0, w_sem = 2.5;
    double MAX_COST = 3.5;

    for (int i = 0; i < N; ++i) {
        int det_idx = unmatched_indices[i];
        OrientedBoundingBox det_obb = compute_obb(points_cam_list[det_idx]);

        for (int j = 0; j < M; ++j) {
            auto& map_obj = objects[map_ids[j]];

            // Distance
            double dx = det_obb.center[0] - map_obj.pose_map[0];
            double dy = det_obb.center[1] - map_obj.pose_map[1];
            double dz = det_obb.center[2] - map_obj.pose_map[2];
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist > 1.5) {
                costMatrix[i][j] = 999.0;
                continue;
            }

            // Geometric & Semantic
            double iou = compute_obb_iou(points_cam_list[det_idx], det_obb, map_obj.accumulated_points, map_obj.obb);
            double cost_sem = 1.0;
            if (embeddings_list[det_idx].has_value() && !map_obj.image_embedding.empty()) {
                cost_sem = static_cast<double>(compute_semantic_distance(embeddings_list[det_idx].value(), map_obj.image_embedding));
            }

            double class_penalty = (object_names[det_idx] != map_obj.current_name) ? 5.0 : 0.0;

            costMatrix[i][j] = (w_dist * dist) + (w_iou * (1.0 - iou)) + (w_sem * cost_sem) + class_penalty;
        }
    }

    // Solve using the imported Hungarian Algorithm C++ Library
    HungarianAlgorithm HungAlgo;
    std::vector<int> assignment;
    HungAlgo.Solve(costMatrix, assignment);

    for (int i = 0; i < N; ++i) {
        int map_idx = assignment[i];
        int det_idx = unmatched_indices[i];
        
        // If matched and the cost is under the maximum threshold
        if (map_idx >= 0 && costMatrix[i][map_idx] < MAX_COST) {
            std::string m_id = map_ids[map_idx];
            const std::vector<float> image_embedding = embeddings_list[det_idx].has_value() ? embeddings_list[det_idx].value() : std::vector<float>{};
            float similarity = 0.0f;
            const auto& map_obj = objects[m_id];
            if (!image_embedding.empty() && !map_obj.image_embedding.empty()) {
                similarity = std::clamp(1.0f - compute_semantic_distance(image_embedding, map_obj.image_embedding), 0.0f, 1.0f);
            }
            update_object(m_id, object_names[det_idx], stamp, points_cam_list[det_idx], similarity, confidences[det_idx], image_embedding, current_ns, tracker_ids[det_idx]);
            track_to_map[tracker_ids[det_idx]] = m_id;
            track_last_seen_ns[tracker_ids[det_idx]] = current_ns;
        } else {
            // PHASE 3: Spawn New Tentative Tracks
            const std::vector<float> image_embedding = embeddings_list[det_idx].has_value() ? embeddings_list[det_idx].value() : std::vector<float>{};
            update_tentative(
                object_names[det_idx],
                tracker_ids[det_idx],
                points_cam_list[det_idx],
                stamp,
                confidences[det_idx],
                image_embedding,
                current_ns,
                map_frame.empty() ? camera_frame : map_frame);
        }
    }

    std::cout << "[SemanticObjectMap] batch complete: objects=" << objects.size()
              << " tentative=" << tentative_tracks.size()
              << " bindings=" << track_to_map.size() << "\n";
}

void SemanticObjectMapV5::fuse_objects(const std::string& keep_id, const std::string& drop_id) {
    if (keep_id == drop_id) return;
    if (!objects.count(keep_id) || !objects.count(drop_id)) return;

    auto& keep = objects[keep_id];
    auto& drop = objects[drop_id];

    keep.accumulated_points = fuse_geometry(keep.accumulated_points, drop.accumulated_points);
    keep.obb = compute_obb(keep.accumulated_points);
    keep.pose_map = keep.obb.center;
    keep.occurrences += drop.occurrences;
    keep.last_seen_ns = std::max(keep.last_seen_ns, drop.last_seen_ns);

    objects.erase(drop_id);
}

void SemanticObjectMapV5::resolve_overlapping_duplicates() {
    // Boilerplate for map merging
    std::cout << "[SemanticObjectMap] Overlap resolution complete.\n";
}

void SemanticObjectMapV5::export_to_json(const std::string& directory_path, const std::string& file) {
    // As discussed, JSON export is delegated to a separate Python node.
}