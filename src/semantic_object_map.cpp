#include "mapper_pkg/semantic_object_map.hpp"
#include "mapper_pkg/Hungarian.h"
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <optional>
#include <array>
#include <set>
#include <sstream>
#include <iomanip>
#include <limits>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <rclcpp/rclcpp.hpp>

namespace {

const rclcpp::Logger semantic_logger = rclcpp::get_logger("SemanticObjectMapV5");

inline bool obb_has_valid_shape(const OrientedBoundingBox& obb) {
    return obb.center.size() >= 3 && obb.extents.size() >= 3 && obb.rotation.size() >= 9 &&
           obb.extents[0] > 0.0f && obb.extents[1] > 0.0f && obb.extents[2] > 0.0f;
}

inline Eigen::Matrix3f rotation_from_obb(const OrientedBoundingBox& obb) {
    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    if (obb.rotation.size() >= 9) {
        R(0, 0) = obb.rotation[0]; R(0, 1) = obb.rotation[1]; R(0, 2) = obb.rotation[2];
        R(1, 0) = obb.rotation[3]; R(1, 1) = obb.rotation[4]; R(1, 2) = obb.rotation[5];
        R(2, 0) = obb.rotation[6]; R(2, 1) = obb.rotation[7]; R(2, 2) = obb.rotation[8];
    }
    return R;
}

inline Eigen::Vector3f center_from_obb(const OrientedBoundingBox& obb) {
    return Eigen::Vector3f(obb.center[0], obb.center[1], obb.center[2]);
}

inline bool point_inside_obb(const pcl::PointXYZ& p, const OrientedBoundingBox& obb, float eps = 1e-3f) {
    if (!obb_has_valid_shape(obb)) {
        return false;
    }
    const Eigen::Matrix3f R = rotation_from_obb(obb);
    const Eigen::Vector3f c = center_from_obb(obb);
    const Eigen::Vector3f w(p.x, p.y, p.z);
    const Eigen::Vector3f local = R.transpose() * (w - c);
    const float hx = 0.5f * obb.extents[0] + eps;
    const float hy = 0.5f * obb.extents[1] + eps;
    const float hz = 0.5f * obb.extents[2] + eps;
    return std::fabs(local.x()) <= hx && std::fabs(local.y()) <= hy && std::fabs(local.z()) <= hz;
}

} // namespace

SemanticObjectMapV5::SemanticObjectMapV5() {
    RCLCPP_INFO(semantic_logger, "[SemanticObjectMap] Initialized with C++ Geometry and Tracking");
}

// Geometry and embedding helpers.

std::vector<float> SemanticObjectMapV5::normalize_embedding(const std::vector<float>& embedding) const {
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

float SemanticObjectMapV5::compute_embedding_similarity(const std::vector<float>& emb1, const std::vector<float>& emb2) {
    if (emb1.empty() || emb2.empty()) return 0.0f;
    // Dot product is cosine similarity because both vectors are normalized.
    float cosine_sim = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);
    return std::max(0.0f, std::min(1.0f, cosine_sim));
}

pcl::PointCloud<pcl::PointXYZ>::Ptr SemanticObjectMapV5::fuse_geometry(
    pcl::PointCloud<pcl::PointXYZ>::Ptr old_points, 
    pcl::PointCloud<pcl::PointXYZ>::Ptr new_points) 
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr combined(new pcl::PointCloud<pcl::PointXYZ>());
    if (!old_points || old_points->empty()) return new_points;
    if (!new_points || new_points->empty()) return old_points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_new(new pcl::PointCloud<pcl::PointXYZ>());
    *aligned_new = *new_points;
    // Simple fusion by concatenation followed by voxel downsampling to reduce duplicates and noise.
    *combined = *old_points;
    *combined += *aligned_new;
    // Adaptive voxelization
    int num_combined = combined->points.size();
    if (num_combined <= min_point_count) {
        voxel_size = min_voxel_size;
        sor_mean_k = min_sor_k;
        sor_stddev_mul_thresh = min_sor_stddev_mul_thresh;
    } else if (num_combined >= max_point_count) {
        voxel_size = max_voxel_size;
        sor_mean_k = max_sor_k;
        sor_stddev_mul_thresh = max_sor_stddev_mul_thresh;
    } else {
        const float t = static_cast<float>(num_combined - min_point_count) /
                        static_cast<float>(max_point_count - min_point_count);
        voxel_size = min_voxel_size + ((max_voxel_size - min_voxel_size) * t);
        sor_mean_k = static_cast<int>(min_sor_k + ((max_sor_k - min_sor_k) * t));
        sor_stddev_mul_thresh = min_sor_stddev_mul_thresh + ((max_sor_stddev_mul_thresh - min_sor_stddev_mul_thresh) * t);
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cleaned_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    // drop noisy "floating" points produced by sensor error.
    if (enable_statistical_outlier_removal) {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(combined);
        sor.setMeanK(sor_mean_k);                       // neighbors used to estimate local density
        sor.setStddevMulThresh(sor_stddev_mul_thresh);  // drop points beyond N std-devs of the mean distance
        sor.filter(*cleaned_cloud);
    } else {
        *cleaned_cloud = *combined;
    }
    if (enable_voxel_filtering) {
        // merge nearby points into voxels to reduce duplicates and density.
        pcl::VoxelGrid<pcl::PointXYZ> grid;
        grid.setInputCloud(cleaned_cloud);
        grid.setLeafSize(voxel_size, voxel_size, voxel_size);
        grid.filter(*downsampled);
    } else {
        *downsampled = *cleaned_cloud;
    }    
    return downsampled;
}

OrientedBoundingBox SemanticObjectMapV5::compute_obb(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    OrientedBoundingBox obb;
    if (!cloud || cloud->points.size() < 4) return obb;
    // Compute cloud centroid.
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cloud, pcaCentroid);
    // Compute PCA basis from covariance.
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    // Enforce a right-handed basis.
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
    // Transform cloud into PCA-aligned local frame.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudProjected(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud, *cloudProjected, projectionTransform);
    // Read extents from axis-aligned bounds in local space.
    pcl::PointXYZ minPoint, maxPoint;
    pcl::getMinMax3D(*cloudProjected, minPoint, maxPoint);
    obb.center = {pcaCentroid(0), pcaCentroid(1), pcaCentroid(2)};
    obb.extents = {maxPoint.x - minPoint.x, maxPoint.y - minPoint.y, maxPoint.z - minPoint.z};
    obb.rotation = {
        eigenVectorsPCA(0, 0), eigenVectorsPCA(0, 1), eigenVectorsPCA(0, 2),
        eigenVectorsPCA(1, 0), eigenVectorsPCA(1, 1), eigenVectorsPCA(1, 2),
        eigenVectorsPCA(2, 0), eigenVectorsPCA(2, 1), eigenVectorsPCA(2, 2)
    };
    return obb;
}

std::array<std::array<float, 3>, 8> SemanticObjectMapV5::compute_obb_corners(const OrientedBoundingBox& obb) const {
    std::array<std::array<float, 3>, 8> corners{};

    if (!obb_has_valid_shape(obb)) {
        return corners;
    }

    const Eigen::Matrix3f R = rotation_from_obb(obb);
    const Eigen::Vector3f c = center_from_obb(obb);
    const float hx = 0.5f * obb.extents[0];
    const float hy = 0.5f * obb.extents[1];
    const float hz = 0.5f * obb.extents[2];

    // Preserve corner ordering expected by downstream visualization and export.
    const std::array<Eigen::Vector3f, 8> local = {
        Eigen::Vector3f(-hx, -hy, -hz),
        Eigen::Vector3f(+hx, -hy, -hz),
        Eigen::Vector3f(-hx, +hy, -hz),
        Eigen::Vector3f(-hx, -hy, +hz),
        Eigen::Vector3f(+hx, +hy, +hz),
        Eigen::Vector3f(-hx, +hy, +hz),
        Eigen::Vector3f(+hx, -hy, +hz),
        Eigen::Vector3f(+hx, +hy, -hz)
    };

    for (size_t i = 0; i < local.size(); ++i) {
        const Eigen::Vector3f w = c + (R * local[i]);
        corners[i] = {w.x(), w.y(), w.z()};
    }

    return corners;
}

float SemanticObjectMapV5::oriented_overlap_ratio(
    pcl::PointCloud<pcl::PointXYZ>::Ptr points1, const OrientedBoundingBox& obb1,
    pcl::PointCloud<pcl::PointXYZ>::Ptr points2, const OrientedBoundingBox& obb2)
{
    if (!points1 || !points2 || points1->empty() || points2->empty()) return 0.0f;
    if (!obb_has_valid_shape(obb1) || !obb_has_valid_shape(obb2)) return 0.0f;

    // Subsample each cloud to at most a fixed number of points so the
    // point-in-OBB test stays cheap regardless of cloud size.
    const size_t max_eval_points = 50;
    
    size_t step1 = std::max<size_t>(1, points1->points.size() / max_eval_points);
    int in_1_in_2 = 0;
    int eval_1 = 0;
    for (size_t i = 0; i < points1->points.size(); i += step1) {
        if (point_inside_obb(points1->points[i], obb2)) {
            ++in_1_in_2;
        }
        ++eval_1;
    }

    size_t step2 = std::max<size_t>(1, points2->points.size() / max_eval_points);
    int in_2_in_1 = 0;
    int eval_2 = 0;
    for (size_t i = 0; i < points2->points.size(); i += step2) {
        if (point_inside_obb(points2->points[i], obb1)) {
            ++in_2_in_1;
        }
        ++eval_2;
    }

    const float r12 = static_cast<float>(in_1_in_2) / static_cast<float>(std::max(eval_1, 1));
    const float r21 = static_cast<float>(in_2_in_1) / static_cast<float>(std::max(eval_2, 1));
    
    return std::clamp(0.5f * (r12 + r21), 0.0f, 1.0f);
}

std::pair<double, double> SemanticObjectMapV5::association_weights(float max_size) const {
    // Small objects are sensitive to drift (high weights); large objects tolerate
    // wider centroid shifts (low weights). In between, interpolate linearly.
    if (max_size < association_small_object_max_size) {
        return {association_small_object_dist_weight, association_small_object_iou_weight};
    }
    if (max_size > association_large_object_min_size) {
        return {association_large_object_dist_weight, association_large_object_iou_weight};
    }
    const double span = std::max(static_cast<double>(association_large_object_min_size - association_small_object_max_size), 1e-6);
    const double ratio = (max_size - association_small_object_max_size) / span;
    const double w_dist = association_small_object_dist_weight - ((association_small_object_dist_weight - association_large_object_dist_weight) * ratio);
    const double w_iou = association_small_object_iou_weight - ((association_small_object_iou_weight - association_large_object_iou_weight) * ratio);
    return {w_dist, w_iou};
}

void SemanticObjectMapV5::refine_object_geometry(const std::string& map_id) {
    if (objects.find(map_id) == objects.end()) return;
    
    auto& obj = objects[map_id];
    if (obj.accumulated_points->points.size() < static_cast<size_t>(refine_min_point_count)) return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr refined_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // Use Euclidean clustering and keep the dominant cluster only.
    if (enable_clustering_refinement) {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(obj.accumulated_points);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(refine_cluster_tolerance);
        ec.setMinClusterSize(refine_min_cluster_size);
        ec.setMaxClusterSize(refine_max_cluster_size);
        ec.setSearchMethod(tree);
        ec.setInputCloud(obj.accumulated_points);
        ec.extract(cluster_indices);

        if (cluster_indices.empty()) return;

        // Cluster ordering is size-descending, so index 0 is the largest cluster.
        for (const auto& idx : cluster_indices[0].indices) {
            refined_cloud->points.push_back(obj.accumulated_points->points[idx]);
        }
    } else {
        refined_cloud = obj.accumulated_points;
    }

    int num_combined = refined_cloud->points.size();

    if (num_combined <= min_point_count) {
        sor_mean_k = min_sor_k;
        sor_stddev_mul_thresh = min_sor_stddev_mul_thresh;
    } else if (num_combined >= max_point_count) {
        sor_mean_k = max_sor_k;
        sor_stddev_mul_thresh = max_sor_stddev_mul_thresh;
    } else {
        const float t = static_cast<float>(num_combined - min_point_count) /
                        static_cast<float>(max_point_count - min_point_count);
        sor_mean_k = static_cast<int>(min_sor_k + ((max_sor_k - min_sor_k) * t));
        sor_stddev_mul_thresh = min_sor_stddev_mul_thresh + ((max_sor_stddev_mul_thresh - min_sor_stddev_mul_thresh) * t);
    }

    if (enable_sor_refinement) {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(refined_cloud);
        sor.setMeanK(sor_mean_k);
        sor.setStddevMulThresh(1.3f);
        sor.filter(*refined_cloud);
    } else {
        refined_cloud = obj.accumulated_points;
    }
    
    refined_cloud->width = refined_cloud->points.size();
    refined_cloud->height = 1;
    refined_cloud->is_dense = true;
    
    obj.accumulated_points = refined_cloud;
    obj.obb = compute_obb(obj.accumulated_points);
    obj.pose_map = obj.obb.center;
}

// State management and class consensus.

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
        return current_name;
    }
    return best_name;
}

void SemanticObjectMapV5::prune_stale_state(long long current_ns) {
    // Remove tentative tracks that haven't been updated recently.
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
            const std::string map_id = track_to_map.count(it->first) ? track_to_map[it->first] : std::string{};
            const std::string class_name = (!map_id.empty() && objects.count(map_id)) ? objects[map_id].current_name : std::string("<unknown>");
            track_to_map.erase(it->first);
            it = track_last_seen_ns.erase(it);
        } else {
            ++it;
        }
    }
}

// Core update logic.

void SemanticObjectMapV5::update_object(
    const std::string& map_id, const std::string& object_name,
    const builtin_interfaces::msg::Time& detection_stamp,
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_map,
    float similarity, float confidence,
    const std::vector<float>& image_embedding_masked,
    const std::vector<float>& image_embedding_unmasked,
    long long current_ns, const std::string& source_track_id) 
{
    auto& entry = objects[map_id];

    entry.accumulated_points = fuse_geometry(entry.accumulated_points, points_map);
    entry.obb = compute_obb(entry.accumulated_points); // recompute OBB after geometry fusion
    if (obb_has_valid_shape(entry.obb)) {
        entry.pose_map = entry.obb.center;
    } else {
        // Keep a valid centroid fallback so association code never indexes empty pose vectors.
        if (entry.accumulated_points && !entry.accumulated_points->empty()) {
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*entry.accumulated_points, centroid);
            entry.pose_map = {centroid[0], centroid[1], centroid[2]};
        } else {
            entry.pose_map = {0.0f, 0.0f, 0.0f};
        }
    }
    // Update class votes and confidence sums for consensus.
    entry.class_votes[object_name] += std::max(confidence, 0.01f);
    entry.class_counts[object_name] += 1;
    entry.class_conf_sums[object_name] += std::max(confidence, 0.01f);

    entry.current_name = choose_consensus_class(entry.class_counts, entry.class_conf_sums, entry.current_name);
    entry.confidence_ema = ((1.0f - confidence_ema_alpha) * entry.confidence_ema) + (confidence_ema_alpha * confidence);

    if (!image_embedding_masked.empty()) {
        entry.image_embedding_masked = fuse_embeddings_running_avg(entry.image_embedding_masked, entry.occurrences, image_embedding_masked, 1);
        entry.embedding_confidence_max = std::max(entry.embedding_confidence_max, confidence);
    }

    if (!image_embedding_unmasked.empty()) {
        entry.image_embedding_unmasked = fuse_embeddings_running_avg(entry.image_embedding_unmasked, entry.occurrences, image_embedding_unmasked, 1);
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
    RCLCPP_DEBUG(semantic_logger,
        "[SemanticObjectMap] map update map_id=%s class='%s' occurrences=%d conf=%.3f geometry_points=%zu",
        map_id.c_str(), entry.current_name.c_str(), entry.occurrences,
        entry.confidence_ema, entry.accumulated_points ? entry.accumulated_points->size() : 0);
}

bool SemanticObjectMapV5::update_tentative(
    const std::string& object_name,
    const std::string& tracker_id,
    pcl::PointCloud<pcl::PointXYZ>::Ptr points_map,
    const builtin_interfaces::msg::Time& detection_stamp,
    float confidence,
    const std::vector<float>& image_embedding_masked,
    const std::vector<float>& image_embedding_unmasked,
    long long current_ns,
    const std::string& frame)
{
    // Ignore detections that cannot contribute usable geometry.
    if (!points_map || points_map->empty()) {
        return false;
    }
    // Confidence gate keeps low-quality detections out of tentative state.
    if (confidence < min_input_confidence) {
        return false;
    }
    // Check if this tracker already has a tentative track record.
    auto it = tentative_tracks.find(tracker_id);
    if (it == tentative_tracks.end()) {
        // First sighting for this tracker: create tentative track record.
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
        if (!image_embedding_masked.empty()) {
            t.image_embedding_masked = normalize_embedding(image_embedding_masked);
            t.embedding_confidence_max = confidence;
        }
        if (!image_embedding_unmasked.empty()) {
            t.image_embedding_unmasked = normalize_embedding(image_embedding_unmasked);
        }
        tentative_tracks[tracker_id] = t;
        return false;
    }
    auto& t = it->second;
    // Existing tentative track: fuse evidence and update running statistics.
    t.accumulated_points = fuse_geometry(t.accumulated_points, points_map);
    t.hits += 1;
    t.last_seen_ns = current_ns;
    t.timestamp = detection_stamp;
    t.confidence_sum += confidence;
    t.confidence_max = std::max(t.confidence_max, confidence);
    if (!image_embedding_masked.empty()) {
        t.embedding_confidence_max = std::max(t.embedding_confidence_max, confidence);
        t.image_embedding_masked = fuse_embeddings_running_avg(t.image_embedding_masked, t.hits - 1, image_embedding_masked, 1);
    }
    if (!image_embedding_unmasked.empty()) {
        t.image_embedding_unmasked = fuse_embeddings_running_avg(t.image_embedding_unmasked, t.hits - 1, image_embedding_unmasked, 1);
    }
    if (confidence >= t.confidence_max || t.class_name.empty()) {
        t.class_name = object_name;
    }
    const double age_sec = static_cast<double>(t.last_seen_ns - t.first_seen_ns) / 1e9;
    const double avg_conf = t.confidence_sum / std::max(1, t.hits);
    // Check promotion criteria: if met, instantiate a new map object and bind the track to it.
    const bool promote =
        t.hits >= confirmation_min_hits &&
        age_sec >= confirmation_min_age_sec &&
        t.confidence_max >= min_confidence_for_promotion &&
        avg_conf >= min_avg_confidence_for_promotion;
    if (!promote) {
        return false;
    }
    std::string map_id = new_map_id();
    // Promotion path: instantiate a stable map object from tentative evidence.
    MapObject obj;
    obj.map_id = map_id;
    obj.frame = frame;
    obj.timestamp = detection_stamp;
    obj.accumulated_points = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    obj.first_seen_ns = current_ns;
    obj.last_seen_ns = current_ns;
    obj.current_name = t.class_name.empty() ? object_name : t.class_name;
    obj.image_embedding_masked = t.image_embedding_masked;
    obj.image_embedding_unmasked = t.image_embedding_unmasked;
    obj.embedding_confidence_max = t.embedding_confidence_max;
    obj.source_track_id = tracker_id;
    objects[map_id] = obj;
    // Finalize the new map object with the accumulated evidence from the tentative track.
    update_object(
        map_id,
        obj.current_name,
        detection_stamp,
        t.accumulated_points,
        0.0f,
        static_cast<float>(avg_conf),
        t.image_embedding_masked,
        t.image_embedding_unmasked,
        current_ns,
        tracker_id);
    track_to_map[tracker_id] = map_id;
    track_last_seen_ns[tracker_id] = current_ns;
    tentative_tracks.erase(it);
    return true;
}

// Batch update and bipartite association.

void SemanticObjectMapV5::add_detections_batch(
    const std::vector<std::string>& object_names,
    const std::vector<std::string>& tracker_ids,
    const std::vector<float>& confidences,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& points_cam_list,
    const std::vector<std::optional<std::vector<float>>>& embeddings_list_masked,
    const std::vector<std::optional<std::vector<float>>>& embeddings_list_unmasked,
    const builtin_interfaces::msg::Time& stamp,
    const std::string& camera_frame,
    const std::string& map_frame)
{

    if (points_cam_list.empty()) {
        RCLCPP_WARN(semantic_logger, "[SemanticObjectMap] batch skipped: empty input list");
        return;
    }

    if (object_names.size() != points_cam_list.size() ||
        tracker_ids.size() != points_cam_list.size() ||
        confidences.size() != points_cam_list.size() ||
        embeddings_list_masked.size() != points_cam_list.size() ||
        embeddings_list_unmasked.size() != points_cam_list.size()) {
        RCLCPP_ERROR(semantic_logger, "[SemanticObjectMap] batch size mismatch error");
        return;
    }
    long long current_ns = stamp_to_ns(stamp);
    if (do_prune) {
        prune_stale_state(current_ns);
    } else {
        RCLCPP_DEBUG(semantic_logger, "[SemanticObjectMap] batch skipping pruning (do_prune=false)");
    }
    std::vector<int> unmatched_indices;
    // First try direct tracker-to-map routing.
    for (size_t i = 0; i < points_cam_list.size(); ++i) {
        std::string t_id = tracker_ids[i];
        bool matched = false;        
        // Tracker is already bound to a map object: update it directly.
        if (track_to_map.count(t_id) && objects.count(track_to_map[t_id])) {
            std::string map_id = track_to_map[t_id];
            // Guard against class flips for reused tracker IDs.
            if (object_names[i] == objects[map_id].current_name) {
                const std::vector<float> image_embedding_masked = embeddings_list_masked[i].has_value() ? embeddings_list_masked[i].value() : std::vector<float>{};
                const std::vector<float> image_embedding_unmasked = embeddings_list_unmasked[i].has_value() ? embeddings_list_unmasked[i].value() : std::vector<float>{};
                float similarity = 0.0f;
                const auto& map_obj = objects[map_id];
                if (!image_embedding_masked.empty() && !map_obj.image_embedding_masked.empty()) {
                    similarity = compute_embedding_similarity(image_embedding_masked, map_obj.image_embedding_masked) * 100.0f;
                }
                update_object(map_id, object_names[i], stamp, points_cam_list[i], similarity, confidences[i], image_embedding_masked, image_embedding_unmasked, current_ns, t_id);
                track_last_seen_ns[t_id] = current_ns;
                matched = true;
            }
        }
        // No usable binding yet: defer to association/tentative routing below.
        if (!matched) {
            unmatched_indices.push_back(i);
        }
    }
    if (unmatched_indices.empty()) return;
    // Associate remaining detections with map objects using Hungarian matching.
    std::vector<std::string> map_ids;
    // Build a list of map IDs in the same order as the cost matrix columns.
    for (const auto& pair : objects) map_ids.push_back(pair.first);
    if (map_ids.empty()) {
    // No existing map objects to associate with, so route all unmatched detections to tentative tracks.
        for (int det_idx : unmatched_indices) {
            const std::vector<float> image_embedding_masked = embeddings_list_masked[det_idx].has_value() ? embeddings_list_masked[det_idx].value() : std::vector<float>{};
            const std::vector<float> image_embedding_unmasked = embeddings_list_unmasked[det_idx].has_value() ? embeddings_list_unmasked[det_idx].value() : std::vector<float>{};
            update_tentative(
                object_names[det_idx],
                tracker_ids[det_idx],
                points_cam_list[det_idx],
                stamp,
                confidences[det_idx],
                image_embedding_masked,
                image_embedding_unmasked,
                current_ns,
                map_frame.empty() ? camera_frame : map_frame);
        }
        return; 
    }
    // N: unmatched detections, M: currently tracked map objects.
    int N = unmatched_indices.size();
    int M = map_ids.size();
    std::vector<std::vector<double>> costMatrix(N, std::vector<double>(M, kBlockedCost));
    for (int i = 0; i < N; ++i) {
        int det_idx = unmatched_indices[i];
        // Compute detection geometry once and reuse it for all candidate objects.
        OrientedBoundingBox det_obb = compute_obb(points_cam_list[det_idx]);
        if (!obb_has_valid_shape(det_obb)) {
            continue;
        }
        // cheap screening to prune impossible/weak candidates.
        struct CheapCandidate {
            double cost;
            int index;
            double dist;
            float max_size;
        };
        std::vector<CheapCandidate> cheap_candidates;
        cheap_candidates.reserve(M);
        for (int j = 0; j < M; ++j) {
            auto& map_obj = objects[map_ids[j]];
            const bool has_pose = map_obj.pose_map.size() >= 3;
            const bool has_obb = obb_has_valid_shape(map_obj.obb);
            if (!has_pose || !has_obb) {
                RCLCPP_WARN(semantic_logger,
                    "[SemanticObjectMap] batch skip geom: invalid map geometry map_id=%s has_pose=%s has_obb=%s",
                    map_ids[j].c_str(), has_pose ? "1" : "0", has_obb ? "1" : "0");
                costMatrix[i][j] = kBlockedCost;
                continue;
            }
            // Center-to-center distance in map frame.
            double dx = det_obb.center[0] - map_obj.pose_map[0];
            double dy = det_obb.center[1] - map_obj.pose_map[1];
            double dz = det_obb.center[2] - map_obj.pose_map[2];
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            // Scale-aware distance weight (IoU weight is only needed in stage 2).
            float max_size_det = std::max({det_obb.extents[0], det_obb.extents[1], det_obb.extents[2]});
            float max_size_map = std::max({map_obj.obb.extents[0], map_obj.obb.extents[1], map_obj.obb.extents[2]});
            float max_size = std::max(max_size_det, max_size_map);
            const double dynamic_w_dist = association_weights(max_size).first;
            // Hard spatial gate to avoid evaluating unrealistic matches.
            double dynamic_max_dist = std::max(0.4f, max_size * 1.2f);
            if (dist > dynamic_max_dist) {
                costMatrix[i][j] = kBlockedCost;
                continue;
            }
            double class_penalty = (object_names[det_idx] != map_obj.current_name) ? max_class_penalty : 0.0;
            const double cheap_cost = (dynamic_w_dist * dist) + class_penalty;
            cheap_candidates.push_back({cheap_cost, j, dist, max_size});
            // Default to blocked; only top-K survivors get a full final score.
            costMatrix[i][j] = kBlockedCost;
        }
        if (cheap_candidates.empty()) {
            continue;
        }
        // Keep only top-K cheapest candidates for expensive IoU+embedding scoring.
        const int keep = std::min<int>(kTopKPerDetection, static_cast<int>(cheap_candidates.size()));
        if (keep < static_cast<int>(cheap_candidates.size())) {
            std::nth_element(
                cheap_candidates.begin(),
                cheap_candidates.begin() + keep,
                cheap_candidates.end(),
                [](const auto& a, const auto& b) { return a.cost < b.cost; });
        } else {
            std::sort(
                cheap_candidates.begin(),
                cheap_candidates.end(),
                [](const auto& a, const auto& b) { return a.cost < b.cost; });
        }
        for (int c = 0; c < keep; ++c) {
            const int j = cheap_candidates[c].index;
            const double dist = cheap_candidates[c].dist;
            const float max_size = cheap_candidates[c].max_size;
            auto& map_obj = objects[map_ids[j]];
            // Compute full cost on shortlisted pairs only.
            const auto [dynamic_w_dist, dynamic_w_iou] = association_weights(max_size);
            // Geometric overlap and semantic similarity terms.
            double iou = oriented_overlap_ratio(points_cam_list[det_idx], det_obb, map_obj.accumulated_points, map_obj.obb);
            double cost_sem = 1.0;
            if (embeddings_list_masked[det_idx].has_value() && !map_obj.image_embedding_masked.empty()) {
                float similarity = compute_embedding_similarity(embeddings_list_masked[det_idx].value(), map_obj.image_embedding_masked);
                cost_sem = 1.0 - static_cast<double>(similarity);
            }
            double class_penalty = (object_names[det_idx] != map_obj.current_name) ? max_class_penalty : 0.0;
            // Final combined assignment cost.
            costMatrix[i][j] = (dynamic_w_dist * dist) + (dynamic_w_iou * (1.0 - iou)) + (w_sem * cost_sem) + class_penalty;
        }
    }
    // Global one-to-one assignment over the final cost matrix.
    HungarianAlgorithm HungAlgo;
    std::vector<int> assignment;
    HungAlgo.Solve(costMatrix, assignment);

    for (int i = 0; i < N; ++i) {
        int map_idx = assignment[i];
        int det_idx = unmatched_indices[i];
        const std::vector<float> image_embedding_masked = embeddings_list_masked[det_idx].has_value() ? embeddings_list_masked[det_idx].value() : std::vector<float>{};
        const std::vector<float> image_embedding_unmasked = embeddings_list_unmasked[det_idx].has_value() ? embeddings_list_unmasked[det_idx].value() : std::vector<float>{};
        // Accept only assignments that exist and pass the quality threshold.
        if (map_idx >= 0 && costMatrix[i][map_idx] < MAX_COST) {
            std::string m_id = map_ids[map_idx];
            float similarity = 0.0f;
            const auto& map_obj = objects[m_id];
            if (!image_embedding_masked.empty() && !map_obj.image_embedding_masked.empty()) {
                similarity = compute_embedding_similarity(image_embedding_masked, map_obj.image_embedding_masked) * 100.0f;
            }
            // Update matched stable object and refresh tracker binding.
            update_object(m_id, object_names[det_idx], stamp, points_cam_list[det_idx], similarity, confidences[det_idx], image_embedding_masked, image_embedding_unmasked, current_ns, tracker_ids[det_idx]);
            track_to_map[tracker_ids[det_idx]] = m_id;
            track_last_seen_ns[tracker_ids[det_idx]] = current_ns;
        } else {
            // Rejected/unassigned detections keep accumulating as tentative tracks.
            update_tentative(
                object_names[det_idx],
                tracker_ids[det_idx],
                points_cam_list[det_idx],
                stamp,
                confidences[det_idx],
                image_embedding_masked,
                image_embedding_unmasked,
                current_ns,
                map_frame.empty() ? camera_frame : map_frame);
        }
    }
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
    if (objects.size() < 2) return;

    // Copy keys first so merges can safely erase from the map.
    std::vector<std::string> ids;
    for (const auto& pair : objects) {
        ids.push_back(pair.first);
    }

    // Track dropped IDs to skip them in the current pass.
    std::set<std::string> to_remove;

    for (size_t i = 0; i < ids.size(); ++i) {
        const std::string& id_a = ids[i];
        if (to_remove.count(id_a)) continue;

        for (size_t j = i + 1; j < ids.size(); ++j) {
            const std::string& id_b = ids[j];
            if (to_remove.count(id_b)) continue;

            auto& obj_a = objects[id_a];
            auto& obj_b = objects[id_b];

            // Merge candidates must agree on class label.
            if (obj_a.current_name != obj_b.current_name) {
                continue;
            }

            float iou = oriented_overlap_ratio(
                obj_a.accumulated_points, obj_a.obb,
                obj_b.accumulated_points, obj_b.obb
            );

            // Require very high overlap to avoid accidental merges.
            if (iou > 0.80f) {
                std::string keep_id, drop_id;
                if (obj_a.occurrences >= obj_b.occurrences) {
                    keep_id = id_a;
                    drop_id = id_b;
                } else {
                    keep_id = id_b;
                    drop_id = id_a;
                }

                fuse_objects(keep_id, drop_id);
                
                to_remove.insert(drop_id);

                if (drop_id == id_a) {
                    break;
                }
            }
        }
    }
    
    RCLCPP_DEBUG(semantic_logger,
        "[SemanticObjectMap] cleanup removed_strict_duplicates=%zu",
        to_remove.size());
}

void SemanticObjectMapV5::remove_wrong_detections() {
    if (objects.empty()) return;

    // Use the newest object timestamp as the reference "now".
    long long current_ns = 0;
    for (const auto& pair : objects) {
        current_ns = std::max(current_ns, pair.second.last_seen_ns);
    }

    int removed_count = 0;

    for (auto it = objects.begin(); it != objects.end(); ) {
        
        double age_sec = static_cast<double>(current_ns - it->second.first_seen_ns) / 1e9;

        if (age_sec > kMaxAgeSec && it->second.occurrences < kMinOccurrences) {
            it = objects.erase(it);
            removed_count++;
            
        } else {
            ++it;
        }
    }

    if (removed_count > 0) {
        RCLCPP_DEBUG(semantic_logger,
            "[SemanticObjectMap] cleanup removed_wrong_detections=%d", removed_count);
    }
}

// Goal text embedding and similarity scoring.

void SemanticObjectMapV5::set_text_embedding(const std::vector<float>& emb, float scale, float bias) {
    goal_text_embedding_ = emb;
    logit_scale_ = scale;
    logit_bias_ = bias;
}

float SemanticObjectMapV5::get_goal_similarity(const std::string& map_id) const {
    if (goal_text_embedding_.empty()) return 0.0f;
    
    auto it = objects.find(map_id);
    if (it == objects.end()) return 0.0f;
    
    const auto& img_emb_masked = it->second.image_embedding_masked;
    const auto& img_emb_unmasked = it->second.image_embedding_unmasked;

    float score_masked = 0.0f;
    if (!img_emb_masked.empty()) {
        float dot_m = std::inner_product(img_emb_masked.begin(), img_emb_masked.end(), goal_text_embedding_.begin(), 0.0f);
        float logits_m = (dot_m * logit_scale_) + logit_bias_;
        float clipped_m = std::clamp(logits_m, -60.0f, 60.0f);
        score_masked = (1.0f / (1.0f + std::exp(-clipped_m))) * 100.0f;
    }

    float score_unmasked = 0.0f;
    if (!img_emb_unmasked.empty()) {
        float dot_u = std::inner_product(img_emb_unmasked.begin(), img_emb_unmasked.end(), goal_text_embedding_.begin(), 0.0f);
        float logits_u = (dot_u * logit_scale_) + logit_bias_;
        float clipped_u = std::clamp(logits_u, -60.0f, 60.0f);
        score_unmasked = (1.0f / (1.0f + std::exp(-clipped_u))) * 100.0f;
    }

    // Keep the same weighting behavior as the paired Python node.
    return (1.0f * score_masked) + (0.0f * score_unmasked);
}