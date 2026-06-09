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
#include <pcl/registration/icp.h>
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

inline void log_detection_fate(
    const char* stage,
    const std::string& track_id,
    const std::string& class_name,
    float confidence,
    const std::string& extra = "")
{
    std::ostringstream oss;
    oss << "[SemanticObjectMap][" << stage << "]"
        << " track=" << track_id
        << " class='" << class_name << "'"
        << " conf=" << std::fixed << std::setprecision(3) << confidence;
    if (!extra.empty()) {
        oss << ' ' << extra;
    }
    RCLCPP_DEBUG_STREAM(semantic_logger, oss.str());
}

inline void log_map_fate(
    const char* stage,
    const std::string& map_id,
    const std::string& class_name,
    const std::string& extra = "")
{
    std::ostringstream oss;
    oss << "[SemanticObjectMap][" << stage << "]"
        << " map_id=" << map_id
        << " class='" << class_name << "'";
    if (!extra.empty()) {
        oss << ' ' << extra;
    }
    RCLCPP_DEBUG_STREAM(semantic_logger, oss.str());
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

    if (enable_icp_fusion_) {
    // ---------------------------------------------------------
    // STEP 1: Alignment (ICP)
    // Align the new points to the existing old points
    // ---------------------------------------------------------
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(new_points);
        icp.setInputTarget(old_points);
    
        // Crucial: Set a maximum distance to prevent it from matching 
        // points that are completely unrelated (e.g., 5cm)
        icp.setMaxCorrespondenceDistance(0.2); 
        icp.setMaximumIterations(20);

    
        icp.align(*aligned_new);
        // If ICP fails to find a good alignment, we skip fusion to avoid corrupting the map
        if (!icp.hasConverged()) {
            // Handle failure (e.g., log a warning and return old points)
            return old_points; 
        }
    } else {
        *aligned_new = *new_points;
    }

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

    // ---------------------------------------------------------
    // STEP 4: Statistical Outlier Removal (Cleanup)
    // Removes noisy "floating" points generated by sensor error
    // ---------------------------------------------------------
    

    if (enable_statistical_outlier_removal) {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(combined);
        sor.setMeanK(sor_mean_k);             // Look at 50 neighbors
        sor.setStddevMulThresh(sor_stddev_mul_thresh);  // Remove points further than 1 standard dev from the mean distance
        sor.filter(*cleaned_cloud);
    } else {
        *cleaned_cloud = *combined;
    }

    if (enable_voxel_filtering) {
        // ---------------------------------------------------------
        // STEP 2: Voxel Grid Downsampling (Duplication Reduction)
        // Reduces the number of points by merging nearby points into voxels.
        // ---------------------------------------------------------
        
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

float SemanticObjectMapV5::compute_obb_iou(
    pcl::PointCloud<pcl::PointXYZ>::Ptr points1, const OrientedBoundingBox& obb1,
    pcl::PointCloud<pcl::PointXYZ>::Ptr points2, const OrientedBoundingBox& obb2) 
{
    // Orientation-aware overlap proxy used by association and duplicate checks.
    return oriented_overlap_ratio(points1, obb1, points2, obb2);
}

float SemanticObjectMapV5::oriented_overlap_ratio(
    pcl::PointCloud<pcl::PointXYZ>::Ptr points1, const OrientedBoundingBox& obb1,
    pcl::PointCloud<pcl::PointXYZ>::Ptr points2, const OrientedBoundingBox& obb2)
{
    if (!points1 || !points2 || points1->empty() || points2->empty()) return 0.0f;
    if (!obb_has_valid_shape(obb1) || !obb_has_valid_shape(obb2)) return 0.0f;

    // SOTA Optimization: Uniformly sample max 100 points per cloud for IoU evaluation.
    // This reduces point-in-OBB matrix operations from ~10,000 to 100 per check.
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
            log_detection_fate(
                "drop_stale_tentative",
                it->first,
                it->second.class_name,
                it->second.confidence_max,
                "reason=stale_timeout");
            it = tentative_tracks.erase(it);
        } else {
            ++it;
        }
    }

    for (auto it = track_last_seen_ns.begin(); it != track_last_seen_ns.end(); ) {
        if (current_ns - it->second > stale_binding_ns) {
            const std::string map_id = track_to_map.count(it->first) ? track_to_map[it->first] : std::string{};
            const std::string class_name = (!map_id.empty() && objects.count(map_id)) ? objects[map_id].current_name : std::string("<unknown>");
            log_map_fate(
                "drop_stale_binding",
                map_id.empty() ? std::string("<unbound>") : map_id,
                class_name,
                std::string("track=") + it->first + " reason=stale_timeout");
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

        RCLCPP_WARN(semantic_logger,
            "[SemanticObjectMap] invalid OBB for map_id=%s after geometry fusion, using centroid fallback pose",
            map_id.c_str());
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

    // Keep the highest goal similarity reported by the vision node for this object.
    if (std::isfinite(similarity)) {
        entry.similarity = std::max(entry.similarity, similarity);
    }

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
    float similarity,
    const std::vector<float>& image_embedding_masked,
    const std::vector<float>& image_embedding_unmasked,
    long long current_ns,
    const std::string& frame)
{
    // Ignore detections that cannot contribute usable geometry.
    if (!points_map || points_map->empty()) {
        log_detection_fate("drop_empty_cloud", tracker_id, object_name, confidence, "reason=empty_geometry");
        return false;
    }

    // Confidence gate keeps low-quality detections out of tentative state.
    if (confidence < min_input_confidence) {
        log_detection_fate("drop_low_confidence", tracker_id, object_name, confidence,
            std::string("threshold=") + std::to_string(min_input_confidence));
        return false;
    }

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
        t.similarity_max = std::isfinite(similarity) ? similarity : 0.0f;
        if (!image_embedding_masked.empty()) {
            t.image_embedding_masked = normalize_embedding(image_embedding_masked);
            t.embedding_confidence_max = confidence;
        }
        if (!image_embedding_unmasked.empty()) {
            t.image_embedding_unmasked = normalize_embedding(image_embedding_unmasked);
        }
        tentative_tracks[tracker_id] = t;

        log_detection_fate("tentative_create", tracker_id, object_name, confidence, "hits=1");
        RCLCPP_DEBUG(semantic_logger,
            "[SemanticObjectMap] new tentative track=%s class='%s' conf=%.3f points=%zu",
            tracker_id.c_str(), object_name.c_str(), confidence, points_map->size());
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
    if (std::isfinite(similarity)) {
        t.similarity_max = std::max(t.similarity_max, similarity);
    }
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

    const bool promote =
        t.hits >= confirmation_min_hits &&
        age_sec >= confirmation_min_age_sec &&
        t.confidence_max >= min_confidence_for_promotion &&
        avg_conf >= min_avg_confidence_for_promotion;

    if (!promote) {
        log_detection_fate("tentative_update", tracker_id, object_name, confidence,
            std::string("hits=") + std::to_string(t.hits) +
            " age_sec=" + std::to_string(age_sec) +
            " avg_conf=" + std::to_string(avg_conf) +
            " max_conf=" + std::to_string(t.confidence_max));
        RCLCPP_DEBUG(semantic_logger,
            "[SemanticObjectMap] update tentative track=%s hits=%d age=%.2fs avg_conf=%.3f (not ready for promotion)",
            tracker_id.c_str(), t.hits, age_sec, avg_conf);
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

    update_object(
        map_id,
        obj.current_name,
        detection_stamp,
        t.accumulated_points,
        t.similarity_max,
        static_cast<float>(avg_conf),
        t.image_embedding_masked,
        t.image_embedding_unmasked,
        current_ns,
        tracker_id);

    track_to_map[tracker_id] = map_id;
    track_last_seen_ns[tracker_id] = current_ns;
    tentative_tracks.erase(it);

    log_map_fate("promote_tentative", map_id, obj.current_name,
        std::string("track=") + tracker_id + " hits=" + std::to_string(obj.occurrences));
    RCLCPP_DEBUG(semantic_logger,
        "[SemanticObjectMap] promoted to map new_map_id=%s class='%s' from_track=%s hits=%d age=%.2fs avg_conf=%.3f",
        map_id.c_str(), obj.current_name.c_str(), tracker_id.c_str(), t.hits, age_sec, avg_conf);
    return true;
}

// Batch update and bipartite association.

void SemanticObjectMapV5::add_detections_batch(
    const std::vector<std::string>& object_names,
    const std::vector<std::string>& tracker_ids,
    const std::vector<float>& confidences,
    const std::vector<float>& similarities,
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& points_cam_list,
    const std::vector<std::optional<std::vector<float>>>& embeddings_list_masked,
    const std::vector<std::optional<std::vector<float>>>& embeddings_list_unmasked,
    const builtin_interfaces::msg::Time& stamp,
    const std::string& camera_frame,
    const std::string& map_frame)
{
    RCLCPP_DEBUG(semantic_logger,
        "[SemanticObjectMap] batch start detections=%zu current_map_objects=%zu tentative_tracks=%zu",
        points_cam_list.size(), objects.size(), tentative_tracks.size());

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
    prune_stale_state(current_ns);

    RCLCPP_DEBUG(semantic_logger,
        "[SemanticObjectMap] batch state objects=%zu tentative=%zu bindings=%zu",
        objects.size(), tentative_tracks.size(), track_to_map.size());

    std::vector<int> unmatched_indices;

    // First try direct tracker-to-map routing.
    for (size_t i = 0; i < points_cam_list.size(); ++i) {
        std::string t_id = tracker_ids[i];
        bool matched = false;

        log_detection_fate("enter", t_id, object_names[i], confidences[i], std::string("batch_idx=") + std::to_string(i));
        
        if (track_to_map.count(t_id) && objects.count(track_to_map[t_id])) {
            //I already have the traker ID in the map
            std::string map_id = track_to_map[t_id];
            
            // Guard against class flips for reused tracker IDs.
            if (object_names[i] == objects[map_id].current_name) {
                // I have an object with same ID and same class in the map, so I will update it with the new detection and embedding.
                const std::vector<float> image_embedding_masked = embeddings_list_masked[i].has_value() ? embeddings_list_masked[i].value() : std::vector<float>{};
                const std::vector<float> image_embedding_unmasked = embeddings_list_unmasked[i].has_value() ? embeddings_list_unmasked[i].value() : std::vector<float>{};
                float similarity = 0.0f;
                const auto& map_obj = objects[map_id];
                if (!image_embedding_masked.empty() && !map_obj.image_embedding_masked.empty()) {
                    similarity = compute_embedding_similarity(image_embedding_masked, map_obj.image_embedding_masked) * 100.0f;
                }
                log_map_fate("merge_direct", map_id, objects[map_id].current_name,
                    std::string("track=") + t_id +
                    " det_class='" + object_names[i] +
                    "' conf=" + std::to_string(confidences[i]) +
                    " emb_similarity=" + std::to_string(similarity) +
                    " reason=existing_track_binding_and_class_match");
                RCLCPP_DEBUG(semantic_logger,
                    "[SemanticObjectMap] added to map track=%s -> map_id=%s class='%s' conf=%.3f",
                    t_id.c_str(), map_id.c_str(), object_names[i].c_str(), confidences[i]);
                update_object(map_id, object_names[i], stamp, points_cam_list[i], similarities[i], confidences[i], image_embedding_masked, image_embedding_unmasked, current_ns, t_id);
                track_last_seen_ns[t_id] = current_ns;
                matched = true;
            } else {
                log_detection_fate("reject_direct", t_id, object_names[i], confidences[i],
                    std::string("reason=class_mismatch map_id=") + map_id + " map_class='" + objects[map_id].current_name + "'");
            }
        } 

        if (!matched) {
            // I don't have the tracker ID in the map
            unmatched_indices.push_back(i);
        }
    }

    if (unmatched_indices.empty()) return;

    RCLCPP_DEBUG(semantic_logger,
        "[SemanticObjectMap] batch unmatched after direct routing=%zu",
        unmatched_indices.size());

    // Associate remaining detections with map objects using Hungarian matching.
    std::vector<std::string> map_ids;
    for (const auto& pair : objects) map_ids.push_back(pair.first);

    if (map_ids.empty()) {
    // No existing map objects to associate with, so route all unmatched detections to tentative tracks.
        RCLCPP_DEBUG(semantic_logger,
            "[SemanticObjectMap] no map objects exist; routing %zu unmatched detections to tentative",
            unmatched_indices.size());
        for (int det_idx : unmatched_indices) {
            const std::vector<float> image_embedding_masked = embeddings_list_masked[det_idx].has_value() ? embeddings_list_masked[det_idx].value() : std::vector<float>{};
            const std::vector<float> image_embedding_unmasked = embeddings_list_unmasked[det_idx].has_value() ? embeddings_list_unmasked[det_idx].value() : std::vector<float>{};
            log_detection_fate("route_tentative", tracker_ids[det_idx], object_names[det_idx], confidences[det_idx], "reason=map_empty");
            RCLCPP_DEBUG(semantic_logger,
                "[SemanticObjectMap] tentative route track=%s class='%s' conf=%.3f points=%zu",
                tracker_ids[det_idx].c_str(), object_names[det_idx].c_str(), confidences[det_idx],
                points_cam_list[det_idx] ? points_cam_list[det_idx]->size() : 0);
            update_tentative(
                object_names[det_idx],
                tracker_ids[det_idx],
                points_cam_list[det_idx],
                stamp,
                confidences[det_idx],
                similarities[det_idx],
                image_embedding_masked,
                image_embedding_unmasked,
                current_ns,
                map_frame.empty() ? camera_frame : map_frame);
        }
        RCLCPP_DEBUG(semantic_logger,
            "[SemanticObjectMap] batch complete via tentative routing objects=%zu tentative=%zu bindings=%zu",
            objects.size(), tentative_tracks.size(), track_to_map.size());
        return; 
    }

    // N: unmatched detections, M: currently tracked map objects.
    int N = unmatched_indices.size();
    int M = map_ids.size();
    RCLCPP_DEBUG(semantic_logger,
        "[SemanticObjectMap] Hungarian matching N=%d unmatched detections vs M=%d map objects", N, M);
    std::vector<std::vector<double>> costMatrix(N, std::vector<double>(M, kBlockedCost));
    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::vector<double>> distMatrix(N, std::vector<double>(M, nan_value));
    std::vector<std::vector<double>> iouMatrix(N, std::vector<double>(M, nan_value));
    std::vector<std::vector<double>> semCostMatrix(N, std::vector<double>(M, nan_value));
    std::vector<std::vector<double>> classPenaltyMatrix(N, std::vector<double>(M, nan_value));
    std::vector<std::vector<double>> wDistMatrix(N, std::vector<double>(M, nan_value));
    std::vector<std::vector<double>> wIouMatrix(N, std::vector<double>(M, nan_value));
    for (int i = 0; i < N; ++i) {
        int det_idx = unmatched_indices[i];
        // Compute detection geometry once and reuse it for all candidate objects.
        OrientedBoundingBox det_obb = compute_obb(points_cam_list[det_idx]);

        if (!obb_has_valid_shape(det_obb)) {
            log_detection_fate("drop_invalid_geom", tracker_ids[det_idx], object_names[det_idx], confidences[det_idx],
                std::string("reason=invalid_detection_obb points=") + std::to_string(points_cam_list[det_idx] ? points_cam_list[det_idx]->size() : 0));
            continue;
        }

        // Stage 1: cheap screening to prune impossible/weak candidates.
        std::vector<std::pair<double, int>> cheap_candidates;
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
            
            // Scale-aware weighting: strict for small objects, permissive for large ones.
            float max_size_det = std::max({det_obb.extents[0], det_obb.extents[1], det_obb.extents[2]});
            float max_size_map = std::max({map_obj.obb.extents[0], map_obj.obb.extents[1], map_obj.obb.extents[2]});
            float max_size = std::max(max_size_det, max_size_map);

            const float small_size = association_small_object_max_size;
            const float large_size = association_large_object_min_size;
            const double small_dist_weight = association_small_object_dist_weight;
            const double small_iou_weight = association_small_object_iou_weight;
            const double large_dist_weight = association_large_object_dist_weight;
            const double large_iou_weight = association_large_object_iou_weight;

            double dynamic_w_dist = small_dist_weight;
            double dynamic_w_iou = small_iou_weight;
            if (max_size < small_size) {
                // Small objects are sensitive to drift.
                dynamic_w_dist = small_dist_weight; 
                dynamic_w_iou = small_iou_weight;
            } else if (max_size > large_size) {
                // Large objects can tolerate wider centroid shifts.
                dynamic_w_dist = large_dist_weight;
                dynamic_w_iou = large_iou_weight;
            } else {
                // Interpolate smoothly between the two regimes.
                const double span = std::max(static_cast<double>(large_size - small_size), 1e-6);
                const double ratio = (max_size - small_size) / span;
                dynamic_w_dist = small_dist_weight - ((small_dist_weight - large_dist_weight) * ratio);
                dynamic_w_iou = small_iou_weight - ((small_iou_weight - large_iou_weight) * ratio);
            }

            // Hard spatial gate to avoid evaluating unrealistic matches.
            double dynamic_max_dist = std::max(0.4f, max_size * 1.2f);
            if (dist > dynamic_max_dist) {
                costMatrix[i][j] = kBlockedCost;
                continue;
            }

            double class_penalty = (object_names[det_idx] != map_obj.current_name) ? max_class_penalty : 0.0;
            const double cheap_cost = (dynamic_w_dist * dist) + class_penalty;
            cheap_candidates.push_back({cheap_cost, j});
            // Default to blocked; only top-K survivors get a full final score.
            costMatrix[i][j] = kBlockedCost;
        }

        if (cheap_candidates.empty()) {
            log_detection_fate("no_candidates", tracker_ids[det_idx], object_names[det_idx], confidences[det_idx], "reason=all_blocked_before_hungarian");
            continue;
        }

        // Keep only top-K cheapest candidates for expensive IoU+embedding scoring.
        const int keep = std::min<int>(kTopKPerDetection, static_cast<int>(cheap_candidates.size()));
        if (keep < static_cast<int>(cheap_candidates.size())) {
            std::nth_element(
                cheap_candidates.begin(),
                cheap_candidates.begin() + keep,
                cheap_candidates.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        } else {
            std::sort(
                cheap_candidates.begin(),
                cheap_candidates.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        }

        for (int c = 0; c < keep; ++c) {
            const int j = cheap_candidates[c].second;
            auto& map_obj = objects[map_ids[j]];

            const bool has_pose = map_obj.pose_map.size() >= 3;
            const bool has_obb = obb_has_valid_shape(map_obj.obb);
            if (!has_pose || !has_obb) {
                costMatrix[i][j] = kBlockedCost;
                continue;
            }

            // Stage 2: compute full cost on shortlisted pairs only.
            double dx = det_obb.center[0] - map_obj.pose_map[0];
            double dy = det_obb.center[1] - map_obj.pose_map[1];
            double dz = det_obb.center[2] - map_obj.pose_map[2];
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

            float max_size_det = std::max({det_obb.extents[0], det_obb.extents[1], det_obb.extents[2]});
            float max_size_map = std::max({map_obj.obb.extents[0], map_obj.obb.extents[1], map_obj.obb.extents[2]});
            float max_size = std::max(max_size_det, max_size_map);

            const float small_size = association_small_object_max_size;
            const float large_size = association_large_object_min_size;
            const double small_dist_weight = association_small_object_dist_weight;
            const double small_iou_weight = association_small_object_iou_weight;
            const double large_dist_weight = association_large_object_dist_weight;
            const double large_iou_weight = association_large_object_iou_weight;

            double dynamic_w_dist = small_dist_weight;
            double dynamic_w_iou = small_iou_weight;
            if (max_size < small_size) {
                dynamic_w_dist = small_dist_weight;
                dynamic_w_iou = small_iou_weight;
            } else if (max_size > large_size) {
                dynamic_w_dist = large_dist_weight;
                dynamic_w_iou = large_iou_weight;
            } else {
                const double span = std::max(static_cast<double>(large_size - small_size), 1e-6);
                const double ratio = (max_size - small_size) / span;
                dynamic_w_dist = small_dist_weight - ((small_dist_weight - large_dist_weight) * ratio);
                dynamic_w_iou = small_iou_weight - ((small_iou_weight - large_iou_weight) * ratio);
            }

            // Geometric overlap and semantic similarity terms.
            double iou = compute_obb_iou(points_cam_list[det_idx], det_obb, map_obj.accumulated_points, map_obj.obb);
            double cost_sem = 1.0;
            if (embeddings_list_masked[det_idx].has_value() && !map_obj.image_embedding_masked.empty()) {
                float similarity = compute_embedding_similarity(embeddings_list_masked[det_idx].value(), map_obj.image_embedding_masked);
                cost_sem = 1.0 - static_cast<double>(similarity);
            }

            double class_penalty = (object_names[det_idx] != map_obj.current_name) ? max_class_penalty : 0.0;

            // Final combined assignment cost.
            costMatrix[i][j] = (dynamic_w_dist * dist) + (dynamic_w_iou * (1.0 - iou)) + (w_sem * cost_sem) + class_penalty;
            distMatrix[i][j] = dist;
            iouMatrix[i][j] = iou;
            semCostMatrix[i][j] = cost_sem;
            classPenaltyMatrix[i][j] = class_penalty;
            wDistMatrix[i][j] = dynamic_w_dist;
            wIouMatrix[i][j] = dynamic_w_iou;
        }
    }

    // Global one-to-one assignment over the final cost matrix.
    HungarianAlgorithm HungAlgo;
    std::vector<int> assignment;
    HungAlgo.Solve(costMatrix, assignment);

    RCLCPP_DEBUG(semantic_logger,
        "[SemanticObjectMap] hungarian solved N=%d M=%d", N, M);

    for (int i = 0; i < N; ++i) {
        int map_idx = assignment[i];
        int det_idx = unmatched_indices[i];
        
        // Accept only assignments that exist and pass the quality threshold.
        if (map_idx >= 0 && costMatrix[i][map_idx] < MAX_COST) {
            std::string m_id = map_ids[map_idx];
            const std::vector<float> image_embedding_masked = embeddings_list_masked[det_idx].has_value() ? embeddings_list_masked[det_idx].value() : std::vector<float>{};
            const std::vector<float> image_embedding_unmasked = embeddings_list_unmasked[det_idx].has_value() ? embeddings_list_unmasked[det_idx].value() : std::vector<float>{};
            float similarity = 0.0f;
            const auto& map_obj = objects[m_id];
            if (!image_embedding_masked.empty() && !map_obj.image_embedding_masked.empty()) {
                similarity = compute_embedding_similarity(image_embedding_masked, map_obj.image_embedding_masked) * 100.0f;
            }
            log_map_fate("associate", m_id, map_obj.current_name,
                std::string("track=") + tracker_ids[det_idx] +
                " det_class='" + object_names[det_idx] +
                " cost=" + std::to_string(costMatrix[i][map_idx]) +
                " dist=" + std::to_string(distMatrix[i][map_idx]) +
                " iou=" + std::to_string(iouMatrix[i][map_idx]) +
                " sem_cost=" + std::to_string(semCostMatrix[i][map_idx]) +
                " class_penalty=" + std::to_string(classPenaltyMatrix[i][map_idx]) +
                " w_dist=" + std::to_string(wDistMatrix[i][map_idx]) +
                " w_iou=" + std::to_string(wIouMatrix[i][map_idx]) +
                " w_sem=" + std::to_string(w_sem) +
                " max_cost=" + std::to_string(MAX_COST));
            // Update matched stable object and refresh tracker binding.
            update_object(m_id, object_names[det_idx], stamp, points_cam_list[det_idx], similarities[det_idx], confidences[det_idx], image_embedding_masked, image_embedding_unmasked, current_ns, tracker_ids[det_idx]);
            track_to_map[tracker_ids[det_idx]] = m_id;
            track_last_seen_ns[tracker_ids[det_idx]] = current_ns;
        } else {
            // Rejected/unassigned detections keep accumulating as tentative tracks.
            const std::vector<float> image_embedding_masked = embeddings_list_masked[det_idx].has_value() ? embeddings_list_masked[det_idx].value() : std::vector<float>{};
            const std::vector<float> image_embedding_unmasked = embeddings_list_unmasked[det_idx].has_value() ? embeddings_list_unmasked[det_idx].value() : std::vector<float>{};
            log_detection_fate("route_tentative", tracker_ids[det_idx], object_names[det_idx], confidences[det_idx], "reason=hungarian_rejected");
            update_tentative(
                object_names[det_idx],
                tracker_ids[det_idx],
                points_cam_list[det_idx],
                stamp,
                confidences[det_idx],
                similarities[det_idx],
                image_embedding_masked,
                image_embedding_unmasked,
                current_ns,
                map_frame.empty() ? camera_frame : map_frame);
        }
    }

    RCLCPP_DEBUG(semantic_logger,
        "[SemanticObjectMap] batch complete map_objects=%zu tentative=%zu bindings=%zu",
        objects.size(), tentative_tracks.size(), track_to_map.size());
}

void SemanticObjectMapV5::fuse_objects(const std::string& keep_id, const std::string& drop_id) {
    if (keep_id == drop_id) return;
    if (!objects.count(keep_id) || !objects.count(drop_id)) return;

    auto& keep = objects[keep_id];
    auto& drop = objects[drop_id];

    log_map_fate("merge_duplicate", keep_id, keep.current_name,
        std::string("drop_id=") + drop_id + " drop_class='" + drop.current_name + "' reason=overlap_cleanup");

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
            
            log_map_fate("drop_wrong_detection", it->first, it->second.current_name,
                std::string("age_sec=") + std::to_string(age_sec) + " occurrences=" + std::to_string(it->second.occurrences));
            
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