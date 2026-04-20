# mapper_pkg

A high-performance **ROS 2 C++ node** that builds and maintains a **real-time 3D semantic object map** from synchronized YOLO instance-segmentation detections and depth images. Designed for robotics applications such as autonomous navigation, object search, and scene understanding.

## Overview

`mapper_pkg` fuses per-frame 2D segmentation masks with aligned depth imagery to produce persistent, class-labeled 3D object representations in a global map frame. Each detected object is tracked across frames, promoted through a tentative confirmation stage, and maintained with geometry refinement, class consensus voting, and duplicate merging.

### Key Capabilities

- **Depth-to-3D projection** — pinhole back-projection of masked depth pixels into camera-frame point clouds.
- **Per-detection filtering** — adaptive voxel downsampling and Euclidean clustering to reject noise and keep only the dominant spatial cluster.
- **TF2 map-frame transformation** — clouds are transformed from the camera optical frame into a stable map frame using live TF2 lookups.
- **Tentative track confirmation** — new detections must survive a configurable number of hits, minimum age, and confidence thresholds before being promoted into the persistent map.
- **Hungarian (bipartite) data association** — unmatched detections are associated with existing map objects via a cost matrix combining centroid distance, oriented bounding-box IoU, image-embedding cosine similarity, and class-label penalties, solved globally with the Hungarian algorithm.
- **Scale-aware association weights** — distance and IoU weights interpolate smoothly between tight (small-object) and permissive (large-object) regimes.
- **Geometry fusion & OBB computation** — accumulated point clouds are merged with adaptive voxel filtering; a PCA-based oriented bounding box (OBB) is recomputed after every update.
- **Class consensus voting** — each object maintains per-class vote counts and confidence sums; a weighted scoring formula with a hysteresis margin prevents spurious class flips.
- **Duplicate merging** — overlapping same-class objects with IoU > 0.80 are fused into a single entry.
- **Stale object pruning** — old objects with very few observations are automatically removed.
- **Image-embedding tracking** — masked and unmasked CLIP-style embeddings are fused with a running average and used for both association and goal-directed similarity scoring.
- **Goal similarity scoring** — an external text embedding is matched against each object's image embedding to produce a per-object relevance score (useful for "find object X" queries).
- **Colored point-cloud visualization** — the full map is published as an XYZRGB cloud with deterministic per-object CRC32-based coloring for RViz visualization.
- **Periodic performance reporting** — detailed per-stage timing statistics are printed every 30 seconds.

## Architecture

### Detailed Block Diagram (mapper internals)

```
Inputs
  ┌─────────────────────────────────────────────┐
  │ /camera/.../camera_info                     │
  │  -> fx, fy, cx, cy                          │
  └──────────────────────┬──────────────────────┘
                         │
  ┌──────────────────────▼──────────────────────┐
  │ /vision/detections (DetectedObjectV3Array)  │
  │ /camera/.../image_raw (depth)               │
  │  -> ApproximateTime sync                    │
  └──────────────────────┬──────────────────────┘
                         │ synced callback
                         ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ PointCloudMapperNodeV5                                                      │
│                                                                              │
│  (A) Depth preprocessing                                                    │
│      - CV conversion (16UC1 mm -> 32FC1 m)                                 │
│      - bilateralFilter(depth, d=5, sigmaColor=0.05, sigmaSpace=5)          │
│      - TF lookup map <- camera                                              │
│                                                                              │
│  (B) Per-detection geometry extraction                                      │
│      - decode mask                                                          │
│      - masked depth back-projection (pinhole)                              │
│      - adaptive voxel + adaptive SOR                                        │
│      - reject tiny clouds (<4 pts)                                          │
│      - transform cloud camera->map                                          │
│                                                                              │
│  (C) Batch handoff                                                          │
│      add_detections_batch(names, track_ids, conf, cloud_map, embeddings)   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ SemanticObjectMapV5                                                         │
│                                                                              │
│  1) Fast route: tracker_id -> map_id (if bound and class-consistent)        │
│  2) For remaining detections: Hungarian association                          │
│      Stage-1 prune: distance + class penalty (Top-K keep)                   │
│      Stage-2 full score: dist + OBB overlap + embedding + class penalty     │
│      Global one-to-one assignment via Hungarian                              │
│      Accept only if cost < MAX_COST                                          │
│  3) Matched => update_object: geometry fuse + OBB recompute + voting/EMA    │
│  4) Unmatched => tentative tracks; promote when maturity constraints pass    │
│  5) Cleanup: duplicate merge (same class, overlap > 0.80), stale pruning     │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼
Outputs
  ┌─────────────────────────────────────────────┐
  │ /vision/semantic_map_v5                     │
  │  - id, class, pose, OBB, embeddings, score │
  └─────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────┐
  │ /vision/semantic_map_v5/points              │
  │  - colored XYZRGB fused map cloud           │
  └─────────────────────────────────────────────┘

Side input (goal relevance):
  /vision/text_embedding -> set_text_embedding(embedding, logit_scale, bias)
  used in get_goal_similarity() during publish
```

## Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/camera/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics (required before processing begins) |
| `/vision/detections` (configurable) | `yolo11_seg_interfaces/DetectedObjectV3Array` | Per-frame instance segmentation detections with masks, class labels, confidence, tracker IDs, and optional image embeddings |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Aligned depth image (16UC1 in mm or 32FC1 in meters) |
| `/vision/text_embedding` | `std_msgs/Float32MultiArray` | Goal text embedding vector (with logit scale and bias appended) for similarity scoring |

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/vision/semantic_map_v5` | `yolo11_seg_interfaces/SemanticObjectArray` | Full semantic object map: IDs, class names, poses, OBB corners, orientations, sizes, occurrences, confidence, embeddings, and similarity scores |
| `/vision/semantic_map_v5/points` (configurable) | `sensor_msgs/PointCloud2` | Colored XYZRGB point cloud of all tracked objects for RViz visualization |

## Parameters

### General

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detection_message` | `/vision/detections` | Topic name for detection input |
| `map_frame` | `map` | Target TF frame for the global map |
| `camera_frame` | `camera_color_optical_frame` | Camera optical frame ID |
| `output_dir` | (workspace path) | Directory for map export output |
| `output_map_file` | `map_v6.json` | Filename for exported map (reserved for future file-based export on shutdown) |
| `stable_pointcloud_topic` | `/vision/semantic_map_v5/points` | Topic for visualization point cloud |
| `publish_stable_pointcloud` | `true` | Enable/disable point cloud publishing |
| `viewer_enabled` | `true` | Enable/disable RViz visualization publisher |

### Depth Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_range` | `0.1` | Minimum valid depth in meters |
| `max_range` | `3.0` | Maximum valid depth in meters |

### Voxel Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `do_voxel_filtering` | `false` | Enable adaptive voxel downsampling on input clouds |
| `voxel_size` | `0.12` | Base voxel leaf size (meters) |
| `min_point_count` | `4000` | Point count below which the minimum voxel size is used |
| `max_point_count` | `30000` | Point count above which the maximum voxel size is used |
| `min_voxel_size` | `0.008` | Smallest adaptive voxel leaf size |
| `max_voxel_size` | `0.050` | Largest adaptive voxel leaf size |

### Euclidean Clustering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cluster_tolerance` | `0.06` | Spatial tolerance for cluster membership (meters) |
| `min_cluster_size` | `5` | Minimum points in a valid cluster |
| `max_cluster_size` | `50000` | Maximum points in a valid cluster |

## Processing Pipeline

Each synchronized callback executes:

1. Convert depth to meters and smooth depth.
2. Get TF transform (`source_frame -> map_frame`) for current depth timestamp.
3. For each detection:
   - decode mask
   - project mask pixels to 3D cloud
   - adaptive voxel/SOR filtering
   - reject if too few points
   - transform cloud to map frame
4. Batch association and state update in `add_detections_batch()`:
   - direct tracker binding update when valid
   - Hungarian association for unresolved detections
   - matched -> update confirmed objects
   - unmatched -> update/create tentative tracks, promote when mature
5. Post-update map maintenance:
   - geometry refinement per object
   - duplicate resolution
   - stale object pruning
6. Publish semantic map and optional colored map cloud.

## Pipeline Math (Detailed)

Notation: pixel `(u,v)`, depth `z`, camera intrinsics `(fx, fy, cx, cy)`, 3D point `p=[x,y,z]^T`.

### 1) Depth conversion and smoothing

- If depth image is `16UC1` (mm):  
  `z_meters = z_raw * 0.001`
- Otherwise depth is already `32FC1` meters.
- 2D bilateral filtering is applied:
  - diameter `d = 5`
  - `sigmaColor = 0.05` (meters)
  - `sigmaSpace = 5.0` (pixels)

### 2) Pinhole back-projection (mask -> 3D cloud)

For every masked pixel with valid depth (`min_range <= z <= max_range`):

- `x = (u - cx) * z / fx`
- `y = (v - cy) * z / fy`
- `z = z`

This creates per-detection cloud `C_det` in camera frame.

### 3) Adaptive filtering

Given raw point count `n`:

- For `n <= min_point_count`: use minimum voxel/SOR settings.
- For `n >= max_point_count`: use maximum voxel/SOR settings.
- Else linear interpolation:
  - `t = (n - min_point_count) / (max_point_count - min_point_count)`
  - `voxel_size = min_voxel_size + t*(max_voxel_size - min_voxel_size)`
  - `sor_mean_k = min_sor_k + t*(max_sor_k - min_sor_k)`
  - `sor_stddev = min_sor_stddev + t*(max_sor_stddev - min_sor_stddev)`

Then:
- optional VoxelGrid downsampling
- optional StatisticalOutlierRemoval

### 4) Camera->map transform

With TF transform `(R, t)` from source frame to map:

- `p_map = R * p_cam + t`

Quaternion from TF is normalized before building `R`.

### 5) OBB (oriented bounding box) from PCA

For cloud `C`:

1. centroid `c = mean(C)`
2. covariance `Σ = cov(C)`
3. eigenvectors `E = [e1 e2 e3]` from `Σ`
4. enforce right-handed basis: `e3 = e1 × e2`
5. project points into local PCA frame:
   - `p_local = E^T (p - c)`
6. local min/max bounds -> extents:
   - `extent_x = max_x - min_x`, similarly for `y,z`

Stored OBB:
- center `c`
- extents `(L,W,H)`
- rotation matrix `E`

### 6) Overlap proxy (used as IoU-like term)

`compute_obb_iou()` uses orientation-aware overlap ratio:

- sample up to ~50 points from each cloud
- compute:
  - `r12 = fraction(points1 inside obb2)`
  - `r21 = fraction(points2 inside obb1)`
- overlap proxy:
  - `overlap = clamp(0.5*(r12 + r21), 0, 1)`

### 7) Embedding normalization and similarity

For embedding vector `e`:

- normalize: `e_hat = e / ||e||_2`

Image-image similarity:

- `sim = dot(e_hat1, e_hat2)` (cosine because normalized)
- clamped to `[0,1]`

Running embedding fusion:

- `fused = normalize((cw*cur + nw*new)/(cw+nw))`

where `cw` and `nw` are effective counts.

### 8) Association math (Hungarian stage)

For each unmatched detection `i` and map object `j`:

1. centroid distance:
   - `dist_ij = ||center_det_i - center_map_j||_2`

2. class mismatch penalty:
   - `penalty_ij = max_class_penalty` if classes differ, else `0`

3. scale-aware weights (object-size dependent):
   - let `s = max(max_extent_det, max_extent_map)`
   - if `s < small_size`: use `(w_dist_small, w_iou_small)`
   - if `s > large_size`: use `(w_dist_large, w_iou_large)`
   - else interpolate linearly with ratio  
     `r = (s-small_size)/(large_size-small_size)`

4. hard distance gate:
   - `dist_ij <= max(0.4, 1.2*s)` else blocked

5. Stage-1 cheap score (for Top-K shortlist):
   - `cheap_ij = w_dist(s)*dist_ij + penalty_ij`

6. Stage-2 full score (only shortlisted pairs):
   - `iou_ij = overlap_proxy(det_i, map_j)`
   - `cost_sem_ij = 1 - sim_embedding_ij` (or `1` if embedding unavailable)
   - Final:
     - `cost_ij = w_dist(s)*dist_ij + w_iou(s)*(1 - iou_ij) + w_sem*cost_sem_ij + penalty_ij`

Hungarian solves global one-to-one assignment on `cost_ij`.

Acceptance rule:
- assigned and `cost_ij < MAX_COST` (default `4.5`) -> match accepted
- otherwise detection goes to tentative pipeline

### 9) Tentative track promotion logic

A tentative track is promoted only when all are true:

- `hits >= confirmation_min_hits` (default `5`)
- `age_sec >= confirmation_min_age_sec` (default `1.0`)
- `confidence_max >= min_confidence_for_promotion` (default `0.6`)
- `avg_conf = confidence_sum/hits >= min_avg_confidence_for_promotion` (default `0.55`)

Also, low-confidence inputs are rejected before tentative update:
- `confidence >= min_input_confidence` (default `0.55`)

### 10) Confirmed object update equations

For accepted detection:

1. geometry fusion:
   - concatenate old/new cloud
   - optional ICP alignment
   - adaptive SOR + adaptive voxel
2. recompute OBB and pose from fused cloud
3. class evidence:
   - `class_count[name] += 1`
   - `class_conf_sum[name] += confidence`
4. class score per label:
   - `score(name) = class_count_weight*count(name) + class_confidence_weight*avg_conf(name)`
5. hysteresis lock:
   - if current label has enough votes and challenger margin is small, keep current class
6. confidence EMA:
   - `conf_ema = (1-α)*conf_ema + α*confidence` with `α = confidence_ema_alpha` (default `0.20`)

### 11) Duplicate merge and stale pruning

- Duplicate merge condition:
  - same class AND overlap proxy `> 0.80`
- Wrong/stale object removal condition:
  - `age_sec > kMaxAgeSec` (default `20.0`)
  - AND `occurrences < kMinOccurrences` (default `50`)
- Tentative and track-binding stale state are also pruned with TTL windows.

### 12) Goal similarity scoring math (published per object)

Given text embedding `t`, object image embedding `e`, logit scale `s`, bias `b`:

1. `dot = <e, t>`
2. `logits = dot*s + b`
3. `logits_clipped = clamp(logits, -60, 60)`
4. `score = sigmoid(logits_clipped) * 100`

Current output uses masked embedding score (`1.0*masked + 0.0*unmasked`).

## Dependencies

### ROS 2 Packages

- `rclcpp`
- `std_msgs`, `sensor_msgs`, `geometry_msgs`
- `message_filters`
- `tf2`, `tf2_ros`, `tf2_geometry_msgs`
- `cv_bridge`
- `pcl_conversions`, `pcl_ros`
- `yolo11_seg_interfaces` — custom message package providing `DetectedObjectV3Array`, `SemanticObjectArray`, and `SemanticObject`

### System Libraries

- **OpenCV** — image conversion and mask processing
- **Eigen3** — fast matrix math for transforms, PCA, and cosine similarity
- **PCL** (Point Cloud Library) — voxel filtering, Euclidean clustering, OBB computation, point cloud transforms
- **zlib** — CRC32 for deterministic per-object coloring

### Third-Party Source

- **Hungarian algorithm** — C++ implementation by Cong Ma (BSD license), included in-tree as `src/Hungarian.cpp` and `include/mapper_pkg/Hungarian.h`

## Building

This package uses the **ament_cmake** build system. Build it inside a ROS 2 workspace:

```bash
# Source your ROS 2 installation
source /opt/ros/<distro>/setup.bash

# Clone into your workspace (ensure yolo11_seg_interfaces is also present)
cd ~/ros2_ws/src
git clone <this-repo-url> mapper_pkg

# Install dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --packages-select mapper_pkg

# Source the workspace
source install/setup.bash
```

## Running

```bash
ros2 run mapper_pkg mapper_node
```

Override parameters at launch:

```bash
ros2 run mapper_pkg mapper_node --ros-args \
  -p detection_message:=/my/detections \
  -p map_frame:=odom \
  -p max_range:=5.0 \
  -p publish_stable_pointcloud:=true
```

Or use a launch file that remaps topics and sets parameters as needed for your robot platform.

## Performance

Timing statistics are printed to stdout every 30 seconds. Example output from a representative run:

| Step | Avg Time (ms) |
|------|---------------|
| Depth Conversion | ~0.5 |
| Point Extraction | ~1.1–1.4 |
| Filtering (Voxel + Cluster) | ~6.6–28.4 |
| Transform to Map | ~0.003 |
| Batch Addition (Matching) | ~40–78 |
| Publishing | ~1.8–4.8 |
| **Total per frame** | **~63–291** |

Performance depends heavily on the number of detections per frame, point cloud density, and the size of the persistent object map.

## License

Apache-2.0 — see [package.xml](package.xml) for details.

The bundled Hungarian algorithm implementation is licensed under the BSD license.
