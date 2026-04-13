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

```
                          ┌───────────────────────┐
  /camera/.../camera_info │   CameraInfo Sub      │ intrinsics (fx, fy, cx, cy)
  ───────────────────────►│                       ├──────────────┐
                          └───────────────────────┘              │
                                                                 ▼
  /vision/detections      ┌───────────────────────┐    ┌──────────────────────┐
  (DetectedObjectV3Array) │  ApproximateTime Sync │    │  PointCloudMapper    │
  ───────────────────────►│                       ├───►│  NodeV5              │
  /camera/.../image_raw   │  (detections + depth) │    │                      │
  (depth Image)           └───────────────────────┘    │  ┌────────────────┐  │
  ───────────────────────►                             │  │ SemanticObject │  │
                                                       │  │ MapV5          │  │
  /vision/text_embedding  ┌───────────────────────┐    │  │                │  │
  (Float32MultiArray)     │  Text Embedding Sub   │    │  │  - Tentative   │  │
  ───────────────────────►│                       ├───►│  │    Tracks      │  │
                          └───────────────────────┘    │  │  - Confirmed   │  │
                                                       │  │    Objects     │  │
                                                       │  │  - Hungarian   │  │
                                                       │  │    Matching    │  │
                                                       │  └────────────────┘  │
                                                       │                      │
                                                       │  Outputs:            │
                                                       │  ├─► /vision/        │
                                                       │  │   semantic_map_v5 │
                                                       │  └─► /vision/        │
                                                       │      .../points      │
                                                       └──────────────────────┘
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
| `output_map_file` | `map_v6.json` | Filename for exported map |
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

Each synchronized callback executes the following steps:

1. **Depth conversion** — the raw depth image is converted to a `CV_32FC1` matrix in meters.
2. **TF lookup** — the camera-to-map transform is obtained via TF2 for the current timestamp.
3. **Per-detection processing** (for each detection in the frame):
   1. Decode the binary segmentation mask.
   2. Back-project masked depth pixels to a 3D point cloud using camera intrinsics.
   3. Optionally apply adaptive voxel downsampling.
   4. Run Euclidean clustering and keep only the largest cluster.
   5. Transform the cleaned cloud into the map frame.
4. **Batch association** — all surviving detections are passed to `SemanticObjectMapV5::add_detections_batch()`:
   1. Detections with a known tracker-ID binding are routed directly to their map object.
   2. Remaining detections enter the Hungarian matching stage:
      - A cost matrix is built against all map objects using centroid distance, class penalty, OBB IoU, and embedding similarity.
      - A two-stage screening first prunes by cheap distance+class cost (top-K), then computes full scores only on survivors.
      - The Hungarian algorithm solves the global assignment.
   3. Matched detections update their map objects (geometry fusion, OBB recomputation, class voting, embedding fusion).
   4. Unmatched detections enter or update tentative tracks; tracks meeting the confirmation criteria are promoted to the persistent map.
5. **Geometry refinement** — each object's accumulated cloud is re-clustered to remove noise.
6. **Duplicate resolution** — same-class objects with OBB overlap > 80% are merged.
7. **Stale pruning** — objects older than 10 s with fewer than 5 observations are removed.
8. **Publishing** — the semantic map message and optional colored point cloud are published.

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
