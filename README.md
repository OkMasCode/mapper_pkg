# mapper_pkg

Semantic mapping layer of the semantic-navigation thesis stack.

`mapper_node` fuses the 2D instance masks from `yolo11_seg_bringup` with aligned depth to build a
persistent **3D semantic object map**: one entry per physical object, tracked across frames, with an
oriented bounding box, a consensus class label, a fused SigLIP embedding, and a goal-relevance score.
It is the bridge between per-frame perception and the room-level reasoning done downstream.

```
yolo11_seg_bringup ──/vision/detections──▶ mapper_node ──/vision/semantic_map_v5──▶ cluster_assignment_node
      (masks, embeddings, tracker IDs)          │                                   cpp_mapper_json_exporter
                                                │                                   bt_pkg
depth + camera_info ────────────────────────────┘
/vision/text_embedding ─────────────────────────┘
```

The package builds **two** executables: `mapper_node` (the object map, described below) and
`sem_mapper` (a separate scene-level grid, see §7).

---

## 1. Models

This package runs **no neural network**. It consumes what `yolo11_seg_bringup` produces, which makes
the interface contract the important thing:

| Input | Used for |
|---|---|
| `image_embedding_masked` (per detection) | fused into the object entry with a running average; the **image-image cosine** term in the association cost |
| `image_embedding_unmasked` | fused and republished; not used in association |
| `similarity` (per detection) | the vision node's SigLIP goal score — kept as a **running max** per object |
| `instance_id` (BoT-SORT tracker ID) | the cheap association fast path |
| `/vision/text_embedding` | stored with its `logit_scale` / `logit_bias` for `get_goal_similarity()` |

**Where the published `similarity` actually comes from.** The mapper stores the goal text embedding and
implements `get_goal_similarity()` (sigmoid over `dot × scale + bias`), but `publish_semantic_map()`
does **not** call it. It publishes `entry.similarity` — the highest per-frame score the vision node ever
reported for that object ([semantic_object_map.cpp:453-456](src/semantic_object_map.cpp#L453-L456),
[mapper_node.cpp:552-555](src/mapper_node.cpp#L552-L555)).

The reason is in the code comment: object embeddings are time-averaged across observations, and SigLIP's
logit scale is steep enough that averaging over good and bad viewpoints collapses the score toward zero.
Taking the max over per-frame scores keeps the evidence from the one good look at the object. The
consequence is that **the score does not decay** — an object that scored high once keeps that score even
after the prompt changes, until the map entry is pruned.

Embeddings are L2-normalized and fused with a count-weighted running average:
`fused = normalize((cw·current + nw·new) / (cw + nw))`.

---

## 2. Pipeline

`/vision/detections` and the depth image are joined by an `ApproximateTime` synchronizer (queue 10).
Per synchronized callback:

1. **Depth prep** — `16UC1` mm → `32FC1` m, then a bilateral filter (`d=5`, `sigmaColor=0.05` m,
   `sigmaSpace=5` px) that smooths noise without blurring across depth discontinuities. TF
   `depth frame → map` is looked up at the depth timestamp (0.10 s tolerance); if the frames match, the
   transform is skipped entirely, which is what makes camera-only testing possible.
2. **Per detection** — decode the `mono8` mask, resize to depth resolution if needed, back-project every
   masked pixel with valid depth through the pinhole model, adaptively voxel-downsample and SOR-filter,
   then transform the cloud into the map frame. Detections that end up with an empty cloud are dropped.
3. **Batch association** — the surviving clouds, names, tracker IDs, confidences, similarities and
   embeddings go to `add_detections_batch()` in one call (§3).
4. **Post-update** — the mutex is **released** before the expensive tail, so new frames can be processed
   while this one finishes: geometry refinement per object, duplicate merge, stale pruning, then publish.

Timing is instrumented per stage and a 12-row table is printed every 30 s; any frame over 150 ms also
triggers an immediate breakdown at ERROR level.

### Adaptive filtering

Filter strength scales with cloud density rather than being fixed — a sparse cloud from a small or
distant object cannot afford aggressive downsampling, a dense nearby one cannot afford not to. Given
raw point count `n`, with `t = (n − min_point_count) / (max_point_count − min_point_count)` clamped to
`[0,1]`:

```
voxel_size  = min_voxel_size + t·(max_voxel_size − min_voxel_size)
sor_mean_k  = min_sor_k      + t·(max_sor_k      − min_sor_k)
sor_stddev  = min_sor_stddev + t·(max_sor_stddev − min_sor_stddev)
```

### Oriented bounding box

PCA on the accumulated cloud: centroid → covariance → eigenvectors, forced right-handed with
`e3 = e1 × e2`, points projected into the local frame, min/max giving the extents. Recomputed after
every geometry fusion. If the OBB is degenerate the pose falls back to the raw centroid.

### Overlap proxy

True OBB-OBB IoU is expensive, so overlap is sampled: up to ~50 points from each cloud, `r12` = fraction
of cloud 1 inside OBB 2, `r21` = the reverse, `overlap = clamp(0.5·(r12 + r21), 0, 1)`.

---

## 3. Association

Three tiers, cheapest first.

**Tier 0 — tracker binding.** If the detection's BoT-SORT ID is already bound to a map object and the
class is consistent, update directly. No geometry, no solver. This is the common case and it is why the
vision node runs `track()` rather than `predict()`.

**Tier 1 — cheap shortlist.** For every remaining detection × object pair: a hard spatial gate
`dist ≤ max(0.4, 1.2·max_size)` blocks unrealistic pairs, then

```
cheap_cost = w_dist(size)·dist + class_penalty        class_penalty = 3.0 if labels differ else 0
```

Only the **top 3** cheapest candidates per detection survive; everything else is set to `kBlockedCost`
(999) and never scored again.

**Tier 2 — full cost on the shortlist**, solved globally one-to-one with the Hungarian algorithm:

```
cost = w_dist(size)·dist + w_iou(size)·(1 − overlap) + 2.5·(1 − cos_sim_embedding) + class_penalty
```

`cos_sim_embedding` uses the **masked** embeddings; when either side has none, the semantic term
defaults to its worst value (1.0). An assignment is accepted only if `cost < MAX_COST` (4.0); otherwise
the detection falls through to the tentative pipeline.

**Scale-aware weights.** A 5 cm cup and a 2 m fridge need different notions of "close". The weights
interpolate on `max_size`, the larger extent of the two candidates:

| | small (`≤ 0.2 m`) | large (`≥ 2.0 m`) |
|---|---|---|
| `w_dist` | 5.0 | 0.2 |
| `w_iou` | 4.0 | 0.3 |

Small objects are dominated by position (a few centimetres off means a different cup); large objects
lean on overlap and embedding, because their centroids move as more of the surface is observed.

**Tentative tracks.** Unmatched detections create tentative tracks that are promoted into the map only
when *all* of: `hits ≥ 5`, `age ≥ 1.0 s`, `max confidence ≥ 0.6`, `mean confidence ≥ 0.55`. Detections
below `min_input_confidence` (0.55) never even reach this stage. Tentative tracks go stale after 2.0 s,
tracker bindings after 4.0 s.

**Confirmed update.** Clouds are concatenated and re-voxelized (no ICP — the alignment step in
`fuse_geometry` is a passthrough), the OBB is recomputed, and class evidence accumulates as
`score(label) = 1.0·count(label) + 2.0·mean_conf(label)`. A **hysteresis lock** keeps the current label
once it has ≥ 4 votes unless a challenger beats it by more than 0.75, which stops objects flickering
between visually similar classes. Confidence is an EMA with `α = 0.20`.

**Cleanup.** Same-class objects with overlap > 0.80 are merged; objects older than 20 s with fewer than
50 occurrences are pruned as false positives.

---

## 4. Interface

### Subscribed

| Topic (default) | Type | Parameter |
|---|---|---|
| `/vision/detections` | `DetectedObjectV3Array` | `detection_message` |
| `/jackal/sensors/camera_0/aligned_depth_to_color/image` | `sensor_msgs/Image` | `depth_topic` |
| `/jackal/sensors/camera_0/aligned_depth_to_color/camera_info` | `sensor_msgs/CameraInfo` | `camera_info_topic` |
| `/vision/text_embedding` | `std_msgs/Float32MultiArray` | `vision_topic` |

Intrinsics are latched from the **first** `CameraInfo` message; frames are skipped until it arrives.
The text-embedding message is the embedding vector with `logit_scale` and `logit_bias` appended as the
last two elements.

### Published

| Topic | Type | Notes |
|---|---|---|
| `/vision/semantic_map_v5` | `SemanticObjectArray` | id, class, poses, OBB (size + quaternion + 8 corners), occurrences, confidence EMA, both embeddings, similarity |
| `/vision/semantic_map_v5/points` | `sensor_msgs/PointCloud2` | XYZRGB fused map cloud, colour from CRC32 of the object ID |

### Parameters

| Parameter | Default | |
|---|---|---|
| `detection_message` | `/vision/detections` | |
| `map_frame` | `map` | |
| `camera_frame` | `camera_0_color_optical_frame` | fallback when the depth header has no `frame_id` |
| `depth_topic` / `camera_info_topic` / `vision_topic` | see table above | |
| `stable_pointcloud_topic` | `/vision/semantic_map_v5/points` | |
| `publish_stable_pointcloud` / `viewer_enabled` | `true` / `true` | both must be true to publish the cloud |
| `export_interval` | `5.0` s | **heartbeat/diagnostic timer, not a file export** |
| `output_dir` / `output_map_file` | — | declared but unused; JSON export is done by `cpp_mapper_json_exporter_node` in `yolo11_seg_bringup` |
| `min_range` / `max_range` | `0.1` / `6.0` m | valid depth window |
| `do_voxel_filtering` | `true` | |
| `voxel_size` | `0.05` | overwritten every frame by the adaptive rule |
| `min_point_count` / `max_point_count` | `800` / `5000` | adaptation endpoints |
| `min_voxel_size` / `max_voxel_size` | `0.01` / `0.08` | |
| `do_sor_` | `true` | note the trailing underscore in the parameter name |
| `sor_mean_k`, `min_sor_k`, `max_sor_k` | `50`, `20`, `120` | |
| `sor_stddev_mul_thresh`, `min_…`, `max_…` | `1.0`, `1.0`, `1.0` | |

Association, promotion, class-consensus and pruning constants are **not** ROS parameters — they are
members of `SemanticObjectMapV5` in
[semantic_object_map.hpp:90-142](include/mapper_pkg/semantic_object_map.hpp#L90-L142). Changing them
means recompiling.

---

## 5. What you need to run it

**Build dependencies.** ROS 2 Humble plus `rclcpp`, `std_msgs`, `sensor_msgs`, `geometry_msgs`,
`message_filters`, `tf2`/`tf2_ros`/`tf2_geometry_msgs`/`tf2_sensor_msgs`, `cv_bridge`,
`pcl_conversions`, `pcl_ros`, `nav_msgs`, `grid_map_ros`/`grid_map_msgs`/`grid_map_core`, and system
OpenCV, Eigen3, PCL (`common io filters features`), zlib. `grid_map_*` is needed even if you only want
`mapper_node`, because both executables are in one CMake project.

**`yolo11_seg_interfaces`** — external message package, required. Must provide `DetectedObjectV3Array`,
`SemanticObject`, `SemanticObjectArray`, `Similarity`, `SimilarityCentroid`, `SimilarityCentroidArray`.

**The Hungarian solver** is vendored in-tree (`src/Hungarian.cpp`, Cong Ma, BSD) — nothing to install.

**At runtime you need all of:**

- `pc_vision_node_v3` publishing `/vision/detections` — without it there is nothing to map
- a depth stream **aligned to colour**, plus its `CameraInfo`
- TF resolving `depth frame → map` at the depth timestamps (i.e. SLAM or localization running)

Build:

```bash
cd /home/workspace/ros2_ws
colcon build --packages-select yolo11_seg_interfaces mapper_pkg
source install/setup.bash
```

Run:

```bash
ros2 run mapper_pkg mapper_node
```

Override for a different camera:

```bash
ros2 run mapper_pkg mapper_node --ros-args \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/aligned_depth_to_color/camera_info \
  -p camera_frame:=camera_color_optical_frame \
  -p max_range:=5.0
```

The node is spun on a `MultiThreadedExecutor` so the heartbeat timer keeps reporting even when the sync
callback is busy.

### Diagnosing a silent mapper

The heartbeat exists because the failure modes are quiet. If nothing appears on
`/vision/semantic_map_v5`, the log tells you which stage is starving:

| Log | Meaning |
|---|---|
| `camera intrinsics are not ready` | no `CameraInfo` — wrong topic |
| `cannot transform … -> map` | TF gap; check SLAM and the depth `frame_id` |
| `no synced callbacks for … ms` with non-zero `mask_msgs`/`depth_msgs` | both streams arrive but timestamps don't align — the usual cause is a camera driver not stamping depth and colour consistently |
| `no detections survived filtering` | clouds too small; check `min_range`/`max_range` and mask quality |
| `no completed synced callbacks yet` | nothing has ever matched — topic names |

---

## 6. Caveats

- **No ICP.** `fuse_geometry` copies the new cloud and concatenates; the `aligned_new` variable is a
  leftover from an alignment step that is not there.
- **Association constants require a rebuild** — they are not ROS parameters.
- **Default topics are Jackal-specific.** `yolo11_seg_bringup` defaults to the same camera; if you remap
  one end, remap both.
- The node-level `voxel_size` / `sor_mean_k` / `sor_stddev_mul_thresh` parameters are **overwritten on
  every detection** by the adaptive rule; only the `min_*`/`max_*` bounds have lasting effect.

---

## 7. `sem_mapper` — scene-level semantic grid (separate executable)

An independent experiment, not part of the object-map pipeline. It maintains a `grid_map` with an
`occupancy` layer and a `similarity` layer: the **scene-level** SigLIP score from
`scene_embedding_node` (`/vision/scene_similarity_raw`) is painted into the camera FOV wedge, so the
grid accumulates "how much did this region look like the prompt". It then extracts frontiers that
border unknown space, filters out those too close to walls or within 0.3 m of the robot, clusters them,
and publishes cluster centroids with their mean similarity.

- **Subscribes:** `/map`, `/jackal/sensors/camera_0/aligned_depth_to_color/camera_info` (FOV is derived
  from `2·atan(width / 2fx)`), `/vision/scene_similarity_raw`
- **Publishes:** `/semantic_grid_map` (`grid_map_msgs/GridMap`, 5 Hz), `/frontiers` and `/clusters`
  (markers), `/similarity_centroids_data` (`SimilarityCentroidArray`)
- **Consumer:** `semantic_navigator.py` in `yolo11_seg_bringup` — see the caveat there about its
  inverted similarity term

```bash
ros2 run mapper_pkg sem_mapper
```

Note it subscribes to `/map`, not `/jackal/map` like the rest of the stack.

---

## License

Apache-2.0. The bundled Hungarian implementation (Cong Ma) is BSD.
