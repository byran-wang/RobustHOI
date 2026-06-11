# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RobustHOI is a unified framework that reconstructs articulated hands and diverse objects from monocular RGB-D videos. It targets these challenges: objects that are textureless, reflective, transparent, or tiny, and mutual hand-object occlusion that leaves large surface regions unobserved.

## Environment

```bash
conda activate robust_hoi  # Python 3.10, PyTorch 2.1.0 (CUDA 11.8)
export DATASET=ho3d         # or: zed, rs_zijian, zed_zijian
export RUN_ON_SERVER=false   # true on server (changes paths in confs/sequence_config.py)
```

## Key Commands

The full pipeline is in `run_rhoi.sh`, ordered top-to-bottom. The stages below mirror that script.

```bash
seq_list="MC1"

# --- Data collection & masks (run on local PC; needs a monitor) ---
# Collect ZED raw data
python run_rhoi.py --execute_list data_read --process_list ZED_read_data --seq_list $seq_list --rebuild
# Parse left/right image, intrinsic and ZED depth from raw data (downsample 3)
python run_rhoi.py --execute_list data_convert --process_list ZED_parse_data --seq_list $seq_list --rebuild --downsample 3
# Convert depth to .ply (check ply_zed in Meshlab afterward)
python run_rhoi.py --execute_list data_convert --process_list convert_depth_to_ply --seq_list $seq_list --rebuild
# Step 1/2: interactively collect & save first-frame SAM3 prompts (text/points/box)
python run_rhoi.py --execute_list data_convert --process_list ho3d_get_obj_mask_prompt --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list data_convert --process_list ho3d_get_hand_mask_prompt --seq_list $seq_list --rebuild
# Step 2/2: run SAM3 mask propagation (loads saved prompt if present, else interactive popup)
python run_rhoi.py --execute_list data_convert --process_list ho3d_get_obj_mask ho3d_get_hand_mask --seq_list $seq_list --rebuild

# --- SAM3D generation, filtering & alignment (run on local PC, 32 GB RAM) ---
python run_rhoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_filter_2D --seq_list $seq_list
python run_rhoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_gen --seq_list $seq_list
python run_rhoi.py --execute_list obj_process --process_list ho3d_obj_SAM3D_filter_3D --seq_list $seq_list
python run_rhoi.py --execute_list obj_process --process_list ho3d_align_SAM3D_mask --seq_list $seq_list
python run_rhoi.py --execute_list obj_process --process_list ho3d_align_SAM3D_pts --seq_list $seq_list
python run_rhoi.py --execute_list obj_process --process_list ho3d_align_SAM3D_fp --seq_list $seq_list --rebuild
# Filter aligned frames by depth 3-axis coverage, drop unused, pick best SAM3D id
python run_rhoi.py --execute_list obj_process --process_list pipeline_sam3d_align_filter --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list obj_process --process_list pipeline_sam3d_delete_unused --seq_list $seq_list
python run_rhoi.py --execute_list obj_process --process_list pipeline_sam3d_best_id --seq_list $seq_list
python run_rhoi.py --execute_list obj_process --process_list pipeline_sam3d_best_id_sum --seq_list $seq_list
python run_rhoi.py --execute_list obj_process --process_list ho3d_SAM3D_post_process --seq_list $seq_list --rebuild

# --- Depth & hand pose (can run on server; no monitor needed) ---
# Foundation Stereo depth (ZED dataset only; check ply_fs in Meshlab afterward)
python run_rhoi.py --execute_list data_convert --process_list get_depth_from_foundation_stereo soft_link_depth --seq_list $seq_list --rebuild
# Estimate & interpolate hand pose (HaMeR)
python run_rhoi.py --execute_list data_convert --process_list ho3d_estimate_hand_pose ho3d_interpolate_hamer --seq_list $seq_list --rebuild
# Fit hand intrinsic & translation (+ vis)
python run_rhoi.py --execute_list hand_pose_postprocess --process_list fit_hand_intrinsic fit_hand_trans --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list hand_pose_postprocess --process_list fit_hand_intrinsic_vis --seq_list $seq_list
python run_rhoi.py --execute_list hand_pose_postprocess --process_list fit_hand_trans_vis --seq_list $seq_list

# --- HOI pipeline: preprocess, correspondence, joint opt, NeuS ---
python run_rhoi.py --execute_list obj_process --process_list hoi_pipeline_data_preprocess hoi_pipeline_get_corres --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list obj_process --process_list hoi_pipeline_eval_corres --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list obj_process --process_list hoi_pipeline_joint_opt --seq_list $seq_list --vis
python run_rhoi.py --execute_list obj_process --process_list hoi_pipeline_neus_global --seq_list $seq_list --rebuild

# --- Hand-object alignment (hand, rotation, object, hand+object) ---
python run_rhoi.py --execute_list obj_process --process_list hoi_pipeline_align_hand_object_h hoi_pipeline_align_hand_object_r hoi_pipeline_align_hand_object_o hoi_pipeline_align_hand_object_ho --seq_list $seq_list --rebuild

# --- Evaluation & visualization ---
python run_rhoi.py --execute_list obj_process --process_list hoi_pipeline_eval --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list obj_process --process_list hoi_pipeline_eval_vis --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list obj_process --process_list eval_sum --seq_list $seq_list
python run_rhoi.py --execute_list obj_process --process_list eval_sum_vis --seq_list $seq_list --rebuild

# --- Baselines ---
python run_rhoi.py --execute_list baseline --process_list foundation_pose_eval_vis --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list baseline --process_list bundle_sdf_eval_vis --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list baseline --process_list hold_eval_vis --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list baseline --process_list gt_eval_vis --seq_list $seq_list --rebuild
python run_rhoi.py --execute_list baseline --process_list ablation_comapre --seq_list $seq_list --rebuild
```

## Architecture

### Orchestration
- **run_rhoi.py**: Central entry point. Maps `--execute_list` + `--process_list` to pipeline functions. Supports `--rebuild` (clear and regenerate), `--vis`, `--eval` flags.
- **confs/sequence_config.py**: Routes to dataset-specific configs (e.g. `sequence_config_ho3d.py`) based on `DATASET` env var. Each sequence has `cond_idx`, `obj_num`, frame ranges.
- **run_rhoi.sh**: Canonical shell script chaining the full pipeline stages end-to-end (see Key Commands). `run.sh` holds legacy/experimental commands.

### Pipeline Stages (`robust_hoi_pipeline/`)
The pipeline processes frames through these stages:

1. **Data preprocessing** (`pipeline_data_preprocess.py`): Load images, depth, masks, intrinsics → cached `.pt` files
2. **SAM3D filtering** (`pipeline_sam3d_filter_2D.py`, `pipeline_sam3d_filter_3D.py`): Filter SAM3D 3D reconstructions by 2D/3D consistency
3. **Alignment** (`pipeline_sam3d_align.py`, `correspondence_alignment.py`): Align SAM3D mesh to image observations
4. **Joint optimization** (`pipeline_joint_opt.py`, ~95KB / 2200 lines): Core module. Registers frames via PnP + RANSAC, refines poses with reprojection + depth + contact/IoU losses, integrates FoundationPose tracking, runs keyframe bundle adjustment. Delegates per-frame logic to sibling modules: `frame_management.py` (`find_next_frame`, `check_frame_invalid`, `check_key_frame`, `process_key_frame`, `_refine_frame_pose_3d`, `save_keyframe_indices`), `optimization.py` (`register_new_frame_by_PnP`), `neus_integration.py` (`prepare_neus_data`, `run_neus_training`, `save_neus_mesh`).
5. **Evaluation** (`pipeline_joint_opt_eval.py`): Computes rotation, translation, intrinsic errors vs GT

### Key Pipeline Module: `pipeline_joint_opt.py`
Top-level flow (`main()` → orchestrates):
- `prepare_joint_opt_inputs()`: Loads preprocessed data + VGGSfM tracks (`pipeline_corres/`) + SAM3D transform (`SAM3D_aligned_post_process/`); masks low-visibility tracks (`vis_thresh`) and aligns hand `o2c` poses to `cond_cam_to_obj` at the condition frame
- `register_first_frame()` / `lift_tracks_to_3d()`: Lifts 2D tracks to 3D object points at the condition frame using depth + intrinsics
- `register_remaining_frames()`: Iteratively registers each remaining frame — PnP (`register_new_frame_by_PnP`), FoundationPose fallback/tracking, depth alignment (`_align_object_with_hand`), outlier track masking, keyframe selection, and incremental NeuS; `check_which_estimate_is_better_and_update()` picks the better of PnP vs FoundationPose poses
- `_joint_optimize_keyframes()`: Bundle adjustment (LBFGS) over keyframes with reprojection + point-to-plane depth + contact + IoU losses

Notable helpers: FoundationPose tracking (`_get_foundation_pose`, `_run_foundation_pose_track`, `_reset_and_track_foundation_pose`); losses (`_compute_contact_loss`, `_compute_iou_loss`); depth-based mesh ICP (`_align_object_with_hand`); pose validity checks (`_check_pose_moved`, `_is_hand_far_from_object`).

- **Logging**: `main()` tees `stdout`/`stderr` through `TeeStream` and adds a `FileHandler` (`PlainFormatter`), both writing to `output/{seq}/pipeline_joint_opt/txt.log`
- **FoundationPose**: estimator is lazily cached in the module-level `_foundation_pose_cache` dict
- **CLI**: `--data_dir`, `--output_dir`, `--cond_index`, `--vis_thresh` (0.3), `--optimize_3D_prior`, `--neus_init_steps` (1000)

### Visualization (`viewer/`)
- **viewer_step.py**: Rerun-based interactive viewer for per-frame results
- **viewer_distance.py**: Hand-object distance visualization (ARCTIC InterField style)
- Pipeline vis modules: `pipeline_joint_opt_vis.py`, `pipeline_joint_opt_eval_vis_rerun.py`, `pipeline_joint_opt_eval_vis_gt.py`

### Coordinate Conventions
- **Extrinsics**: `o2c` = object-to-camera (4x4), `c2o = inv(o2c)` = camera-to-object (world)
- **SAM3D poses**: `camera.json["blw2cvc"]` contains scaled o2c; extract scale from rotation columns
- **Depth scale**: `depth_scale` from preprocessed data scales GT space to SAM3D space (divide by it)
- **GT data** (`gt.load_data`): Returns xdict with `o2c`, `is_valid`, `K`, `v3d_c.right` (hand verts in cam), `mesh_name.object`, etc.

### Coding Rules
- **Logging**: Never use `print()`. Always use `from utils_simba.logger import get_logger; logger = get_logger(__name__)` for colored terminal output. `ColoredFormatter` in `third_party/utils_simba/utils_simba/logger.py` provides level-colored output. `pipeline_joint_opt.py` adds a `FileHandler` to also write logs to `pipeline_joint_opt/txt.log`.
- **Rerun visualization**: Use helper functions from `utils_simba.rerun` (`log_camera_frame`, `load_mesh_as_trimesh`, `get_vertex_colors`, `stamp_frame_text`, `backproject_depth_to_points`). Do not call raw `rr.log` for cameras/meshes when a helper exists.
- **Depth processing**: Use functions from `utils_simba.depth` (`get_depth`, `depth2xyzmap`). Do not write custom depth loading/conversion code.
- **Debug helpers**: All debug/visualization functions for `pipeline_joint_opt.py` live in `robust_hoi_pipeline/pipeline_joint_opt_debug.py`. Do not add debug functions directly to `pipeline_joint_opt.py`; add them to the debug module and import them.
- **Commit & push**: Commit the change to the remote after each change (commit and `git push` to `origin`).

## Data Structure

### HO3D_v3 Dataset (`ho3d_v3/`)
```
ho3d_v3/
├── calibration/{subject}/     # Camera calibration
├── train/{seq_id}/            # rgb/, depth/, meta/, hands/, mask_object/, mask_hand/
├── models/{object_id}/        # YCB 3D models (textured.obj, etc.)
├── processed/{seq_id}.pt      # Preprocessed GT data (used by gt.load_data)
└── evaluation/
```

### Output (`output/{seq_id}/`)
```
output/{seq_id}/
├── pipeline_preprocess/       # Preprocessed frames, frame_list.txt
├── SAM3D_aligned_post_process/# SAM3D results with camera.json per frame
├── pipeline_joint_opt/txt.log # Pipeline log (tee'd stdout/stderr + FileHandler)
├── results/{frame_id}/        # results.pkl, points.ply, mesh.obj, reproj_error.png
└── metrics_summary/           # Aggregated eval metrics
```

## Third-Party Dependencies
- `third_party/utils_simba/`: Rendering, depth processing, logging utilities
- `third_party/FoundationPose/`: Object pose estimation
- `third_party/instant-nsr-pl/`: NeuS neural SDF (entry: `launch.py`)
- `third_party/SAM3/`: Segment Anything 3D
- `third_party/HAMER/`: Hand pose estimation
- `dependency/LightGlue/`: Feature matching (install: `cd dependency/LightGlue && pip install -e .`)

## HO3D Sequences
19 sequences configured in `confs/sequence_config_ho3d.py`: ABF12, ABF14, GPMF12, GPMF14, MC1, MC4, MDF12, MDF14, ShSu10, ShSu12, ShSu14, SM2, SM4, SMu1, SMu40, BB12, BB13, GSF12, GSF13. Each has a `cond_idx` (conditioning frame) and frame range.

**Note**: ABF12/ABF14 frames beyond 1135 have bad GT annotations (marked invalid in `gt.load_data`).

## Best SAM3D IDs generated by auto mode (2026-04-24)

| Sequence | BestID | Score    | Coverage | Faces             |
|----------|--------|----------|----------|-------------------|
| ABF12    | 0120   | N/A      | 3/6      | X+,Y-,Z+          |
| ABF14    | 0405   | 0.007471 | 4/6      | X+,X-,Y+,Z+       |
| GPMF12   | 0239   | 0.010542 | 3/6      | X-,Y-,Z+          |
| GPMF14   | 0410   | 0.001299 | 3/6      | X+,Y+,Z+          |
| MC1      | 0700   | 0.005518 | 3/6      | X-,Y+,Z+          |
| MC4      | 0155   | 0.008594 | 2/6      | X+,Z+             |
| MDF12    | 1755   | N/A      | 4/6      | X-,Y+,Z+,Z-       |
| MDF14    | 1790   | 0.004230 | 4/6      | X+,Y-,Z+,Z-       |
| ShSu10   | 0518   | 0.001345 | 3/6      | X-,Y-,Z+          |
| ShSu12   | 0550   | N/A      | 3/6      | X-,Y+,Z+          |
| SM2      | 0018   | 0.002805 | 3/6      | X-,Y+,Z+          |
| SM4      | 0630   | N/A      | 3/6      | X+,Y-,Z-          |
| SMu1     | 0780   | 0.001863 | 3/6      | X-,Y-,Z+          |
| SMu40    | 0400   | 0.009716 | 3/6      | X+,Y+,Z-          |
| BB12     | 0485   | N/A      | 5/6      | X+,Y+,Y-,Z+,Z-    |
| BB13     | 1024   | 0.004680 | 3/6      | X+,Y-,Z+          |
| GSF12    | 0940   | 0.014783 | 4/6      | X-,Y-,Z+,Z-       |
| GSF13    | 0755   | 0.033960 | 2/6      | X-,Y-             |

## SOTA Results (2026-06-11)

Source: `output[4_25_11_09][879f][sam3d_auto_selection]/metrics_summary/eval.txt`

| Sequence | ADD AUC | ADD-S AUC | Total Frames | Reg Frames | Keyframes | SAM3D CD | SAM3D F5 | NeuS CD | NeuS F5 | MPJPE RA | CD Right |
|----------|---------|-----------|--------------|------------|-----------|----------|----------|---------|---------|----------|----------|
| ABF12    | 92.54   | 96.54     | 277          | 277        | 165       | 0.72     | 73.27    | 0.34    | 97.39   | 23.45    | 6.70     |
| ABF14    | 88.26   | 94.34     | 277          | 277        | 190       | 0.87     | 62.45    | 0.62    | 79.11   | 29.65    | 3.46     |
| BB12     | 70.91   | 88.81     | 322          | 322        | 277       | 1.13     | 68.75    | 0.50    | 89.86   | 23.11    | 3.97     |
| BB13     | 85.19   | 95.04     | 323          | 323        | 285       | 0.71     | 80.63    | 0.31    | 95.55   | 14.65    | 2.54     |
| GPMF12   | 58.99   | 90.90     | 220          | 220        | 169       | 0.39     | 96.03    | 0.90    | 61.55   | 20.82    | 5.70     |
| GPMF14   | 89.82   | 96.23     | 219          | 219        | 165       | 0.21     | 100.00   | 0.38    | 98.27   | 26.65    | 2.31     |
| GSF12    | 86.20   | 93.42     | 299          | 299        | 234       | 0.38     | 96.39    | 0.42    | 92.79   | 17.26    | 4.49     |
| GSF13    | 90.76   | 96.10     | 319          | 319        | 222       | 0.40     | 92.33    | 0.31    | 98.67   | 13.59    | 2.02     |
| MC1      | 94.24   | 96.85     | 180          | 180        | 172       | 0.67     | 79.43    | 0.43    | 95.47   | 9.55     | 2.28     |
| MC4      | 86.49   | 93.42     | 180          | 180        | 162       | 1.00     | 70.10    | 0.82    | 77.57   | 14.87    | 15.34    |
| MDF12    | 94.97   | 97.43     | 562          | 562        | 425       | 1.10     | 52.24    | 0.58    | 87.17   | 11.89    | 2.11     |
| MDF14    | 94.76   | 97.20     | 562          | 562        | 426       | 0.48     | 91.12    | 0.49    | 91.64   | 19.34    | 1.46     |
| ShSu10   | 92.14   | 95.73     | 371          | 371        | 191       | 0.40     | 96.74    | 0.30    | 99.15   | 8.18     | 2.10     |
| ShSu12   | 84.02   | 92.05     | 370          | 370        | 201       | 0.40     | 96.44    | 0.57    | 88.27   | 9.89     | 25.14    |
| SM2      | 92.31   | 96.78     | 181          | 181        | 168       | 0.46     | 90.29    | 0.45    | 93.93   | 10.31    | 3.01     |
| SM4      | 94.79   | 97.64     | 180          | 180        | 178       | 0.54     | 84.55    | 0.33    | 97.73   | 9.78     | 18.94    |
| SMu1     | 96.23   | 98.17     | 359          | 359        | 299       | 0.28     | 97.55    | 0.56    | 87.70   | 12.84    | 16.52    |
| SMu40    | 93.22   | 97.35     | 400          | 400        | 274       | 0.36     | 97.21    | 0.66    | 84.37   | 11.93    | 1.32     |
| **Avg**  | **88.10** | **95.22** | **311.17** | **311.17** | **233.50** | **0.58** | **84.75** | **0.50** | **89.79** | **15.99** | **6.63** |

Metrics: ADD AUC / ADD-S AUC (object pose, higher=better), SAM3D CD / NeuS CD (chamfer dist cm, lower=better), SAM3D F5 / NeuS F5 (F-score@5mm %, higher=better), MPJPE RA (hand mm, lower=better), CD Right (hand cm, lower=better).
