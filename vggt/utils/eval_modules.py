import numpy as np
import torch
from scipy.spatial import cKDTree as KDTree
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
import common.metrics as metrics
import trimesh


def compute_bounding_box_centers(vertices):
    """
    Compute the centers of the tight bounding box for a moving point cloud.

    Parameters:
    - vertices: A numpy array of shape (frames, num_verts, 3) representing the vertices of the object over time.

    Returns:
    - A numpy array of shape (frames, 3) where each row represents the center of the bounding box for each frame.
    """

    if isinstance(vertices, list):
        bbox_centers = []
        for verts in vertices:
            assert verts.shape[1] == 3
            bmin = np.min(verts, axis=0)
            bmax = np.max(verts, axis=0)
            bbox_center = (bmin + bmax) / 2
            bbox_centers.append(bbox_center)
        bbox_centers = np.stack(bbox_centers, axis=0)
    else:
        bbox_min = np.min(vertices, axis=1)
        bbox_max = np.max(vertices, axis=1)
        bbox_centers = (bbox_min + bbox_max) / 2
    return bbox_centers


def convert_to_tensors(data):
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint32:
                data[key] = torch.from_numpy(value.astype(np.int64))
            elif value.dtype.kind == "f":  # Check if it's a floating-point type
                data[key] = torch.from_numpy(
                    value.astype(np.float32)
                )  # Convert to Float32
            else:
                data[key] = torch.from_numpy(value)
    return data

def convert_to_absolute_scale(scale):
    if scale < 1:
        return 1/scale - 1
    else:
        return scale - 1


def _get_model_pts(model_pts, idx):
    if model_pts.ndim == 3:
        return model_pts[min(idx, model_pts.shape[0] - 1)]
    return model_pts


def eval_image_info(data_pred, data_gt, metric_dict):
    metric_dict["total_frames"] = data_pred.get("total_frames", 0)
    metric_dict["registered_frames"] = data_pred.get("registered_frames", 0)
    metric_dict["keyframe_count"] = data_pred.get("keyframe_count", 0)
    metric_dict["invalid_frames"] = data_pred.get("invalid_frames", 0)
    return metric_dict

def eval_add_object(data_pred, data_gt, metric_dict):
    pred_o2c = data_pred.get("extrinsics")
    gt_o2c = data_gt.get("o2c")
    model_pts = data_gt.get("v3d_can.object")
    is_valid = data_gt.get("is_valid") * data_pred.get("is_valid")

    if pred_o2c is None or gt_o2c is None or model_pts is None:
        print("[WARN][eval_add_object] missing extrinsics/gt poses/model points; skipping ADD.")
        return metric_dict

    pred_o2c = _to_numpy(pred_o2c)
    gt_o2c = _to_numpy(gt_o2c)
    model_pts = _to_numpy(model_pts)
    valid_flags = _to_numpy(is_valid) if is_valid is not None else None

    # Align predicted sequence to GT using first valid frame (match BundleSDF)
    pred_poses = [_build_pose_4x4(p) for p in pred_o2c]
    gt_poses = [np.array(g) for g in gt_o2c]
    n = min(len(gt_poses), len(pred_poses))
    # pred_poses and gt_poses has been aligned so we do not need to do additional alignment here.
    # if n > 0:
    #     align_tf = np.linalg.inv(pred_poses[0]) @ gt_poses[0]
    #     pred_poses = [p @ align_tf for p in pred_poses]

    add_vals = []
    for i in range(n):
        if valid_flags is not None and not bool(valid_flags[i]):
            add_vals.append(np.nan)
            continue
        pred_pose = pred_poses[i]
        gt_pose = gt_poses[i]
        cur_model_pts = _get_model_pts(model_pts, i)
        add_vals.append(add_err(pred_pose, gt_pose, cur_model_pts))

    metric_dict["add"] = np.array(add_vals)
    return metric_dict


def eval_add_s_object(data_pred, data_gt, metric_dict):
    pred_extr = data_pred.get("extrinsics")
    gt_o2c = data_gt.get("o2c")
    model_pts = data_gt.get("v3d_can.object")
    is_valid = data_gt.get("is_valid") * data_pred.get("is_valid")

    if pred_extr is None or gt_o2c is None or model_pts is None:
        print("[WARN][eval_add_s_object] missing extrinsics/gt poses/model points; skipping ADD-S.")
        return metric_dict

    pred_extr = _to_numpy(pred_extr)
    gt_o2c = _to_numpy(gt_o2c)
    model_pts = _to_numpy(model_pts)
    valid_flags = _to_numpy(is_valid) if is_valid is not None else None

    pred_poses = [_build_pose_4x4(p) for p in pred_extr]
    gt_poses = [np.array(g) for g in gt_o2c]
    n = min(len(gt_poses), len(pred_poses))
    # pred_poses and gt_poses has been aligned so we do not need to do additional alignment here.
    # if n > 0:
    #     align_tf = np.linalg.inv(pred_poses[0]) @ gt_poses[0]
    #     pred_poses = [p @ align_tf for p in pred_poses]

    adds_vals = []
    for i in range(n):
        if valid_flags is not None and not bool(valid_flags[i]):
            adds_vals.append(np.nan)
            continue
        pred_pose = pred_poses[i]
        gt_pose = gt_poses[i]
        cur_model_pts = _get_model_pts(model_pts, i)
        adds_vals.append(adi_err(pred_pose, gt_pose, cur_model_pts))

    metric_dict["add_s"] = np.array(adds_vals)
    return metric_dict


def eval_add_auc_object(data_pred, data_gt, metric_dict):
    metric_dict = eval_add_object(data_pred, data_gt, metric_dict)
    add_vals = np.array(metric_dict.get("add", []), dtype=float)
    add_vals = add_vals[np.isfinite(add_vals)]
    metric_dict["add_auc"] = compute_auc(add_vals) * 100.0 if add_vals.size else np.nan
    return metric_dict


def eval_add_s_auc_object(data_pred, data_gt, metric_dict):
    metric_dict = eval_add_s_object(data_pred, data_gt, metric_dict)
    adds_vals = np.array(metric_dict.get("add_s", []), dtype=float)
    adds_vals = adds_vals[np.isfinite(adds_vals)]
    metric_dict["add_s_auc"] = compute_auc(adds_vals) * 100.0 if adds_vals.size else np.nan
    return metric_dict


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def to_homo(pts):
    return np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)


def add_err(pred, gt, model_pts):
    pred_pts = (pred @ to_homo(model_pts).T).T[:, :3]
    gt_pts = (gt @ to_homo(model_pts).T).T[:, :3]
    return np.linalg.norm(pred_pts - gt_pts, axis=1).mean()


def adi_err(pred, gt, model_pts):
    pred_pts = (pred @ to_homo(model_pts).T).T[:, :3]
    gt_pts = (gt @ to_homo(model_pts).T).T[:, :3]
    nn_index = KDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
    return nn_dists.mean()


def _build_pose_4x4(pose):
    pose = np.array(pose)
    if pose.shape == (4, 4):
        return pose
    out = np.eye(4)
    out[:3] = pose
    return out


def compute_auc(rec, max_val=0.1):
    """
    Area under recall-threshold curve for pose errors.
    Matches BundleSDF: integrate fraction of samples below max_val (default 0.1m).
    """
    if len(rec) == 0:
        return 0.0
    rec = np.sort(np.array(rec))
    n = len(rec)
    prec = np.arange(1, n + 1) / float(n)
    # keep only errors within threshold
    idx = np.where(rec < max_val)[0]
    rec = rec[idx].reshape(-1)
    prec = prec[idx].reshape(-1)
    if rec.size == 0:
        return 0.0
    # add endpoints
    mrec = np.array([0.0, *rec.tolist(), max_val])
    mpre = np.array([0.0, *prec.tolist(), prec[-1]])
    # enforce monotonic precision
    for i in range(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])
    i = np.where(mrec[1:] != mrec[0:len(mrec) - 1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) / max_val
    return ap

def eval_icp_first_frame(data_pred, data_gt, metric_dict, debug=False):
    faces = data_pred["faces"]["object"]
    from vggt.utils.icp import compute_icp_metrics
    from open3d.geometry import TriangleMesh
    from open3d.utility import Vector3dVector, Vector3iVector
    selected_index = 0
    v3d_o_ra = Vector3dVector(data_pred["v3d_ra.object"][selected_index].numpy())
    faces_o = Vector3iVector(faces.cpu().numpy())
    if debug:
        # Create a Trimesh mesh object
        vertices = data_pred["v3d_ra.object"][selected_index].cpu().numpy()  # Get vertices as numpy array
        faces = faces.cpu().numpy()  # Get faces as numpy array
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Save the mesh to a file
        mesh.export("object_mesh.obj")  # Export to OBJ format or any other format supported by trimesh

        # Create a Trimesh mesh object
        vertices_gt = data_gt["v3d_ra.object"][selected_index].cpu().numpy()  # Get vertices as numpy array
        faces_gt = data_gt["faces.object"].numpy()  # Get faces as numpy array
        mesh_gt = trimesh.Trimesh(vertices=vertices_gt, faces=faces_gt, process=False)

        # Save the mesh to a file
        mesh_gt.export("gt_mesh.obj")  # Export to OBJ format or any other format supported by trimesh

    
    v3d_o_ra_gt = Vector3dVector(data_gt["v3d_ra.object"][selected_index].numpy())
    faces_o_gt = Vector3iVector(data_gt["faces.object"].numpy())
    source_mesh = TriangleMesh(v3d_o_ra, faces_o)
    target_mesh = TriangleMesh(v3d_o_ra_gt, faces_o_gt)
    best_cd, best_f5, best_f10, best_cd_no_scale, best_f5_no_scale, best_f10_no_scale, scale = compute_icp_metrics(
        target_mesh, source_mesh, num_iters=60, out_dir=data_pred["out_dir"]
    )
    metric_dict["cd_icp"] = best_cd
    metric_dict["f5_icp"] = best_f5 * 100.0
    metric_dict["f10_icp"] = best_f10 * 100.0
    metric_dict["cd_icp_no_scale"] = best_cd_no_scale
    metric_dict["f5_icp_no_scale"] = best_f5_no_scale * 100.0
    metric_dict["f10_icp_no_scale"] = best_f10_no_scale * 100.0
    metric_dict["scale"] = convert_to_absolute_scale(scale)
    return metric_dict


def eval_icp_every_frame(data_pred, data_gt, metric_dict):
    from src.utils.icp import compute_icp_metrics
    from open3d.geometry import TriangleMesh
    from open3d.utility import Vector3dVector, Vector3iVector

    is_valid = data_gt["is_valid"]

    num_frames = len(data_pred["v3d_o_ra"])
    cd_list = []
    f5_list = []
    f10_list = []
    cd_no_scale_list = []
    f5_no_scale_list = []
    f10_no_scale_list = []
    scale_list = []
    assert num_frames == len(data_gt["v3d_o_ra"])
    assert num_frames == len(is_valid)

    for idx in tqdm(range(num_frames)):
        if is_valid[idx]:
            v3d_o_ra = Vector3dVector(data_pred["v3d_o_ra"][idx].numpy())
            faces_o = Vector3iVector(data_pred["faces_o"][idx].numpy())
            v3d_o_ra_gt = Vector3dVector(data_gt["v3d_o_ra"][idx].numpy())
            faces_o_gt = Vector3iVector(data_gt["faces_o"].numpy())
            source_mesh = TriangleMesh(v3d_o_ra, faces_o)
            target_mesh = TriangleMesh(v3d_o_ra_gt, faces_o_gt)

            cd, f5, f10, cd_no_scale, f5_no_scale, f10_no_scale, scale = compute_icp_metrics(
                target_mesh, source_mesh, num_iters=10, no_tqdm=True, out_dir=data_pred["out_dir"]
            )
        else:
            cd = float("nan")
            f5 = float("nan")
            f10 = float("nan")
            cd_no_scale = float("nan")
            f5_no_scale = float("nan")
            f10_no_scale = float("nan")
            scale = float("nan")
        cd_list.append(cd)
        f5_list.append(f5)
        f10_list.append(f10)
        cd_no_scale_list.append(cd_no_scale)
        f5_no_scale_list.append(f5_no_scale)
        f10_no_scale_list.append(f10_no_scale)
        scale_list.append(scale)

    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)
    cd_no_scale_list = np.array(cd_no_scale_list)
    f5_no_scale_list = np.array(f5_no_scale_list)
    f10_no_scale_list = np.array(f10_no_scale_list)
    scale_list = np.array(scale_list)
    mean_cd = np.nanmean(cd_list)
    mean_f5 = np.nanmean(f5_list)
    mean_f10 = np.nanmean(f10_list)
    mean_cd_no_scale = np.nanmean(cd_no_scale_list)
    mean_f5_no_scale = np.nanmean(f5_no_scale_list)
    mean_f10_no_scale = np.nanmean(f10_no_scale_list)
    mean_scale = np.nanmean(scale_list)
    
    metric_dict["cd_icp"] = mean_cd
    metric_dict["f5_icp"] = mean_f5 * 100.0
    metric_dict["f10_icp"] = mean_f10 * 100.0
    metric_dict["cd_icp_no_scale"] = mean_cd_no_scale
    metric_dict["f5_icp_no_scale"] = mean_f5_no_scale * 100.0
    metric_dict["f10_icp_no_scale"] = mean_f10_no_scale * 100.0
    metric_dict["scale"] = convert_to_absolute_scale(mean_scale)
    return metric_dict


def eval_mrrpe_ho_right(data_pred, data_gt, metric_dict):
    j3d_h_c_pred = data_pred["j3d_c.right"]
    root_o_pred = data_pred["root.object"]

    j3d_h_c_gt = data_gt["j3d_c.right"]
    root_o_gt = data_gt["root.object"]
    is_valid = data_gt["is_valid"]

    root_h_gt = j3d_h_c_gt[:, 0]
    root_h_pred = j3d_h_c_pred[:, 0]
    mrrpe_ho = (
        metrics.compute_mrrpe(
            root_h_gt,
            root_o_gt,
            root_h_pred,
            root_o_pred,
            is_valid,
        )
        * 1000
    )
    not_valid = (1 - is_valid).numpy().astype(bool)
    mrrpe_ho[not_valid] = np.nan

    metric_dict["mrrpe_ho"] = mrrpe_ho
    return metric_dict


def calculate_chamfer_f_scores(vertices_source, vertices_target, is_sqrt=True):
    vertices_source = vertices_source * 100
    vertices_target = vertices_target * 100

    gen_points_kd_tree = KDTree(vertices_source)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(vertices_target)

    if is_sqrt: # square-root chamfer
        gt_to_gen_chamfer = np.mean(one_distances)
    else: # squared chamfer
        gt_to_gen_chamfer = np.mean(np.square(one_distances))
    # other direction
    gt_points_kd_tree = KDTree(vertices_target)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(vertices_source)
    if is_sqrt: # square-root chamfer
        gen_to_gt_chamfer = np.mean(two_distances)
    else: # squared chamfer
        gen_to_gt_chamfer = np.mean(np.square(two_distances))
    
    chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer
    threshold = 0.5  # 5 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

    threshold = 1.0  # 10 mm
    precision_1 = np.mean(one_distances < threshold).astype(np.float32)
    precision_2 = np.mean(two_distances < threshold).astype(np.float32)
    fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
    return chamfer_obj, fscore_obj_5, fscore_obj_10


def compute_iou_per_frame(insta_map_pred, insta_map_gt):
    classes = [0, 100, 200]
    ious = []

    for frame_idx in range(insta_map_pred.shape[0]):
        iou_per_class = []
        for cls in classes:
            pred_mask = insta_map_pred[frame_idx] == cls
            gt_mask = insta_map_gt[frame_idx] == cls
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            iou = intersection / union if union != 0 else 0
            iou_per_class.append(iou)
        ious.append(
            np.mean(iou_per_class)
        )  # Assuming you want the mean IoU for all classes per frame

    return np.array(ious)


def eval_cd_ra(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_o_c_ra"]
    v3d_o_c_gt_ra = data_gt["v3d_o_c_ra"]

    torch.manual_seed(1)
    rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_ra = (
        chamfer_distance(
            v3d_o_c_pred_ra[:, rand_pred_idx],
            v3d_o_c_gt_ra[:, rand_gt_idx],
            batch_reduction=None,
        )[0]
        * 1000
    )

    metric_dict["cd_ra"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    return metric_dict


def eval_cd_f(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_o_c"]
    v3d_o_c_gt_ra = data_gt["v3d_o_c"]
    is_valid = data_gt["is_valid"]

    torch.manual_seed(1)
    rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(v3d_o_c_pred_ra.shape[0]):
        cd_error, f5, f10 = calculate_chamfer_f_scores(
            v3d_o_c_pred_ra[idx, rand_pred_idx].numpy(),
            v3d_o_c_gt_ra[idx, rand_gt_idx].numpy(),
        )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    metric_dict["cd"] = cd_list
    metric_dict["f5"] = f5_list * 100.0
    metric_dict["f10"] = f10_list * 100.0
    return metric_dict


def eval_cd_f_right(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_right.object"]
    v3d_o_c_gt_ra = data_gt["v3d_right.object"]
    is_valid = data_gt["is_valid"]

    torch.manual_seed(1)

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(len(v3d_o_c_pred_ra)):
        v3d_pred = v3d_o_c_pred_ra[idx]
        v3d_gt = v3d_o_c_gt_ra[idx]

        if torch.isnan(v3d_pred.mean()) or torch.isnan(v3d_gt.mean()):
            cd_error = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        else:
            rand_pred_idx = torch.randperm(v3d_pred.shape[0])[:3000]
            rand_gt_idx = torch.randperm(v3d_gt.shape[0])[:3000]
            cd_error, f5, f10 = calculate_chamfer_f_scores(
                v3d_pred[rand_pred_idx].numpy(), v3d_gt[rand_gt_idx].numpy()
            )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    metric_dict["cd_right"] = cd_list
    metric_dict["f5_right"] = f5_list * 100.0
    metric_dict["f10_right"] = f10_list * 100.0
    return metric_dict


def eval_cd_f_ra(data_pred, data_gt, metric_dict):
    v3d_o_c_pred_ra = data_pred["v3d_ra.object"]
    v3d_o_c_gt_ra = data_gt["v3d_ra.object"]
    is_valid = data_gt["is_valid"]

    torch.manual_seed(1)
    # rand_gt_idx = torch.randperm(v3d_o_c_gt_ra.shape[1])[:3000]
    # rand_pred_idx = torch.randperm(v3d_o_c_pred_ra.shape[1])[:3000]

    cd_list = []
    f5_list = []
    f10_list = []
    for idx in range(len(v3d_o_c_pred_ra)):
        v3d_pred = v3d_o_c_pred_ra[idx]

        if torch.isnan(v3d_pred.mean()):
            cd_error = float("nan")
            f5 = float("nan")
            f10 = float("nan")
        else:
            v3d_gt = v3d_o_c_gt_ra[idx]

            num_pts = min(3000, v3d_pred.shape[0])
            rand_pred_idx = torch.randperm(v3d_pred.shape[0])[:num_pts]
            rand_gt_idx = torch.randperm(v3d_gt.shape[0])[:3000]
            cd_error, f5, f10 = calculate_chamfer_f_scores(
                v3d_pred[rand_pred_idx].numpy(), v3d_gt[rand_gt_idx].numpy()
            )
        cd_list.append(cd_error)
        f5_list.append(f5)
        f10_list.append(f10)
    cd_list = np.array(cd_list)
    f5_list = np.array(f5_list)
    f10_list = np.array(f10_list)

    not_valid = (1 - is_valid).numpy().astype(bool)
    cd_list[not_valid] = np.nan
    f5_list[not_valid] = np.nan
    f10_list[not_valid] = np.nan

    # metric_dict["cd_rh"] = cd_ra.numpy()  # Assuming cd_ra is a 1-element tensor
    metric_dict["cd_ra"] = cd_list
    metric_dict["f5_ra"] = f5_list * 100.0
    metric_dict["f10_ra"] = f10_list * 100.0
    return metric_dict


def eval_mpjpe_right(data_pred, data_gt, metric_dict):
    j3d_h_c_pred_ra = data_pred["j3d_ra.right"]
    j3d_h_c_gt_ra = data_gt["j3d_ra.right"]
    is_valid = data_gt["is_valid"]

    mpjpe_ra_r = metrics.compute_joint3d_error(j3d_h_c_gt_ra, j3d_h_c_pred_ra, is_valid)
    mpjpe_ra_r = mpjpe_ra_r.mean(axis=1) * 1000  # Use dim instead of axis for PyTorch

    metric_dict["mpjpe_ra_r"] = mpjpe_ra_r
    return metric_dict


def _rigid_align_transform(src, dst):
    """Least-squares rigid transform (rotation + translation, no scale).

    Solves for (R, t) minimizing ||(R @ src + t) - dst|| over the point sets
    using the Kabsch/Umeyama algorithm. Matches SLAHMR/GLAMR world-frame
    alignment used for global hand-motion metrics.

    Args:
        src: (N, 3) source points.
        dst: (N, 3) target points.

    Returns:
        (R, t): rotation (3, 3) and translation (3,) such that
        ``(R @ src.T).T + t`` aligns ``src`` onto ``dst``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    # Reflection-safe rotation (ensure det(R) = +1)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def _apply_rigid(pts, R, t):
    """Apply a rigid transform to (T, J, 3) joints. Returns (T, J, 3)."""
    flat = pts.reshape(-1, 3)
    out = (R @ flat.T).T + t
    return out.reshape(pts.shape)


def eval_global_mpjpe_right(data_pred, data_gt, metric_dict):
    """Global hand-motion metrics: G-MPJPE and GA-MPJPE (mm).

    Both operate on the right-hand joint trajectory expressed in the world
    (object) frame, following the SLAHMR/GLAMR convention used by Dyn-HaMR:

    - ``g_mpjpe``  (G-MPJPE):  rigidly align the predicted trajectory to GT
      using only the **first valid frame**, then per-frame MPJPE over the whole
      sequence. Captures accumulated global drift.
    - ``ga_mpjpe`` (GA-MPJPE): rigidly align the predicted trajectory to GT with
      a **single transform fit over all valid frames** (Procrustes over the full
      trajectory, rotation + translation only), then per-frame MPJPE. Captures
      global motion consistency independent of a global rigid offset.

    Expects ``data_pred["j3d_glob.right"]`` and ``data_gt["j3d_glob.right"]`` as
    (T, J, 3) world-frame joints in meters; results are returned in mm.
    """
    pred = data_pred.get("j3d_glob.right")
    gt = data_gt.get("j3d_glob.right")
    if pred is None or gt is None:
        print("[WARN][eval_global_mpjpe_right] missing world-frame hand joints; skipping G/GA-MPJPE.")
        return metric_dict

    pred = _to_numpy(pred).astype(np.float64)  # (T, J, 3)
    gt = _to_numpy(gt).astype(np.float64)      # (T, J, 3)
    T = min(len(pred), len(gt))
    pred = pred[:T]
    gt = gt[:T]

    is_valid = _to_numpy(data_gt.get("is_valid"))
    if is_valid is not None:
        valid = is_valid[:T].astype(bool)
    else:
        valid = np.ones(T, dtype=bool)
    # Drop frames with non-finite or dummy (-1000) joints just in case.
    finite = np.isfinite(pred).all(axis=(1, 2)) & np.isfinite(gt).all(axis=(1, 2))
    not_dummy = np.abs(gt).reshape(T, -1).max(axis=1) < 100.0
    valid = valid & finite & not_dummy

    g_mpjpe = np.full(T, np.nan)
    ga_mpjpe = np.full(T, np.nan)

    if valid.sum() >= 1:
        vp = pred[valid]
        vg = gt[valid]

        # G-MPJPE: align using the first valid frame only.
        R0, t0 = _rigid_align_transform(vp[0], vg[0])
        pred_g = _apply_rigid(pred, R0, t0)
        err_g = np.linalg.norm(pred_g - gt, axis=2).mean(axis=1) * 1000.0
        g_mpjpe[valid] = err_g[valid]

        # GA-MPJPE: single rigid alignment over the whole valid trajectory.
        Rg, tg = _rigid_align_transform(
            vp.reshape(-1, 3), vg.reshape(-1, 3)
        )
        pred_ga = _apply_rigid(pred, Rg, tg)
        err_ga = np.linalg.norm(pred_ga - gt, axis=2).mean(axis=1) * 1000.0
        ga_mpjpe[valid] = err_ga[valid]

    metric_dict["g_mpjpe"] = g_mpjpe
    metric_dict["ga_mpjpe"] = ga_mpjpe
    return metric_dict


def eval_ious(data_pred, data_gt, metric_dict):
    masks_pred = data_pred["masks_pred"].long().numpy()
    masks_gt = data_gt["masks_gt"].long().numpy()
    is_valid = data_gt["is_valid"]
    ious = compute_iou_per_frame(masks_pred, masks_gt)
    not_valid = (1 - is_valid).numpy().astype(bool)
    ious[not_valid] = np.nan
    metric_dict["ious"] = ious * 100.0
    return metric_dict


# ---------------------------------------------------------------------------
# Rendering-based hand/object metrics (mask IOU, depth consistency,
# penetration volume). These render GT hand + object meshes via nvdiffrast and
# compare against per-frame GT masks/depth loaded from the preprocessed data.
# ---------------------------------------------------------------------------
_glctx_cache = {}


def _get_glctx():
    if "glctx" not in _glctx_cache:
        import nvdiffrast.torch as dr
        _glctx_cache["glctx"] = dr.RasterizeCudaContext()
    return _glctx_cache["glctx"]


# Color codes (0-255) used to tag each part in the merged mesh render.
_OBJ_COLOR = np.array([0, 0, 255], dtype=np.uint8)      # blue  -> object
_HAND_COLOR = np.array([128, 0, 128], dtype=np.uint8)   # purple -> hand


def _render_merged_mask_depth(verts_obj, faces_obj, verts_hand, faces_hand, K, H, W):
    """Render object (blue) + hand (purple) as one merged mesh, then split by color.

    Rendering both parts together lets the z-buffer resolve occlusion, so each
    returned mask/depth only covers the *visible* surface of that part.

    Returns dict with keys ``obj_mask``, ``obj_depth``, ``hand_mask``,
    ``hand_depth`` (any entry is None if that part was not provided). Returns
    None on failure.
    """
    try:
        from utils_simba.render import nvdiffrast_render

        verts_list, faces_list, colors_list = [], [], []
        offset = 0
        if verts_obj is not None and faces_obj is not None:
            verts_obj = np.asarray(verts_obj, dtype=np.float32)
            faces_obj = np.asarray(faces_obj, dtype=np.int64)
            verts_list.append(verts_obj)
            faces_list.append(faces_obj + offset)
            colors_list.append(np.tile(_OBJ_COLOR, (len(verts_obj), 1)))
            offset += len(verts_obj)
        if verts_hand is not None and faces_hand is not None:
            verts_hand = np.asarray(verts_hand, dtype=np.float32)
            faces_hand = np.asarray(faces_hand, dtype=np.int64)
            verts_list.append(verts_hand)
            faces_list.append(faces_hand + offset)
            colors_list.append(np.tile(_HAND_COLOR, (len(verts_hand), 1)))
            offset += len(verts_hand)
        if not verts_list:
            return None

        merged = trimesh.Trimesh(
            vertices=np.concatenate(verts_list, axis=0),
            faces=np.concatenate(faces_list, axis=0),
            process=False,
        )
        merged.visual.vertex_colors = np.concatenate(colors_list, axis=0)

        # Identity ob_in_cvcam since verts are already in camera space
        ob_in_cvcam = torch.eye(4, dtype=torch.float32, device="cuda").unsqueeze(0)
        glctx = _get_glctx()
        color, depth, _ = nvdiffrast_render(
            K=K, H=H, W=W,
            ob_in_cvcams=ob_in_cvcam,
            glctx=glctx,
            mesh=merged,
            get_normal=False,
        )
        color_np = color[0].cpu().numpy()   # (H, W, 3) in [0, 1]
        depth_np = depth[0].cpu().numpy()   # (H, W)
        rendered = depth_np > 0

        # Classify each rendered pixel by nearest reference color.
        ref_obj = _OBJ_COLOR.astype(np.float32) / 255.0
        ref_hand = _HAND_COLOR.astype(np.float32) / 255.0
        d_obj = np.linalg.norm(color_np - ref_obj, axis=-1)
        d_hand = np.linalg.norm(color_np - ref_hand, axis=-1)

        out = {"obj_mask": None, "obj_depth": None, "hand_mask": None, "hand_depth": None}
        if verts_obj is not None and faces_obj is not None:
            obj_mask = rendered & (d_obj <= d_hand)
            out["obj_mask"] = obj_mask
            out["obj_depth"] = np.where(obj_mask, depth_np, 0.0).astype(np.float32)
        if verts_hand is not None and faces_hand is not None:
            hand_mask = rendered & (d_hand < d_obj)
            out["hand_mask"] = hand_mask
            out["hand_depth"] = np.where(hand_mask, depth_np, 0.0).astype(np.float32)
        return out
    except Exception:
        return None


def _save_mask_debug(save_dir, fid, tag, pred_mask, gt_mask):
    """Save pred/gt masks and an overlay (R=pred, G=gt, yellow=overlap) as PNGs."""
    from pathlib import Path
    if pred_mask is None or gt_mask is None:
        return
    try:
        import imageio.v2 as imageio
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        p = pred_mask.astype(bool)
        g = np.asarray(gt_mask).astype(bool)
        imageio.imwrite(save_dir / f"{fid:04d}_{tag}_pred.png", (p * 255).astype(np.uint8))
        imageio.imwrite(save_dir / f"{fid:04d}_{tag}_gt.png", (g * 255).astype(np.uint8))
        overlay = np.zeros((*p.shape, 3), dtype=np.uint8)
        overlay[..., 0] = p * 255  # pred -> red
        overlay[..., 1] = g * 255  # gt -> green (overlap -> yellow)
        imageio.imwrite(save_dir / f"{fid:04d}_{tag}_overlay.png", overlay)
    except Exception:
        pass


def _backproject_depth(depth, K, mask=None):
    """Back-project a depth map (HxW, meters) to camera-space points (M, 3)."""
    H, W = depth.shape[:2]
    valid = depth > 0
    if mask is not None:
        valid = valid & mask.astype(bool)
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[ys, xs].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _subsample_points(pts, max_pts=20000, seed=0):
    """Randomly subsample a point cloud so KD-tree queries stay cheap."""
    if len(pts) <= max_pts:
        return pts
    rng = np.random.default_rng(seed)
    return pts[rng.choice(len(pts), max_pts, replace=False)]


def _depth_chamfer_err(pred_depth, pred_mask, gt_depth, gt_mask, K, max_pts=10000):
    """Symmetric chamfer distance (cm) between depth-backprojected point clouds.

    Both depth maps are lifted to camera space with the same intrinsics, so the
    metric measures 3D surface disagreement rather than per-pixel depth offset
    (per-pixel differences are only defined where the two masks overlap and are
    dominated by silhouette mismatch at object boundaries).
    """
    if pred_depth is None or gt_depth is None or K is None:
        return float('nan')
    pred_pts = _backproject_depth(np.asarray(pred_depth), K, mask=pred_mask)
    gt_pts = _backproject_depth(np.asarray(gt_depth), K, mask=gt_mask if gt_mask is not None else pred_mask)
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float('nan')
    pred_pts = _subsample_points(pred_pts, max_pts)
    gt_pts = _subsample_points(gt_pts, max_pts)
    cd, _, _ = calculate_chamfer_f_scores(pred_pts.astype(np.float64), gt_pts.astype(np.float64))
    return float(cd)


def _save_depth_ply(save_dir, fid, tag, pred_depth, gt_depth, K, mask=None, gt_mask=None):
    """Back-project pred/gt depth to camera-space point clouds and save as PLY.

    Pred points are colored red, GT points green, for side-by-side inspection.
    """
    from pathlib import Path
    if K is None or (pred_depth is None and gt_depth is None):
        return
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if pred_depth is not None:
            pts = _backproject_depth(pred_depth, K, mask=mask)
            if len(pts) > 0:
                colors = np.tile(np.array([255, 0, 0], dtype=np.uint8), (len(pts), 1))
                trimesh.PointCloud(pts, colors=colors).export(save_dir / f"{fid:04d}_{tag}_pred_depth.ply")
        if gt_depth is not None:
            pts = _backproject_depth(np.asarray(gt_depth), K, mask=gt_mask if gt_mask is not None else mask)
            if len(pts) > 0:
                colors = np.tile(np.array([0, 255, 0], dtype=np.uint8), (len(pts), 1))
                trimesh.PointCloud(pts, colors=colors).export(save_dir / f"{fid:04d}_{tag}_gt_depth.ply")
    except Exception:
        pass


def _save_mesh_debug(save_dir, fid, tag, verts, faces, color=None):
    """Save a mesh (verts in camera space + faces) as a PLY for debug."""
    from pathlib import Path
    if verts is None or faces is None:
        return
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)
        if color is not None:
            mesh.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.vertices), 1))
        mesh.export(save_dir / f"{fid:04d}_{tag}_mesh.ply")
    except Exception:
        pass


def _save_merged_mesh_debug(save_dir, fid, tag, verts_obj, faces_obj, verts_hand, faces_hand,
                            obj_color=(0, 0, 255), hand_color=(128, 0, 128)):
    """Merge object (blue) + hand (purple) meshes into one PLY for debug."""
    from pathlib import Path
    try:
        verts_list, faces_list, colors_list = [], [], []
        offset = 0
        if verts_obj is not None and faces_obj is not None:
            vo = np.asarray(verts_obj, dtype=np.float64)
            fo = np.asarray(faces_obj, dtype=np.int64)
            verts_list.append(vo)
            faces_list.append(fo + offset)
            colors_list.append(np.tile(np.asarray(obj_color, dtype=np.uint8), (len(vo), 1)))
            offset += len(vo)
        if verts_hand is not None and faces_hand is not None:
            vh = np.asarray(verts_hand, dtype=np.float64)
            fh = np.asarray(faces_hand, dtype=np.int64)
            verts_list.append(vh)
            faces_list.append(fh + offset)
            colors_list.append(np.tile(np.asarray(hand_color, dtype=np.uint8), (len(vh), 1)))
            offset += len(vh)
        if not verts_list:
            return
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        merged = trimesh.Trimesh(
            vertices=np.concatenate(verts_list, axis=0),
            faces=np.concatenate(faces_list, axis=0),
            process=False,
        )
        merged.visual.vertex_colors = np.concatenate(colors_list, axis=0)
        merged.export(save_dir / f"{fid:04d}_{tag}_merged_mesh.ply")
    except Exception:
        pass


def eval_mask_iou(data_pred, data_gt, metric_dict):
    """Mask IOU of hand and object between rendered mask and GT mask."""
    from pathlib import Path
    from robust_hoi_pipeline.pipeline_utils import load_preprocessed_frame
    frame_indices = data_pred["valid_frame_indices"]
    K = data_gt["K"].numpy() if torch.is_tensor(data_gt["K"]) else np.array(data_gt["K"])
    v3d_c_obj = data_gt.get("v3d_c.object")
    v3d_c_hand = data_gt.get("v3d_c.right")
    faces_obj = data_gt.get("faces.object")
    faces_hand = data_gt.get("faces.right")
    data_preprocess_dir = Path(data_pred["data_preprocess_dir"]) if "data_preprocess_dir" in data_pred else None
    debug_dir = (data_preprocess_dir.parent / "mask_iou_debug") if data_preprocess_dir is not None else None

    def _iou(pred_mask, gt_mask):
        p, g = pred_mask.astype(bool), gt_mask.astype(bool)
        inter = (p & g).sum()
        union = (p | g).sum()
        return float(inter) / float(union) if union > 0 else float('nan')

    fo = faces_obj.numpy() if torch.is_tensor(faces_obj) else (np.array(faces_obj) if faces_obj is not None else None)
    fh = faces_hand.numpy() if torch.is_tensor(faces_hand) else (np.array(faces_hand) if faces_hand is not None else None)

    iou_obj_list, iou_hand_list = [], []
    for i, fid in enumerate(frame_indices):
        gt_frame = None
        if data_preprocess_dir is not None:
            try:
                gt_frame = load_preprocessed_frame(data_preprocess_dir, fid)
            except Exception:
                pass
        H, W = (gt_frame["image"].shape[:2] if gt_frame is not None and gt_frame.get("image") is not None else (480, 640))

        vo = (v3d_c_obj[i].numpy() if torch.is_tensor(v3d_c_obj[i]) else np.array(v3d_c_obj[i])) if v3d_c_obj is not None else None
        vh = (v3d_c_hand[i].numpy() if torch.is_tensor(v3d_c_hand[i]) else np.array(v3d_c_hand[i])) if v3d_c_hand is not None else None
        rendered = _render_merged_mask_depth(vo, fo, vh, fh, K, H, W) if gt_frame is not None else None

        if rendered is not None and rendered["obj_mask"] is not None:
            gt_mask = gt_frame.get("mask_obj")
            # if gt_mask is not None:
            #     _save_mask_debug(debug_dir, int(fid), "obj", rendered["obj_mask"], gt_mask > 0)
            iou_obj_list.append(_iou(rendered["obj_mask"], gt_mask > 0) if gt_mask is not None else float('nan'))
        else:
            iou_obj_list.append(float('nan'))

        if rendered is not None and rendered["hand_mask"] is not None:
            gt_mask = gt_frame.get("mask_hand")
            # if gt_mask is not None:
            #     _save_mask_debug(debug_dir, int(fid), "hand", rendered["hand_mask"], gt_mask > 0)
            iou_hand_list.append(_iou(rendered["hand_mask"], gt_mask > 0) if gt_mask is not None else float('nan'))
        else:
            iou_hand_list.append(float('nan'))

    metric_dict["mask_iou_obj"] = np.array(iou_obj_list, dtype=np.float32)
    metric_dict["mask_iou_hand"] = np.array(iou_hand_list, dtype=np.float32)
    return metric_dict


def eval_depth_consistency(data_pred, data_gt, metric_dict):
    """Depth consistency of hand and object as chamfer distance (cm).

    The predicted mesh is rendered to a depth map, both the rendered depth and the
    GT sensor depth are back-projected to camera-space point clouds (restricted to
    the predicted / GT mask respectively), and the symmetric chamfer distance
    between the two clouds is reported.
    """
    from pathlib import Path
    from robust_hoi_pipeline.pipeline_utils import load_preprocessed_frame
    frame_indices = data_pred["valid_frame_indices"]
    K = data_gt["K"].numpy() if torch.is_tensor(data_gt["K"]) else np.array(data_gt["K"])
    gt_v3d_c_obj = data_gt.get("v3d_c.object")
    gt_v3d_c_hand = data_gt.get("v3d_c.right")
    faces_obj = data_gt.get("faces.object")
    faces_hand = data_gt.get("faces.right")
    data_preprocess_dir = Path(data_pred["data_preprocess_dir"]) if "data_preprocess_dir" in data_pred else None
    debug_dir = (data_preprocess_dir.parent / "depth_consistency_debug") if data_preprocess_dir is not None else None

    gt_fo = faces_obj.numpy() if torch.is_tensor(faces_obj) else (np.array(faces_obj) if faces_obj is not None else None)
    gt_fh = faces_hand.numpy() if torch.is_tensor(faces_hand) else (np.array(faces_hand) if faces_hand is not None else None)

    err_obj_list, err_hand_list = [], []

    for i, fid in enumerate(frame_indices):
        gt_frame = None
        if data_preprocess_dir is not None:
            try:
                gt_frame = load_preprocessed_frame(data_preprocess_dir, fid)
            except Exception:
                pass
        H, W = (gt_frame["image"].shape[:2] if gt_frame is not None and gt_frame.get("image") is not None else (480, 640))

        gt_vo = (gt_v3d_c_obj[i].numpy() if torch.is_tensor(gt_v3d_c_obj[i]) else np.array(gt_v3d_c_obj[i])) if gt_v3d_c_obj is not None else None
        gt_vh = (gt_v3d_c_hand[i].numpy() if torch.is_tensor(gt_v3d_c_hand[i]) else np.array(gt_v3d_c_hand[i])) if gt_v3d_c_hand is not None else None
        
        gt_depth = gt_frame.get("depth") * gt_frame.get("depth_scale") if gt_frame is not None else None

        # Save merged GT (object+hand) and merged predicted ( object+hand) meshes for debug.
        if gt_frame is not None:
            # _save_merged_mesh_debug(debug_dir, int(fid), "gt", gt_vo, gt_fo, gt_vh, gt_fh)

            pred_vo_all = data_pred.get("v3d_c.object")
            pred_vh_all = data_pred.get("v3d_c.right")
            pred_fo = data_pred.get("faces.object")
            pred_fh = data_pred.get("faces.right")
            pred_fo = pred_fo.numpy() if torch.is_tensor(pred_fo) else (np.array(pred_fo) if pred_fo is not None else None)
            pred_fh = pred_fh.numpy() if torch.is_tensor(pred_fh) else (np.array(pred_fh) if pred_fh is not None else None)
            pred_vo = None
            if pred_vo_all is not None and i < len(pred_vo_all):
                pred_vo = pred_vo_all[i].numpy() if torch.is_tensor(pred_vo_all[i]) else np.asarray(pred_vo_all[i])
            pred_vh = None
            if pred_vh_all is not None and i < len(pred_vh_all):
                pred_vh = pred_vh_all[i].numpy() if torch.is_tensor(pred_vh_all[i]) else np.asarray(pred_vh_all[i])
            # _save_merged_mesh_debug(debug_dir, int(fid), "pred", pred_vo, pred_fo, pred_vh, pred_fh)

        rendered = _render_merged_mask_depth(pred_vo, pred_fo, pred_vh, pred_fh, K, H, W) if gt_frame is not None else None

        gt_mask_obj = (gt_frame.get("mask_obj") > 0) if (gt_frame is not None and gt_frame.get("mask_obj") is not None) else None
        gt_mask_hand = (gt_frame.get("mask_hand") > 0) if (gt_frame is not None and gt_frame.get("mask_hand") is not None) else None

        if rendered is not None and rendered["obj_mask"] is not None and gt_depth is not None:
            # _save_depth_ply(debug_dir, int(fid), "obj", rendered["obj_depth"], gt_depth, K,
            #                 mask=rendered["obj_mask"], gt_mask=gt_mask_obj)
            err_obj_list.append(_depth_chamfer_err(
                rendered["obj_depth"], rendered["obj_mask"], gt_depth, gt_mask_obj, K))
        else:
            err_obj_list.append(float('nan'))

        if rendered is not None and rendered["hand_mask"] is not None and gt_depth is not None:
            # _save_depth_ply(debug_dir, int(fid), "hand", rendered["hand_depth"], gt_depth, K,
            #                 mask=rendered["hand_mask"], gt_mask=gt_mask_hand)
            err_hand_list.append(_depth_chamfer_err(
                rendered["hand_depth"], rendered["hand_mask"], gt_depth, gt_mask_hand, K))
        else:
            err_hand_list.append(float('nan'))

    metric_dict["depth_err_obj"] = np.array(err_obj_list, dtype=np.float32)
    metric_dict["depth_err_hand"] = np.array(err_hand_list, dtype=np.float32)
    return metric_dict


def eval_penetration_volume(data_pred, data_gt, metric_dict):
    """Penetration volume between hand and object meshes (cm^3)."""
    v3d_c_obj = data_gt.get("v3d_c.object")
    v3d_c_hand = data_gt.get("v3d_c.right")
    faces_obj = data_gt.get("faces.object")
    faces_hand = data_gt.get("faces.right")
    frame_indices = data_pred["valid_frame_indices"]

    pen_list = []
    for i in range(len(frame_indices)):
        if any(x is None for x in [v3d_c_obj, v3d_c_hand, faces_obj, faces_hand]):
            pen_list.append(float('nan'))
            continue
        try:
            verts_obj = v3d_c_obj[i].numpy() if torch.is_tensor(v3d_c_obj[i]) else np.array(v3d_c_obj[i])
            verts_hand = v3d_c_hand[i].numpy() if torch.is_tensor(v3d_c_hand[i]) else np.array(v3d_c_hand[i])
            f_obj = faces_obj.numpy() if torch.is_tensor(faces_obj) else np.array(faces_obj)
            f_hand = faces_hand.numpy() if torch.is_tensor(faces_hand) else np.array(faces_hand)
            mesh_obj = trimesh.Trimesh(vertices=verts_obj, faces=f_obj, process=False)
            mesh_hand = trimesh.Trimesh(vertices=verts_hand, faces=f_hand, process=False)
            pts_hand, _ = trimesh.sample.sample_surface(mesh_hand, 5000)
            inside = mesh_obj.contains(pts_hand)
            if not inside.any():
                pen_list.append(0.0)
                continue
            hand_vol = abs(float(mesh_hand.volume)) if mesh_hand.is_watertight else float('nan')
            pen_list.append(float('nan') if np.isnan(hand_vol) else (inside.sum() / len(pts_hand)) * hand_vol * 1e6)
        except Exception:
            pen_list.append(float('nan'))

    metric_dict["penetration_volume"] = np.array(pen_list, dtype=np.float32)
    return metric_dict
