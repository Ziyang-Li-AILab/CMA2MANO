# -*- coding: utf-8 -*-
"""MANO model helpers, copied verbatim from the original run_pipeline.py.

Kept identical on purpose so that the exported parameters (beta, global_orient,
hand_pose residual convention, trans, cam_R/cam_t) mean exactly the same thing
as in the original conversion. Source: baseline_original_code/run_pipeline.py
(MANO_TO_OPENPOSE ... initial_pose, export_hand, point_error_mm).
"""
from __future__ import annotations

import hashlib
import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

DEFAULT_MANO = Path(r"C:\Users\11068\Desktop\项目材料\ace-ego-hand\Ace-Ego-Hand-main\repro\2608.20308\code\data\mano\mano")
TIP_INDEX = np.array((4, 8, 12, 16, 20))
BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20),
         (5, 9), (9, 13), (13, 17)]

MANO_TO_OPENPOSE = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
MANO_PARENTS = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
TIP_VERTEX_IDS = np.array([744, 320, 443, 554, 671])
NATIVE_FROM_OP = np.array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3])
TIP_FOR_NATIVE = {3: 8, 6: 12, 9: 20, 12: 16, 15: 4}


def rodrigues(axis_angle):
    vectors = np.asarray(axis_angle, dtype=np.float64)
    return Rotation.from_rotvec(vectors.reshape(-1, 3)).as_matrix().reshape(vectors.shape[:-1] + (3, 3))


def matrix_to_axis_angle(matrices):
    return Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(matrices.shape[:-2] + (3,))


def physical_to_residual(model, hand_pose):
    """Residual is physical axis-angle minus hands_mean, subtracted once."""
    physical = np.asarray(hand_pose, dtype=np.float64)
    return physical - model["hands_mean"]


class _ChumpyStub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state)


class _SelectStub(_ChumpyStub):
    pass


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _SelectStub if name == "Select" else _ChumpyStub
        if module.startswith("scipy.sparse") and name in ("csc_matrix", "csr_matrix"):
            return getattr(sparse, name)
        return super().find_class(module, name)


def _resolve(value):
    if isinstance(value, _SelectStub):
        selected = np.asarray(_resolve(value.a)).ravel()[np.asarray(value.idxs)]
        return selected.reshape(value.preferred_shape)
    if isinstance(value, _ChumpyStub):
        return np.asarray(_resolve(value.x))
    if hasattr(value, "toarray"):
        return np.asarray(value.toarray())
    return value


def read_model_arrays(path, side):
    if side not in ("left", "right"):
        raise ValueError(f"side must be left or right, got {side!r}")
    path = Path(path)
    filename = f"MANO_{side.upper()}.pkl"
    if path.is_dir():
        candidates = list(path.rglob(filename))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one {filename} below {path}, found {len(candidates)}")
        payload = candidates[0].read_bytes()
        member = str(candidates[0].relative_to(path))
    else:
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in archive.namelist() if name.rsplit("/", 1)[-1] == filename]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one {filename} in {path}, found {len(candidates)}")
            member = candidates[0]
            payload = archive.read(member)
    raw = _ArrayUnpickler(io.BytesIO(payload), encoding="latin1").load()
    arrays = {key: _resolve(value) for key, value in raw.items()}
    metadata = {"side": side, "pkl_sha256": hashlib.sha256(payload).hexdigest(), "source_member": member}
    return arrays, metadata


def load_mano(path, side="left", mirror_left_shapedirs=True):
    arrays, metadata = read_model_arrays(path, side)
    shapedirs = np.asarray(arrays["shapedirs"], dtype=np.float64)[..., :10].copy()
    applied = False
    gap = None
    if side == "left":
        right, _ = read_model_arrays(path, "right")
        gap = float(np.abs(shapedirs[:, 0] - np.asarray(right["shapedirs"])[:, 0, :10]).sum())
        applied = bool(mirror_left_shapedirs and gap < 1.0)
        if applied:
            shapedirs[:, 0] *= -1
    parents = np.asarray(arrays["kintree_table"])[0].astype(np.int64)
    parents[0] = -1
    if not np.array_equal(parents, MANO_PARENTS):
        raise ValueError(f"Unexpected MANO kinematic tree: {parents.tolist()}")
    metadata.update({
        "mirror_applied": applied,
        "left_right_x_gap_before_correction": gap,
        "flat_hand_mean": False,
    })
    model = {
        "v_template": np.asarray(arrays["v_template"], dtype=np.float64),
        "shapedirs": shapedirs,
        "posedirs": np.asarray(arrays["posedirs"], dtype=np.float64).reshape(778, 3, 135),
        "J_regressor": np.asarray(arrays["J_regressor"], dtype=np.float64),
        "parents": parents,
        "weights": np.asarray(arrays["weights"], dtype=np.float64),
        "f": np.asarray(arrays["f"], dtype=np.int32),
        "hands_mean": np.asarray(arrays["hands_mean"], dtype=np.float64).reshape(45),
        "metadata": metadata,
    }
    if model["v_template"].shape != (778, 3) or model["J_regressor"].shape != (16, 778):
        raise ValueError("Unexpected MANO template or joint-regressor shape")
    return model


def _validate_inputs(beta, global_orient, hand_pose, transl):
    beta = np.asarray(beta, dtype=np.float64)
    orient = np.asarray(global_orient, dtype=np.float64)
    pose = np.asarray(hand_pose, dtype=np.float64)
    trans = np.asarray(transl, dtype=np.float64)
    if orient.ndim == 1:
        orient, pose, trans = orient[None], pose[None], trans[None]
    if beta.shape != (10,):
        raise ValueError(f"beta must be one shared (10,) vector, got {beta.shape}")
    count = orient.shape[0]
    if orient.shape != (count, 3) or pose.shape != (count, 45) or trans.shape != (count, 3):
        raise ValueError(f"Pose/translation shapes mismatch: {orient.shape}, {pose.shape}, {trans.shape}")
    if count == 0 or not all(np.isfinite(value).all() for value in (beta, orient, pose, trans)):
        raise ValueError("MANO inputs must be nonempty and finite")
    return beta, orient, pose, trans


def mano_forward(model, beta, global_orient, hand_pose, transl, pose_is_residual=True):
    """Metres. transl places the rest wrist at J0+transl. Residual pose adds hands_mean once."""
    beta, orient, pose, trans = _validate_inputs(beta, global_orient, hand_pose, transl)
    v_shaped = model["v_template"] + model["shapedirs"] @ beta
    rest = model["J_regressor"] @ v_shaped
    physical = pose + model["hands_mean"] if pose_is_residual else pose
    local = rodrigues(np.concatenate([orient[:, None], physical.reshape(-1, 15, 3)], axis=1))
    feature = (local[:, 1:] - np.eye(3)).reshape(-1, 135)
    v_posed = v_shaped[None] + np.einsum("vck,tk->tvc", model["posedirs"], feature)
    transforms = np.zeros((len(orient), 16, 4, 4), dtype=np.float64)
    transforms[:, :, 3, 3] = 1
    transforms[:, 0, :3, :3] = local[:, 0]
    transforms[:, 0, :3, 3] = rest[0]
    for joint in range(1, 16):
        parent = int(model["parents"][joint])
        relative = np.broadcast_to(np.eye(4), (len(orient), 4, 4)).copy()
        relative[:, :3, :3] = local[:, joint]
        relative[:, :3, 3] = rest[joint] - rest[parent]
        transforms[:, joint] = transforms[:, parent] @ relative
    native_joints = transforms[:, :, :3, 3] + trans[:, None]
    skin = transforms.copy()
    skin[:, :, :3, 3] -= np.einsum("tjab,jb->tja", transforms[:, :, :3, :3], rest)
    blended = np.einsum("vj,tjab->tvab", model["weights"], skin)
    vertices = np.einsum("tvab,tvb->tva", blended[:, :, :3, :3], v_posed)
    vertices += blended[:, :, :3, 3] + trans[:, None]
    joints = np.concatenate([native_joints, vertices[:, TIP_VERTEX_IDS]], axis=1)[:, MANO_TO_OPENPOSE]
    return vertices, joints, transforms[:, :, :3, :3], rest


def _unit(vector):
    return vector / max(np.linalg.norm(vector), 1e-12)


def _frame(axis, up):
    x = _unit(axis)
    z = np.cross(x, up)
    if np.linalg.norm(z) < 1e-7:
        z = np.cross(x, [1.0, 0.0, 0.0] if abs(x[0]) < 0.8 else [0.0, 1.0, 0.0])
    z = _unit(z)
    return np.stack([x, np.cross(z, x), z], axis=-1)


def _align(source, target):
    u, _, vt = np.linalg.svd(source.T @ target)
    fix = np.eye(3)
    fix[2, 2] = np.linalg.det(vt.T @ u.T)
    return vt.T @ fix @ u.T


def initial_pose(model, points_m, quaternions_R):
    """Bone-length beta, frozen palm calibration C, residual pose. Points are metres."""
    from scipy.optimize import least_squares

    measured = points_m[:, NATIVE_FROM_OP]
    q = quaternions_R[:, NATIVE_FROM_OP]
    parents = model["parents"]
    target_lengths = np.linalg.norm(measured[:, 1:] - measured[:, parents[1:]], axis=-1).mean(0)

    def bone_error(beta):
        rest = model["J_regressor"] @ (model["v_template"] + model["shapedirs"] @ beta)
        residual = np.linalg.norm(rest[1:] - rest[parents[1:]], axis=-1) - target_lengths
        return np.r_[residual * 1000, beta * 10]

    beta = least_squares(bone_error, np.zeros(10), bounds=(-3, 3)).x
    shaped = model["v_template"] + model["shapedirs"] @ beta
    rest = model["J_regressor"] @ shaped
    reference = int(np.argmax(np.linalg.norm(points_m[:, [4, 8, 12, 16, 20]] - points_m[:, :1], axis=-1).sum(-1)))
    palm = [1, 4, 10, 7, 13]
    rest_dirs = np.array([_unit(rest[j] - rest[0]) for j in palm])
    dirs = np.array([_unit(measured[reference, j] - measured[reference, 0]) for j in palm])
    calibration = np.zeros((16, 3, 3))
    calibration[0] = q[reference, 0].T @ _align(rest_dirs, dirs)
    normal = np.cross(rest[1] - rest[0], rest[7] - rest[0])
    for joint in range(1, 16):
        if joint in TIP_FOR_NATIVE:
            op_tip = TIP_FOR_NATIVE[joint]
            tip_vertex = TIP_VERTEX_IDS[[15, 3, 6, 12, 9].index(joint)]
            axis = shaped[tip_vertex] - rest[joint]
            target = points_m[reference, op_tip] - measured[reference, joint]
        else:
            axis = rest[joint + 1] - rest[joint]
            target = measured[reference, joint + 1] - measured[reference, joint]
        desired = _frame(target, q[reference, joint, :, 1]) @ _frame(axis, normal).T
        calibration[joint] = q[reference, joint].T @ desired
    global_R = q @ calibration
    local = global_R.copy()
    for joint in range(1, 16):
        local[:, joint] = global_R[:, parents[joint]].transpose(0, 2, 1) @ global_R[:, joint]
    go = matrix_to_axis_angle(local[:, 0])
    physical = matrix_to_axis_angle(local[:, 1:]).reshape(-1, 45)
    hp = physical_to_residual(model, physical)
    return dict(
        beta=beta, go=go, hp=hp, trans=measured[:, 0] - rest[0],
        calibration=calibration, reference_frame=reference, rest_wrist=rest[0],
    )


def export_hand(model, fit, wrist_m, origin_mm, rig_R, mount_R, mount_t):
    beta = np.asarray(fit["beta"], dtype=np.float32).astype(np.float64)
    go = np.asarray(fit["go"], dtype=np.float32).astype(np.float64)
    hp = np.asarray(fit["hp"], dtype=np.float32).astype(np.float64)
    _, _, _, rest = mano_forward(model, beta, go[:1], hp[:1], np.zeros((1, 3)))
    trans = (wrist_m + fit["delta_mm"] / 1000 - rest[0]).astype(np.float32).astype(np.float64)
    vertices, joints, _, _ = mano_forward(model, beta, go, hp, trans)
    rotation = np.asarray(mount_R) @ np.swapaxes(rig_R, -1, -2)
    translation = np.asarray(mount_t) / 1000 - np.einsum("tij,tj->ti", rotation, origin_mm / 1000)
    joints_cam = np.einsum("tij,tkj->tki", rotation, joints) + translation[:, None, :]
    return {
        "beta": beta.astype(np.float32),
        "global_orient": go.astype(np.float32),
        "hand_pose": hp.astype(np.float32),
        "trans": trans.astype(np.float32),
        "joints_world": joints.astype(np.float32),
        "vertices_mm": (vertices * 1000).astype(np.float32),
        "faces": np.asarray(model["f"], dtype=np.int32),
        "joints_cam": joints_cam.astype(np.float32),
        "cam_R": rotation.astype(np.float32),
        "cam_t": translation.astype(np.float32),
        "rest_wrist": rest[0].astype(np.float64),
    }


def wrist_error_mm(joints_world, wrist_m):
    return np.linalg.norm(joints_world[:, 0] - wrist_m, axis=-1) * 1000


def point_error_mm(joints_world, points_mm):
    return np.linalg.norm(joints_world - points_mm / 1000, axis=-1) * 1000

