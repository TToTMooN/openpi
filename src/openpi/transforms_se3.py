"""SE(3) chunk-relative action transforms for EE-rel (rot6d) profiles.

VENDORED from vlaforge (geometry.py / staterep.py) — pure numpy, no openpi/jax
imports, so the hub can parity-test this exact file without a JAX environment
(vlaforge/tests/test_fork_parity.py). Keep changes mirrored.

Conventions: rot6d = first two ROWS of R (UMI/PyTorch3D/LeRobot; RLinf uses
columns — transposed). Datasets store ABSOLUTE poses; these transforms
relativize at train time (A_h = inv(T_state) @ T_h) and invert at inference
(T_h = T_state @ A_h). Component-wise subtraction on rotation dims would yield
non-orthogonal garbage — hence this file instead of `DeltaActions`.

Layout entries mirror RLinf's declarative style:
    {"kind": "pose6d", "xyz": (s, e), "rot6d": (s, e)}
    {"kind": "scalar_abs", "idx": i}          # grippers — always absolute
Dims not covered by the layout (e.g. padding) pass through untouched.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

_EPS = 1e-8

DUAL_ARM_EE_LAYOUT_20: tuple[dict[str, Any], ...] = (
    {"kind": "pose6d", "xyz": (0, 3), "rot6d": (3, 9)},
    {"kind": "scalar_abs", "idx": 9},
    {"kind": "pose6d", "xyz": (10, 13), "rot6d": (13, 19)},
    {"kind": "scalar_abs", "idx": 19},
)


# -- vendored math (mirror of vlaforge.geometry; rows convention) -------------


def rot6d_from_mat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    return R[..., :2, :].reshape(R.shape[:-2] + (6,)).copy()


def mat_from_rot6d(d6: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    bad1 = n1[..., 0] < _EPS
    b1 = a1 / np.maximum(n1, _EPS)
    a2p = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    n2 = np.linalg.norm(a2p, axis=-1, keepdims=True)
    bad2 = n2[..., 0] < _EPS
    b2 = a2p / np.maximum(n2, _EPS)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=-2)
    bad = bad1 | bad2
    if bad.any():
        if fallback is None:
            raise ValueError(f"degenerate rot6d input at {int(bad.sum())} element(s)")
        R = np.where(bad[..., None, None], np.broadcast_to(fallback, R.shape), R)
    return R


def make_se3(xyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    T = np.zeros(xyz.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = xyz
    T[..., 3, 3] = 1.0
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def pose9_to_mat(p: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return make_se3(p[..., :3], mat_from_rot6d(p[..., 3:9], fallback=fallback))


def mat_to_pose9(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[..., :3, 3], rot6d_from_mat(T[..., :3, :3])], axis=-1)


# -- transforms ---------------------------------------------------------------


def _group_pose9(vec: np.ndarray, entry: dict[str, Any]) -> np.ndarray:
    xs, xe = entry["xyz"]
    rs, re = entry["rot6d"]
    return np.concatenate([vec[..., xs:xe], vec[..., rs:re]], axis=-1)


def _write_pose9(vec: np.ndarray, entry: dict[str, Any], pose9: np.ndarray) -> None:
    xs, xe = entry["xyz"]
    rs, re = entry["rot6d"]
    vec[..., xs:xe] = pose9[..., :3]
    vec[..., rs:re] = pose9[..., 3:9]


@dataclasses.dataclass(frozen=True)
class RigidBodyDeltaActions:
    """abs -> chunk-relative (body frame): A_h = inv(T_state) @ T_h.

    `state` is (D,), `actions` is (H, D); the whole chunk shares the one state
    anchor. scalar_abs slots and dims beyond the layout pass through.
    """

    layout: Sequence[dict[str, Any]] = DUAL_ARM_EE_LAYOUT_20

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        # rel_ee_history mode relativizes the state SLOTS — the absolute SE(3)
        # anchor then travels as `state_anchor` (stashed by HubEEInputs).
        state = np.asarray(data.get("state_anchor", data["state"]), dtype=np.float64)
        if state.ndim == 2:  # stacked [past, current] from state-history loading
            state = state[-1]
        actions = np.array(data["actions"], dtype=np.float64, copy=True)
        for entry in self.layout:
            if entry["kind"] != "pose6d":
                continue
            T_state = pose9_to_mat(_group_pose9(state, entry))
            T_abs = pose9_to_mat(_group_pose9(actions, entry))
            A = se3_inv(T_state) @ T_abs
            _write_pose9(actions, entry, mat_to_pose9(A))
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class RigidBodyAbsoluteActions:
    """chunk-relative -> abs: T_h = T_state @ A_h. Inverse of the above; used on
    the output side at inference (after Unnormalize, before robot Outputs)."""

    layout: Sequence[dict[str, Any]] = DUAL_ARM_EE_LAYOUT_20

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        # rel_ee_history mode relativizes the state SLOTS — the absolute SE(3)
        # anchor then travels as `state_anchor` (stashed by HubEEInputs).
        state = np.asarray(data.get("state_anchor", data["state"]), dtype=np.float64)
        if state.ndim == 2:  # stacked [past, current] from state-history loading
            state = state[-1]
        actions = np.array(data["actions"], dtype=np.float64, copy=True)
        for entry in self.layout:
            if entry["kind"] != "pose6d":
                continue
            T_state = pose9_to_mat(_group_pose9(state, entry))
            fallback = T_state[..., :3, :3]
            A = pose9_to_mat(_group_pose9(actions, entry), fallback=fallback)
            T_abs = T_state @ A
            _write_pose9(actions, entry, mat_to_pose9(T_abs))
        data["actions"] = actions
        return data


# -- state extras (vendored mirror of vlaforge.staterep) ----------------------


def rel_history_state(
    state: np.ndarray, state_past: np.ndarray, layout: Sequence[dict[str, Any]]
) -> np.ndarray:
    """UMI-correct model state (frame-invariant; mirror of vlaforge.staterep
    rel_ee_history — parity-tested): the pose SLOTS carry the PAST pose in the
    CURRENT frame, T_t^-1 T_past (latest = identity), gripper slots keep the
    current width; inter-gripper (from current absolutes) appends; then each
    arm's PAST gripper width. Observations must never contain absolute pose —
    the UMI/SLAM map frame is arbitrary per session."""
    state = np.asarray(state, dtype=np.float64)
    past = np.asarray(state_past, dtype=np.float64)
    with_ig = append_inter_gripper_pose(state, layout)
    out = with_ig.copy()
    past_widths = []
    for entry in layout:
        if entry["kind"] == "pose6d":
            T_cur = pose9_to_mat(_group_pose9(state, entry))
            T_past = pose9_to_mat(_group_pose9(past, entry))
            _write_pose9(out, entry, mat_to_pose9(se3_inv(T_cur) @ T_past))
        elif entry["kind"] == "scalar_abs":
            past_widths.append(past[..., entry["idx"] : entry["idx"] + 1])
    return np.concatenate([out, *past_widths], axis=-1)


def append_inter_gripper_pose(state: np.ndarray, layout: Sequence[dict[str, Any]]) -> np.ndarray:
    """Append the second arm's pose expressed in the first arm's frame (9 dims)."""
    state = np.asarray(state, dtype=np.float64)
    poses = [e for e in layout if e["kind"] == "pose6d"]
    if len(poses) < 2:
        raise ValueError("inter_gripper_pose needs two pose6d groups")
    T_first = pose9_to_mat(_group_pose9(state, poses[0]))
    T_second = pose9_to_mat(_group_pose9(state, poses[1]))
    rel = mat_to_pose9(se3_inv(T_first) @ T_second)
    return np.concatenate([state, rel], axis=-1)
