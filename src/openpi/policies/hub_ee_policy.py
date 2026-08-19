"""Generic Inputs/Outputs for hub EE-rel profiles (vlaforge).

Profile-driven instead of one-class-per-robot: the camera slots and native
action dim come from the hub profile; state extras (inter-gripper pose) are
computed on the fly so datasets stay in the native layout.

Expected post-repack keys (training) == client obs keys (inference):
    observation/left_wrist_image, observation/right_wrist_image   uint8 HWC or CHW
    state       (D_native,) absolute [xyz, rot6d, grip] per arm
    actions     (H, D_native) absolute — RigidBodyDeltaActions relativizes next
    prompt      str
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi import transforms_se3
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class HubEEInputs(transforms.DataTransformFn):
    action_dim: int  # model action dim (32) — state/actions pad later
    model_type: _model.ModelType = _model.ModelType.PI05
    add_inter_gripper_pose: bool = True
    layout: tuple = transforms_se3.DUAL_ARM_EE_LAYOUT_20
    # This rig has no exterior camera. A masked zero-image is excluded from
    # attention, so OMITTING the slot entirely is mathematically equivalent and
    # skips its SigLIP pass + dead attention width (~1/3 of vision compute).
    # True restores the padded 3-slot form (needed for pi0-FAST, which does not
    # mask padding images).
    include_masked_base: bool = False

    # "ee_pose_gripper" (historical, absolute pose slots) or "rel_ee_history"
    # (UMI-correct: pose slots carry the past pose in the current frame; obs
    # never contain absolute pose). Train time delivers a stacked
    # [past, current] state via the loader's state-history timestamps; serve
    # time delivers the current state + obs["state_history"] on the wire
    # (missing history degrades to identity motion).
    state_mode: str = "ee_pose_gripper"

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float64)
        past = None
        if state.ndim == 2:  # stacked [past, current] (training loader)
            past, state = state[0], state[-1]
        elif "state_history" in data and data["state_history"] is not None:
            hist = np.asarray(data["state_history"], dtype=np.float64)
            past = np.atleast_2d(hist)[0]
        anchor = state.copy()
        if self.state_mode == "rel_ee_history":
            if past is None:
                past = state  # identity motion — degrade, never crash
            state = transforms_se3.rel_history_state(state, past, self.layout)
        elif self.add_inter_gripper_pose:
            state = transforms_se3.append_inter_gripper_pose(state, self.layout)

        left = _parse_image(data["observation/left_wrist_image"])
        right = _parse_image(data["observation/right_wrist_image"])

        images = {"left_wrist_0_rgb": left, "right_wrist_0_rgb": right}
        masks = {"left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_}
        if self.include_masked_base or self.model_type == _model.ModelType.PI0_FAST:
            images["base_0_rgb"] = np.zeros_like(left)
            masks["base_0_rgb"] = (
                np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
            )

        inputs = {
            "state": state,
            # absolute native anchor for the SE(3) action transforms — in
            # rel_ee_history mode the state slots no longer carry it
            "state_anchor": anchor,
            "image": images,
            "image_mask": masks,
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float64)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class HubEEOutputs(transforms.DataTransformFn):
    native_action_dim: int = 20

    def __call__(self, data: dict) -> dict:
        # Keep state so the client can cross-check the anchor it sent; actions
        # here are already absolute (RigidBodyAbsoluteActions ran before us).
        return {"actions": np.asarray(data["actions"][..., : self.native_action_dim])}
