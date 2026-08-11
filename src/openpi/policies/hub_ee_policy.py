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

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float64)
        if self.add_inter_gripper_pose:
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
