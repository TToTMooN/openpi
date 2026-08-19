"""vlahub hub configs — additive registration of profile-driven TrainConfigs.

Spliced into `_CONFIGS` via one line in config.py (the roboarena/polaris
pattern; imports are deferred into the function to avoid circular imports).
Each hub profile (embodiment x representation family) gets a config here; the
hub's pipeline computes norm stats into assets/<config>/<repo_id>/ before
training (parquet-native, openpi-format — compute_norm_stats.py also works).
"""

import dataclasses

from typing_extensions import override

from openpi import transforms as _transforms
from openpi import transforms_se3
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.hub_ee_policy as hub_ee_policy
from openpi.training.weight_loaders import CheckpointWeightLoader

_HUB_EE_REPO_ID = "vlahub/cardboard_box_tcp_curated_10s_ee_rel"
_PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"


def get_hub_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import DataConfigFactory
    from openpi.training.config import ModelTransformFactory
    from openpi.training.config import TrainConfig

    @dataclasses.dataclass(frozen=True)
    class LeRobotHubEEDataConfig(DataConfigFactory):
        """EE-rel (chunk-anchored SE(3), rot6d rows) profile over a LeRobot
        dataset whose state/action are absolute [xyz, rot6d, grip] per arm
        (20-dim bimanual, produced by vlahub's teleop_ee adapter)."""

        default_prompt: str | None = None
        # "ee_pose_gripper" (historical, absolute pose in state) or
        # "rel_ee_history" (UMI-correct: pose slots carry the past pose in the
        # current frame; obs never contain absolute pose). Must match the hub
        # profile's repspec.state.mode.
        state_mode: str = "ee_pose_gripper"
        state_history_stride: int = 5  # frames; ~167 ms at 30 fps (UMI ~150 ms)

        @override
        def create(self, assets_dirs, model_config: _model.BaseModelConfig) -> DataConfig:
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/left_wrist_image": "observation.images.left_head",
                            "observation/right_wrist_image": "observation.images.right_head",
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            )

            data_transforms = _transforms.Group(
                inputs=[
                    hub_ee_policy.HubEEInputs(
                        action_dim=model_config.action_dim, model_type=model_config.model_type,
                        state_mode=self.state_mode,
                    )
                ],
                outputs=[hub_ee_policy.HubEEOutputs()],
            )
            # SE(3) body-frame chunk delta — component-wise DeltaActions would
            # corrupt the rot6d dims. Norm stats are computed downstream of this.
            data_transforms = data_transforms.push(
                inputs=[transforms_se3.RigidBodyDeltaActions()],
                outputs=[transforms_se3.RigidBodyAbsoluteActions()],
            )

            model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
                model_config
            )

            base = self.create_base_config(assets_dirs, model_config)
            # Fail at construction, not first batch/infer: stats computed under
            # the other state_mode (pre-flip checkpoint served under the rel
            # config, or stale assets) would otherwise surface as a cryptic
            # broadcast error deep in Normalize. Bimanual 20-dim native state:
            # 29 = 20 + 9 inter-gripper; 31 = rel history layout (+2 past widths).
            expected = 31 if self.state_mode == "rel_ee_history" else 29
            if base.norm_stats is not None and "state" in base.norm_stats:
                got = base.norm_stats["state"].mean.shape[-1]
                if got != expected:
                    raise ValueError(
                        f"norm stats state dim {got} != {expected} expected for "
                        f"state_mode={self.state_mode!r}. A 29-dim stats file under the "
                        "rel config means a pre-flip checkpoint: serve it with config "
                        "hub_portable_bimanual_ee_v1 (or re-run the hub pipeline to "
                        "regenerate stats)."
                    )

            return dataclasses.replace(
                base,
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=("action",),
                state_history_frames=(
                    (-self.state_history_stride, 0)
                    if self.state_mode == "rel_ee_history" else None
                ),
            )

    lora_model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=24,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )

    return [
        # Full finetune — needs ~70GB+ (A100-80G/H100) or fsdp_devices.
        TrainConfig(
            name="hub_portable_bimanual_ee",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=24),
            data=LeRobotHubEEDataConfig(
                repo_id=_HUB_EE_REPO_ID,
                assets=AssetsConfig(),
                base_config=DataConfig(prompt_from_task=True),
                # UMI-correct state (profile v2, repspec rel_ee_history):
                # frame-invariant obs — past-in-current pose slots + inter-
                # gripper + past widths. Pre-flip checkpoints serve via
                # hub_portable_bimanual_ee_v1 (vla-hub's serve path selects it
                # from the checkpoint's shipped representation.json).
                state_mode="rel_ee_history",
            ),
            weight_loader=CheckpointWeightLoader(_PI05_BASE),
            num_train_steps=20_000,
            batch_size=32,
        ),
        # Legacy serving config for pre-flip checkpoints (absolute pose in the
        # state, 29-dim stats). Never train this — it exists so old checkpoints
        # keep serving/evaluating after the production config moved to
        # rel_ee_history under the same assets.
        TrainConfig(
            name="hub_portable_bimanual_ee_v1",
            model=pi0_config.Pi0Config(pi05=True, action_horizon=24),
            data=LeRobotHubEEDataConfig(
                repo_id=_HUB_EE_REPO_ID,
                assets=AssetsConfig(),
                base_config=DataConfig(prompt_from_task=True),
                state_mode="ee_pose_gripper",
            ),
            weight_loader=CheckpointWeightLoader(_PI05_BASE),
            num_train_steps=20_000,
            batch_size=32,
        ),
        # LoRA variant for a single 24-32GB GPU (RTX 4090/5090 class).
        # NOTE: pi05 + LoRA is not an upstream-published recipe — treat as
        # experimental; PI found LoRA weaker than full FT at DROID scale.
        TrainConfig(
            name="hub_portable_bimanual_ee_lora",
            model=lora_model,
            data=LeRobotHubEEDataConfig(
                repo_id=_HUB_EE_REPO_ID,
                assets=AssetsConfig(),
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=CheckpointWeightLoader(_PI05_BASE),
            freeze_filter=lora_model.get_freeze_filter(),
            ema_decay=None,
            num_train_steps=20_000,
            batch_size=16,
        ),
    ]
