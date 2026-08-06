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
                        action_dim=model_config.action_dim, model_type=model_config.model_type
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

            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack_transform,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=("action",),
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
