# openpi fork (TToTMooN/openpi) — context for AI sessions

Fork of Physical-Intelligence/openpi. Branch `main` tracks upstream; branch
**`hub`** carries the vla-hub integration (github.com/TToTMooN/vla-hub — read
its CLAUDE.md and docs/EXTENDING.md first; this fork is the training core the
hub drives, not the place hub features live).

## Hub-branch rules

- **Additive files only** (upstream rebases must stay trivial). Allowed
  upstream-file edits (keep them tiny; each is flagged inline):
  `src/openpi/training/config.py` (2-line hub_configs splice + the
  `state_history_frames` DataConfig field), `src/openpi/models/model.py`
  (masked-camera skip), `scripts/serve_policy.py` (`--num-steps`),
  `src/openpi/training/data_loader.py` (guarded state-history
  delta_timestamps for rel_ee_history profiles),
  `src/openpi/policies/policy.py` (guarded `state_anchor` passthrough into
  the infer outputs dict — without it RigidBodyAbsoluteActions anchors on
  the rel-mode state and serves mis-anchored chunks).
- Hub files: `src/openpi/transforms_se3.py` (SE(3) chunk-relative EE
  transforms, pure numpy, NO openpi/jax imports — it is a VENDORED copy of
  vla-hub's geometry, parity-tested from vla-hub's test suite; keep them in
  sync), `src/openpi/policies/hub_ee_policy.py` (profile-driven
  Inputs/Outputs; computes the inter-gripper-pose state extra on the fly),
  `src/openpi/training/hub_configs.py` (TrainConfig registration; lazy imports
  inside `get_hub_configs()` to avoid circular imports; repack maps MUST
  include `"prompt": "prompt"` when `prompt_from_task` is used).
- Conventions: rot6d = first two ROWS of R; datasets store ABSOLUTE poses and
  are relativized by `RigidBodyDeltaActions` in `data_transforms` (so norm
  stats land in relative space); component-wise `DeltaActions` is correct ONLY
  for joint-space configs, never for rot6d dims.
- Norm stats are written by vla-hub's parquet-native writer into
  `assets/<config_name>/<repo_id>/norm_stats.json` (openpi's
  `compute_norm_stats.py` also works but decodes video — slow).
- pi05 + LoRA configs here are experimental (not an upstream recipe).

## Workflow

- Train: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py <config>
  --exp-name <name> --overwrite` (this checkout is the default
  `VLAHUB_OPENPI_ROOT`).
- Serve: `uv run scripts/serve_policy.py --port=N policy:checkpoint
  --policy.config=<config> --policy.dir=<step dir>` — top-level flags BEFORE
  the subcommand (tyro).
- After pushing `hub`, bump the submodule in vla-hub.
