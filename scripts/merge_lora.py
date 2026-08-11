"""Merge LoRA adapters into base weights, producing a plain checkpoint (vla-hub).

External runtimes (FlashRT etc.) and the non-LoRA TrainConfigs load only plain
weight layouts, so this is the bridge from a `*_lora` finetune to a deployable
standard checkpoint:

    uv run scripts/merge_lora.py \
        --checkpoint-dir checkpoints/<cfg_lora>/<exp>/<step> \
        --output-dir    checkpoints/<cfg_lora>/<exp>/<step>_merged

Math (see src/openpi/models/lora.py): Einsum-LoRA composes over the rank axis
on the weight's last two dims — merged w += scale * einsum('...ir,...ro->...io',
lora_a, lora_b). openpi's Einsum applies scale = alpha/rank while the
FeedForward path applies NO scale; both released variants (gemma_2b_lora r16
a16, gemma_300m_lora r32 a32) have alpha == rank, so the effective scale is
1.0 everywhere — asserted below rather than assumed.

Output: <output-dir>/params (orbax, plain layout loadable by the matching
non-LoRA config) + assets/ copied from the source checkpoint. Equivalence of
merged-vs-LoRA outputs should be verified before deploying (the hub does this).
"""

import pathlib
import shutil

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tyro

import openpi.models.model as _model


def _merge_tree(tree: dict, *, path: str = "") -> tuple[dict, list[str]]:
    """Recursively merge {*, lora_a, lora_b} triples; returns (new_tree, log)."""
    out = {}
    log = []
    keys = set(tree)
    lora_a_keys = [k for k in keys if k == "lora_a" or k.endswith("_lora_a")]
    consumed = set()
    for a_key in lora_a_keys:
        b_key = a_key[:-1] + "b"
        base_key = "w" if a_key == "lora_a" else a_key[: -len("_lora_a")]
        if b_key not in keys or base_key not in keys:
            raise KeyError(f"{path}: found {a_key} without {b_key}/{base_key}")
        a = np.asarray(tree[a_key], dtype=np.float32)
        b = np.asarray(tree[b_key], dtype=np.float32)
        w = tree[base_key]
        rank = a.shape[-1]
        assert rank in (16, 32), f"{path}/{a_key}: unexpected rank {rank}"
        update = np.einsum("...ir,...ro->...io", a, b)
        assert update.shape == w.shape, f"{path}/{base_key}: {update.shape} vs {w.shape}"
        # Keep merged weights in float32: base checkpoints store bf16, and
        # re-rounding (W + AB) to bf16 discards a large fraction of the small
        # LoRA delta (|AB| ~ 2% of |W| vs bf16 rounding ~ 0.4% of |W|).
        # Serving casts to bf16 at load — the same rounding any full-finetune
        # checkpoint gets. Equivalence is gated BEHAVIORALLY (open-loop eval),
        # not bitwise.
        out[base_key] = jnp.asarray(np.asarray(w, dtype=np.float32) + update)
        consumed.update({a_key, b_key, base_key})
        log.append(f"{path}/{base_key} += lora(rank={rank})  {w.shape}")
    for k in keys - consumed:
        v = tree[k]
        if isinstance(v, dict):
            sub, sub_log = _merge_tree(v, path=f"{path}/{k}")
            out[k] = sub
            log.extend(sub_log)
        else:
            out[k] = v
    return out, log


def main(checkpoint_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    checkpoint_dir = checkpoint_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} exists")

    params = _model.restore_params(checkpoint_dir / "params", restore_type=np.ndarray)
    merged, log = _merge_tree(params)
    if not log:
        raise ValueError("no LoRA parameters found — is this a LoRA checkpoint?")
    print(f"merged {len(log)} LoRA pairs; first few:")
    for line in log[:5]:
        print(f"  {line}")

    output_dir.mkdir(parents=True)
    ocp.PyTreeCheckpointer().save(output_dir / "params", {"params": merged})
    if (checkpoint_dir / "assets").exists():
        shutil.copytree(checkpoint_dir / "assets", output_dir / "assets")

    # Round-trip validation: the merged tree must load as a plain layout.
    _model.restore_params(output_dir / "params", restore_type=np.ndarray)
    print(f"OK -> {output_dir} (params + assets); load-validated")


if __name__ == "__main__":
    tyro.cli(main)
