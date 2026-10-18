"""Merge a trained PaliGemma LoRA adapter into a MolmoBot-Pi0 safetensors
checkpoint, producing a new checkpoint that can be loaded via
`--pytorch-weight-path` by `src/agent/train_action.py`.

Pipeline:
  1. Load base PaliGemma (google/paligemma-3b-mix-224).
  2. Overwrite its weights with the `paligemma_with_expert.paligemma.*` slice
     of the source MolmoBot-Pi0 checkpoint (same logic as src/agent/train_reasoning.py).
  3. Apply the LoRA adapter on top and call `merge_and_unload()`.
  4. Stream the original MolmoBot-Pi0 safetensors out, replacing every
     `paligemma_with_expert.paligemma.<sub>` entry with the merged tensor.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# HF token read from the environment (export HF_TOKEN=...) for the gated
# google/paligemma-3b-mix-224 weights.
_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _HF_TOKEN:
    os.environ["HF_TOKEN"] = _HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _HF_TOKEN

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import PaliGemmaForConditionalGeneration

PREFIX = "paligemma_with_expert.paligemma."


@torch.no_grad()
def _overwrite_paligemma_from_molmobot(model, ckpt_path: str) -> None:
    own = dict(model.named_parameters())
    own.update(dict(model.named_buffers()))
    with safe_open(ckpt_path, framework="pt") as f:
        keys = [k for k in f.keys() if k.startswith(PREFIX)]
        for full in keys:
            sub = full[len(PREFIX):]
            if sub not in own:
                continue
            tgt = own[sub]
            src = f.get_tensor(full).to(tgt.dtype)
            if tgt.shape == src.shape:
                tgt.data.copy_(src)
            elif (
                src.dim() == 2
                and tgt.dim() == 2
                and src.shape[1] == tgt.shape[1]
                and src.shape[0] < tgt.shape[0]
            ):
                tgt.data[: src.shape[0]].copy_(src)
            del src


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--molmobot-ckpt",
        required=True,
    )
    p.add_argument(
        "--lora-adapter",
        required=True,
    )
    p.add_argument("--base-model", default="google/paligemma-3b-mix-224")
    p.add_argument(
        "--output-ckpt",
        required=True,
    )
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_path = Path(args.output_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16
    print(f"[1/4] load base {args.base_model}")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True,
    )

    print(f"[2/4] overlay paligemma slice from {args.molmobot_ckpt}")
    _overwrite_paligemma_from_molmobot(model, args.molmobot_ckpt)

    print(f"[3/4] apply + merge LoRA adapter from {args.lora_adapter}")
    model = PeftModel.from_pretrained(model, args.lora_adapter)
    model = model.merge_and_unload()
    merged = {k: v.detach().to(dtype).cpu() for k, v in model.state_dict().items()}
    del model
    print(f"       merged param count: {len(merged)}")

    print(f"[4/4] write {out_path}")
    out: dict[str, torch.Tensor] = {}
    replaced = kept = 0
    with safe_open(args.molmobot_ckpt, framework="pt") as f:
        for full in f.keys():
            if full.startswith(PREFIX):
                sub = full[len(PREFIX):]
                if sub in merged:
                    src = merged[sub]
                    tgt = f.get_tensor(full)
                    if tgt.shape == src.shape:
                        out[full] = src.to(tgt.dtype).contiguous()
                        replaced += 1
                        continue
                    if (
                        src.dim() == 2
                        and tgt.dim() == 2
                        and src.shape[1] == tgt.shape[1]
                        and src.shape[0] <= tgt.shape[0]
                    ):
                        new = tgt.clone()
                        new[: src.shape[0]].copy_(src.to(tgt.dtype))
                        out[full] = new.contiguous()
                        replaced += 1
                        continue
            out[full] = f.get_tensor(full).contiguous()
            kept += 1
    print(f"       replaced={replaced}  kept={kept}")
    save_file(out, str(out_path))

    src_meta = Path(args.molmobot_ckpt).parent / "metadata.pt"
    if src_meta.exists():
        import shutil
        shutil.copy2(src_meta, out_path.parent / "metadata.pt")
        print(f"       copied metadata.pt -> {out_path.parent / 'metadata.pt'}")

    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
