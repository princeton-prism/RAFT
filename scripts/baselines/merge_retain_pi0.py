"""RETAIN baseline for MolmoBot-Pi0 (arXiv:2512.08333).

Robust finetuning via parameter merging: interpolate the *language-model*
weights of a finetuned pi0 back toward the pretrained generalist, while keeping
the vision encoder and action expert fully finetuned:

    theta_tilde_llm = (1 - alpha) * theta_base_llm + alpha * theta_ft_llm
    (vision encoder, multimodal projector, action expert: kept fully finetuned)

alpha = 1.0  -> pure finetuned model (no merge)
alpha = 0.0  -> language model reset to the base generalist's LLM

Here the "pretrained generalist" (--base) is the base MolmoBot-Pi0-DROID, and the
finetuned specialist (--finetuned) is a full-FT checkpoint.
Both are flat model.safetensors sharing the pi0 key namespace:

    paligemma_with_expert.paligemma.model.language_model.*  <- merged (alpha)
    paligemma_with_expert.paligemma.lm_head.weight          <- merged (alpha)
    paligemma_with_expert.paligemma.model.vision_tower.*     <- finetuned
    paligemma_with_expert.paligemma.model.multi_modal_*      <- finetuned
    paligemma_with_expert.gemma_expert.*                     <- finetuned
    action_*/state_proj.*                                    <- finetuned

Output is a flat model.safetensors + metadata.pt + assets/ copied from the
FINETUNED checkpoint (so eval de-normalizes with the task's norm_stats), which
pi0 eval loads directly.

Usage (single alpha):
    python scripts/baselines/merge_retain_pi0.py \
        --base       checkpoints/MolmoBot-Pi0-DROID/model.safetensors \
        --finetuned  <finetuned checkpoint dir> \
        --alpha <alpha> \
        --output-dir <output dir>

Usage (sweep -- one output dir per alpha):
    python scripts/baselines/merge_retain_pi0.py --base ... --finetuned ... \
        --alpha <alpha> [<alpha> ...] --output-root <output root>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("merge_retain")

# The language-model subspace RETAIN interpolates. Everything else is kept
# finetuned. `lm_head` is the LLM output projection (tied to embed_tokens); it
# belongs with the language model.
LLM_MARKERS = (
    "paligemma_with_expert.paligemma.model.language_model.",
    "paligemma_with_expert.paligemma.lm_head.",
)
# Optional extras you can fold into the merged (language) side via --also-merge.
EXTRA_MARKERS = {
    "projector": "paligemma_with_expert.paligemma.model.multi_modal_projector.",
    "embed": "paligemma_with_expert.paligemma.model.language_model.embed_tokens.",  # already in LLM
}


def _resolve_weights(spec: str) -> str:
    p = Path(spec)
    if p.is_dir():
        cand = p / "model.safetensors"
        if not cand.exists():
            raise SystemExit(f"no model.safetensors under {p}")
        return str(cand)
    return str(p)


def _is_llm(key: str, extra_markers: tuple[str, ...]) -> bool:
    return key.startswith(LLM_MARKERS) or (bool(extra_markers) and key.startswith(extra_markers))


@torch.no_grad()
def merge_one(base_f, ft_f, alpha: float, extra_markers: tuple[str, ...], save_dtype):
    out: dict[str, torch.Tensor] = {}
    n_merged = n_kept = 0
    drift_merged = drift_kept = 0.0
    for k in ft_f.keys():
        ft = ft_f.get_tensor(k)
        if _is_llm(k, extra_markers) and k in base_f.keys():
            b = base_f.get_tensor(k).float()
            f = ft.float()
            out[k] = ((1.0 - alpha) * b + alpha * f).to(save_dtype)
            drift_merged += torch.norm(f - b).item() ** 2
            n_merged += 1
        else:
            out[k] = ft.to(save_dtype)
            if k in base_f.keys():
                drift_kept += torch.norm(ft.float() - base_f.get_tensor(k).float()).item() ** 2
            n_kept += 1
    log.info(f"[alpha={alpha}] merged(LLM)={n_merged} kept(vision/expert)={n_kept} "
             f"|ft-base|_LLM={drift_merged ** 0.5:.3f} |ft-base|_kept={drift_kept ** 0.5:.3f}")
    return out


def _copy_assets(assets_src: Path, out_dir: Path) -> list[str]:
    if assets_src.is_file():
        assets_src = assets_src.parent
    copied = []
    if (assets_src / "metadata.pt").exists():
        shutil.copy2(assets_src / "metadata.pt", out_dir / "metadata.pt")
        copied.append("metadata.pt")
    if (assets_src / "assets").exists():
        dst = out_dir / "assets"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(assets_src / "assets", dst)
        copied.append("assets/")
    return copied


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True,
                   help="Pretrained generalist model.safetensors (base MolmoBot-Pi0-DROID).")
    p.add_argument("--finetuned", required=True,
                   help="Finetuned specialist: model.safetensors or its dir.")
    p.add_argument("--alpha", type=float, nargs="+", required=True,
                   help="LLM interpolation coeff(s). theta=(1-a)*base + a*ft.")
    p.add_argument("--also-merge", nargs="*", default=[], choices=sorted(EXTRA_MARKERS),
                   help="Extra param groups to interpolate alongside the LLM (default: LLM only).")
    p.add_argument("--output-dir", default=None,
                   help="Single-alpha output dir (requires exactly one --alpha).")
    p.add_argument("--output-root", default=None,
                   help="Sweep root; each alpha -> <root>/alpha<a>/.")
    p.add_argument("--assets-from", default=None,
                   help="Dir for metadata.pt + assets/ (default: --finetuned dir; must carry the "
                        "task norm_stats the FT policy expects).")
    p.add_argument("--save-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    save_dtype = torch.bfloat16 if args.save_dtype == "bfloat16" else torch.float32
    extra = tuple(EXTRA_MARKERS[e] for e in args.also_merge)

    base_path = _resolve_weights(args.base)
    ft_path = _resolve_weights(args.finetuned)
    assets_src = Path(args.assets_from).resolve() if args.assets_from else Path(ft_path).resolve()

    if args.output_dir and len(args.alpha) != 1:
        raise SystemExit("--output-dir needs exactly one --alpha (use --output-root for a sweep).")
    if not args.output_dir and not args.output_root:
        raise SystemExit("pass --output-dir (single) or --output-root (sweep).")

    base_f = safe_open(base_path, framework="pt")
    ft_f = safe_open(ft_path, framework="pt")
    log.info(f"[base] {base_path}")
    log.info(f"[ft]   {ft_path}")
    log.info(f"[merge] LLM markers + extras={args.also_merge or '[]'}")

    for a in args.alpha:
        out_dir = (Path(args.output_dir).resolve() if args.output_dir
                   else Path(args.output_root).resolve() / f"alpha{a}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = merge_one(base_f, ft_f, a, extra, save_dtype)
        weights = out_dir / "model.safetensors"
        log.info(f"[save] {weights} ({len(out)} tensors, {args.save_dtype})")
        save_file({k: v.contiguous() for k, v in out.items()}, str(weights))
        copied = _copy_assets(assets_src, out_dir)
        if not copied:
            log.warning(f"[assets] none found under {assets_src} -- eval needs assets/.../norm_stats.json")
        else:
            log.info(f"[assets] copied {copied} from {assets_src}")
        (out_dir / "retain_merge.json").write_text(json.dumps({
            "method": "RETAIN (language-model interpolation)",
            "base": base_path, "finetuned": ft_path,
            "alpha": a, "also_merge": args.also_merge,
            "formula": "theta_llm = (1-alpha)*base + alpha*ft; vision+expert kept finetuned",
        }, indent=2))
        log.info(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
