"""LoRA finetune the PaliGemma VLM used inside MolmoBot-Pi0.

Loads google/paligemma-3b-mix-224, overwrites the common weights with the
PaliGemma sub-state-dict from a trained MolmoBot-Pi0 checkpoint, applies LoRA
to the language-model attention/MLP, and finetunes on a paligemma-style
prefix/suffix JSONL dataset (image + text in -> text out).
"""

from __future__ import annotations

import argparse
import json
import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Disable TensorFlow before importing transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# HF token read from the environment (export HF_TOKEN=...) for the gated
# google/paligemma-3b-mix-224 weights.
_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _HF_TOKEN:
    os.environ["HF_TOKEN"] = _HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _HF_TOKEN

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from torch.utils.data import Dataset
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import LoraConfig, get_peft_model

from src.data.reasoning_augment import AugConfig, augment

MOLMOBOT_PALIGEMMA_PREFIX = "paligemma_with_expert.paligemma."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        required=True,
        help="Reasoning dataset root, or several comma-separated roots to train one "
             "LoRA on a mixture. Each root keeps its own images/ dir; records are "
             "tagged with their root so image paths resolve per-record.",
    )
    p.add_argument(
        "--molmobot-ckpt",
        required=True,
    )
    p.add_argument("--base-model", default="google/paligemma-3b-mix-224")
    p.add_argument(
        "--output-dir",
        required=True,
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--run-name", default="lora_paligemma_pickplace")
    # Image augmentation (src/data/reasoning_augment.py); off by default, train split only.
    p.add_argument("--aug-photometric", action="store_true",
                   help="colour/blur/JPEG/noise jitter; leaves coordinates untouched")
    p.add_argument("--aug-geometric", action="store_true",
                   help="random resized crop; REWRITES bbox/keypoint/trace coordinates to match")
    p.add_argument("--aug-pad-geometry", action="store_true",
                   help="letterbox like the action stage's resize_with_pad instead of the "
                        "processor's squash-resize; rewrites coordinates to match")
    p.add_argument("--aug-strength", type=float, default=1.0)
    p.add_argument("--aug-seed", type=int, default=0)
    p.add_argument(
        "--resume",
        choices=["auto", "yes", "no"],
        default="auto",
        help="auto: resume from the latest checkpoint in output-dir if one exists "
        "(restores model+optimizer+scheduler+RNG for clean preemption recovery); "
        "yes: require a checkpoint; no: always start fresh.",
    )
    return p.parse_args()


@torch.no_grad()
def overwrite_paligemma_weights(
    model: PaliGemmaForConditionalGeneration,
    ckpt_path: str,
) -> None:
    """Stream `paligemma_with_expert.paligemma.*` tensors from the MolmoBot
    safetensors file directly into the model's parameters in-place. Avoids
    holding a second full state_dict copy in CPU RAM.
    """
    own = dict(model.named_parameters())
    own_buffers = dict(model.named_buffers())
    own.update(own_buffers)

    matched = partial_rows = mismatch = missing_in_model = 0
    extras: list[str] = []

    with safe_open(ckpt_path, framework="pt") as f:
        ckpt_keys = [k for k in f.keys() if k.startswith(MOLMOBOT_PALIGEMMA_PREFIX)]
        # Mirror lm_head onto embed_tokens (tied weights, ckpt only stores lm_head).
        embed_target = None
        for cand in (
            "language_model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ):
            if cand in own:
                embed_target = cand
                break

        for full_key in ckpt_keys:
            sub = full_key[len(MOLMOBOT_PALIGEMMA_PREFIX):]
            if sub not in own:
                missing_in_model += 1
                if len(extras) < 5:
                    extras.append(sub)
                continue
            tgt = own[sub]
            src = f.get_tensor(full_key).to(tgt.dtype)
            if tgt.shape == src.shape:
                tgt.data.copy_(src)
                matched += 1
            elif (
                src.dim() == 2
                and tgt.dim() == 2
                and src.shape[1] == tgt.shape[1]
                and src.shape[0] < tgt.shape[0]
            ):
                # Vocab-trim case: copy first N rows; leave the rest pretrained.
                tgt.data[: src.shape[0]].copy_(src)
                partial_rows += 1
            else:
                mismatch += 1
            del src

        # Replicate lm_head onto tied embed_tokens.
        lm_head_full = MOLMOBOT_PALIGEMMA_PREFIX + "lm_head.weight"
        if embed_target is not None and lm_head_full in ckpt_keys:
            tgt = own[embed_target]
            src = f.get_tensor(lm_head_full).to(tgt.dtype)
            if tgt.shape == src.shape:
                tgt.data.copy_(src)
                matched += 1
            elif src.shape[1] == tgt.shape[1] and src.shape[0] < tgt.shape[0]:
                tgt.data[: src.shape[0]].copy_(src)
                partial_rows += 1
            del src

    print(
        f"[ckpt-overwrite] matched={matched} partial_rows={partial_rows} "
        f"unrecoverable_mismatch={mismatch} ckpt_keys_absent_in_model={missing_in_model}"
    )
    if extras:
        print(f"[ckpt-overwrite] sample absent keys: {extras}")


@dataclass
class PaliGemmaJsonlDataset(Dataset):
    records: list[dict[str, Any]]
    images_root: Path
    aug: "AugConfig | None" = None
    aug_seed: int = 0
    _epoch: int = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.records[idx]
        # Mixture runs stamp each record with the root it came from; single-root
        # runs fall back to the shared images_root, unchanged.
        img_path = Path(r["_root"]) / r["image"] if "_root" in r else self.images_root / r["image"]
        image = Image.open(img_path).convert("RGB")
        suffix = r["suffix"]
        if self.aug is not None and self.aug.enabled:
            # Seeded per (record, epoch): reproducible, but a record is not shown the same
            # augmentation every epoch. Workers each get their own generator, so this must not
            # depend on worker id.
            rng = np.random.default_rng((self.aug_seed, idx, self._epoch))
            image, suffix = augment(image, suffix, self.aug, rng)
        return {"image": image, "prefix": r["prefix"], "suffix": suffix}


def make_collator(processor: PaliGemmaProcessor, dtype: torch.dtype):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [b["image"] for b in batch]
        prefixes = [b["prefix"] for b in batch]
        suffixes = [b["suffix"] for b in batch]
        enc = processor(
            text=prefixes,
            images=images,
            suffix=suffixes,
            return_tensors="pt",
            padding="longest",
        )
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(dtype)
        return enc

    return collate


class LossAccTrainer(Trainer):
    """Trainer that also logs token-level next-token-prediction accuracy.

    Accuracy is computed only over tokens whose label != -100 (i.e. the
    suffix tokens that PaliGemmaProcessor unmasked for us).
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._acc_sum = 0.0
        self._acc_n = 0

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss
        if labels is not None and getattr(outputs, "logits", None) is not None:
            with torch.no_grad():
                logits = outputs.logits[..., :-1, :]
                tgt = labels[..., 1:]
                mask = tgt.ne(-100)
                if mask.any():
                    pred = logits.argmax(-1)
                    correct = (pred.eq(tgt) & mask).sum().float()
                    total = mask.sum().float()
                    self._acc_sum += correct.item()
                    self._acc_n += total.item()
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs):  # type: ignore[override]
        if "loss" in logs and self._acc_n > 0:
            logs["token_acc"] = self._acc_sum / self._acc_n
            self._acc_sum = 0.0
            self._acc_n = 0
        return super().log(logs, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = outputs.loss
            tok_acc = None
            if labels is not None and getattr(outputs, "logits", None) is not None:
                logits = outputs.logits[..., :-1, :]
                tgt = labels[..., 1:]
                mask = tgt.ne(-100)
                if mask.any():
                    pred = logits.argmax(-1)
                    tok_acc = (
                        (pred.eq(tgt) & mask).sum().float() / mask.sum().float()
                    )
        # Stash accuracy on the trainer so an EvalAccCallback can collect it.
        if tok_acc is not None:
            if not hasattr(self, "_eval_acc_buf"):
                self._eval_acc_buf = []
            self._eval_acc_buf.append(tok_acc.item())
        return (loss.detach(), None, None)


class EvalAccCallback(TrainerCallback):
    """Inject mean token_acc into the eval log dict."""

    def __init__(self, trainer: "LossAccTrainer"):
        super().__init__()
        self.trainer = trainer

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        buf = getattr(self.trainer, "_eval_acc_buf", [])
        if metrics is not None and buf:
            metrics["eval_token_acc"] = sum(buf) / len(buf)
        self.trainer._eval_acc_buf = []


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    roots = [Path(r.strip()) for r in str(args.data_root).split(",") if r.strip()]
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    for root in roots:
        tr = load_jsonl(root / "train.jsonl")
        va = load_jsonl(root / "val.jsonl") if (root / "val.jsonl").exists() else []
        for rec in (*tr, *va):
            rec["_root"] = str(root)
        train_records += tr
        val_records += va
        print(f"[data] {root.name}: train={len(tr)} val={len(va)}")
    if len(roots) > 1:
        # Interleave the roots before any truncation, so --max-train-samples and the
        # first epoch's early steps see the mixture rather than whichever root was
        # listed first.
        random.Random(args.seed).shuffle(train_records)
    data_root = roots[0]
    if args.max_train_samples:
        train_records = train_records[: args.max_train_samples]
    mix = ", ".join(f"{r.name}" for r in roots)
    print(f"[data] TOTAL train={len(train_records)}  val={len(val_records)}  roots=[{mix}]")

    print(f"[model] loading base {args.base_model}")
    dtype = torch.bfloat16
    processor = PaliGemmaProcessor.from_pretrained(args.base_model)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    print(f"[model] streaming common weights from {args.molmobot_ckpt}")
    overwrite_paligemma_weights(model, args.molmobot_ckpt)

    # Freeze everything except the language model (LoRA will re-enable a slice).
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad = False

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    aug_cfg = AugConfig(
        photometric=args.aug_photometric,
        geometric=args.aug_geometric,
        pad_geometry=args.aug_pad_geometry,
        strength=args.aug_strength,
    )
    if aug_cfg.enabled:
        print(f"[aug] photometric={aug_cfg.photometric} geometric={aug_cfg.geometric} "
              f"pad_geometry={aug_cfg.pad_geometry} strength={aug_cfg.strength} "
              f"seed={args.aug_seed}  (train split only)")
    train_ds = PaliGemmaJsonlDataset(train_records, data_root, aug=aug_cfg, aug_seed=args.aug_seed)
    val_ds = PaliGemmaJsonlDataset(val_records, data_root)      # never augmented
    collator = make_collator(processor, dtype)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=1e-6,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to="none",
        run_name=args.run_name,
        seed=args.seed,
    )

    trainer = LossAccTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    trainer.add_callback(EvalAccCallback(trainer))

    # Resume from the latest checkpoint (model+optimizer+scheduler+RNG) so a
    # preempted run continues exactly where it left off.
    resume_ckpt = None
    if args.resume != "no" and os.path.isdir(args.output_dir):
        resume_ckpt = get_last_checkpoint(args.output_dir)
    if args.resume == "yes" and resume_ckpt is None:
        raise SystemExit(f"--resume=yes but no checkpoint found in {args.output_dir}")
    if resume_ckpt is not None:
        print(f"[resume] continuing from {resume_ckpt}")
    else:
        print("[resume] no checkpoint found — starting fresh")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    final_dir = Path(args.output_dir) / "final_adapter"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[done] saved LoRA adapter to {final_dir}")


if __name__ == "__main__":
    main()
