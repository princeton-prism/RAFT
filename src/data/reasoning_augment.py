"""Image augmentation for the reasoning-LoRA datasets, with label-consistent coordinate rewriting.

The paligemma_finetune suffixes carry spatial supervision as literal numbers:

    Relevant Objects:  blue toy: <bbox>[449.0, 476.0, 485.0, 516.0]</bbox>
      grasp_point: [467.0, 496.0]
      [[467.0, 496.0], [467.0, 446.0], ...]        (the trace line)

Coordinates are [x, y] normalized to 0-1000 with the origin at the top left (not pixels), so a
geometric transform is arithmetic in [0, 1000] with no image-size bookkeeping.

Any geometric augmentation therefore has to rewrite these numbers or it silently corrupts the
labels — an image cropped without moving its boxes teaches the model the wrong grounding. That is
what this module exists to prevent. Photometric augmentation leaves them alone.

Three tiers, independently switchable:
  photometric   brightness/contrast/saturation/hue, grayscale, blur, JPEG recompression. Coords
                untouched, so it is label-safe by construction.
  geometric     random resized crop + translation. Coords transformed; a crop is REJECTED and
                resampled if it would push any keypoint or trace vertex out of frame, because
                dropping a labelled point teaches the model to invent one.
  pad_geometry  systematic, not random: pre-apply the action stage's resize_with_pad letterbox so
                the reasoning LoRA trains on the same geometry the policy sees at run time. The
                reasoning stage otherwise squash-resizes frames to 224x224 via PaliGemmaProcessor
                while the action stage letterboxes them, a mismatch between the two stages.

NOT included: horizontal flip. The suffix is natural language with lateral references ("to the
right of the bowl"); flipping pixels without rewriting the prose yields self-contradictory
supervision, and rewriting it reliably is not worth the risk.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# a bracketed list of 2 (point) or 4 (box) numbers; the trace line's outer bracket contains
# brackets so it never matches, only its inner pairs do
_COORD = re.compile(r"\[\s*(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)+)\s*\]")
COORD_MAX = 1000.0


@dataclass
class AugConfig:
    photometric: bool = False
    geometric: bool = False
    pad_geometry: bool = False
    strength: float = 1.0          # scales every random magnitude; 0 = identity
    max_tries: int = 8             # crop resamples before giving up and passing through
    pad_size: int = 1000           # square canvas for pad_geometry, in the same units as coords

    @property
    def enabled(self) -> bool:
        return self.photometric or self.geometric or self.pad_geometry


def parse_coords(suffix: str) -> list[list[float]]:
    """Every bracketed numeric group, in order: 4-lists are boxes, 2-lists are points."""
    return [[float(t) for t in m.group(1).split(",")] for m in _COORD.finditer(suffix)]


def map_coords(suffix: str, fn) -> str:
    """Rewrite every coordinate group through fn(x, y) -> (x, y), preserving all other bytes.

    Serialization matches annotate_dataset_paligemma.build_assistant_response exactly ('.1f',
    ', ' separator), so an identity fn round-trips byte-for-byte.
    """
    def sub(m: re.Match) -> str:
        v = [float(t) for t in m.group(1).split(",")]
        out: list[float] = []
        for i in range(0, len(v), 2):
            x, y = fn(v[i], v[i + 1])
            out += [x, y]
        return "[" + ", ".join(f"{c:.1f}" for c in out) + "]"

    return _COORD.sub(sub, suffix)


def _points_and_boxes(suffix: str):
    pts, boxes = [], []
    for v in parse_coords(suffix):
        (boxes if len(v) == 4 else pts).append(v)
    return pts, boxes


# ── tier 1: photometric ──────────────────────────────────────────────────────
def apply_photometric(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    def jit(enh, lo, hi):
        return enh(img).enhance(float(rng.uniform(1 + (lo - 1) * s, 1 + (hi - 1) * s)))

    img = jit(ImageEnhance.Brightness, 0.7, 1.3)
    img = jit(ImageEnhance.Contrast, 0.7, 1.3)
    img = jit(ImageEnhance.Color, 0.6, 1.4)
    if rng.random() < 0.15 * s:
        img = img.convert("L").convert("RGB")
    if rng.random() < 0.25 * s:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.6) * s)))
    if rng.random() < 0.30 * s:                      # JPEG recompression artifacts
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(rng.integers(35, 80)))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    if rng.random() < 0.20 * s:                      # sensor noise
        a = np.asarray(img).astype(np.float32)
        a += rng.normal(0, float(rng.uniform(3, 10) * s), a.shape)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img


# ── tier 2: geometric ────────────────────────────────────────────────────────
def _sample_crop(rng: np.random.Generator, s: float) -> tuple[float, float, float, float]:
    """Crop window in 0-1000 coordinate space (x0, y0, x1, y1)."""
    scale = 1.0 - float(rng.uniform(0.0, 0.25 * s))       # keep >= 75% of each side at s=1
    w = h = None
    w = COORD_MAX * scale
    h = COORD_MAX * scale * float(rng.uniform(1 - 0.08 * s, 1 + 0.08 * s))
    h = min(h, COORD_MAX)
    x0 = float(rng.uniform(0, COORD_MAX - w))
    y0 = float(rng.uniform(0, COORD_MAX - h))
    return x0, y0, x0 + w, y0 + h


def apply_geometric(img: Image.Image, suffix: str, rng: np.random.Generator, s: float,
                    max_tries: int) -> tuple[Image.Image, str]:
    pts, boxes = _points_and_boxes(suffix)
    for _ in range(max_tries):
        x0, y0, x1, y1 = _sample_crop(rng, s)
        # every keypoint / trace vertex must survive the crop, with a small margin
        if any(not (x0 + 5 <= p[0] <= x1 - 5 and y0 + 5 <= p[1] <= y1 - 5) for p in pts):
            continue
        # every box must retain real area after clipping
        ok = True
        for b in boxes:
            bx0, by0 = max(b[0], x0), max(b[1], y0)
            bx1, by1 = min(b[2], x1), min(b[3], y1)
            if (bx1 - bx0) < 0.4 * (b[2] - b[0]) or (by1 - by0) < 0.4 * (b[3] - b[1]):
                ok = False
                break
        if not ok:
            continue

        sx, sy = COORD_MAX / (x1 - x0), COORD_MAX / (y1 - y0)

        def fn(x, y):
            return (
                float(np.clip((x - x0) * sx, 0.0, COORD_MAX)),
                float(np.clip((y - y0) * sy, 0.0, COORD_MAX)),
            )

        W, H = img.size
        img = img.crop((int(x0 / COORD_MAX * W), int(y0 / COORD_MAX * H),
                        int(x1 / COORD_MAX * W), int(y1 / COORD_MAX * H)))
        return img, map_coords(suffix, fn)
    return img, suffix          # no valid crop found: pass through unchanged


# ── tier 3: match the action stage's letterbox ───────────────────────────────
def resize_with_pad_params(w: int, h: int, size: int) -> tuple[float, float, float, float]:
    """(x_off, y_off, scaled_w, scaled_h) in `size` units — mirrors openpi's resize_with_pad
    convention."""
    ratio = max(w / size, h / size)
    rw, rh = w / ratio, h / ratio
    return (size - rw) / 2, (size - rh) / 2, rw, rh


def apply_pad_geometry(img: Image.Image, suffix: str, size: int = 224) -> tuple[Image.Image, str]:
    W, H = img.size
    x_off, y_off, rw, rh = resize_with_pad_params(W, H, size)
    out = Image.new("RGB", (size, size), (0, 0, 0))
    out.paste(img.resize((int(round(rw)), int(round(rh))), Image.BILINEAR),
              (int(round(x_off)), int(round(y_off))))

    # same mapping in normalized 0-1000 space
    fx0, fy0, fw, fh = resize_with_pad_params(W, H, COORD_MAX)

    def fn(x, y):
        return (fx0 + x / COORD_MAX * fw, fy0 + y / COORD_MAX * fh)

    return out, map_coords(suffix, fn)


def augment(img: Image.Image, suffix: str, cfg: AugConfig,
            rng: np.random.Generator) -> tuple[Image.Image, str]:
    """Order matters: crop in original geometry, then letterbox, then recolor."""
    if cfg.geometric:
        img, suffix = apply_geometric(img, suffix, rng, cfg.strength, cfg.max_tries)
    if cfg.pad_geometry:
        img, suffix = apply_pad_geometry(img, suffix)
    if cfg.photometric:
        img = apply_photometric(img, rng, cfg.strength)
    return img, suffix
