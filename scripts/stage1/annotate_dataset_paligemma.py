"""Annotate the MolmoSpaces pick-and-place datagen output for PaliGemma VLM
fine-tuning (PEFT/LoRA).

Source data layout (h5-format datagen):
  <DATAGEN_ROOT>/
    valid_trajectory_index.json   { house_K: { rel_h5: { traj_X: n_steps, ... } } }
    house_K/
      trajectories_batch_1_of_1.h5
        traj_X/  (one per trajectory)
          obs_scene  -> JSON bytes; has task_description, object_name, place_receptacle_name, ...
      episode_XXXXXXXX_<camera>_batch_1_of_1.mp4    # X = traj index zero-padded

Gemini annotates each frame (one image -> JSON with reasoning, objects, keypoints,
trace), which is rendered as a PaliGemma record: flat (image, prefix, suffix) JSONL where
the loss is masked on `prefix` and computed on `suffix`. `prefix` should
*include* an explicit `<image>` token so the data collator splices the image
embeddings in at the right place.

Output:
  <OUT_DIR>/
    images/houseK_epXXXX_fYYYYY.jpg ...
    train.jsonl
    val.jsonl
    config.json    # echo of args/prompt for traceability
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
from PIL import Image
import google.generativeai as genai


# ── Task schemas ─────────────────────────────────────────────────────────────
# The reasoning target is task-family specific. Pick-and-place keypoints describe a
# grasp-then-place motion; opening tasks describe grasping a handle and swinging it
# through an arc, so "target/lift/pre_place" have no referent there. Everything
# downstream is key-agnostic -- the JSON contract, parse_annotation() and
# build_assistant_response() iterate whatever keys come back -- so a schema only has
# to supply the keypoint block and the phrase naming where the trace goes.
KEYPOINT_SCHEMAS = {
    "pickplace": (
        '    "grasp_point":     [x, y],\n'
        '    "target_point":    [x, y],\n'
        '    "lift_point":      [x, y],\n'
        '    "pre_grasp_point": [x, y],\n'
        '    "pre_place_point": [x, y]\n'
    ),
    "open": (
        '    "handle_point":     [x, y],\n'
        '    "pre_grasp_point":  [x, y],\n'
        '    "grasp_point":      [x, y],\n'
        '    "pull_start_point": [x, y],\n'
        '    "pull_end_point":   [x, y]\n'
    ),
}
TRACE_DESTINATIONS = {
    "pickplace": "to the goal",
    "open": "along the opening arc",
}

DEFAULT_TASK_SCHEMA = "pickplace"


# ── PaliGemma-style prompts ──────────────────────────────────────────────────
# Gemini returns the JSON schema below; it is rendered into PaliGemma prefix/suffix records.
def system_prompt(schema: str = DEFAULT_TASK_SCHEMA) -> str:
    return (
        "You are a robot-manipulation vision assistant. "
        "Given a single RGB frame from a robot episode, output ONLY a JSON object "
        "with the structure shown below. "
        "All coordinates are integers on a 0-1000 scale with (0,0) at the top-left corner "
        "and (1000,1000) at the bottom-right, i.e. normalized to the image size. "
        "Do NOT use a 0-100 scale and do NOT use pixel coordinates.\n\n"
        "Required JSON structure:\n"
        "{\n"
        '  "reasoning": "<one or two sentences about relevant objects and positions>",\n'
        '  "objects": [\n'
        '    {"name": "<label>", "bbox": [x1, y1, x2, y2]}\n'
        "  ],\n"
        '  "keypoints": {\n'
        f"{KEYPOINT_SCHEMAS[schema]}"
        "  },\n"
        '  "trace": [[x, y], ...]'
        + "\n}\n\n"
        + "No markdown fences. No extra keys. No extra text outside the JSON object."
    )


# PaliGemma's processor convention: a single <image> token is consumed by the
# vision encoder and the rest of `prefix` is text. The model is trained with
# loss masked on the prefix, so the suffix is what it learns to produce.
def paligemma_task_template(schema: str = DEFAULT_TASK_SCHEMA) -> str:
    return (
        "<image>\n"
        "answer en Task: {task}\n"
        "Identify the relevant objects (with bounding boxes), mark the five "
        "semantic keypoints, and output a smooth {n_waypoints}-point trace from "
        f"the current end-effector position {TRACE_DESTINATIONS[schema]}."
    )


COORD_SCALE = 1000.0          # the convention every emitted dataset must be on
_SCALE_STATS = {"rescaled": 0, "already": 0, "degenerate": 0}


def _ann_coords(ann: dict) -> list[float]:
    vals: list[float] = []
    for obj in ann.get("objects", []) or []:
        b = obj.get("bbox") or []
        if len(b) == 4:
            vals += [float(c) for c in b]
    for v in (ann.get("keypoints") or {}).values():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            vals += [float(c) for c in v]
    for p in ann.get("trace") or []:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            vals += [float(c) for c in p]
    return vals


def normalize_annotation_scale(ann: dict) -> dict:
    """Force one coordinate convention (0-COORD_SCALE) regardless of what the model returned.

    The model may return either 0-100 or 0-1000 coordinates regardless of the requested scale.
    An annotation whose largest coordinate is <= 100 is treated as 0-100 and rescaled.
    """
    vals = _ann_coords(ann)
    if not vals:
        return ann
    hi = max(vals)
    if hi == 0.0:
        _SCALE_STATS["degenerate"] += 1     # all-zero: the model gave up; caller may want to drop
        return ann
    if hi > 100.0:
        _SCALE_STATS["already"] += 1
        return ann
    _SCALE_STATS["rescaled"] += 1

    k = COORD_SCALE / 100.0
    for obj in ann.get("objects", []) or []:
        b = obj.get("bbox") or []
        if len(b) == 4:
            obj["bbox"] = [float(c) * k for c in b]
    kps = ann.get("keypoints") or {}
    for name, v in list(kps.items()):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            kps[name] = [float(v[0]) * k, float(v[1]) * k]
    ann["trace"] = [
        [float(p[0]) * k, float(p[1]) * k]
        if isinstance(p, (list, tuple)) and len(p) == 2 else p
        for p in (ann.get("trace") or [])
    ]
    return ann


def scale_stats() -> dict:
    """Counts since import — worth printing at the end of an annotation run."""
    return dict(_SCALE_STATS)


def _fmt_points(pts) -> str:
    return "[" + ", ".join(
        f"[{p[0]:.1f}, {p[1]:.1f}]"
        for p in (pts or [])
        if isinstance(p, (list, tuple)) and len(p) == 2
    ) + "]"


def build_assistant_response(task: str, ann: dict) -> str:
    """Render the Gemini JSON into a tagged target string the VLM should learn."""
    ann = normalize_annotation_scale(ann)
    lines = [f"Task: {task}", "<reasoning>"]
    if (r := (ann.get("reasoning") or "").strip()):
        lines.append(r)
    obj_parts = []
    for obj in ann.get("objects", []):
        bbox = obj.get("bbox", [])
        if len(bbox) == 4:
            obj_parts.append(
                f"{obj.get('name', 'object')}: <bbox>[{', '.join(f'{c:.1f}' for c in bbox)}]</bbox>"
            )
    if obj_parts:
        lines.append("Relevant Objects:  " + "  ".join(obj_parts))
    lines += ["</reasoning>", "<keypoints>"]
    for k, v in (ann.get("keypoints") or {}).items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            lines.append(f"  {k}: [{v[0]:.1f}, {v[1]:.1f}]")
    lines += ["</keypoints>", "<trace>"]
    lines += [f"  {_fmt_points(ann.get('trace'))}", "</trace>"]
    return "\n".join(lines)


def build_paligemma_record(rel_image_path: str, prefix: str, suffix: str) -> dict:
    return {"image": rel_image_path, "prefix": prefix, "suffix": suffix}


# ── h5 / mp4 plumbing ────────────────────────────────────────────────────────
def discover_episodes(datagen_root: Path) -> list[dict]:
    """Yield {house, traj_idx, h5_path, video_path, language_task} per valid traj."""
    idx = json.loads((datagen_root / "valid_trajectory_index.json").read_text())
    out: list[dict] = []
    for house, files in idx.items():
        for rel_h5, trajs in files.items():
            h5_path = datagen_root / rel_h5
            if not h5_path.exists():
                print(f"  [skip] missing h5: {h5_path}")
                continue
            with h5py.File(h5_path, "r") as f:
                for traj_name in trajs:
                    g = f.get(traj_name)
                    if g is None:
                        continue
                    obs_scene = g["obs_scene"][()]
                    if isinstance(obs_scene, (bytes, bytearray)):
                        obs_scene = obs_scene.decode("utf-8", "replace")
                    try:
                        scene = json.loads(obs_scene)
                    except json.JSONDecodeError:
                        scene = {}
                    task_desc = scene.get("task_description") or scene.get("text") or ""
                    traj_idx = int(traj_name.split("_")[1])
                    out.append(
                        {
                            "house": house,
                            "traj_name": traj_name,
                            "traj_idx": traj_idx,
                            "h5_path": str(h5_path),
                            "house_dir": str(h5_path.parent),
                            "language_task": task_desc,
                        }
                    )
    return out


def find_video(house_dir: Path, traj_idx: int, camera: str) -> Path | None:
    pat = f"episode_{traj_idx:08d}_{camera}_batch_*.mp4"
    matches = sorted(house_dir.glob(pat))
    return matches[0] if matches else None


def extract_frames(video_path: str, sample_rate: int):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % sample_rate == 0:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            frames.append((idx, pil))
        idx += 1
    cap.release()
    return frames, total


# ── Gemini ───────────────────────────────────────────────────────────────────
FINISH_REASON_MAX_TOKENS = 2


def _hit_token_cap(response) -> bool:
    """True if the response was cut off by max_output_tokens.

    Thinking models count their reasoning against the same output budget, so a
    small budget can truncate the JSON and parse_annotation() would drop the frame.
    Detecting it lets the caller retry with a larger budget.
    """
    try:
        return int(response.candidates[0].finish_reason) == FINISH_REASON_MAX_TOKENS
    except Exception:
        return False


def call_gemini(model, prompt: str, pil_images, *, max_retries: int, delay: float,
                 max_output_tokens: int = 8192) -> str:
    """One Gemini call. `pil_images` can be a single PIL.Image or a list of them.

    Truncated (MAX_TOKENS) responses are retried with a doubled budget; other
    failures are retried with backoff.
    """
    if isinstance(pil_images, Image.Image):
        contents = [prompt, pil_images]
    else:
        contents = [prompt, *pil_images]
    budget = max_output_tokens
    for attempt in range(max_retries + 1):
        try:
            r = model.generate_content(
                contents,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0, max_output_tokens=budget
                ),
            )
            if _hit_token_cap(r) and attempt < max_retries:
                budget *= 2
                print(f"      Gemini hit max_output_tokens; retrying with {budget}")
                continue
            return r.text.strip()
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = delay * (attempt + 2)
            print(f"      Gemini error (attempt {attempt+1}): {exc}. retry in {wait:.0f}s")
            time.sleep(wait)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_annotation(raw: str) -> dict | None:
    text = _FENCE.sub("", raw).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def parse_batch_annotation(raw: str, expected_n: int) -> list[dict] | None:
    """Parse a JSON array of `expected_n` per-frame annotation objects."""
    text = _FENCE.sub("", raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "frames" in data and isinstance(data["frames"], list):
        data = data["frames"]
    if not isinstance(data, list) or len(data) != expected_n:
        return None
    if not all(isinstance(d, dict) for d in data):
        return None
    return data


def batch_system_prompt(schema: str = DEFAULT_TASK_SCHEMA) -> str:
    """Batch variant. Still carries {n} for .format(n=...) at the call site; the
    keypoint block contains no braces, so interpolating it here is format-safe."""
    return (
        "You are a robot-manipulation vision assistant. "
        "You will be given {n} ordered RGB frames from the SAME robot episode. "
        "Output ONLY a JSON array of exactly {n} objects, one per frame in the same "
        "order as the frames are provided. "
        "All coordinates are integers on a 0-1000 scale with (0,0) at the top-left corner "
        "and (1000,1000) at the bottom-right, i.e. normalized to the image size. "
        "Do NOT use a 0-100 scale and do NOT use pixel coordinates.\n\n"
        "Each object MUST have this structure:\n"
        "{{\n"
        '  "reasoning": "<one or two sentences about relevant objects and positions>",\n'
        '  "objects": [\n'
        '    {{"name": "<label>", "bbox": [x1, y1, x2, y2]}}\n'
        "  ],\n"
        '  "keypoints": {{\n'
        f"{KEYPOINT_SCHEMAS[schema]}"
        "  }},\n"
        '  "trace": [[x, y], ...]'
        + "\n}}\n\n"
        + "Return a top-level JSON array: [obj_frame_1, obj_frame_2, ...]. "
        + "No markdown fences. No extra keys. No extra text outside the JSON array."
    )


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datagen-root",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--camera",
        default="droid_shoulder_light_randomization",
        help="Which exo camera to pull frames from (matches mp4 filename).",
    )
    ap.add_argument("--frame-sample-rate", type=int, default=5)
    ap.add_argument("--num-waypoints", type=int, default=6)
    ap.add_argument("--train-split", type=float, default=0.9)
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--api-delay", type=float, default=1.0)
    ap.add_argument("--gemini-model", default="gemini-3.1-pro-preview")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument(
        "--frame-workers", type=int, default=8,
        help="Concurrent in-flight Gemini calls per episode (threads).",
    )
    ap.add_argument(
        "--frame-batch-size", type=int, default=1,
        help="Number of frames packed into ONE Gemini call. "
             ">1 sends multiple images per request and expects a JSON array back.",
    )
    ap.add_argument(
        "--batch-max-output-tokens", type=int, default=8192,
        help="Token budget for batched responses (scaled with batch size).",
    )
    ap.add_argument(
        "--episode-workers", type=int, default=1,
        help="Number of episodes to process in parallel. Frame-workers fan out under each.",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Skip frames already present in existing train/val jsonl checkpoints.",
    )
    ap.add_argument(
        "--task-schema", choices=sorted(KEYPOINT_SCHEMAS), default=DEFAULT_TASK_SCHEMA,
        help="Which keypoint vocabulary the reasoning supervises. 'pickplace' is the "
             "grasp/place schema; 'open' supervises handle/pull keypoints and "
             "an opening-arc trace.",
    )
    args = ap.parse_args()
    SYSTEM_PROMPT = system_prompt(args.task_schema)
    BATCH_SYSTEM_PROMPT = batch_system_prompt(args.task_schema)
    PALIGEMMA_TASK_TEMPLATE = paligemma_task_template(args.task_schema)
    TRACE_DEST = TRACE_DESTINATIONS[args.task_schema]

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY before running.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.gemini_model)

    out_dir: Path = args.out_dir
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl = out_dir / "train.jsonl"
    val_jsonl = out_dir / "val.jsonl"

    # Save run config so the dataset is self-describing.
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "system_prompt": SYSTEM_PROMPT,
                "batch_system_prompt": BATCH_SYSTEM_PROMPT,
                "paligemma_task_template": PALIGEMMA_TASK_TEMPLATE,
                "format": "paligemma_prefix_suffix_jsonl",
            },
            indent=2,
        )
    )

    print(f"Scanning datagen root: {args.datagen_root}")
    episodes = discover_episodes(args.datagen_root)
    print(f"Found {len(episodes)} valid trajectories")
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    all_records: list[dict] = []
    records_lock = threading.Lock()
    print_lock = threading.Lock()

    def _log(msg: str) -> None:
        with print_lock:
            print(msg, flush=True)

    # Resume: collect (image-path) already in existing jsonl checkpoints so we
    # don't re-call Gemini for finished frames. If --resume, also seed
    # all_records with the prior entries so a new shuffled split is written.
    done_images: set[str] = set()
    if args.resume:
        for ckpt in (train_jsonl, val_jsonl):
            if not ckpt.exists():
                continue
            for line in ckpt.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "image" in r:
                    done_images.add(r["image"])
                    all_records.append(r)
        print(f"[resume] loaded {len(done_images)} pre-existing records")

    def _record_from_ann(ep: dict, frame_idx: int, total: int, video_path: Path,
                         rel_path: str, ann: dict) -> dict:
        task = ep["language_task"]
        suffix = build_assistant_response(task, ann)
        prefix = PALIGEMMA_TASK_TEMPLATE.format(task=task, n_waypoints=args.num_waypoints)
        rec = build_paligemma_record(rel_path, prefix, suffix)
        rec["_meta"] = {
            "house": ep["house"],
            "traj_name": ep["traj_name"],
            "traj_idx": ep["traj_idx"],
            "frame_idx": frame_idx,
            "total_frames": total,
            "video_path": str(video_path),
            "h5_path": ep["h5_path"],
            "language_task": task,
        }
        return rec

    def _save_frame(ep: dict, frame_idx: int, pil_image: Image.Image) -> str | None:
        slug = f"{ep['house']}_traj{ep['traj_idx']:04d}"
        img_filename = f"{slug}__f{frame_idx:05d}.jpg"
        rel_path = f"images/{img_filename}"
        if args.resume and rel_path in done_images:
            return None
        try:
            pil_image.save(images_dir / img_filename, format="JPEG", quality=args.jpeg_quality)
        except Exception:
            _log(f"    {slug} f{frame_idx}: save failed")
            return None
        return rel_path

    def process_frame(ep: dict, frame_idx: int, pil_image: Image.Image,
                       total: int, video_path: Path) -> dict | None:
        task = ep["language_task"]
        slug = f"{ep['house']}_traj{ep['traj_idx']:04d}"
        rel_path = _save_frame(ep, frame_idx, pil_image)
        if rel_path is None:
            return None
        user_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Task: {task}\nFrame: {frame_idx} / {max(total-1,0)}\n\n"
            f"Identify the objects relevant to the task, provide their bounding boxes, "
            f"mark the five semantic keypoints, and generate a smooth {args.num_waypoints}-point "
            f"trace from the current end-effector position {TRACE_DEST}."
        )
        try:
            raw = call_gemini(
                model, user_text, pil_image,
                max_retries=args.max_retries, delay=args.api_delay,
            )
        except Exception:
            _log(f"    {slug} f{frame_idx:05d} GEMINI ERROR")
            return None
        ann = parse_annotation(raw)
        if ann is None:
            _log(f"    {slug} f{frame_idx:05d} PARSE FAILED")
            return None
        _log(
            f"    {slug} f{frame_idx:05d} ✓ "
            f"objs={len(ann.get('objects',[]))} "
            f"kp={len(ann.get('keypoints',{}))} "
            f"trace={len(ann.get('trace',[]))}"
        )
        return _record_from_ann(ep, frame_idx, total, video_path, rel_path, ann)

    def process_batch(ep: dict, batch: list[tuple[int, Image.Image]],
                       total: int, video_path: Path) -> list[dict]:
        """Send a list of (frame_idx, pil) as one Gemini call. Returns records (may be empty)."""
        task = ep["language_task"]
        slug = f"{ep['house']}_traj{ep['traj_idx']:04d}"
        saved: list[tuple[int, Image.Image, str]] = []
        for fi, pil in batch:
            rel = _save_frame(ep, fi, pil)
            if rel is not None:
                saved.append((fi, pil, rel))
        if not saved:
            return []
        if len(saved) == 1:  # nothing to gain from batching one frame
            rec = process_frame(ep, saved[0][0], saved[0][1], total, video_path)
            return [rec] if rec is not None else []

        n = len(saved)
        frame_ids_str = ", ".join(str(fi) for fi, _, _ in saved)
        user_text = (
            BATCH_SYSTEM_PROMPT.format(n=n)
            + f"\n\nTask: {task}\nFrame indices (in order): [{frame_ids_str}]  /  total {max(total-1,0)}"
            + f"\n\nFor EACH frame: identify the relevant objects with bounding boxes, "
            + f"mark the five semantic keypoints, and produce a smooth {args.num_waypoints}-point "
            + f"trace from the current end-effector position {TRACE_DEST}."
        )
        try:
            raw = call_gemini(
                model, user_text, [pil for _, pil, _ in saved],
                max_retries=args.max_retries, delay=args.api_delay,
                max_output_tokens=args.batch_max_output_tokens,
            )
        except Exception:
            _log(f"    {slug} batch[{frame_ids_str}] GEMINI ERROR → falling back per-frame")
            return [r for r in (
                process_frame(ep, fi, pil, total, video_path) for fi, pil, _ in saved
            ) if r is not None]

        anns = parse_batch_annotation(raw, expected_n=n)
        if anns is None:
            _log(f"    {slug} batch[{frame_ids_str}] PARSE FAILED → falling back per-frame")
            return [r for r in (
                process_frame(ep, fi, pil, total, video_path) for fi, pil, _ in saved
            ) if r is not None]

        out: list[dict] = []
        for (fi, _, rel), ann in zip(saved, anns):
            out.append(_record_from_ann(ep, fi, total, video_path, rel, ann))
        _log(
            f"    {slug} batch[{frame_ids_str}] ✓ "
            f"{n} frames; objs/kp/trace from Gemini batch"
        )
        return out

    def process_episode(ep_cnt: int, ep: dict) -> int:
        house = ep["house"]; traj_idx = ep["traj_idx"]; task = ep["language_task"]
        slug = f"{house}_traj{traj_idx:04d}"
        video_path = find_video(Path(ep["house_dir"]), traj_idx, args.camera)
        _log(f"\n[{ep_cnt+1}/{len(episodes)}] {slug}  \"{task}\"")
        if video_path is None:
            _log(f"  ✗ no mp4 for camera {args.camera!r} in {ep['house_dir']}")
            return 0
        try:
            frames, total = extract_frames(str(video_path), args.frame_sample_rate)
        except Exception:
            _log(f"  ✗ frame extraction failed:\n{traceback.format_exc()}")
            return 0
        _log(
            f"  video: {video_path.name}  "
            f"frames: {total} → {len(frames)} sampled (every {args.frame_sample_rate})"
        )

        ep_records: list[dict] = []
        bsz = max(1, args.frame_batch_size)
        if bsz <= 1:
            with ThreadPoolExecutor(max_workers=max(1, args.frame_workers)) as pool:
                futs = [
                    pool.submit(process_frame, ep, fi, pil, total, video_path)
                    for fi, pil in frames
                ]
                for fut in as_completed(futs):
                    rec = fut.result()
                    if rec is not None:
                        ep_records.append(rec)
        else:
            batches = [frames[i:i + bsz] for i in range(0, len(frames), bsz)]
            with ThreadPoolExecutor(max_workers=max(1, args.frame_workers)) as pool:
                futs = [
                    pool.submit(process_batch, ep, batch, total, video_path)
                    for batch in batches
                ]
                for fut in as_completed(futs):
                    ep_records.extend(fut.result())

        if ep_records:
            with records_lock:
                all_records.extend(ep_records)
                n = len(all_records)
                n_train = math.ceil(n * args.train_split)
                write_jsonl(all_records[:n_train], train_jsonl)
                write_jsonl(all_records[n_train:], val_jsonl)
            _log(f"  [checkpoint] +{len(ep_records)} records ({slug}); total={n}")
        return len(ep_records)

    if args.episode_workers <= 1:
        for ep_cnt, ep in enumerate(episodes):
            process_episode(ep_cnt, ep)
    else:
        with ThreadPoolExecutor(max_workers=args.episode_workers) as ep_pool:
            futs = [ep_pool.submit(process_episode, i, ep) for i, ep in enumerate(episodes)]
            for fut in as_completed(futs):
                fut.result()

    random.seed(42)
    random.shuffle(all_records)
    n_train = math.ceil(len(all_records) * args.train_split)
    write_jsonl(all_records[:n_train], train_jsonl)
    write_jsonl(all_records[n_train:], val_jsonl)

    print(f"\n{'='*60}")
    print(f"Total records : {len(all_records)}")
    print(f"Train         : {n_train}  →  {train_jsonl}")
    print(f"Val           : {len(all_records)-n_train}  →  {val_jsonl}")
    print(f"Images        : {images_dir}/")
    st = scale_stats()
    print(f"Coord scale   : {st['already']} already 0-1000, {st['rescaled']} rescaled "
          f"from 0-100, {st['degenerate']} all-zero (degenerate annotations)")


if __name__ == "__main__":
    main()
