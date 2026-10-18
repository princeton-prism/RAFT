#!/usr/bin/env python
"""Split an MlSpaces action dataset into train / held-out-val roots for a TRUE validation loss.

src/agent/train_action.py's VAL_BATCHES metric is, by default, computed on batches drawn from the
training loader -- comparable across steps, but blind to overfitting. Point VAL_DATA_ROOT at the
val root this script writes and the loss becomes a real generalization estimate.

Both roots are symlink farms over the SAME house dirs; what differs is `valid_trajectory_index.json`,
which restricts each root to its own trajectories. The val root receives a COPY of the train root's
`aggregated_stats.json` so both normalize identically -- computing separate stats on the val split
would leak the val distribution into its own normalizer and make the two losses incomparable.

Trajectories are held out stratified by scene (the symlink target's own house_<K>), so every scene
contributes to val in proportion.

Usage:
    python scripts/data/build_open_holdout_split.py --src <action_train dir> --val-frac 0.10
    # then: calculate_stats.py on the train root, and copy it to the val root (--copy-stats)
"""
import argparse, json, os, random, re, shutil


def scene_of(src, house):
    return int(re.search(r"house_(\d+)$", os.path.realpath(os.path.join(src, house))).group(1))


def write_root(base, name, idx, recs, src):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    for h in {r[0] for r in recs}:
        link = os.path.join(d, h)
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(os.path.join(src, h)), link)
    sub = {}
    for h, f, t in recs:
        sub.setdefault(h, {}).setdefault(f, {})[t] = idx[h][f][t]
    json.dump(sub, open(os.path.join(d, "valid_trajectory_index.json"), "w"), indent=1)
    n = sum(len(ts) for fs in sub.values() for ts in fs.values())
    print(f"  {name}: {len(sub)} houses, {n} trajectories -> {d}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="preprocessed action dataset root")
    ap.add_argument("--train-name", default=None)
    ap.add_argument("--val-name", default=None)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-stats", action="store_true",
                    help="copy the TRAIN root's aggregated_stats.json to the val root "
                         "(run after calculate_stats.py on the train root)")
    args = ap.parse_args()

    base = os.path.dirname(os.path.normpath(args.src))
    idx = json.load(open(os.path.join(args.src, "valid_trajectory_index.json")))
    flat = [(h, f, t) for h, fs in idx.items() for f, ts in fs.items() for t in ts]

    by_scene = {}
    for rec in flat:
        by_scene.setdefault(scene_of(args.src, rec[0]), []).append(rec)
    print(f"total {len(flat)} trajectories over {len(by_scene)} scenes "
          f"{{{', '.join(f'{k}:{len(v)}' for k, v in sorted(by_scene.items()))}}}")

    rng = random.Random(args.seed)
    val = []
    for s in sorted(by_scene):
        v = by_scene[s][:]
        rng.shuffle(v)
        val += v[: max(1, round(args.val_frac * len(v)))]
    vset = set(val)
    train = [r for r in flat if r not in vset]

    tn = args.train_name or f"{os.path.basename(os.path.normpath(args.src))}_h{len(train)}"
    vn = args.val_name or f"{os.path.basename(os.path.normpath(args.src))}_val{len(val)}"
    tr = write_root(base, tn, idx, train, args.src)
    va = write_root(base, vn, idx, val, args.src)
    print(f"  val per scene: "
          f"{ {s: sum(1 for r in val if scene_of(args.src, r[0]) == s) for s in sorted(by_scene)} }")

    if args.copy_stats:
        s = os.path.join(tr, "aggregated_stats.json")
        if not os.path.exists(s):
            raise SystemExit(f"run calculate_stats.py on {tr} first")
        shutil.copy2(s, os.path.join(va, "aggregated_stats.json"))
        print(f"  copied train normalizer -> {vn}/aggregated_stats.json")
    else:
        print(f"\nnext: calculate_stats.py on {tr}, then re-run with --copy-stats")


if __name__ == "__main__":
    main()
