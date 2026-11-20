"""Prépare les fenêtres de trajectoires cyclistes du SDD (train/val/test) dans data/processed/.

Usage :
    python scripts/prepare_sdd.py --sdd-root /chemin/vers/OpenTraj/datasets/SDD

Les annotations du SDD se récupèrent depuis le dépôt OpenTraj (voir README, section Données).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybemo.data import extract_windows, load_scales, load_video, split_videos  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sdd-root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    scales = load_scales(args.sdd_root / "H_SDD.txt")
    videos = []
    for ann in sorted(args.sdd_root.glob("*/video*/annotations.txt")):
        scene, vid = ann.parent.parent.name, int(ann.parent.name.replace("video", ""))
        if (scene, vid) in scales:
            videos.append((scene, vid))
        else:
            print(f"[ignorée] {scene}/video{vid} : pas d'échelle pixel->mètre disponible")

    splits = split_videos(videos)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, vids in splits.items():
        parts = []
        for scene, vid in vids:
            df = load_video(args.sdd_root / scene / f"video{vid}" / "annotations.txt", scales[(scene, vid)])
            w = extract_windows(df, "Biker")
            w["scene"] = np.array([scene] * len(w["traj"]))
            parts.append(w)
            print(f"{name:5s} {scene}/video{vid}: {len(w['traj'])} fenêtres")
        data = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
        np.savez_compressed(args.out / f"{name}.npz", **data)
        summary[name] = {"videos": [f"{s}/video{v}" for s, v in vids], "windows": int(len(data["traj"]))}
        print(f"==> {name}: {len(data['traj'])} fenêtres")
    (args.out / "splits.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
