"""Chargement du Stanford Drone Dataset (SDD) et extraction des fenêtres de trajectoires cyclistes.

Protocole (standard en prédiction de trajectoires) :
- échantillonnage toutes les 12 images (29,97 fps -> pas de 0,4 s, soit 2,5 Hz) ;
- 8 positions observées (3,2 s) -> 12 positions à prédire (4,8 s) ;
- positions = centre de la boîte englobante, converties de pixels en mètres
  avec le ratio fourni pour chaque vidéo (fichier H_SDD.txt d'OpenTraj).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FPS = 29.97
FRAME_STEP = 12                  # 12 images entre deux positions
DT = FRAME_STEP / FPS            # ~0,4 s
OBS_LEN = 8
PRED_LEN = 12
SEQ_LEN = OBS_LEN + PRED_LEN
NEIGHBOR_RADIUS = 10.0           # m : voisins retenus autour du cycliste au dernier instant observé
MAX_NEIGHBORS = 32

COLUMNS = ["track", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]


def load_scales(h_sdd_path: Path) -> dict[tuple[str, int], float]:
    """Ratio mètres/pixel moyen pour chaque (scène, vidéo)."""
    h = pd.read_csv(h_sdd_path, sep="\t")
    h["scene"] = h["File"].str.split("_").str[0]
    return h.groupby(["scene", "Nr"])["Ratio"].mean().to_dict()


def load_video(annotation_path: Path, ratio: float) -> pd.DataFrame:
    """Lit un fichier annotations.txt et renvoie les positions (en mètres) échantillonnées à 2,5 Hz."""
    df = pd.read_csv(annotation_path, sep=" ", header=None, names=COLUMNS)
    df = df[(df["lost"] == 0) & (df["frame"] % FRAME_STEP == 0)].copy()
    df["x"] = (df["xmin"] + df["xmax"]) / 2 * ratio
    df["y"] = (df["ymin"] + df["ymax"]) / 2 * ratio
    df["label"] = df["label"].str.strip('"')
    return df[["track", "frame", "x", "y", "label"]].sort_values(["track", "frame"])


def _consecutive_runs(frames: np.ndarray) -> list[slice]:
    """Découpe une piste en segments sans trou temporel."""
    breaks = np.where(np.diff(frames) != FRAME_STEP)[0] + 1
    bounds = np.concatenate([[0], breaks, [len(frames)]])
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def extract_windows(df: pd.DataFrame, target_label: str = "Biker") -> dict[str, np.ndarray]:
    """Extrait toutes les fenêtres de 20 positions des agents `target_label`, avec leurs voisins.

    Renvoie :
      traj      (N, 20, 2)     trajectoire du cycliste (8 observées + 12 futures)
      neighbors (N, K, 20, 2)  positions des autres usagers aux mêmes instants (NaN si absents)
      nbr_label (N, K)         classe des voisins (0 = piéton, 1 = cycliste, 2 = autre, -1 = vide)
    """
    label_code = {"Pedestrian": 0, "Biker": 1}
    # Table frame -> positions de tous les agents, pour retrouver les voisins rapidement
    by_frame = {f: g for f, g in df.groupby("frame")}
    track_label = df.groupby("track")["label"].agg(lambda s: s.mode().iloc[0])
    # Table piste -> {frame: (x, y)} pour lire la position d'un voisin en O(1)
    track_pos = {t: dict(zip(g["frame"].to_numpy(), g[["x", "y"]].to_numpy()))
                 for t, g in df.groupby("track")}

    trajs, nbrs, nbr_labels = [], [], []
    for track, g in df[df["track"].isin(track_label[track_label == target_label].index)].groupby("track"):
        frames = g["frame"].to_numpy()
        xy = g[["x", "y"]].to_numpy()
        for run in _consecutive_runs(frames):
            f_run, xy_run = frames[run], xy[run]
            for s in range(0, len(f_run) - SEQ_LEN + 1):
                win_frames = f_run[s:s + SEQ_LEN]
                win_xy = xy_run[s:s + SEQ_LEN]
                # Voisins présents au dernier instant observé, dans un rayon de NEIGHBOR_RADIUS
                t_obs = win_frames[OBS_LEN - 1]
                at_obs = by_frame[t_obs]
                at_obs = at_obs[at_obs["track"] != track]
                d = np.hypot(at_obs["x"] - win_xy[OBS_LEN - 1, 0], at_obs["y"] - win_xy[OBS_LEN - 1, 1])
                ids = at_obs["track"].to_numpy()[np.argsort(d.to_numpy())]
                ids = ids[np.sort(d.to_numpy()) < NEIGHBOR_RADIUS][:MAX_NEIGHBORS]

                nb = np.full((MAX_NEIGHBORS, SEQ_LEN, 2), np.nan, dtype=np.float32)
                nl = np.full(MAX_NEIGHBORS, -1, dtype=np.int8)
                for k, nid in enumerate(ids):
                    nl[k] = label_code.get(track_label[nid], 2)
                    pos = track_pos[nid]
                    for t, f in enumerate(win_frames):
                        p = pos.get(f)
                        if p is not None:
                            nb[k, t] = p
                trajs.append(win_xy.astype(np.float32))
                nbrs.append(nb)
                nbr_labels.append(nl)

    if not trajs:
        return {"traj": np.zeros((0, SEQ_LEN, 2), np.float32),
                "neighbors": np.zeros((0, MAX_NEIGHBORS, SEQ_LEN, 2), np.float32),
                "nbr_label": np.zeros((0, MAX_NEIGHBORS), np.int8)}
    return {"traj": np.stack(trajs), "neighbors": np.stack(nbrs), "nbr_label": np.stack(nbr_labels)}


def split_videos(videos: list[tuple[str, int]], seed: int = 0) -> dict[str, list[tuple[str, int]]]:
    """Découpage train/val/test PAR VIDÉO (aucune vidéo partagée entre deux ensembles),
    stratifié par scène : ~20 % des vidéos de chaque scène en test, ~10 % en validation."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    scenes = sorted({s for s, _ in videos})
    for scene in scenes:
        vids = sorted(v for v in videos if v[0] == scene)
        rng.shuffle(vids)
        n = len(vids)
        n_test = max(1, round(0.2 * n)) if n >= 3 else 0
        n_val = max(1, round(0.1 * n)) if n >= 4 else 0
        out["test"] += vids[:n_test]
        out["val"] += vids[n_test:n_test + n_val]
        out["train"] += vids[n_test + n_val:]
    return out
