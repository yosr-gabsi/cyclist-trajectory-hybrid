"""Simulation SFM d'un espace partagé piétons + cyclistes, exportée en GIF animé.

Scène : une allée de 30 m x 10 m. Des piétons la traversent dans les deux sens (~1,3 m/s),
des cyclistes la parcourent plus vite (~4 m/s) avec un rayon plus grand et doivent les contourner.
Les murs de l'allée exercent aussi une répulsion.

Usage :
    python scripts/simulate_sfm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybemo.sfm import SFMParams, step  # noqa: E402

LENGTH, WIDTH = 30.0, 10.0
DT = 0.05
DURATION = 24.0


def spawn(rng, n_ped=24, n_bike=6):
    """Positions, vitesses désirées et rayons initiaux."""
    pos, v0, radius, is_bike = [], [], [], []
    for i in range(n_ped):
        direction = 1 if i % 2 == 0 else -1
        x = rng.uniform(0, LENGTH)
        pos.append([x, rng.uniform(1, WIDTH - 1)])
        v0.append([direction * rng.normal(1.3, 0.15), 0.0])
        radius.append(0.3)
        is_bike.append(False)
    for i in range(n_bike):
        direction = 1 if i % 2 == 0 else -1
        pos.append([rng.uniform(0, LENGTH), rng.uniform(2, WIDTH - 2)])
        v0.append([direction * rng.normal(4.0, 0.4), 0.0])
        radius.append(0.6)
        is_bike.append(True)
    return np.array(pos), np.array(v0), np.array(radius), np.array(is_bike)


def wall_force(pos, p):
    """Répulsion des deux murs (y = 0 et y = WIDTH)."""
    f = np.zeros_like(pos)
    f[:, 1] += p.A * np.exp((0.3 - pos[:, 1]) / p.B)
    f[:, 1] -= p.A * np.exp((0.3 - (WIDTH - pos[:, 1])) / p.B)
    return f


def main() -> None:
    rng = np.random.default_rng(7)
    pos, v0, radius, is_bike = spawn(rng)
    vel = v0.copy()
    p = SFMParams(A=3.0, B=0.4, tau=0.5, lam=0.35, radius_self=0.0, radius_other=0.0)
    n = len(pos)
    frames = []
    for _ in range(int(DURATION / DT)):
        # chaque agent voit tous les autres ; rayon d'interaction = somme des deux rayons
        others = np.repeat(pos[None], n, axis=0)
        others[np.arange(n), np.arange(n)] = np.nan
        r_pair = radius[:, None] + radius[None, :]
        pos_new, vel = step(pos, vel, v0, others, p, DT, radius_other=r_pair)
        vel = vel + wall_force(pos, p) * DT
        pos = pos_new
        pos[:, 1] = np.clip(pos[:, 1], 0.3, WIDTH - 0.3)
        pos[:, 0] = np.mod(pos[:, 0], LENGTH)       # les agents qui sortent réapparaissent de l'autre côté
        frames.append(pos.copy())

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.set_xlim(0, LENGTH); ax.set_ylim(0, WIDTH); ax.set_aspect("equal")
    ax.axhline(0, c="k", lw=2); ax.axhline(WIDTH, c="k", lw=2)
    ax.set_xticks([]); ax.set_yticks([])
    peds = ax.scatter([], [], s=40, c="#1f77b4", label="Piétons")
    bikes = ax.scatter([], [], s=150, c="#d62728", marker="s", label="Cyclistes")
    ax.legend(loc="upper right", fontsize=8, framealpha=.9)
    title = ax.set_title("")

    def update(k):
        f = frames[k]
        peds.set_offsets(f[~is_bike]); bikes.set_offsets(f[is_bike])
        title.set_text(f"Social Force Model – espace partagé piétons / cyclistes   t = {k * DT:4.1f} s")
        return peds, bikes, title

    stride = 2   # 1 image GIF sur 2 pas de simulation -> 10 images/s
    out = Path("results/figures/sfm_shared_space.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=range(0, len(frames), stride))
    anim.save(out, writer=PillowWriter(fps=int(1 / (DT * stride))), dpi=80)
    print(f"GIF écrit : {out}")


if __name__ == "__main__":
    main()
