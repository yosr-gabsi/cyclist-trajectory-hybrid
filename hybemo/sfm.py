"""Social Force Model (Helbing & Molnár, 1995), version vectorisée NumPy.

Chaque agent i subit :
  - une force motrice qui le ramène vers sa vitesse désirée :  (v0_i * e_i - v_i) / tau
  - une force de répulsion de chaque autre agent j :
        A * exp((r_ij - d_ij) / B) * n_ij * w(phi)
    où d_ij est la distance, r_ij la somme des rayons, n_ij la direction j -> i,
    et w(phi) = lam + (1 - lam) * (1 + cos phi) / 2 un facteur d'anisotropie
    (un agent réagit plus à ce qui est devant lui qu'à ce qui est derrière).

Ce module sert à la fois :
  - à simuler une scène complète (scripts/simulate_sfm.py) ;
  - à prédire la trajectoire d'un cycliste du SDD, voisins extrapolés à vitesse constante
    (sfm_predict), avec une vitesse désirée venant soit de l'historique, soit d'un LSTM (hybride).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DT, OBS_LEN, PRED_LEN


@dataclass
class SFMParams:
    tau: float = 0.5          # s   : temps de relaxation vers la vitesse désirée
    A: float = 2.0            # m/s²: intensité de la répulsion
    B: float = 0.3            # m   : portée de la répulsion
    lam: float = 0.35         # -   : anisotropie (0 = ne voit que devant, 1 = isotrope)
    radius_self: float = 0.6  # m   : rayon d'un cycliste
    radius_other: float = 0.3 # m   : rayon moyen des autres usagers
    max_speed_factor: float = 1.3  # vitesse max = facteur x vitesse désirée


def repulsion(pos: np.ndarray, vel: np.ndarray, others: np.ndarray, p: SFMParams,
              radius_other: np.ndarray | float | None = None) -> np.ndarray:
    """Force de répulsion totale exercée par `others` sur chaque agent.

    pos, vel : (N, 2) ; others : (N, K, 2) avec NaN pour les voisins absents.
    """
    r_other = p.radius_other if radius_other is None else radius_other
    diff = pos[:, None, :] - others                                   # (N, K, 2)
    dist = np.linalg.norm(diff, axis=-1)                              # (N, K)
    valid = ~np.isnan(dist) & (dist > 1e-6)
    dist_safe = np.where(valid, dist, 1.0)
    n = diff / dist_safe[..., None]
    mag = p.A * np.exp((p.radius_self + r_other - dist_safe) / p.B)

    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    heading = np.where(speed > 1e-6, vel / np.maximum(speed, 1e-6), 0.0)
    cos_phi = -(n * heading[:, None, :]).sum(-1)                      # autre agent devant -> cos_phi > 0
    w = p.lam + (1 - p.lam) * (1 + cos_phi) / 2

    f = (mag * w)[..., None] * n
    f = np.where(valid[..., None], f, 0.0)
    return f.sum(axis=1)


def step(pos: np.ndarray, vel: np.ndarray, desired_vel: np.ndarray, others: np.ndarray,
         p: SFMParams, dt: float, radius_other=None) -> tuple[np.ndarray, np.ndarray]:
    """Un pas d'intégration (Euler semi-implicite)."""
    force = (desired_vel - vel) / p.tau + repulsion(pos, vel, others, p, radius_other)
    vel = vel + force * dt
    v_max = p.max_speed_factor * np.linalg.norm(desired_vel, axis=-1, keepdims=True)
    speed = np.linalg.norm(vel, axis=-1, keepdims=True)
    vel = np.where(speed > v_max, vel / np.maximum(speed, 1e-9) * v_max, vel)
    return pos + vel * dt, vel


def extrapolate_neighbors(neighbors_obs: np.ndarray, pred_len: int = PRED_LEN) -> np.ndarray:
    """Futur des voisins supposé à vitesse constante (on ne connaît pas leur vrai futur).

    neighbors_obs : (N, K, 8, 2) -> (N, K, pred_len, 2) ; NaN si le voisin n'est pas visible au dernier instant.
    """
    last, prev = neighbors_obs[:, :, -1], neighbors_obs[:, :, -2]
    v = np.where(np.isnan(prev), 0.0, last - prev)
    steps = np.arange(1, pred_len + 1)[None, None, :, None]
    return last[:, :, None, :] + steps * v[:, :, None, :]


def sfm_predict(obs: np.ndarray, neighbors_obs: np.ndarray, p: SFMParams,
                desired_path: np.ndarray | None = None, substeps: int = 4) -> np.ndarray:
    """Prédit 12 positions par simulation SFM.

    obs : (N, 8, 2) ; neighbors_obs : (N, K, 8, 2).
    desired_path : None -> vitesse désirée = dernière vitesse observée (SFM « pur ») ;
                   (N, 12, 2) -> vitesse désirée à chaque pas = celle du chemin fourni (hybride LSTM + SFM).
    """
    pos = obs[:, -1].astype(np.float64)
    vel = (obs[:, -1] - obs[:, -2]) / DT
    nbr_future = extrapolate_neighbors(neighbors_obs)
    nbr_now = neighbors_obs[:, :, -1]
    if desired_path is not None:
        anchors = np.concatenate([obs[:, -1:], desired_path], axis=1)
        desired_seq = np.diff(anchors, axis=1) / DT                    # (N, 12, 2)
    dt = DT / substeps
    out = np.zeros((len(obs), PRED_LEN, 2))
    for k in range(PRED_LEN):
        desired = desired_seq[:, k] if desired_path is not None else (obs[:, -1] - obs[:, -2]) / DT
        for s in range(1, substeps + 1):
            a = s / substeps  # interpolation linéaire des voisins entre deux instants
            prev_n = nbr_now if k == 0 else nbr_future[:, :, k - 1]
            others = (1 - a) * prev_n + a * nbr_future[:, :, k]
            pos, vel = step(pos, vel, desired, others, p, dt)
        out[:, k] = pos
    return out.astype(np.float32)
