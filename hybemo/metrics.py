"""Métriques d'évaluation : précision (ADE/FDE), collisions et réalisme physique."""
from __future__ import annotations

import numpy as np

from .data import DT, OBS_LEN


def ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Average Displacement Error : erreur moyenne (m) sur les 12 pas prédits, par fenêtre."""
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def fde(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Final Displacement Error : erreur (m) sur la dernière position prédite, par fenêtre."""
    return np.linalg.norm(pred[:, -1] - gt[:, -1], axis=-1)


def collision(pred: np.ndarray, neighbors_future: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """1 si la trajectoire prédite passe à moins de `threshold` m d'un voisin réel (au même instant).

    pred : (N, 12, 2) ; neighbors_future : (N, K, 12, 2) avec NaN pour les voisins absents.
    """
    d = np.linalg.norm(neighbors_future - pred[:, None], axis=-1)    # (N, K, 12)
    d = np.where(np.isnan(d), np.inf, d)
    return (d.min(axis=(1, 2)) < threshold).astype(float)


def max_acceleration(pred: np.ndarray, last_obs: np.ndarray, prev_obs: np.ndarray) -> np.ndarray:
    """Accélération maximale (m/s²) le long de la trajectoire prédite, en incluant la jonction
    avec les deux dernières positions observées."""
    full = np.concatenate([prev_obs[:, None], last_obs[:, None], pred], axis=1)
    vel = np.diff(full, axis=1) / DT
    acc = np.diff(vel, axis=1) / DT
    return np.linalg.norm(acc, axis=-1).max(axis=-1)


def summarize(pred: np.ndarray, traj: np.ndarray, neighbors: np.ndarray,
              coll_threshold: float = 0.5, acc_threshold: float = 4.0) -> dict[str, float]:
    """Toutes les métriques sur un ensemble de fenêtres."""
    gt = traj[:, OBS_LEN:]
    return {
        "ADE (m)": float(ade(pred, gt).mean()),
        "FDE (m)": float(fde(pred, gt).mean()),
        "Collisions (%)": float(100 * collision(pred, neighbors[:, :, OBS_LEN:], coll_threshold).mean()),
        f"Accél. > {acc_threshold:g} m/s² (%)": float(
            100 * (max_acceleration(pred, traj[:, OBS_LEN - 1], traj[:, OBS_LEN - 2]) > acc_threshold).mean()),
    }
