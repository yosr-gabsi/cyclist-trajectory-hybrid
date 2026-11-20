"""Stratégies hybrides combinant le LSTM (précision sur données réelles) et le SFM (interactions).

1. Fusion : le LSTM fournit l'intention (vitesse désirée à chaque pas), le SFM ajoute
   les forces de répulsion des voisins et intègre la dynamique.
2. Sélection : on garde la prédiction du LSTM, sauf lorsqu'elle passe trop près d'un voisin
   (distance < d_switch) ; dans ce cas on bascule sur la fusion.
"""
from __future__ import annotations

import numpy as np

from .sfm import SFMParams, extrapolate_neighbors, sfm_predict


def hybrid_fusion(obs: np.ndarray, neighbors_obs: np.ndarray, lstm_pred: np.ndarray,
                  p: SFMParams) -> np.ndarray:
    return sfm_predict(obs, neighbors_obs, p, desired_path=lstm_pred)


def min_distance_to_neighbors(path: np.ndarray, neighbors_obs: np.ndarray) -> np.ndarray:
    """Distance minimale entre un chemin prédit et les voisins extrapolés à vitesse constante."""
    nbr = extrapolate_neighbors(neighbors_obs)                          # (N, K, 12, 2)
    d = np.linalg.norm(nbr - path[:, None], axis=-1)
    return np.where(np.isnan(d), np.inf, d).min(axis=(1, 2))


def hybrid_selection(obs: np.ndarray, neighbors_obs: np.ndarray, lstm_pred: np.ndarray,
                     fusion_pred: np.ndarray, d_switch: float) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie la prédiction choisie et le masque des fenêtres où le SFM a été activé."""
    use_sfm = min_distance_to_neighbors(lstm_pred, neighbors_obs) < d_switch
    return np.where(use_sfm[:, None, None], fusion_pred, lstm_pred), use_sfm
