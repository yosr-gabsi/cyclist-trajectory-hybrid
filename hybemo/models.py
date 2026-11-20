"""Modèles de prédiction : baseline à vitesse constante et LSTM encodeur-décodeur."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import OBS_LEN, PRED_LEN


def constant_velocity(obs: np.ndarray, pred_len: int = PRED_LEN) -> np.ndarray:
    """Prolonge le dernier déplacement observé. obs : (N, 8, 2) -> (N, 12, 2)."""
    v = obs[:, -1] - obs[:, -2]
    steps = np.arange(1, pred_len + 1)[None, :, None]
    return obs[:, -1:, :] + steps * v[:, None, :]


class Seq2SeqLSTM(nn.Module):
    """LSTM encodeur-décodeur sur les déplacements.

    - Entrée : les 7 déplacements observés (invariance par translation).
    - L'encodeur résume l'historique dans son état caché.
    - Le décodeur génère les 12 déplacements futurs un par un, en réinjectant
      sa propre prédiction (génération autorégressive).
    """

    def __init__(self, hidden: int = 128, emb: int = 64, layers: int = 1, scale: float = 1.5):
        super().__init__()
        self.scale = scale  # normalisation des déplacements (~ ordre de grandeur en m par pas)
        self.embed = nn.Sequential(nn.Linear(2, emb), nn.ReLU())
        self.encoder = nn.LSTM(emb, hidden, layers, batch_first=True)
        self.decoder = nn.LSTMCell(emb, hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(self, obs: torch.Tensor, pred_len: int = PRED_LEN) -> torch.Tensor:
        """obs : (B, 8, 2) en mètres -> positions prédites (B, 12, 2) en mètres."""
        d = (obs[:, 1:] - obs[:, :-1]) / self.scale                 # (B, 7, 2)
        _, (h, c) = self.encoder(self.embed(d))
        h, c = h[-1], c[-1]
        step = d[:, -1]
        out = []
        for _ in range(pred_len):
            h, c = self.decoder(self.embed(step), (h, c))
            step = step + self.head(h)       # le réseau prédit une CORRECTION du dernier déplacement
            out.append(step)
        disp = torch.stack(out, dim=1) * self.scale                  # (B, 12, 2)
        return obs[:, -1:, :] + torch.cumsum(disp, dim=1)


def random_rotation(traj: torch.Tensor) -> torch.Tensor:
    """Augmentation de données : rotation aléatoire de chaque fenêtre autour de son dernier point observé."""
    theta = torch.rand(traj.shape[0], device=traj.device) * 2 * np.pi
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)   # (B, 2, 2)
    origin = traj[:, OBS_LEN - 1:OBS_LEN]
    return torch.einsum("bij,btj->bti", rot, traj - origin) + origin
