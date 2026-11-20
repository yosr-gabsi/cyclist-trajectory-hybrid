"""Entraîne le LSTM encodeur-décodeur sur les trajectoires cyclistes du SDD.

Usage :
    python scripts/train_lstm.py --epochs 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybemo.data import OBS_LEN  # noqa: E402
from hybemo.models import Seq2SeqLSTM, random_rotation  # noqa: E402


def evaluate_ade(model: Seq2SeqLSTM, traj: torch.Tensor, batch: int = 4096) -> float:
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(traj), batch):
            t = traj[i:i + batch]
            pred = model(t[:, :OBS_LEN])
            errs.append(torch.linalg.norm(pred - t[:, OBS_LEN:], dim=-1).mean(dim=1))
    return float(torch.cat(errs).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = torch.from_numpy(np.load(args.data / "train.npz")["traj"])
    val = torch.from_numpy(np.load(args.data / "val.npz")["traj"])

    model = Seq2SeqLSTM(hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    best, best_epoch, history = float("inf"), 0, []
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        perm = torch.randperm(len(train))
        for i in range(0, len(train), args.batch):
            t = random_rotation(train[perm[i:i + args.batch]])
            pred = model(t[:, :OBS_LEN])
            loss = torch.linalg.norm(pred - t[:, OBS_LEN:], dim=-1).mean()   # perte = ADE
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        val_ade = evaluate_ade(model, val)
        sched.step(val_ade)
        history.append({"epoch": epoch, "train_ade": float(np.mean(losses)), "val_ade": val_ade})
        print(f"epoch {epoch:3d}  train ADE {np.mean(losses):.3f} m  val ADE {val_ade:.3f} m  ({time.time() - t0:.0f}s)")
        if val_ade < best:
            best, best_epoch = val_ade, epoch
            torch.save({"state_dict": model.state_dict(), "hidden": args.hidden}, args.out / "lstm_sdd_biker.pt")
        elif epoch - best_epoch >= args.patience:
            print(f"Arrêt anticipé : pas d'amélioration depuis {args.patience} époques")
            break

    (args.out / "train_history.json").write_text(json.dumps(history, indent=2))
    print(f"Meilleur modèle : époque {best_epoch}, val ADE {best:.3f} m")


if __name__ == "__main__":
    main()
