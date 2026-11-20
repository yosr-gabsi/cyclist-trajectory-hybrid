"""Calibre le SFM et les stratégies hybrides sur la VALIDATION, puis compare toutes les approches sur le TEST.

Usage :
    python scripts/evaluate.py

Règle de calibrage (fixée avant de regarder le test) :
    parmi les réglages dont le taux de collisions sur la validation ne dépasse pas celui des
    trajectoires réelles, on garde celui qui a le plus petit ADE.
Sorties : results/metrics_test.json, results/calibration_val.json, figures dans results/figures/.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybemo.data import OBS_LEN  # noqa: E402
from hybemo.hybrid import hybrid_fusion, hybrid_selection  # noqa: E402
from hybemo.metrics import summarize  # noqa: E402
from hybemo.models import Seq2SeqLSTM, constant_velocity  # noqa: E402
from hybemo.sfm import SFMParams, sfm_predict  # noqa: E402

RES = Path("results")
FIG = RES / "figures"


def load(split: str, n: int | None = None, seed: int = 0):
    d = np.load(f"data/processed/{split}.npz")
    idx = np.arange(len(d["traj"]))
    if n is not None and n < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    return d["traj"][idx], d["neighbors"][idx]


def lstm_predict(obs: np.ndarray) -> np.ndarray:
    ck = torch.load(RES / "lstm_sdd_biker.pt")
    m = Seq2SeqLSTM(hidden=ck["hidden"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(obs)).numpy()


def calibrate(traj, nbr) -> dict:
    obs, nobs = traj[:, :OBS_LEN], nbr[:, :, :OBS_LEN]
    lstm = lstm_predict(obs)
    gt_coll = summarize(traj[:, OBS_LEN:], traj, nbr)["Collisions (%)"]
    grid = list(itertools.product([0.5, 1.0, 2.0, 3.0, 5.0], [0.2, 0.3, 0.5], [0.3, 0.5, 1.0]))
    rows = []
    for A, B, tau in grid:
        p = SFMParams(A=A, B=B, tau=tau)
        for name, pred in [("SFM", sfm_predict(obs, nobs, p)), ("Hybride fusion", hybrid_fusion(obs, nobs, lstm, p))]:
            m = summarize(pred, traj, nbr)
            rows.append({"method": name, "A": A, "B": B, "tau": tau, **m})
        print(f"A={A} B={B} tau={tau} : {rows[-2]['ADE (m)']:.3f} / {rows[-1]['ADE (m)']:.3f} m")

    def best(method):
        ok = [r for r in rows if r["method"] == method and r["Collisions (%)"] <= gt_coll]
        return min(ok or [r for r in rows if r["method"] == method], key=lambda r: r["ADE (m)"])

    best_sfm, best_fus = best("SFM"), best("Hybride fusion")
    p_fus = SFMParams(A=best_fus["A"], B=best_fus["B"], tau=best_fus["tau"])
    fusion = hybrid_fusion(obs, nobs, lstm, p_fus)
    sel_rows = []
    for d in [0.5, 1.0, 1.5, 2.0, 3.0]:
        pred, used = hybrid_selection(obs, nobs, lstm, fusion, d)
        sel_rows.append({"d_switch": d, "SFM activé (%)": float(100 * used.mean()), **summarize(pred, traj, nbr)})
    ok = [r for r in sel_rows if r["Collisions (%)"] <= gt_coll]
    best_sel = min(ok or sel_rows, key=lambda r: r["ADE (m)"])
    return {"gt_collisions_val (%)": gt_coll, "grid": rows, "selection": sel_rows,
            "chosen": {"SFM": best_sfm, "Hybride fusion": best_fus, "Hybride sélection": best_sel}}


def plot_pareto(cal: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method, color in [("SFM", "#7f7f7f"), ("Hybride fusion", "#1f77b4")]:
        r = [x for x in cal["grid"] if x["method"] == method]
        ax.scatter([x["Collisions (%)"] for x in r], [x["ADE (m)"] for x in r], s=18, c=color, alpha=.7, label=method)
    s = cal["selection"]
    ax.plot([x["Collisions (%)"] for x in s], [x["ADE (m)"] for x in s], "o-", c="#d62728", ms=4, label="Hybride sélection (d_switch)")
    ax.axvline(cal["gt_collisions_val (%)"], ls="--", c="k", lw=1)
    ax.text(cal["gt_collisions_val (%)"], ax.get_ylim()[1], " trajectoires réelles", va="top", fontsize=8)
    ax.set_xlabel("Collisions (% des fenêtres, seuil 0,5 m)")
    ax.set_ylabel("ADE (m)")
    ax.set_title("Compromis précision / interactions (validation)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "pareto_val.png", dpi=150)


def plot_examples(traj, nbr, preds: dict, n: int = 4, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    # exemples où au moins un voisin est proche, pour voir l'effet des interactions
    dist = np.linalg.norm(nbr[:, :, OBS_LEN - 1] - traj[:, None, OBS_LEN - 1], axis=-1)
    close = np.where(np.where(np.isnan(dist), np.inf, dist).min(axis=1) < 3)[0]
    idx = rng.choice(close, n, replace=False)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    colors = {"Vitesse constante": "#7f7f7f", "LSTM": "#2ca02c", "SFM": "#9467bd", "Hybride fusion": "#1f77b4"}
    for ax, i in zip(axes, idx):
        for k in range(nbr.shape[1]):
            if not np.isnan(nbr[i, k, OBS_LEN - 1, 0]):
                ax.plot(nbr[i, k, :, 0], nbr[i, k, :, 1], c="#dddddd", lw=1)
        ax.plot(traj[i, :OBS_LEN, 0], traj[i, :OBS_LEN, 1], "k.-", label="Observé")
        ax.plot(traj[i, OBS_LEN - 1:, 0], traj[i, OBS_LEN - 1:, 1], "k--", label="Réel")
        for name, c in colors.items():
            p = np.concatenate([traj[i, OBS_LEN - 1:OBS_LEN], preds[name][i]])
            ax.plot(p[:, 0], p[:, 1], c=c, lw=1.6, label=name)
        ax.set_aspect("equal", "datalim")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Exemples de prédictions (test) – voisins en gris")
    fig.tight_layout()
    fig.savefig(FIG / "examples_test.png", dpi=130)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    val_traj, val_nbr = load("val", n=4000)
    cal = calibrate(val_traj, val_nbr)
    (RES / "calibration_val.json").write_text(json.dumps(cal, indent=2, ensure_ascii=False))
    plot_pareto(cal)

    traj, nbr = load("test")
    obs, nobs = traj[:, :OBS_LEN], nbr[:, :, :OBS_LEN]
    c = cal["chosen"]
    p_sfm = SFMParams(A=c["SFM"]["A"], B=c["SFM"]["B"], tau=c["SFM"]["tau"])
    p_fus = SFMParams(A=c["Hybride fusion"]["A"], B=c["Hybride fusion"]["B"], tau=c["Hybride fusion"]["tau"])
    lstm = lstm_predict(obs)
    fusion = hybrid_fusion(obs, nobs, lstm, p_fus)
    selection, used = hybrid_selection(obs, nobs, lstm, fusion, c["Hybride sélection"]["d_switch"])
    preds = {
        "Vitesse constante": constant_velocity(obs),
        "LSTM": lstm,
        "SFM": sfm_predict(obs, nobs, p_sfm),
        "Hybride fusion": fusion,
        "Hybride sélection": selection,
    }
    table = {"Trajectoires réelles": summarize(traj[:, OBS_LEN:], traj, nbr)}
    table |= {k: summarize(v, traj, nbr) for k, v in preds.items()}
    out = {"n_windows_test": int(len(traj)), "sfm_activation_selection (%)": float(100 * used.mean()),
           "params": {"SFM": vars(p_sfm), "Hybride fusion": vars(p_fus),
                      "d_switch": c["Hybride sélection"]["d_switch"]},
           "results": table}
    (RES / "metrics_test.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    plot_examples(traj, nbr, preds)

    print(f"\nTEST ({len(traj)} fenêtres)")
    cols = list(next(iter(table.values())).keys())
    print(f"{'Méthode':22s}" + "".join(f"{c:>24s}" for c in cols))
    for k, m in table.items():
        print(f"{k:22s}" + "".join(f"{m[c]:24.3f}" for c in cols))


if __name__ == "__main__":
    main()
