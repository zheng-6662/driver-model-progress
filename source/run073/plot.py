from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
BASE = "B_all3"
TRUE = "B_all3_eye_true"
SHIFT = "B_all3_eye_shift_control"


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    subject = pd.read_csv(TABLES / "subject_metrics.csv")
    pivot = subject.pivot(index="subject", columns="model", values="curve_mae_deg")
    pivot = pivot.loc[:, [BASE, TRUE, SHIFT]].sort_index()
    gain_true = pivot[BASE] - pivot[TRUE]
    gain_shift = pivot[BASE] - pivot[SHIFT]

    x = np.arange(len(pivot))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, gain_true, width, label="True eye vs B_all3", color="#1976D2")
    ax.bar(x + width / 2, gain_shift, width, label="Shift control vs B_all3", color="#9E9E9E")
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x, pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Subject mean MAE improvement (deg; positive is better)")
    ax.set_title("Run73 subject-level effect of pre-anchor eye tracking")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_1_subject_eye_gain.png", dpi=180)
    plt.close(fig)

    groups = pd.read_csv(TABLES / "stratified_metrics.csv")
    labels = ["10_20", "20_30", "30_45", "45_70", "ge70"]
    models = [BASE, TRUE, SHIFT]
    colors = {BASE: "#37474F", TRUE: "#1976D2", SHIFT: "#9E9E9E"}
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for model in models:
        values = []
        for label in labels:
            row = groups.loc[
                groups["model"].eq(model) & groups["group"].eq(f"amplitude_bin::{label}"),
                "subject_macro_relative_mae_over_median_release_amplitude",
            ]
            values.append(float(row.iloc[0]))
        ax.plot(labels, values, marker="o", linewidth=2, label=model, color=colors[model])
    ax.set_ylabel("Subject-macro relative MAE")
    ax.set_xlabel("Release-amplitude bin (deg)")
    ax.set_title("Run73 amplitude-stratified eye-tracking comparison")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_2_amplitude_relative_mae.png", dpi=180)
    plt.close(fig)
    print("PLOT_PASS")


if __name__ == "__main__":
    main()
