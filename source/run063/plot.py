from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "B_all3": "#4C78A8",
    "R_lowrank_residual": "#54A24B",
    "G_soft_expert_gate": "#F58518",
}
ORDER = ["B_all3", "R_lowrank_residual", "G_soft_expert_gate"]


def generate_plots(out_dir: Path) -> list[Path]:
    tables = out_dir / "tables"
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    aggregate = pd.read_csv(tables / "aggregate_metrics_reference_only.csv", encoding="utf-8-sig")
    floor = pd.read_csv(tables / "floor_ratio_by_support_class.csv", encoding="utf-8-sig")
    subject = pd.read_csv(tables / "per_subject_harm.csv", encoding="utf-8-sig")
    selection = pd.read_csv(tables / "model_selection.csv", encoding="utf-8-sig")

    aggregate_index = aggregate.set_index("model")
    dense_index = floor.loc[floor["support_class"].eq("dense_overlap")].set_index("model")
    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    mae = [float(aggregate_index.loc[model, "subject_macro_mae_deg_reference_only"]) for model in ORDER]
    ratio = [float(dense_index.loc[model, "subject_macro_floor_ratio"]) for model in ORDER]
    axes[0].bar(x, mae, color=[COLORS[model] for model in ORDER], width=0.68)
    axes[0].set_xticks(x, ["B_all3", "Low-rank\nresidual", "Soft expert\ngate"])
    axes[0].set_ylabel("Subject-macro curve MAE (deg)")
    axes[0].set_title("Point prediction reference")
    axes[0].grid(axis="y", alpha=0.25)
    for position, value in enumerate(mae):
        axes[0].text(position, value + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    axes[1].bar(x, ratio, color=[COLORS[model] for model in ORDER], width=0.68)
    axes[1].axhline(2.01, color="#D62728", linestyle="--", linewidth=1.2, label="deployment reference 2.01")
    axes[1].set_xticks(x, ["B_all3", "Low-rank\nresidual", "Soft expert\ngate"])
    axes[1].set_ylabel("Dense subject-macro floor ratio")
    axes[1].set_title("Run59-scaled dense error")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    for position, value in enumerate(ratio):
        axes[1].text(position, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Run63 protected second-stage combinations", fontsize=13)
    path1 = figures / "Figure_1_model_comparison.png"
    fig.savefig(path1, dpi=220)
    plt.close(fig)

    pivot = subject.pivot(index="subject", columns="candidate_model", values="subject_mae_improvement_deg")
    pivot = pivot.sort_values("R_lowrank_residual", ascending=False)
    subjects = pivot.index.astype(str).tolist()
    x = np.arange(len(subjects))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13.0, 5.0), constrained_layout=True)
    ax.bar(
        x - width / 2,
        pivot["R_lowrank_residual"].to_numpy(float),
        width,
        color=COLORS["R_lowrank_residual"],
        label="Low-rank residual",
    )
    ax.bar(
        x + width / 2,
        pivot["G_soft_expert_gate"].to_numpy(float),
        width,
        color=COLORS["G_soft_expert_gate"],
        label="Soft expert gate",
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x, subjects, rotation=45, ha="right")
    ax.set_ylabel("Subject MAE improvement vs B_all3 (deg)")
    ax.set_title("Run63 improvement and harm by held-out subject")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    path2 = figures / "Figure_2_subject_improvement.png"
    fig.savefig(path2, dpi=220)
    plt.close(fig)

    # 第三图只展示训练侧选择，不把外折结果反向用于选参。
    selection["label"] = selection["selected_candidate_id"].astype(str)
    families = ["R_lowrank_residual", "G_soft_expert_gate"]
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 5.8), constrained_layout=True)
    for axis, family in zip(axes, families):
        frame = selection.loc[selection["family"].eq(family)].sort_values("outer_fold")
        axis.scatter(frame["outer_fold"], np.arange(len(frame)), s=55, color=COLORS[family])
        for row_number, row in enumerate(frame.itertuples(index=False)):
            axis.text(float(row.outer_fold) + 0.05, row_number, str(row.selected_candidate_id), va="center", fontsize=8)
        axis.set_yticks([])
        axis.set_xticks(range(1, 6))
        axis.set_xlim(0.8, 5.8)
        axis.set_title(family)
        axis.grid(axis="x", alpha=0.2)
    axes[-1].set_xlabel("Outer subject fold")
    path3 = figures / "Figure_3_train_side_selections.png"
    fig.savefig(path3, dpi=220)
    plt.close(fig)
    return [path1, path2, path3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="run_1")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = here / out_dir
    paths = generate_plots(out_dir.resolve())
    print("Generated figures:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

