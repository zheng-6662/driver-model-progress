from __future__ import annotations

"""从已经完成的 Run68 训练侧结果生成两幅确定性诊断图。

本脚本不训练模型、不读取 outer-test。输出目录必须已经包含 Run68 的
``tables/arm_summary.csv`` 与 ``tables/risk_coverage_curve.csv``。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ORDER = ["U_V", "U_VQ", "U_Vshift", "U_VP"]
COLORS = {
    "U_V": "#4C78A8",
    "U_VQ": "#A0CBE8",
    "U_Vshift": "#F58518",
    "U_VP": "#54A24B",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = HERE / run_dir
    summary_path = run_dir / "tables" / "arm_summary.csv"
    risk_path = run_dir / "tables" / "risk_coverage_curve.csv"
    if not summary_path.is_file() or not risk_path.is_file():
        raise FileNotFoundError("Run68 result tables are incomplete")
    summary = pd.read_csv(summary_path).set_index("arm").loc[ORDER].reset_index()
    risk = pd.read_csv(risk_path)
    figures = run_dir / "figures"
    if figures.exists():
        raise FileExistsError(f"append-only figure directory already exists: {figures}")
    figures.mkdir()

    x = np.arange(len(summary))
    fig, left = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    bars = left.bar(
        x - 0.18,
        summary["subject_macro_simultaneous_coverage"],
        width=0.36,
        color=[COLORS[arm] for arm in summary["arm"]],
        alpha=0.85,
        label="Simultaneous coverage",
    )
    left.axhspan(0.77, 0.83, color="#D9EAD3", alpha=0.6, label="Target band 0.77–0.83")
    left.set_ylim(0.0, 1.0)
    left.set_ylabel("Subject-macro simultaneous coverage")
    left.set_xticks(x, summary["arm"])
    right = left.twinx()
    right.plot(
        x + 0.18,
        summary["subject_macro_mean_width_deg"],
        marker="o",
        linewidth=2,
        color="#7A1FA2",
        label="Mean interval width",
    )
    right.set_ylabel("Subject-macro mean width (degree)")
    left.set_title("Run68 coverage and interval width")
    handles = [bars]
    labels = ["Simultaneous coverage"]
    line = right.lines[0]
    handles.append(line)
    labels.append("Mean interval width")
    left.legend(handles, labels, loc="upper left", frameon=False)
    fig.savefig(figures / "Figure_1_coverage_width.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for arm in ORDER:
        part = risk.loc[risk["arm"].eq(arm)].sort_values("retention")
        axis.plot(
            100.0 * part["retention"],
            part["subject_macro_selective_tail_mae_deg"],
            marker="o",
            linewidth=2,
            color=COLORS[arm],
            label=arm,
        )
    axis.set_xlabel("Retained events (%)")
    axis.set_ylabel("Subject-macro tail MAE (degree)")
    axis.set_title("Run68 risk–coverage curves (risk = interval width)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.savefig(figures / "Figure_2_risk_coverage.png", dpi=180)
    plt.close(fig)
    print(str(figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
