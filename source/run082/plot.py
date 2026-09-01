from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
RUN = HERE / "run_2" if (HERE / "run_2/tables/aggregate_metrics.csv").exists() else HERE / "run_1"
TABLES = RUN / "tables"
FIGURES = RUN / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

if (TABLES / "aggregate_metrics.csv").exists():
    aggregate = pd.read_csv(TABLES / "aggregate_metrics.csv")
    domain = pd.read_csv(TABLES / "metrics_by_domain.csv")
    paired = pd.read_csv(TABLES / "paired_subject_improvements.csv")
    lag = pd.read_csv(TABLES / "lag_perturbation.csv")

    order = ["ExtraTrees_134D", "Plain_Raw_TCN", "Role_TCN", "LGRS_lambda0", "LGRS"]
    aggregate = aggregate.set_index("model").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        aggregate["model"],
        aggregate["subject_macro_curve_mae_deg"],
        color=["#4D4D4D", "#9E9E9E", "#D95F02", "#66A61E", "#1976D2"],
    )
    ax.set_ylabel("subject-macro curve MAE (deg)")
    ax.set_title("Run82: combined 38-subject OOF")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, aggregate["subject_macro_curve_mae_deg"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_1_model_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    role = paired.loc[paired["comparison"].eq("LGRS_vs_Role_TCN")].sort_values("subject")
    extra = paired.loc[paired["comparison"].eq("LGRS_vs_ExtraTrees")].sort_values("subject")
    x = range(len(role))
    ax.bar([value - 0.2 for value in x], role["improvement"], width=0.4, label="LGRS vs Role-TCN", color="#1976D2")
    ax.bar([value + 0.2 for value in x], extra["improvement"], width=0.4, label="LGRS vs ExtraTrees", color="#D95F02")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x), role["subject"], rotation=45)
    ax.set_ylabel("comparator - LGRS MAE improvement (deg)")
    ax.set_title("Subject-level paired improvement")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_2_subject_improvement.png", dpi=180)
    plt.close(fig)

    selected_domain = domain.loc[domain["model"].isin(["ExtraTrees_134D", "Role_TCN", "LGRS"])]
    pivot = selected_domain.pivot(index="stratum", columns="model", values="subject_macro_mae_deg")
    pivot = pivot.loc[["original_domain", "august_all", "august_new_subjects", "combined"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=["#4D4D4D", "#1976D2", "#D95F02"])
    ax.set_ylabel("subject-macro curve MAE (deg)")
    ax.set_xlabel("")
    ax.set_title("Performance by data domain")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_3_domain_comparison.png", dpi=180)
    plt.close(fig)

    lag_summary = lag.groupby("shift_ms")["mae_change_deg"].agg(["mean", "std", "count"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(lag_summary["shift_ms"], lag_summary["mean"], yerr=lag_summary["std"], marker="o", capsize=4, color="#1976D2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("command time shift (ms)")
    ax.set_ylabel("shifted - original subject-macro MAE (deg)")
    ax.set_title("LGRS lag perturbation diagnostic")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_4_lag_perturbation.png", dpi=180)
    plt.close(fig)
    print("Run82 full figures generated")
    raise SystemExit(0)

trace = pd.read_csv(TABLES / "smoke_training_trace.csv")
summary = pd.read_csv(TABLES / "smoke_model_summary.csv")

fig, ax = plt.subplots(figsize=(8, 5))
for model, group in trace.groupby("model", sort=False):
    ax.plot(group["epoch"], group["validation_subject_macro_mae_deg"], marker="o", label=model)
ax.set_xlabel("smoke epoch")
ax.set_ylabel("inner validation subject-macro MAE (deg)")
ax.set_title("Run82 smoke: validation trace (not a scientific comparison)")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "Figure_1_smoke_validation_trace.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(summary["model"], summary["parameter_count"], color=["#6E6E6E", "#1976D2", "#4CAF50"])
ax.set_ylabel("trainable parameters")
ax.set_title("Run82 smoke: parameter matching")
for index, value in enumerate(summary["parameter_count"]):
    ax.text(index, value + max(summary["parameter_count"]) * 0.015, str(int(value)), ha="center")
fig.tight_layout()
fig.savefig(FIGURES / "Figure_2_parameter_matching.png", dpi=180)
plt.close(fig)

print("Run82 smoke figures generated")
