from __future__ import annotations

"""训练侧LUPI筛选：事件后0–5秒生理教师是否真的比专家分歧控制更强。

该结果只决定是否值得做教师到预测起点前学生的蒸馏。事件后生理永远不进入outer
测试部署输入。
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_local("run64_base_post_teacher", RUN_DIR / "experiment.py")
tcn = load_local("run64_tcn_post_teacher", RUN_DIR / "scripts" / "screen_tcn_inner.py")

OUT_DIR = RUN_DIR / "run_7_inner_post_teacher"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "outputs").mkdir(parents=True, exist_ok=True)
POST_PATH = RUN_DIR / "cache" / "physio_post_sequence_10hz_teacher_only.npz"
ARMS = {"D_control": "D_tiny", "T_post_physio": "P_tcn", "TS_post_style": "PS_film"}


def main():
    print(f"[Run64 post teacher screen] device={tcn.DEVICE}; post-event data are teacher-only.")
    pfull = pd.read_csv(base.PFULL_PATH).set_index("event_uid", verify_integrity=True)
    features = pd.read_csv(base.FEATURE_PATH).set_index("event_uid", verify_integrity=True)
    inner_all = pd.read_csv(base.INNER_PATH)
    cache = np.load(POST_PATH, allow_pickle=True)
    cache_ids = cache["event_uid"].astype(str)
    cache_map = {uid: i for i, uid in enumerate(cache_ids)}
    post_seq = cache["post_sequence"].astype(np.float32)
    post_mask = cache["channel_mask"].astype(np.float32)
    rows = []
    for outer_fold in range(1, 6):
        frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        ids = frame["event_uid"].astype(str).to_numpy()
        idx = np.asarray([cache_map[x] for x in ids], dtype=int)
        seq = post_seq[idx]
        mask = post_mask[idx]
        curves = base.load_inner_curves(frame)
        truth = pfull.loc[ids, base.truth_columns()].to_numpy(float)
        regret, losses = base.centered_regret(curves, truth)
        tau = max(0.5, float(np.median(np.std(losses, axis=1))))
        teacher_logits = -regret / tau
        teacher_logits -= teacher_logits.mean(axis=1, keepdims=True)
        d = base.disagreement_features(curves)
        style = features.reindex(pd.Index(ids))[base.STYLE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        base_curve = curves.mean(axis=1)
        base_error = base.event_mae(base_curve, truth)
        for label, internal_arm in ARMS.items():
            pred_curve = np.full_like(truth, np.nan)
            params = 0
            for inner_fold in sorted(frame["inner_fold"].unique()):
                val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
                fit = ~val
                logits, params = tcn.train_predict(
                    internal_arm,
                    d[fit],
                    d[val],
                    seq[fit],
                    seq[val],
                    mask[fit],
                    mask[val],
                    style[fit],
                    style[val],
                    teacher_logits[fit],
                    frame.loc[fit, "subject"].astype(str).to_numpy(),
                    base.SEED + 25000 + outer_fold * 100 + inner_fold * 10 + list(ARMS).index(label),
                )
                q = base.stable_softmax(logits)
                weights = (1.0 - base.TRUST_UPDATE) / 3.0 + base.TRUST_UPDATE * q
                pred_curve[val] = base.curves_from_weights(curves[val], weights)
            model_error = base.event_mae(pred_curve, truth)
            cert = base.certification(
                frame["subject"].astype(str).to_numpy(),
                base_error,
                model_error,
                base.SEED + 27000 + outer_fold * 100 + list(ARMS).index(label),
            )
            cert.update(
                {
                    "outer_fold": outer_fold,
                    "teacher_arm": label,
                    "internal_architecture": internal_arm,
                    "parameter_count": params,
                    "event_improved_fraction": float(np.mean(model_error < base_error - 1e-12)),
                    "teacher_post_any_valid_fraction": float(np.mean(mask.any(axis=1))),
                    "teacher_only": label != "D_control",
                }
            )
            rows.append(cert)
            print(
                f"outer={outer_fold} arm={label} gain={cert['subject_macro_mae_improvement_deg']:+.4f} "
                f"ci_lo={cert['bootstrap_ci_lower_deg']:+.4f}"
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "tables" / "inner_post_teacher_screen.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby("teacher_arm", as_index=False)
        .agg(
            mean_inner_gain_deg=("subject_macro_mae_improvement_deg", "mean"),
            min_inner_gain_deg=("subject_macro_mae_improvement_deg", "min"),
            outer_positive_count=("subject_macro_mae_improvement_deg", lambda x: int((x > 0).sum())),
            outer_base_gate_pass_count=("base_gate_pass", "sum"),
            mean_ci_lower_deg=("bootstrap_ci_lower_deg", "mean"),
            mean_leave_top_gain_deg=("leave_top_subject_improvement_deg", "mean"),
            mean_event_improved_fraction=("event_improved_fraction", "mean"),
            parameter_count=("parameter_count", "max"),
        )
        .sort_values("mean_inner_gain_deg", ascending=False)
    )
    control = float(summary.loc[summary["teacher_arm"] == "D_control", "mean_inner_gain_deg"].iloc[0])
    summary["gain_over_D_control_deg"] = summary["mean_inner_gain_deg"] - control
    summary.to_csv(OUT_DIR / "tables" / "inner_post_teacher_summary.csv", index=False, encoding="utf-8-sig")
    best_post = summary.loc[summary["teacher_arm"] != "D_control"].iloc[0]
    advance = bool(
        best_post["gain_over_D_control_deg"] >= 0.02
        and best_post["outer_positive_count"] >= 4
    )
    decision = {
        "status": "inner_only_post_teacher_screen_complete",
        "outer_test_opened": False,
        "best_post_teacher": best_post.to_dict(),
        "advance_to_privileged_distillation": advance,
        "all_arms": summary.to_dict(orient="records"),
        "boundary": "Post-event physiology is training-only privileged information and is forbidden at deployment.",
    }
    (OUT_DIR / "outputs" / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "outputs" / "RESULT_CN.md").write_text(
        "# Run64 事件后生理特权教师训练侧筛选\n\n"
        "事件后0–5秒数据只允许训练教师；不进入部署输入。\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + f"\n\n是否进入蒸馏：**{advance}**\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

