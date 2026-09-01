from __future__ import annotations

"""只在 outer-train / inner-OOF 上筛选第二波结构，不读取新候选 outer-test 表现。

第一波发现 D/P/S/PS 的约0.047°外层改善几乎全部来自专家分歧块。这里检验两个
未被第一波覆盖的机制：

1. 生理/风格与专家分歧的显式低阶交互；
2. 只使用已经揭晓的同一驾驶员先前事件真值形成暖启动历史。

脚本只输出训练侧screen，不生成任何outer-test预测。
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
SPEC = importlib.util.spec_from_file_location("run64_base", RUN_DIR / "experiment.py")
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)


SCREEN_DIR = RUN_DIR / "run_2_inner_screen"
SCREEN_DIR.mkdir(parents=True, exist_ok=True)
(SCREEN_DIR / "tables").mkdir(parents=True, exist_ok=True)
(SCREEN_DIR / "outputs").mkdir(parents=True, exist_ok=True)


CANDIDATES = {
    "D": {"blocks": (), "interactions": ()},
    "P_add": {"blocks": ("phys",), "interactions": ()},
    "S_add": {"blocks": ("style",), "interactions": ()},
    "PS_add": {"blocks": ("phys", "style"), "interactions": ()},
    "P_x_D": {"blocks": ("phys",), "interactions": ("phys",)},
    "S_x_D": {"blocks": ("style",), "interactions": ("style",)},
    "PS_x_D": {"blocks": ("phys", "style"), "interactions": ("phys", "style")},
    "H_add": {"blocks": ("history",), "interactions": ()},
    "HP_add": {"blocks": ("history", "phys"), "interactions": ()},
    "HS_add": {"blocks": ("history", "style"), "interactions": ()},
    "HPS_add": {"blocks": ("history", "phys", "style"), "interactions": ()},
    "HPS_x_D": {
        "blocks": ("history", "phys", "style"),
        "interactions": ("history", "phys", "style"),
    },
}


def event_absolute_seconds(frame: pd.DataFrame) -> np.ndarray:
    session = pd.to_datetime(frame["session_stamp"], format="%Y_%m_%d_%H_%M_%S", errors="coerce")
    return session.astype("int64").to_numpy(float) / 1e9 + frame["primary_release_s"].to_numpy(float)


def history_features(
    event_ids: np.ndarray,
    curves: np.ndarray,
    truth: np.ndarray,
    pfull_index: pd.DataFrame,
) -> np.ndarray:
    """在预测事件前，以EWMA聚合已经完整揭晓的过去专家相对后悔。"""
    ids = pd.Index(event_ids.astype(str))
    meta = pfull_index.reindex(ids)[["subject", "session_stamp", "primary_release_s"]].copy()
    if meta.isna().any().any():
        raise ValueError("history metadata join failed")
    _, losses = base.centered_regret(curves, truth)
    regrets = losses - losses.mean(axis=1, keepdims=True)
    meta["row_index"] = np.arange(len(meta))
    meta["anchor_abs_s"] = event_absolute_seconds(meta)
    result = np.zeros((len(meta), 4), dtype=float)
    for _, group in meta.groupby("subject", sort=False):
        order = group.sort_values(["anchor_abs_s", "primary_release_s", "row_index"])
        pending: list[tuple[float, np.ndarray]] = []
        ewma = np.zeros(3, dtype=float)
        count = 0
        for row in order.itertuples():
            now = float(row.anchor_abs_s)
            ready = [x for x in pending if x[0] <= now + 1e-9]
            pending = [x for x in pending if x[0] > now + 1e-9]
            for _, value in sorted(ready, key=lambda x: x[0]):
                ewma = value.copy() if count == 0 else 0.5 * value + 0.5 * ewma
                ewma = ewma - ewma.mean()
                count += 1
            idx = int(row.row_index)
            result[idx, 0] = min(count, 3) / 3.0
            result[idx, 1:] = ewma
            pending.append((now + 1.0, regrets[idx]))
    return result


def feature_matrix(
    ids: np.ndarray,
    curves: np.ndarray,
    feature_index: pd.DataFrame,
    history: np.ndarray,
    spec: dict[str, tuple[str, ...]],
) -> np.ndarray:
    disagreement = base.disagreement_features(curves)
    rows = feature_index.reindex(pd.Index(ids.astype(str)))
    values = {
        "phys": rows[base.PHYS_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float),
        "style": rows[base.STYLE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float),
        "history": history,
    }
    blocks = [disagreement]
    for name in spec["blocks"]:
        blocks.append(values[name])
    # 只让模态与5个最直观的专家分歧量相乘，避免完整二阶多项式爆炸。
    d_small = disagreement[:, :5]
    for name in spec["interactions"]:
        block = values[name]
        blocks.append((d_small[:, :, None] * block[:, None, :]).reshape(len(ids), -1))
    return np.concatenate(blocks, axis=1)


def evaluate_candidate(
    outer_fold: int,
    candidate: str,
    spec: dict[str, tuple[str, ...]],
    frame: pd.DataFrame,
    curves: np.ndarray,
    truth: np.ndarray,
    feature_index: pd.DataFrame,
    pfull_index: pd.DataFrame,
) -> dict[str, object]:
    history = history_features(frame["event_uid"].astype(str).to_numpy(), curves, truth, pfull_index)
    predictions = np.full_like(truth, np.nan)
    for inner_fold in sorted(frame["inner_fold"].unique()):
        val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
        fit = ~val
        x_all = feature_matrix(
            frame["event_uid"].astype(str).to_numpy(), curves, feature_index, history, spec
        )
        target, losses = base.centered_regret(curves[fit], truth[fit])
        model = base.fit_student(
            x_all[fit], target, losses, frame.loc[fit, "subject"].astype(str).to_numpy()
        )
        _, weights = base.predict_student(model, x_all[val])
        predictions[val] = base.curves_from_weights(curves[val], weights)
    base_pred = curves.mean(axis=1)
    base_error = base.event_mae(base_pred, truth)
    model_error = base.event_mae(predictions, truth)
    cert = base.certification(
        frame["subject"].astype(str).to_numpy(),
        base_error,
        model_error,
        base.SEED + 9000 + outer_fold * 100 + list(CANDIDATES).index(candidate),
    )
    cert.update(
        {
            "outer_fold": outer_fold,
            "candidate": candidate,
            "feature_dimension_before_imputation": int(
                feature_matrix(
                    frame["event_uid"].astype(str).to_numpy(), curves, feature_index, history, spec
                ).shape[1]
            ),
            "event_improved_fraction": float(np.mean(model_error < base_error - 1e-12)),
            "warm_history_available_fraction": float(np.mean(history[:, 0] > 0)),
        }
    )
    return cert


def main() -> int:
    print("[Run64 inner screen] 目标：只用outer-train结果筛选状态×特质交互和暖启动候选。")
    pfull = pd.read_csv(base.PFULL_PATH)
    pfull_index = pfull.set_index("event_uid", verify_integrity=True)
    features = pd.read_csv(base.FEATURE_PATH).set_index("event_uid", verify_integrity=True)
    inner_all = pd.read_csv(base.INNER_PATH)
    truth_lookup = pfull_index[base.truth_columns()]
    rows = []
    for outer_fold in range(1, 6):
        frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        curves = base.load_inner_curves(frame)
        truth = truth_lookup.loc[frame["event_uid"]].to_numpy(float)
        for candidate, spec in CANDIDATES.items():
            row = evaluate_candidate(
                outer_fold, candidate, spec, frame, curves, truth, features, pfull_index
            )
            rows.append(row)
            print(
                f"outer={outer_fold} candidate={candidate} "
                f"gain={row['subject_macro_mae_improvement_deg']:+.4f} "
                f"ci_lo={row['bootstrap_ci_lower_deg']:+.4f}"
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(SCREEN_DIR / "tables" / "inner_candidate_screen.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby("candidate", as_index=False)
        .agg(
            mean_inner_gain_deg=("subject_macro_mae_improvement_deg", "mean"),
            min_inner_gain_deg=("subject_macro_mae_improvement_deg", "min"),
            outer_positive_count=("subject_macro_mae_improvement_deg", lambda x: int((x > 0).sum())),
            outer_base_gate_pass_count=("base_gate_pass", "sum"),
            mean_ci_lower_deg=("bootstrap_ci_lower_deg", "mean"),
            mean_leave_top_gain_deg=("leave_top_subject_improvement_deg", "mean"),
            mean_event_improved_fraction=("event_improved_fraction", "mean"),
            mean_warm_history_available_fraction=("warm_history_available_fraction", "mean"),
            feature_dimension=("feature_dimension_before_imputation", "max"),
        )
        .sort_values(["mean_inner_gain_deg", "outer_positive_count"], ascending=False)
    )
    summary.to_csv(SCREEN_DIR / "tables" / "inner_candidate_summary.csv", index=False, encoding="utf-8-sig")
    best = summary.iloc[0].to_dict()
    decision = {
        "status": "inner_only_screen_complete",
        "outer_test_new_candidates_opened": False,
        "best_candidate": best,
        "all_candidates": summary.to_dict(orient="records"),
        "selection_rule": (
            "Highest mean inner subject-macro gain, at least 4/5 positive outer contexts; "
            "no outer-test result was used for this screen."
        ),
    }
    (SCREEN_DIR / "outputs" / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (SCREEN_DIR / "outputs" / "RESULT_CN.md").write_text(
        "# Run64 第二波训练侧候选筛选\n\n"
        + "本表只使用outer训练侧inner-OOF，不读取新候选outer-test结果。\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

