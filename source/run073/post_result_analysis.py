from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PRED_PATH = HERE / "predictions" / "per_event_predictions.csv"
OUT_TABLE = HERE / "tables" / "post_result_availability_diagnostic.csv"
OUT_MD = HERE / "outputs" / "POST_RESULT_DIAGNOSTIC_CN.md"
BASE = "B_all3"
TRUE = "B_all3_eye_true"
SHIFT = "B_all3_eye_shift_control"
SEED = 20260830


def pred(frame: pd.DataFrame, model: str) -> np.ndarray:
    return frame[[f"{model}_pred_t{i:02d}_deg" for i in range(1, 21)]].to_numpy(float)


def event_mae(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(prediction - truth), axis=1)


def compare(subject: np.ndarray, reference: np.ndarray, candidate: np.ndarray, name: str) -> dict[str, float | int | str]:
    frame = pd.DataFrame({"subject": subject, "improvement": reference - candidate})
    values = frame.groupby("subject")["improvement"].mean().to_numpy(float)
    rng = np.random.default_rng(SEED + sum(ord(char) for char in name))
    draws = np.asarray([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(2000)])
    return {
        "comparison": name,
        "subject_macro_improvement_deg": float(values.mean()),
        "bootstrap_ci_lower_deg": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_upper_deg": float(np.quantile(draws, 0.975)),
        "improved_subjects": int((values > 1e-12).sum()),
        "harmed_subjects": int((values < -1e-12).sum()),
    }


def main() -> None:
    frame = pd.read_csv(PRED_PATH, low_memory=False)
    truth = frame[[f"true_t{i:02d}_deg" for i in range(1, 21)]].to_numpy(float)
    base = pred(frame, BASE)
    true = pred(frame, TRUE)
    shift = pred(frame, SHIFT)
    base_mae = event_mae(truth, base)
    rows: list[dict[str, object]] = []
    masks = {
        "eye_available": frame["eye_available"].eq(1).to_numpy(),
        "pupil_available": frame["pupil_available"].eq(1).to_numpy(),
        "gaze_available": frame["gaze_available"].eq(1).to_numpy(),
        "pupil_and_gaze_available": (frame["pupil_available"].eq(1) & frame["gaze_available"].eq(1)).to_numpy(),
    }
    for label, mask in masks.items():
        protected_true = np.where(mask[:, None], true, base)
        protected_shift = np.where(mask[:, None], shift, base)
        protected_true_mae = event_mae(truth, protected_true)
        protected_shift_mae = event_mae(truth, protected_shift)
        direct = compare(frame["subject"].to_numpy(), base_mae, protected_true_mae, f"protected_true_vs_B::{label}")
        control = compare(frame["subject"].to_numpy(), protected_shift_mae, protected_true_mae, f"protected_true_vs_shift::{label}")
        for item in [direct, control]:
            rows.append(
                {
                    "availability_rule": label,
                    "available_event_count": int(mask.sum()),
                    "unavailable_fallback_event_count": int((~mask).sum()),
                    **item,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    lines = [
        "# Run73 结果后可用性回退诊断",
        "",
        "本诊断没有重训模型，只把无相应眼动质量的事件精确替换回冻结B_all3预测。规则在看到Run73正式结果后才评估，因此不能改写正式no-go，只能用于决定下一轮是否预注册保护门。",
        "",
        "| 回退规则 | 比较 | 可用事件 | 改善° | 95%CI | 改善被试 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['availability_rule']} | {row['comparison']} | {row['available_event_count']} | "
            f"{row['subject_macro_improvement_deg']:+.4f} | [{row['bootstrap_ci_lower_deg']:+.4f}, {row['bootstrap_ci_upper_deg']:+.4f}] | "
            f"{row['improved_subjects']}/18 |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
