from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
PREDICTIONS = HERE / "predictions" / "per_event_predictions.csv"
DECISION = HERE / "outputs" / "decision.json"
BASE = "B_all3"
TRUE = "B_all3_eye_true"
SHIFT = "B_all3_eye_shift_control"


def subject_macro(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("subject")[column].mean().mean())


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS, low_memory=False)
    eye = pd.read_csv(TABLES / "eye_features_by_event.csv", low_memory=False)
    recordings = pd.read_csv(TABLES / "eye_recording_coverage.csv", low_memory=False)
    shifts = pd.read_csv(TABLES / "eye_shift_control_mapping.csv", low_memory=False)
    aggregate = pd.read_csv(TABLES / "aggregate_metrics.csv")
    decision = json.loads(DECISION.read_text(encoding="utf-8"))

    checks: dict[str, bool] = {}
    checks["events_2323"] = len(predictions) == 2323 == len(eye)
    checks["unique_event_uid"] = predictions["event_uid"].is_unique and eye["event_uid"].is_unique
    checks["subjects_18_no_zyl"] = predictions["subject"].nunique() == 18 and "zyl" not in set(predictions["subject"])
    checks["fold_counts"] = predictions["outer_fold"].value_counts().sort_index().astype(int).to_dict() == {1: 352, 2: 471, 3: 435, 4: 539, 5: 526}
    checks["eye_join_exact"] = predictions["event_uid"].equals(eye["event_uid"])
    checks["matched_recordings_68"] = int(recordings["eye_matched"].sum()) == 68 and len(recordings) == 85
    checks["future_eye_samples_zero"] = int(eye["eye_window_future_sample_count"].sum()) == 0
    checks["window_stops_at_anchor"] = float(pd.to_numeric(eye["eye_window_max_relative_s"], errors="coerce").max()) <= 1e-9
    checks["all_events_retained"] = decision["eye"]["all_events_retained"] is True

    source_recording = shifts.set_index("event_uid")["recording_uid"]
    target_recording = predictions.set_index("event_uid")["recording_uid"]
    source_uid = shifts.set_index("event_uid")["control_source_event_uid"]
    source_uid_recording = target_recording.reindex(source_uid.to_numpy()).to_numpy()
    checks["shift_stays_within_recording"] = bool(np.all(source_recording.to_numpy() == source_uid_recording))
    checks["shift_changes_events"] = int(shifts["control_was_shifted"].sum()) == int(decision["eye"]["shift_control_changed_event_count"])

    true_columns = [f"true_t{i:02d}_deg" for i in range(1, 21)]
    for model in [BASE, TRUE, SHIFT]:
        pred_columns = [f"{model}_pred_t{i:02d}_deg" for i in range(1, 21)]
        recomputed = np.mean(np.abs(predictions[pred_columns].to_numpy(float) - predictions[true_columns].to_numpy(float)), axis=1)
        stored = predictions[f"{model}_curve_mae_deg"].to_numpy(float)
        checks[f"{model}_event_mae_exact"] = bool(np.max(np.abs(recomputed - stored)) < 1e-10)
        macro = subject_macro(predictions.assign(_mae=recomputed), "_mae")
        expected = float(aggregate.loc[aggregate["model"].eq(model), "subject_macro_curve_mae_deg"].iloc[0])
        checks[f"{model}_aggregate_exact"] = abs(macro - expected) < 1e-10

    checks["decision_supported_equals_all_gates"] = bool(decision["eye_state_increment_supported"]) == bool(all(decision["gates"].values()))
    payload = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "all_pass": bool(all(checks.values()))}
    (HERE / "outputs" / "validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"Run73 validation failed: {failed}")
    print("VALIDATION_PASS")


if __name__ == "__main__":
    main()
