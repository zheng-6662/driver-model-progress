"""Classify subject harm with the same 1e-12 numerical tolerance used for events.

This is a read-only post-result interpretation audit. It does not alter predictions,
metrics, gates, bootstrap values, model selection, or decision.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "run_1"
TOLERANCE = 1e-12


def main() -> None:
    source = pd.read_csv(OUT / "tables" / "per_subject_harm.csv", encoding="utf-8-sig")
    rows = []
    for candidate, group in source.groupby("candidate_model", sort=True):
        delta = group["subject_mae_improvement_deg"].astype(float)
        rows.append(
            {
                "candidate_model": candidate,
                "tolerance_deg": TOLERANCE,
                "materially_harmed_subject_count": int((delta < -TOLERANCE).sum()),
                "numerically_tied_subject_count": int((delta.abs() <= TOLERANCE).sum()),
                "materially_improved_subject_count": int((delta > TOLERANCE).sum()),
                "worst_material_subject_change_candidate_minus_base_deg": float(
                    -delta.loc[delta < -TOLERANCE].min() if (delta < -TOLERANCE).any() else 0.0
                ),
                "producer_raw_negative_count": int((delta < 0.0).sum()),
                "decision_affected": False,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables" / "harm_summary_tolerance_audit.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS",
        "source": "tables/per_subject_harm.csv",
        "tolerance_deg": TOLERANCE,
        "reason": "Exact B_all3 fallback can differ by machine floating summation at about 1e-16 degree; those rows are numerical ties, not material harm.",
        "decision_or_gate_changed": False,
        "rows": table.to_dict(orient="records"),
    }
    (OUT / "audit" / "harm_tolerance_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

