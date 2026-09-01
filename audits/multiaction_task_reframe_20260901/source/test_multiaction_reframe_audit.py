from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--public-audit-root", type=Path, required=True)
    args = parser.parse_args()

    events = pd.read_csv(args.package_root / "tables_private/stimulus_events_private.csv", low_memory=False)
    actions = pd.read_csv(args.package_root / "tables_private/action_labels_private.csv", low_memory=False)
    thresholds = pd.read_csv(args.package_root / "tables_private/label_threshold_sensitivity_private.csv")
    physiology = pd.read_csv(args.package_root / "tables_private/physiology_join_private.csv", low_memory=False)

    included = events["included_candidate"].astype(bool)
    assert included.any()
    assert len(actions) == int(included.sum())
    assert actions["event_id"].is_unique
    assert actions["no_response"].astype(bool).any()
    assert set(thresholds["threshold_set"]) == {"lenient", "primary", "strict"}
    assert set(thresholds["window_s"]) == {1, 2, 3, 5}
    assert len(physiology) == len(actions)
    mu_events = events.loc[events["trigger_signal"].eq("mu")]
    assert not mu_events.empty
    assert mu_events["stimulus_type"].eq("enter_low_mu_scene").all()
    assert mu_events.groupby("recording_alias").size().le(1).all()
    assert not mu_events["trigger_rule"].str.contains(
        r"1\.000 -> 0\.800|0\.800 -> 1\.000|0\.400 -> 0\.200"
    ).any()
    assert int(included.sum()) <= 526

    forbidden = re.compile(r"(?i)[A-Z]:[\\/]|/home/|subject_alias,|event_id,")
    public_files = 0
    for path in args.public_audit_root.rglob("*"):
        if path.suffix.lower() not in {".md", ".json", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        assert not forbidden.search(text), path
        if path.suffix.lower() == ".json":
            json.loads(text)
        elif path.suffix.lower() == ".csv":
            assert not pd.read_csv(path).empty, path
        public_files += 1
    assert public_files >= 10
    print(
        json.dumps(
            {
                "stimulus_event_contract": "PASS",
                "action_label_contract": "PASS",
                "threshold_sensitivity_contract": "PASS",
                "physiology_join_contract": "PASS",
                "public_privacy_and_parse": "PASS",
                "models_trained": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
