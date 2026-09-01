# Run79 Research Card

## Status

- Machine/research status: `preprocessing`
- Original local directory: `run79_august_physio_preprocessing_20260831`

## One-line conclusion

完成27位、188个recording的四通道生理正式预处理。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`event_feature_coverage.csv`](artifacts/event_feature_coverage.csv)
- [`processing_summary.json`](artifacts/processing_summary.json)

## Published figures

- [`Figure_1_channel_sampling_rates.png`](figures/Figure_1_channel_sampling_rates.png)
- [`Figure_2_channel_quality_coverage.png`](figures/Figure_2_channel_quality_coverage.png)
- [`Figure_3_event_feature_coverage.png`](figures/Figure_3_event_feature_coverage.png)

## Source code and contracts

- [`config.json`](../../source/run079/config.json)
- [`notes.txt`](../../source/run079/notes.txt)
- [`process_august_physio.py`](../../source/run079/process_august_physio.py)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
