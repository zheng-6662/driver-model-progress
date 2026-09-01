# Current Project Status

## Outcome first

The project now has a stronger data foundation (2598 events, 38 drivers) and clean physiology/eye/EEG inventories, but the strongest practical release-time predictor is still ExtraTrees on causal vehicle summaries.

## Main practical baseline

| Population | Model | Subject-macro curve MAE |
|---|---|---:|
| Original 18 | ExtraTrees / no-rate comparable line | about 12.62-12.97 deg depending exact cohort protocol |
| Combined 38 | ExtraTrees-134D | 14.1103 deg |
| August new 20 | ExtraTrees-134D | 15.1089 deg |

## Latest scientific decision

Run82 decision: `LGRS_NOT_EFFECTIVE`.

The lag-gain relation bottleneck is better than parameter-matched Role-TCN, but it does not beat ExtraTrees. The result supports a representation mechanism, not a new main predictor.

## Data expansion status

- 29 August subject directories inspected.
- 27 subjects have four-channel physiology.
- 26 subjects have eligible vehicle events.
- 20 are truly new drivers relative to the original cohort.
- 275 August events satisfy the current screening contract.
- Combined rows: 2598; combined distinct drivers: 38.

## Current model-use recommendation

- Keep ExtraTrees as the main release-time predictor.
- Keep Run79 physiology for mechanism or future changed-estimand studies, not direct mean correction.
- Do not continue tuning LGRS/TCN/Transformer/Mamba merely to rescue the neural family.
- A future experiment must introduce genuinely new information or a clearly different legal data-use protocol.
