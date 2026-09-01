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

## New pedal and multi-action audit

The current strongest causal summaries exclude accelerator and brake. A new read-only audit confirms that both signals are present and nontrivial in every original and August vehicle recording used by the current cohorts.

- Brake changes occur in 598/2323 original steering events and 109/275 August events.
- Full continuous recordings contain 1973 stable high-speed brake onsets.
- Signal quantity is sufficient to build braking-dominant and combined-action candidate pools.
- The current event tables are steering-selected, so multi-action labels are not yet frozen.
- Original traffic stimulus timing is partly recoverable; August traffic stimulus timing remains uncertain.

This makes pedals a genuinely new input for the current mainline, but not a proven model increment.
