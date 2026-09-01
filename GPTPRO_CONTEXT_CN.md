# GPTPro Direct Project Context

## Required reading order

1. Read this file completely.
2. Read [PROJECT_BACKGROUND_CN.md](PROJECT_BACKGROUND_CN.md).
3. Read [CURRENT_STATUS_CN.md](CURRENT_STATUS_CN.md).
4. Read the new [pedal, stimulus and multi-action audit](audits/pedal_multiaction_audit_20260901/AUDIT_CN.md).
5. Read [RUN_INDEX.md](RUN_INDEX.md).
6. Read [REQUEST_TO_GPTPRO_CN.md](REQUEST_TO_GPTPRO_CN.md).
7. When an earlier design decision needs reconstruction, read [Claude history index](claude_analysis/CLAUDE_ANALYSIS_INDEX_CN.md) and then open only the relevant raw JSONL or extracted conclusion.
8. Open only the run cards needed to support your proposed next plan.

Do not infer current status from an older run. Do not treat a diagnostic result as a validated model gain.

## Bottom-line research problem

At the release time of a high-dynamic / near-instability proxy driving event, use only information available at or before release to predict the complete 20-point steering response curve over release +0.05 to +1.00 seconds, with subject-disjoint generalization.

Aligned target convention:

```text
aligned_deg = degrees((target - release_value) * direction)
```

## Current data

- Original strict-causal population: 2323 events, 18 subjects, 85 recordings.
- August cohort: 275 eligible events, 26 subjects with eligible events.
- Truly new August drivers relative to the original cohort: 20.
- Combined modeling population: 2598 events, 38 distinct drivers, 190 recordings.
- Shared vehicle input across cohorts: steering angle, steering rate, speed, lateral acceleration, roll, curvature, lateral distance, plus road-missing mask.
- August four-channel physiology: 27 subjects, 188 recordings preprocessed in Run79.
- Processed physiology can be joined to 265/275 August events.

## New legal input evidence: accelerator and brake

The current Run57–Run82 mainline does not include accelerator or brake in its 134D/172D causal vehicle summaries. A read-only audit has now confirmed that both pedal fields exist in all 85 original and all 136 August vehicle recordings.

- Original P_full: accelerator active in 87.5% and brake active in 26.9% of release-2s to release+1s event windows; 598/2323 events have a brake range change of at least 0.05.
- August eligible: accelerator active in 76.4% and brake active in 45.1%; 109/275 events have a brake range change of at least 0.05.
- Full recordings contain 1973 stable brake onsets at speed >=60 km/h, including 551 with less than 5 deg steering in the next second and 788 with at least 20 deg steering.
- Original traffic-stimulus timing is partly recoverable from recorded target-distance thresholds; August traffic trigger timing is not present, although road-mu transitions are recorded.

This establishes availability and candidate quantity, not predictive gain. Read the audit before proposing the next experiment.

## Current strongest practical model

ExtraTrees using the 134-dimensional no-yaw-rate/no-roll-rate summary remains the strongest model on the combined 38-driver development protocol:

```text
subject-macro curve MAE = 14.1103 deg
```

By domain:

```text
original 18 drivers = 12.9685 deg
August all = 15.1924 deg
August truly new 20 drivers = 15.1089 deg
```

## Latest method result: Run82

```text
ExtraTrees-134D = 14.1103 deg
Plain Raw-TCN   = 19.1768 deg
Role-TCN        = 18.0616 deg
LGRS-lambda0    = 17.9139 deg
LGRS            = 17.5390 deg
```

LGRS versus parameter-matched Role-TCN:

- subject-macro gain +0.5226 deg
- 95% subject-bootstrap CI [+0.0443, +1.0023]
- 5/5 outer folds positive
- 26/38 subjects improved
- amplitude protection passed

LGRS versus ExtraTrees:

- subject-macro gain -3.4287 deg
- 95% CI [-4.6142, -2.4021]
- 0/5 folds positive
- only 3/38 subjects improved
- all amplitude bins were worse

Interpretation: explicit command-response lag/gain modeling is useful inside neural sequence models, but the neural family remains substantially worse than the tree baseline. LGRS is not allowed to replace ExtraTrees.

## Important negative evidence

- Changing ExtraTrees to LightGBM/HistGBM did not close the gap (Run60).
- Road preview residual correction did not produce an independent gain (Run61).
- Amplitude-shape factorization and eight hand-crafted phase features failed (Run62).
- Low-rank residual and soft gating did not pass frozen gates (Run63).
- Direct physiology/style, physiology TCN, FiLM, BIOT and related residual routes did not pass (Run64-68).
- Eye tracking did not improve the original 18-driver model (Run73).
- Directly pooling early August data into original training harmed original-driver performance (Run76).
- Formally preprocessed physiology still did not improve vehicle prediction (Run79-80).
- Raw vehicle TCNs and LGRS remained much worse than ExtraTrees (Run82).

## Positive but bounded evidence

- Removing missing yaw/roll-rate channels has negligible overall MAE impact, so August vehicle data are not blocked by those channels (Run75).
- Additional August data increase the independent driver count, but naive pooling is harmful (Run76/78).
- Clean physiology signals are now available for mechanism analysis, although they are not a direct mean-prediction increment (Run79/80).
- LGRS lag perturbation increases error, confirming that command-response timing is used; this is mechanism evidence only (Run82).
- Waiting until t0+0.4 s and observing points 1-8 greatly improves tail prediction, but this is a changed estimand and failed a road-missing protection gate (Run69).

## Current evidence boundary

- Do not claim theoretical unpredictability or causal irreducible noise.
- Do not use release-post information in the release-time task.
- Do not promote a mechanism gain if the full model loses to ExtraTrees.
- Subject-macro and amplitude-stratified relative error are primary; pooled MAE is reference only.
- Any next plan must use subject-disjoint validation and keep same-driver sessions in one fold.
- Do not retry a closed route by changing only its name or backbone.
