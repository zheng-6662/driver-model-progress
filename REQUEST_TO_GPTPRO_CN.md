# Request to GPTPro: Produce the Next Executable Plan

Read `GPTPRO_CONTEXT_CN.md`, `PROJECT_BACKGROUND_CN.md`, `CURRENT_STATUS_CN.md`, `audits/pedal_multiaction_audit_20260901/AUDIT_CN.md`, `RUN_INDEX.md`, and the relevant run cards/source code before answering. Claude's earlier analyses and sanitized raw JSONL are indexed at `claude_analysis/CLAUDE_ANALYSIS_INDEX_CN.md`; use them to reconstruct old reasoning when needed, but resolve conflicts in favor of current run evidence.

## Your task

Propose the single best next experiment or research pivot that is still justified by the evidence. The output must be implementation-ready, not a general brainstorming list.

## Required questions to answer

1. Why did ExtraTrees remain much stronger than all raw-sequence neural models after expanding to 38 drivers?
2. Given that the current 134D/172D mainline excludes accelerator and brake, and the new audit confirms substantial pedal activity, should the single next candidate be `ExtraTrees-134D + pedal representation`? If not, explain why the new evidence is still insufficient.
3. Choose exactly one next legal information source: pedals, road/event semantics, sequential personalization from completed prior events, self-supervised use of continuous unlabeled recordings, or a changed prediction anchor.
4. How should the 20 truly new drivers be used without repeating the harmful naive pooling in Run76?
5. Is there any defensible role for physiology/style now that direct mean-prediction increments repeatedly failed?
6. If pedals are selected, choose one primary representation only: raw 2 s sequence, low-dimensional summary, or current value/onset/change-rate representation. Explain how continuous throttle maintenance will be separated from a stimulus response.

## Required output format

- One-sentence hypothesis.
- Exact population and folds.
- Exact input representation available at inference.
- Exact target and prediction anchor.
- One baseline and one candidate only, plus essential ablations.
- Training-only model selection procedure.
- Subject-level and amplitude-level metrics.
- Predeclared success and stop rules.
- Files/code that need to be changed.
- Expected compute and run order.
- Evidence boundary and claims that remain prohibited.

Do not recommend a broad hyperparameter search. Do not reopen a no-go route without identifying the genuinely new information or mechanism that makes it different.
