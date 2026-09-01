# Request to GPTPro: Produce the Next Executable Plan

Read `GPTPRO_CONTEXT_CN.md`, `PROJECT_BACKGROUND_CN.md`, `CURRENT_STATUS_CN.md`, `RUN_INDEX.md`, and the relevant run cards/source code before answering.

## Your task

Propose the single best next experiment or research pivot that is still justified by the evidence. The output must be implementation-ready, not a general brainstorming list.

## Required questions to answer

1. Why did ExtraTrees remain much stronger than all raw-sequence neural models after expanding to 38 drivers?
2. Does the Run82 relation-state signal justify extracting a small number of legal relation features for the tree model, or would that merely repeat Run62 hand-crafted phase features?
3. Should the next legal information source be road/event semantics, sequential personalization from completed prior events, self-supervised use of continuous unlabeled recordings, or a changed prediction anchor? Choose one, not all.
4. How should the 20 truly new drivers be used without repeating the harmful naive pooling in Run76?
5. Is there any defensible role for physiology/style now that direct mean-prediction increments repeatedly failed?

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
