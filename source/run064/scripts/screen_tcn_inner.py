from __future__ import annotations

"""GPU训练侧筛选：小型生理TCN + 既往session风格FiLM -> 专家后悔学生。

只使用 outer-train 的 inner-OOF 结果。固定架构、固定80 epochs、固定25%信任域；
不读取新候选 outer-test 表现。目的是检验手工16维可能丢失的生理动态，而不是
扩大车辆主模型。
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
SPEC = importlib.util.spec_from_file_location("run64_base_tcn", RUN_DIR / "experiment.py")
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

OUT_DIR = RUN_DIR / "run_5_inner_tcn_screen"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "outputs").mkdir(parents=True, exist_ok=True)
SEQ_PATH = RUN_DIR / "cache" / "physio_sequence_10hz.npz"

ARMS = ("D_tiny", "P_tcn", "S_mlp", "PS_film")
EPOCHS = 80
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 1e-2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RegretStudent(nn.Module):
    def __init__(self, arm: str, d_dim: int, style_dim: int):
        super().__init__()
        self.arm = arm
        self.d_net = nn.Sequential(nn.Linear(d_dim, 8), nn.ReLU())
        if arm in {"P_tcn", "PS_film"}:
            self.phys_net = nn.Sequential(
                nn.Conv1d(4, 8, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(8, 8, kernel_size=5, dilation=2, padding=4),
                nn.ReLU(),
            )
        if arm in {"S_mlp", "PS_film"}:
            self.style_net = nn.Sequential(nn.Linear(style_dim, 8), nn.ReLU())
        if arm == "PS_film":
            self.film = nn.Linear(8, 16)
            head_in = 8 + 8 + 8 + 4
        elif arm == "P_tcn":
            head_in = 8 + 8 + 4
        elif arm == "S_mlp":
            head_in = 8 + 8
        else:
            head_in = 8
        self.head = nn.Linear(head_in, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, d, seq, mask, style):
        d_emb = self.d_net(d)
        parts = [d_emb]
        if self.arm in {"P_tcn", "PS_film"}:
            p = self.phys_net(seq * mask.unsqueeze(-1)).mean(dim=2)
            if self.arm == "PS_film":
                s_emb = self.style_net(style)
                gamma, beta = self.film(s_emb).chunk(2, dim=1)
                p = p * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * beta
                parts.extend([p, s_emb, mask])
            else:
                parts.extend([p, mask])
        elif self.arm == "S_mlp":
            parts.append(self.style_net(style))
        logits = self.head(torch.cat(parts, dim=1))
        return logits - logits.mean(dim=1, keepdim=True)


def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_tabular(fit, val):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    return scaler.fit_transform(imp.fit_transform(fit)), scaler.transform(imp.transform(val))


def train_predict(arm, d_fit, d_val, seq_fit, seq_val, mask_fit, mask_val, style_fit, style_val, target, subjects, seed):
    seed = int(seed)
    seed_all(seed)
    d_fit, d_val = prepare_tabular(d_fit, d_val)
    style_fit, style_val = prepare_tabular(style_fit, style_val)
    model = RegretStudent(arm, d_fit.shape[1], style_fit.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    weights = base.subject_weights(subjects).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(d_fit.astype(np.float32)),
        torch.from_numpy(seq_fit.astype(np.float32)),
        torch.from_numpy(mask_fit.astype(np.float32)),
        torch.from_numpy(style_fit.astype(np.float32)),
        torch.from_numpy(target.astype(np.float32)),
        torch.from_numpy(weights),
    )
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model.train()
    for _ in range(EPOCHS):
        for d, seq, mask, style, y, w in loader:
            d, seq, mask, style, y, w = [x.to(DEVICE) for x in (d, seq, mask, style, y, w)]
            pred = model(d, seq, mask, style)
            loss = (((pred - y) ** 2).mean(dim=1) * w).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(d_val.astype(np.float32)).to(DEVICE),
            torch.from_numpy(seq_val.astype(np.float32)).to(DEVICE),
            torch.from_numpy(mask_val.astype(np.float32)).to(DEVICE),
            torch.from_numpy(style_val.astype(np.float32)).to(DEVICE),
        ).cpu().numpy()
    return logits, sum(p.numel() for p in model.parameters())


def main():
    print(f"[Run64 TCN inner screen] device={DEVICE} torch={torch.__version__}")
    pfull = pd.read_csv(base.PFULL_PATH)
    pfull_index = pfull.set_index("event_uid", verify_integrity=True)
    features = pd.read_csv(base.FEATURE_PATH).set_index("event_uid", verify_integrity=True)
    inner_all = pd.read_csv(base.INNER_PATH)
    # 缓存由本项目本轮脚本生成；event_uid沿用pandas object字符串，因此本地读取
    # 需要allow_pickle。数值序列和mask仍为普通float32数组。
    cache = np.load(SEQ_PATH, allow_pickle=True)
    seq_ids = cache["event_uid"].astype(str)
    seq_map = {uid: i for i, uid in enumerate(seq_ids)}
    sequence = cache["sequence"].astype(np.float32)
    channel_mask = cache["channel_mask"].astype(np.float32)
    rows = []
    for outer_fold in range(1, 6):
        frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        ids = frame["event_uid"].astype(str).to_numpy()
        idx = np.asarray([seq_map[x] for x in ids], dtype=int)
        seq = sequence[idx]
        mask = channel_mask[idx]
        curves = base.load_inner_curves(frame)
        truth = pfull_index.loc[ids, base.truth_columns()].to_numpy(float)
        regret, losses = base.centered_regret(curves, truth)
        tau = max(0.5, float(np.median(np.std(losses, axis=1))))
        teacher_logits = -regret / tau
        teacher_logits -= teacher_logits.mean(axis=1, keepdims=True)
        d = base.disagreement_features(curves)
        style = features.reindex(pd.Index(ids))[base.STYLE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        base_curve = curves.mean(axis=1)
        base_error = base.event_mae(base_curve, truth)
        for arm in ARMS:
            pred_curve = np.full_like(truth, np.nan)
            params = 0
            for inner_fold in sorted(frame["inner_fold"].unique()):
                val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
                fit = ~val
                logits, params = train_predict(
                    arm,
                    d[fit],
                    d[val],
                    seq[fit],
                    seq[val],
                    mask[fit],
                    mask[val],
                    style[fit],
                    style[val],
                    teacher_logits[fit],
                    frame.loc[fit, "subject"].astype(str).to_numpy(),
                    base.SEED + outer_fold * 100 + inner_fold * 10 + ARMS.index(arm),
                )
                q = base.stable_softmax(logits)
                weights = (1.0 - base.TRUST_UPDATE) / 3.0 + base.TRUST_UPDATE * q
                pred_curve[val] = base.curves_from_weights(curves[val], weights)
            model_error = base.event_mae(pred_curve, truth)
            cert = base.certification(
                frame["subject"].astype(str).to_numpy(),
                base_error,
                model_error,
                base.SEED + 21000 + outer_fold * 100 + ARMS.index(arm),
            )
            cert.update(
                {
                    "outer_fold": outer_fold,
                    "arm": arm,
                    "parameter_count": params,
                    "event_improved_fraction": float(np.mean(model_error < base_error - 1e-12)),
                    "physio_any_valid_fraction": float(np.mean(mask.any(axis=1))),
                    "physio_all_valid_fraction": float(np.mean(mask.all(axis=1))),
                    "device": str(DEVICE),
                }
            )
            rows.append(cert)
            print(
                f"outer={outer_fold} arm={arm} gain={cert['subject_macro_mae_improvement_deg']:+.4f} "
                f"ci_lo={cert['bootstrap_ci_lower_deg']:+.4f} params={params}"
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "tables" / "inner_tcn_screen.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby("arm", as_index=False)
        .agg(
            mean_inner_gain_deg=("subject_macro_mae_improvement_deg", "mean"),
            min_inner_gain_deg=("subject_macro_mae_improvement_deg", "min"),
            outer_positive_count=("subject_macro_mae_improvement_deg", lambda x: int((x > 0).sum())),
            outer_base_gate_pass_count=("base_gate_pass", "sum"),
            mean_ci_lower_deg=("bootstrap_ci_lower_deg", "mean"),
            mean_leave_top_gain_deg=("leave_top_subject_improvement_deg", "mean"),
            mean_event_improved_fraction=("event_improved_fraction", "mean"),
            parameter_count=("parameter_count", "max"),
        )
        .sort_values(["mean_inner_gain_deg", "outer_positive_count"], ascending=False)
    )
    summary.to_csv(OUT_DIR / "tables" / "inner_tcn_summary.csv", index=False, encoding="utf-8-sig")
    d_gain = float(summary.loc[summary["arm"] == "D_tiny", "mean_inner_gain_deg"].iloc[0])
    for arm in ("P_tcn", "S_mlp", "PS_film"):
        summary.loc[summary["arm"] == arm, "gain_over_D_tiny_deg"] = (
            summary.loc[summary["arm"] == arm, "mean_inner_gain_deg"] - d_gain
        )
    decision = {
        "status": "inner_only_gpu_screen_complete",
        "outer_test_new_candidates_opened": False,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "best_arm": summary.iloc[0].to_dict(),
        "all_arms": summary.to_dict(orient="records"),
        "advance_rule": "A modality arm must beat D_tiny by >=0.02 deg mean inner subject-macro gain and be positive in at least 4/5 outer contexts.",
    }
    (OUT_DIR / "outputs" / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "outputs" / "RESULT_CN.md").write_text(
        "# Run64 原始生理TCN与风格FiLM训练侧筛选\n\n"
        "仅使用outer训练侧inner-OOF；固定小模型与固定训练轮数。\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
