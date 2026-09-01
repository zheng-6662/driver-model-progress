from __future__ import annotations

"""Run65：完整20点多尺度残差教师—学生训练侧筛选。

冻结B_all3作为先验。车辆-only、+生理、+风格、联合学生都预测20点残差；完整
教师额外读取事件后0–5秒生理（训练期特权信息）。VPS_KD学生在相同因果输入上
同时拟合真值残差与教师残差/事件关系。脚本只运行outer-train的inner-OOF，不生成
新候选outer-test结果。
"""

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run65_multimodal_residual_distillation_20260830"
RUN64 = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
RUN57 = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run57_a_full_release_population_causal_baseline_20260827" / "run_1"


def load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run64 = load_local("run64_helpers_for_run65", RUN64 / "experiment.py")

PFULL_PATH = run64.PFULL_PATH
INNER_PATH = run64.INNER_PATH
CAUSAL_CACHE = RUN57 / "tables" / "causal_input_cache.npz"
PRE_SEQ_PATH = RUN64 / "cache" / "physio_sequence_10hz.npz"
POST_SEQ_PATH = RUN64 / "cache" / "physio_post_sequence_10hz_teacher_only.npz"
PRIOR_STYLE_PATH = RUN64 / "tables" / "multimodal_features.csv"
RECENT_STYLE_PATH = RUN64 / "tables" / "recent_style_features.csv"

SEED = 20260830
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-2
TRUST = 0.25
KD_WEIGHT = 0.5
RELATION_WEIGHT = 0.05
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ARMS = ("T_privileged", "V_vehicle", "VP_physio", "VS_style", "VPS_joint", "VPS_KD")


@dataclass
class Bundle:
    event_ids: np.ndarray
    subjects: np.ndarray
    summary: np.ndarray
    vehicle_seq: np.ndarray
    pre_seq: np.ndarray
    pre_mask: np.ndarray
    post_seq: np.ndarray
    post_mask: np.ndarray
    style: np.ndarray
    base_curve: np.ndarray
    truth: np.ndarray


@dataclass
class FoldData:
    summary: np.ndarray
    vehicle_seq: np.ndarray
    pre_seq: np.ndarray
    pre_mask: np.ndarray
    post_seq: np.ndarray
    post_mask: np.ndarray
    style: np.ndarray
    base_curve: np.ndarray
    target_norm: np.ndarray
    weights: np.ndarray


class SeqEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, dilation=2, padding=4),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return torch.cat([z.mean(dim=2), z.amax(dim=2)], dim=1)


class ResidualNet(nn.Module):
    def __init__(self, arm: str, summary_dim: int, style_dim: int):
        super().__init__()
        self.arm = arm
        self.use_phys = arm in {"T_privileged", "VP_physio", "VPS_joint", "VPS_KD"}
        self.use_style = arm in {"T_privileged", "VS_style", "VPS_joint", "VPS_KD"}
        self.use_post = arm == "T_privileged"
        self.summary_net = nn.Sequential(nn.Linear(summary_dim, 32), nn.GELU())
        self.vehicle_encoder = SeqEncoder(9, 16)
        self.base_net = nn.Sequential(nn.Linear(20, 16), nn.GELU())
        parts = 32 + 32 + 16
        if self.use_phys:
            self.phys_encoder = SeqEncoder(4, 8)
            parts += 16 + 4
        if self.use_style:
            self.style_net = nn.Sequential(nn.Linear(style_dim, 16), nn.GELU())
            parts += 16
        if self.use_phys and self.use_style:
            self.style_to_film = nn.Linear(16, 32)
        if self.use_post:
            self.post_encoder = SeqEncoder(4, 8)
            parts += 16 + 4
        self.trunk = nn.Sequential(
            nn.Linear(parts, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.head = nn.Linear(32, 20)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, summary, vehicle_seq, pre_seq, pre_mask, post_seq, post_mask, style, base_curve):
        pieces = [self.summary_net(summary), self.vehicle_encoder(vehicle_seq), self.base_net(base_curve)]
        style_emb = self.style_net(style) if self.use_style else None
        if self.use_phys:
            phys = self.phys_encoder(pre_seq * pre_mask.unsqueeze(-1))
            if style_emb is not None:
                gamma, beta = self.style_to_film(style_emb).chunk(2, dim=1)
                phys = phys * (1.0 + 0.20 * torch.tanh(gamma)) + 0.20 * beta
            pieces.extend([phys, pre_mask])
        if style_emb is not None:
            pieces.append(style_emb)
        if self.use_post:
            post = self.post_encoder(post_seq * post_mask.unsqueeze(-1))
            pieces.extend([post, post_mask])
        latent = self.trunk(torch.cat(pieces, dim=1))
        return self.head(latent), latent


def seed_all(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def truth_cols() -> list[str]:
    return [f"target_t{i:02d}_deg" for i in range(1, 21)]


def load_cache_by_uid(path: Path, value_key: str):
    z = np.load(path, allow_pickle=True)
    ids = z["event_uid"].astype(str)
    return {uid: i for i, uid in enumerate(ids)}, z[value_key].astype(np.float32), z["channel_mask"].astype(np.float32)


def load_bundle(frame: pd.DataFrame, pfull: pd.DataFrame, causal: dict, prior: pd.DataFrame, recent: pd.DataFrame, pre_cache, post_cache) -> Bundle:
    ids = frame["event_uid"].astype(str).to_numpy()
    p_pos = pfull["row_pos"].reindex(pd.Index(ids)).to_numpy(int)
    pre_map, pre_seq_all, pre_mask_all = pre_cache
    post_map, post_seq_all, post_mask_all = post_cache
    pre_idx = np.asarray([pre_map[x] for x in ids], dtype=int)
    post_idx = np.asarray([post_map[x] for x in ids], dtype=int)
    prior_cols = run64.STYLE_COLS
    recent_cols = [c for c in recent.columns if c.startswith("recent_")]
    style = np.concatenate(
        [
            prior.reindex(pd.Index(ids))[prior_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float),
            recent.reindex(pd.Index(ids))[recent_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float),
        ],
        axis=1,
    )
    curves = run64.load_inner_curves(frame)
    base_curve = curves.mean(axis=1)
    truth = pfull.loc[ids, truth_cols()].to_numpy(float)
    return Bundle(
        event_ids=ids,
        subjects=frame["subject"].astype(str).to_numpy(),
        summary=causal["summary"][p_pos].astype(float),
        vehicle_seq=causal["sequence"][p_pos].astype(np.float32),
        pre_seq=pre_seq_all[pre_idx],
        pre_mask=pre_mask_all[pre_idx],
        post_seq=post_seq_all[post_idx],
        post_mask=post_mask_all[post_idx],
        style=style,
        base_curve=base_curve,
        truth=truth,
    )


class Preprocessor:
    def __init__(self):
        self.summary_imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        self.summary_scaler = StandardScaler()
        self.style_imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        self.style_scaler = StandardScaler()
        self.v_mean = None
        self.v_std = None
        self.base_mean = None
        self.base_std = None
        self.target_mean = None
        self.target_std = None

    def fit(self, b: Bundle, idx: np.ndarray):
        self.summary_scaler.fit(self.summary_imp.fit_transform(b.summary[idx]))
        self.style_scaler.fit(self.style_imp.fit_transform(b.style[idx]))
        v = b.vehicle_seq[idx]
        self.v_mean = np.nanmean(v, axis=(0, 2), keepdims=True)
        self.v_std = np.nanstd(v, axis=(0, 2), keepdims=True)
        self.v_std = np.where(self.v_std > 1e-6, self.v_std, 1.0)
        self.base_mean = np.mean(b.base_curve[idx], axis=0, keepdims=True)
        self.base_std = np.std(b.base_curve[idx], axis=0, keepdims=True)
        self.base_std = np.where(self.base_std > 1e-6, self.base_std, 1.0)
        residual = b.truth[idx] - b.base_curve[idx]
        self.target_mean = np.mean(residual, axis=0, keepdims=True)
        self.target_std = np.std(residual, axis=0, keepdims=True)
        self.target_std = np.where(self.target_std > 1e-6, self.target_std, 1.0)
        return self

    def transform(self, b: Bundle, idx: np.ndarray) -> FoldData:
        summary = self.summary_scaler.transform(self.summary_imp.transform(b.summary[idx])).astype(np.float32)
        style = self.style_scaler.transform(self.style_imp.transform(b.style[idx])).astype(np.float32)
        vehicle = np.nan_to_num((b.vehicle_seq[idx] - self.v_mean) / self.v_std, nan=0.0).astype(np.float32)
        base_curve = ((b.base_curve[idx] - self.base_mean) / self.base_std).astype(np.float32)
        target = ((b.truth[idx] - b.base_curve[idx] - self.target_mean) / self.target_std).astype(np.float32)
        return FoldData(
            summary=summary,
            vehicle_seq=vehicle,
            pre_seq=b.pre_seq[idx].astype(np.float32),
            pre_mask=b.pre_mask[idx].astype(np.float32),
            post_seq=b.post_seq[idx].astype(np.float32),
            post_mask=b.post_mask[idx].astype(np.float32),
            style=style,
            base_curve=base_curve,
            target_norm=target,
            weights=run64.subject_weights(b.subjects[idx]).astype(np.float32),
        )

    def invert_residual(self, pred_norm: np.ndarray) -> np.ndarray:
        return pred_norm * self.target_std + self.target_mean


def tensor_dataset(d: FoldData) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(d.summary),
        torch.from_numpy(d.vehicle_seq),
        torch.from_numpy(d.pre_seq),
        torch.from_numpy(d.pre_mask),
        torch.from_numpy(d.post_seq),
        torch.from_numpy(d.post_mask),
        torch.from_numpy(d.style),
        torch.from_numpy(d.base_curve),
        torch.from_numpy(d.target_norm),
        torch.from_numpy(d.weights),
    )


def multiscale_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    point = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
    diff = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], reduction="none").mean(dim=1)
    p2 = F.avg_pool1d(pred.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
    t2 = F.avg_pool1d(target.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)
    p4 = F.avg_pool1d(pred.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)
    t4 = F.avg_pool1d(target.unsqueeze(1), kernel_size=4, stride=4).squeeze(1)
    coarse = F.smooth_l1_loss(p2, t2, reduction="none").mean(dim=1) + F.smooth_l1_loss(p4, t4, reduction="none").mean(dim=1)
    return ((point + 0.20 * diff + 0.10 * coarse) * weights).mean()


def relation_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    s = F.normalize(student - student.mean(dim=1, keepdim=True), dim=1)
    t = F.normalize(teacher - teacher.mean(dim=1, keepdim=True), dim=1)
    return F.mse_loss(s @ s.T, t @ t.T)


def train_model(arm: str, train: FoldData, seed: int, teacher: ResidualNet | None = None) -> ResidualNet:
    seed_all(seed)
    model = ResidualNet(arm, train.summary.shape[1], train.style.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(tensor_dataset(train), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(int(seed)))
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    model.train()
    for _ in range(EPOCHS):
        for batch in loader:
            summary, vehicle, pre, pre_mask, post, post_mask, style, base_curve, target, weights = [x.to(DEVICE) for x in batch]
            pred, _ = model(summary, vehicle, pre, pre_mask, post, post_mask, style, base_curve)
            loss = multiscale_loss(pred, target, weights)
            if teacher is not None:
                with torch.no_grad():
                    t_pred, _ = teacher(summary, vehicle, pre, pre_mask, post, post_mask, style, base_curve)
                loss = loss + KD_WEIGHT * multiscale_loss(pred, t_pred, weights) + RELATION_WEIGHT * relation_loss(pred, t_pred)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def predict(model: ResidualNet, d: FoldData) -> np.ndarray:
    model.eval()
    dataset = tensor_dataset(d)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    outputs = []
    with torch.inference_mode():
        for batch in loader:
            summary, vehicle, pre, pre_mask, post, post_mask, style, base_curve, _, _ = [x.to(DEVICE) for x in batch]
            pred, _ = model(summary, vehicle, pre, pre_mask, post, post_mask, style, base_curve)
            outputs.append(pred.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main(out_dir: Path) -> int:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)
    (out_dir / "outputs").mkdir(exist_ok=True)
    print(f"[Run65 inner screen] device={DEVICE} epochs={EPOCHS} batch={BATCH_SIZE}")
    pfull_frame = pd.read_csv(PFULL_PATH)
    pfull_frame["row_pos"] = np.arange(len(pfull_frame))
    pfull = pfull_frame.set_index("event_uid", verify_integrity=True)
    causal_npz = np.load(CAUSAL_CACHE, allow_pickle=True)
    causal = {"summary": causal_npz["summary"], "sequence": causal_npz["sequence"]}
    prior = pd.read_csv(PRIOR_STYLE_PATH).set_index("event_uid", verify_integrity=True)
    recent = pd.read_csv(RECENT_STYLE_PATH).set_index("event_uid", verify_integrity=True)
    pre_cache = load_cache_by_uid(PRE_SEQ_PATH, "sequence")
    post_cache = load_cache_by_uid(POST_SEQ_PATH, "post_sequence")
    inner_all = pd.read_csv(INNER_PATH)
    rows = []
    for outer_fold in range(1, 6):
        frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        bundle = load_bundle(frame, pfull, causal, prior, recent, pre_cache, post_cache)
        base_error = run64.event_mae(bundle.base_curve, bundle.truth)
        fold_preds = {arm: np.full_like(bundle.truth, np.nan) for arm in ARMS}
        params = {}
        for inner_fold in sorted(frame["inner_fold"].unique()):
            val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
            fit = ~val
            prep = Preprocessor().fit(bundle, fit)
            train = prep.transform(bundle, fit)
            valid = prep.transform(bundle, val)
            teacher = train_model(
                "T_privileged", train, SEED + outer_fold * 1000 + inner_fold * 100
            )
            t_norm = predict(teacher, valid)
            t_residual = prep.invert_residual(t_norm)
            fold_preds["T_privileged"][val] = bundle.base_curve[val] + TRUST * t_residual
            params["T_privileged"] = sum(p.numel() for p in teacher.parameters())
            for ai, arm in enumerate(ARMS[1:], start=1):
                kd_teacher = teacher if arm == "VPS_KD" else None
                model = train_model(
                    "VPS_joint" if arm == "VPS_KD" else arm,
                    train,
                    SEED + outer_fold * 1000 + inner_fold * 100 + ai,
                    teacher=kd_teacher,
                )
                pred_norm = predict(model, valid)
                residual = prep.invert_residual(pred_norm)
                fold_preds[arm][val] = bundle.base_curve[val] + TRUST * residual
                params[arm] = sum(p.numel() for p in model.parameters())
        for arm in ARMS:
            err = run64.event_mae(fold_preds[arm], bundle.truth)
            cert = run64.certification(
                bundle.subjects,
                base_error,
                err,
                SEED + 50000 + outer_fold * 100 + ARMS.index(arm),
            )
            cert.update(
                {
                    "outer_fold": outer_fold,
                    "arm": arm,
                    "parameter_count": params[arm],
                    "event_improved_fraction": float(np.mean(err < base_error - 1e-12)),
                    "physio_any_valid_fraction": float(bundle.pre_mask.any(axis=1).mean()),
                    "style_feature_dimension": int(bundle.style.shape[1]),
                    "teacher_uses_post_event_physio": arm == "T_privileged",
                    "deployment_uses_post_event_physio": False,
                }
            )
            rows.append(cert)
            print(
                f"outer={outer_fold} arm={arm} gain={cert['subject_macro_mae_improvement_deg']:+.4f} "
                f"ci_lo={cert['bootstrap_ci_lower_deg']:+.4f} params={params[arm]}"
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "tables" / "inner_multiscale_distillation.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby("arm", as_index=False)
        .agg(
            mean_gain=("subject_macro_mae_improvement_deg", "mean"),
            min_gain=("subject_macro_mae_improvement_deg", "min"),
            positive_outer_count=("subject_macro_mae_improvement_deg", lambda x: int((x > 0).sum())),
            gate_pass_count=("base_gate_pass", "sum"),
            mean_ci_lower=("bootstrap_ci_lower_deg", "mean"),
            mean_leave_top=("leave_top_subject_improvement_deg", "mean"),
            mean_event_improved_fraction=("event_improved_fraction", "mean"),
            parameter_count=("parameter_count", "max"),
        )
        .sort_values("mean_gain", ascending=False)
    )
    v_gain = float(summary.loc[summary["arm"] == "V_vehicle", "mean_gain"].iloc[0])
    summary["gain_over_vehicle_student"] = summary["mean_gain"] - v_gain
    summary.to_csv(out_dir / "tables" / "inner_multiscale_distillation_summary.csv", index=False, encoding="utf-8-sig")
    vp = summary.set_index("arm").loc["VP_physio"]
    vs = summary.set_index("arm").loc["VS_style"]
    vps = summary.set_index("arm").loc["VPS_joint"]
    kd = summary.set_index("arm").loc["VPS_KD"]
    advance = bool(
        vp["gain_over_vehicle_student"] >= 0.02
        and vs["gain_over_vehicle_student"] >= 0.02
        and vps["mean_gain"] >= max(vp["mean_gain"], vs["mean_gain"])
        and kd["mean_gain"] >= vps["mean_gain"]
        and kd["positive_outer_count"] >= 4
    )
    decision = {
        "status": "inner_only_multiscale_distillation_complete",
        "outer_test_opened": False,
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "advance_to_outer": advance,
        "all_arms": summary.to_dict(orient="records"),
        "advance_rule": "VP and VS must each beat V by >=0.02 deg; joint must beat both; KD must not reduce joint gain; >=4/5 outer contexts positive.",
        "elapsed_seconds": float(time.time() - started),
        "boundary": "T_privileged may read post-event physiology only during training-side diagnosis. All deployable arms use prediction-anchor-or-earlier inputs only.",
    }
    (out_dir / "outputs" / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "outputs" / "RESULT_CN.md").write_text(
        "# Run65 多尺度残差教师—学生训练侧筛选\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + f"\n\n是否进入outer：**{advance}**\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="run_1_inner_screen")
    args = parser.parse_args()
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = RUN_DIR / out
    raise SystemExit(main(out))

