from __future__ import annotations

"""Run65 v2：严格双层基预测、成对模态门、影子控制与精确回退。

仅在 outer-train/meta-validation 上运行。基曲线来自 ``nested_base``：当前meta验证
被试永久不进入任何拟合侧基专家，meta-fit残差标签也由内部subject-crossfit产生。
"""

import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run65_multimodal_residual_distillation_20260830"
RUN64 = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
RUN57 = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run57_a_full_release_population_causal_baseline_20260827" / "run_1"


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = load_local("run65_v1_components", RUN_DIR / "experiment.py")
run64 = v1.run64

PFULL_PATH = v1.PFULL_PATH
CAUSAL_CACHE = v1.CAUSAL_CACHE
PRE_SEQ_PATH = v1.PRE_SEQ_PATH
POST_SEQ_PATH = v1.POST_SEQ_PATH
PRIOR_STYLE_PATH = v1.PRIOR_STYLE_PATH
RECENT_STYLE_PATH = v1.RECENT_STYLE_PATH
NESTED_DIR = RUN_DIR / "cache" / "nested_base"

SEED = 20260830
EPOCHS = 40
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-2
ALPHAS = (0.0, 0.10, 0.25)
KD_WEIGHT = 0.5
RELATION_WEIGHT = 0.05
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ARMS = (
    "T_privileged",
    "T_post_shadow",
    "V_vehicle",
    "VP_physio",
    "VP_shadow",
    "VS_style",
    "VS_shadow",
    "VPS_joint",
    "VPS_point",
    "VPS_KD",
)


@dataclass
class Bundle:
    indices: np.ndarray
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
        self.target_scale = None

    def fit(self, b: Bundle):
        self.summary_scaler.fit(self.summary_imp.fit_transform(b.summary))
        self.style_scaler.fit(self.style_imp.fit_transform(b.style))
        self.v_mean = np.nanmean(b.vehicle_seq, axis=(0, 2), keepdims=True)
        self.v_std = np.nanstd(b.vehicle_seq, axis=(0, 2), keepdims=True)
        self.v_std = np.where(self.v_std > 1e-6, self.v_std, 1.0)
        self.base_mean = np.mean(b.base_curve, axis=0, keepdims=True)
        self.base_std = np.std(b.base_curve, axis=0, keepdims=True)
        self.base_std = np.where(self.base_std > 1e-6, self.base_std, 1.0)
        residual = b.truth - b.base_curve
        # 单一全曲线标度保持相邻差分与多尺度平均的物理度数关系；零输出精确代表零残差。
        self.target_scale = float(np.std(residual))
        if not np.isfinite(self.target_scale) or self.target_scale <= 1e-6:
            self.target_scale = 1.0
        return self

    def transform(self, b: Bundle) -> FoldData:
        return FoldData(
            summary=self.summary_scaler.transform(self.summary_imp.transform(b.summary)).astype(np.float32),
            vehicle_seq=np.nan_to_num((b.vehicle_seq - self.v_mean) / self.v_std, nan=0.0).astype(np.float32),
            pre_seq=b.pre_seq.astype(np.float32),
            pre_mask=b.pre_mask.astype(np.float32),
            post_seq=b.post_seq.astype(np.float32),
            post_mask=b.post_mask.astype(np.float32),
            style=self.style_scaler.transform(self.style_imp.transform(b.style)).astype(np.float32),
            base_curve=((b.base_curve - self.base_mean) / self.base_std).astype(np.float32),
            target_norm=((b.truth - b.base_curve) / self.target_scale).astype(np.float32),
            weights=run64.subject_weights(b.subjects).astype(np.float32),
        )

    def invert(self, pred_norm):
        return pred_norm * self.target_scale


def truth_cols():
    return [f"target_t{i:02d}_deg" for i in range(1, 21)]


def load_cache(path, key):
    z = np.load(path, allow_pickle=True)
    ids = z["event_uid"].astype(str)
    return {u: i for i, u in enumerate(ids)}, z[key].astype(np.float32), z["channel_mask"].astype(np.float32)


def make_bundle(indices, base_predictions, pfull_frame, causal, prior, recent, pre_cache, post_cache):
    indices = np.asarray(indices, dtype=int)
    meta = pfull_frame.iloc[indices]
    ids = meta["event_uid"].astype(str).to_numpy()
    pre_map, pre_all, pre_mask_all = pre_cache
    post_map, post_all, post_mask_all = post_cache
    pre_idx = np.asarray([pre_map[x] for x in ids], dtype=int)
    post_idx = np.asarray([post_map[x] for x in ids], dtype=int)
    prior_cols = run64.STYLE_COLS
    recent_cols = [c for c in recent.columns if c.startswith("recent_")]
    style = np.concatenate(
        [
            prior.reindex(pd.Index(ids))[prior_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float),
            recent.reindex(pd.Index(ids))[recent_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float),
        ], axis=1,
    )
    return Bundle(
        indices=indices,
        event_ids=ids,
        subjects=meta["subject"].astype(str).to_numpy(),
        summary=causal["summary"][indices].astype(float),
        vehicle_seq=causal["sequence"][indices].astype(np.float32),
        pre_seq=pre_all[pre_idx],
        pre_mask=pre_mask_all[pre_idx],
        post_seq=post_all[post_idx],
        post_mask=post_mask_all[post_idx],
        style=style,
        base_curve=np.asarray(base_predictions, dtype=float).mean(axis=1),
        truth=meta[truth_cols()].to_numpy(float),
    )


def architecture(arm):
    return {
        "T_post_shadow": "T_privileged",
        "VP_shadow": "VP_physio",
        "VS_shadow": "VS_style",
        "VPS_point": "VPS_joint",
        "VPS_KD": "VPS_joint",
    }.get(arm, arm)


def shuffled(d: FoldData, kind: str, seed: int) -> FoldData:
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(d.summary))
    if kind == "phys":
        return replace(d, pre_seq=d.pre_seq[order], pre_mask=d.pre_mask[order])
    if kind == "style":
        return replace(d, style=d.style[order])
    if kind == "post":
        return replace(d, post_seq=d.post_seq[order], post_mask=d.post_mask[order])
    return d


def dataset(d):
    return TensorDataset(*[
        torch.from_numpy(x) for x in (
            d.summary, d.vehicle_seq, d.pre_seq, d.pre_mask, d.post_seq, d.post_mask,
            d.style, d.base_curve, d.target_norm, d.weights,
        )
    ])


def seed_all(seed):
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def multi_loss(pred, target, weights, point_only=False):
    point = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
    if point_only:
        return (point * weights).mean()
    diff = F.smooth_l1_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], reduction="none").mean(dim=1)
    p2 = F.avg_pool1d(pred.unsqueeze(1), 2, 2).squeeze(1)
    t2 = F.avg_pool1d(target.unsqueeze(1), 2, 2).squeeze(1)
    p4 = F.avg_pool1d(pred.unsqueeze(1), 4, 4).squeeze(1)
    t4 = F.avg_pool1d(target.unsqueeze(1), 4, 4).squeeze(1)
    coarse = F.smooth_l1_loss(p2, t2, reduction="none").mean(dim=1) + F.smooth_l1_loss(p4, t4, reduction="none").mean(dim=1)
    return ((point + 0.20 * diff + 0.10 * coarse) * weights).mean()


def weighted_relation(student, teacher, weights):
    s = F.normalize(student - student.mean(dim=1, keepdim=True), dim=1)
    t = F.normalize(teacher - teacher.mean(dim=1, keepdim=True), dim=1)
    diff = (s @ s.T - t @ t.T) ** 2
    pair = weights[:, None] * weights[None, :]
    return (diff * pair).sum() / pair.sum().clamp_min(1e-12)


def train(arm, train_data, seed, teacher=None):
    seed_all(seed)
    model = v1.ResidualNet(architecture(arm), train_data.summary.shape[1], train_data.style.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loader = DataLoader(dataset(train_data), batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(int(seed)))
    if teacher is not None:
        teacher.eval()
        for p in teacher.parameters(): p.requires_grad_(False)
    point_only = arm == "VPS_point"
    model.train()
    for _ in range(EPOCHS):
        for batch in loader:
            summary, vehicle, pre, pm, post, pom, style, base_curve, target, weights = [x.to(DEVICE) for x in batch]
            pred, _ = model(summary, vehicle, pre, pm, post, pom, style, base_curve)
            loss = multi_loss(pred, target, weights, point_only=point_only)
            if teacher is not None:
                with torch.no_grad(): tpred, _ = teacher(summary, vehicle, pre, pm, post, pom, style, base_curve)
                loss = loss + 0.5 * multi_loss(pred, tpred, weights) + 0.05 * weighted_relation(pred, tpred, weights)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


def predict(model, d):
    model.eval(); out=[]
    with torch.inference_mode():
        for batch in DataLoader(dataset(d), batch_size=256, shuffle=False):
            summary, vehicle, pre, pm, post, pom, style, base_curve, _, _ = [x.to(DEVICE) for x in batch]
            y, _ = model(summary, vehicle, pre, pm, post, pom, style, base_curve)
            out.append(y.cpu().numpy())
    return np.concatenate(out)


def select_alpha(subjects, base_curve, truth, residual):
    base_err = run64.event_mae(base_curve, truth)
    rows=[]
    for alpha in ALPHAS:
        err = run64.event_mae(base_curve + alpha * residual, truth)
        d = run64.subject_delta_frame(subjects, base_err, err)
        rows.append((float(d["improvement"].mean()), alpha, err))
    rows.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return rows[0][1], rows[0][2], rows


def pair_cert(subjects, reference_error, candidate_error, seed):
    return run64.certification(subjects, reference_error, candidate_error, seed)


def main(out_dir: Path):
    started=time.time(); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"tables").mkdir(exist_ok=True); (out_dir/"outputs").mkdir(exist_ok=True)
    print(f"[Run65 v2] device={DEVICE} epochs={EPOCHS}; strict nested base cache", flush=True)
    pfull_frame=pd.read_csv(PFULL_PATH); pfull_frame["row_pos"]=np.arange(len(pfull_frame))
    if len(pfull_frame)!=2323 or pfull_frame["event_uid"].nunique()!=2323 or pfull_frame["subject"].nunique()!=18:
        raise RuntimeError("P_full identity mismatch")
    causal_z=np.load(CAUSAL_CACHE, allow_pickle=True)
    causal={"summary":causal_z["summary"],"sequence":causal_z["sequence"]}
    if causal["summary"].shape!=(2323,172) or causal["sequence"].shape!=(2323,9,101):
        raise RuntimeError("causal cache shape mismatch")
    prior=pd.read_csv(PRIOR_STYLE_PATH).set_index("event_uid",verify_integrity=True)
    recent=pd.read_csv(RECENT_STYLE_PATH).set_index("event_uid",verify_integrity=True)
    pre_cache=load_cache(PRE_SEQ_PATH,"sequence"); post_cache=load_cache(POST_SEQ_PATH,"post_sequence")
    outer_rows=[]; pair_rows=[]; alpha_rows=[]
    for outer in range(1,6):
        checkpoint_arms = out_dir / "tables" / f"outer_{outer}_arms.csv"
        checkpoint_pairs = out_dir / "tables" / f"outer_{outer}_pairs.csv"
        checkpoint_alpha = out_dir / "tables" / f"outer_{outer}_alpha.csv"
        if checkpoint_arms.exists() and checkpoint_pairs.exists() and checkpoint_alpha.exists():
            outer_rows.extend(pd.read_csv(checkpoint_arms).to_dict(orient="records"))
            pair_rows.extend(pd.read_csv(checkpoint_pairs).to_dict(orient="records"))
            alpha_rows.extend(pd.read_csv(checkpoint_alpha).to_dict(orient="records"))
            print(f"outer={outer}: loaded completed checkpoint", flush=True)
            continue
        outer_row_start = len(outer_rows); pair_row_start = len(pair_rows); alpha_row_start = len(alpha_rows)
        expected=set(np.flatnonzero(pfull_frame["outer_fold"].astype(int).to_numpy()!=outer))
        outer_subjects=set(pfull_frame.loc[pfull_frame["outer_fold"].astype(int).eq(outer),"subject"].astype(str))
        store={arm:{} for arm in ARMS}; base_store={}; truth_store={}; subject_store={}
        for meta in (1,2,3):
            path=NESTED_DIR/f"outer_{outer}_meta_{meta}.npz"
            with np.load(path,allow_pickle=False) as z:
                fit_idx=z["fit_indices"].astype(int); val_idx=z["validation_indices"].astype(int)
                fit_pred=z["fit_base_predictions"].astype(float); val_pred=z["validation_base_predictions"].astype(float)
            if set(fit_idx)&set(val_idx) or outer_subjects & set(pfull_frame.iloc[np.r_[fit_idx,val_idx]]["subject"].astype(str)):
                raise RuntimeError("nested context subject/event overlap")
            fit_bundle=make_bundle(fit_idx,fit_pred,pfull_frame,causal,prior,recent,pre_cache,post_cache)
            val_bundle=make_bundle(val_idx,val_pred,pfull_frame,causal,prior,recent,pre_cache,post_cache)
            prep=Preprocessor().fit(fit_bundle); train_data=prep.transform(fit_bundle); val_data=prep.transform(val_bundle)
            context_seed=SEED+outer*100+meta
            teacher=train("T_privileged",train_data,context_seed)
            for arm in ARMS:
                if arm=="T_privileged": model=teacher; val_used=val_data
                else:
                    tr=train_data; va=val_data
                    if arm=="T_post_shadow": tr=shuffled(tr,"post",context_seed+1); va=shuffled(va,"post",context_seed+2)
                    elif arm=="VP_shadow": tr=shuffled(tr,"phys",context_seed+1); va=shuffled(va,"phys",context_seed+2)
                    elif arm=="VS_shadow": tr=shuffled(tr,"style",context_seed+1); va=shuffled(va,"style",context_seed+2)
                    model=train(arm,tr,context_seed,teacher=teacher if arm=="VPS_KD" else None); val_used=va
                residual=prep.invert(predict(model,val_used))
                for g,r in zip(val_idx,residual): store[arm][int(g)]=r
            for li,g in enumerate(val_idx):
                base_store[int(g)]=val_bundle.base_curve[li]; truth_store[int(g)]=val_bundle.truth[li]; subject_store[int(g)]=val_bundle.subjects[li]
        if set(base_store)!=expected or any(set(x)!=expected for x in store.values()):
            raise RuntimeError(f"outer {outer} meta validation coverage mismatch")
        order=np.asarray(sorted(expected),dtype=int)
        subjects=np.asarray([subject_store[int(g)] for g in order]); base_curve=np.stack([base_store[int(g)] for g in order]); truth=np.stack([truth_store[int(g)] for g in order])
        base_err=run64.event_mae(base_curve,truth); selected_errors={}; selected_alpha={}
        for arm in ARMS:
            residual=np.stack([store[arm][int(g)] for g in order])
            alpha,err,candidates=select_alpha(subjects,base_curve,truth,residual)
            selected_errors[arm]=err; selected_alpha[arm]=alpha
            cert=pair_cert(subjects,base_err,err,SEED+60000+outer*100+ARMS.index(arm))
            outer_rows.append({"outer_fold":outer,"arm":arm,"selected_alpha":alpha,**cert,"event_improved_fraction":float(np.mean(err<base_err-1e-12))})
            for gain,a,_ in candidates: alpha_rows.append({"outer_fold":outer,"arm":arm,"alpha":a,"subject_macro_gain_deg":gain,"selected":a==alpha})
        pairs={
            "VP_minus_V":("V_vehicle","VP_physio"),
            "VP_minus_shadow":("VP_shadow","VP_physio"),
            "VS_minus_V":("V_vehicle","VS_style"),
            "VS_minus_shadow":("VS_shadow","VS_style"),
            "VPS_minus_VP":("VP_physio","VPS_joint"),
            "VPS_minus_VS":("VS_style","VPS_joint"),
            "KD_minus_VPS":("VPS_joint","VPS_KD"),
            "Teacher_minus_no_post":("VPS_joint","T_privileged"),
            "Teacher_minus_post_shadow":("T_post_shadow","T_privileged"),
            "Multiscale_minus_point":("VPS_point","VPS_joint"),
        }
        for name,(ref,cand) in pairs.items():
            cert=pair_cert(subjects,selected_errors[ref],selected_errors[cand],SEED+70000+outer*100+list(pairs).index(name))
            pair_rows.append({"outer_fold":outer,"comparison":name,"reference":ref,"candidate":cand,**cert})
        pd.DataFrame(outer_rows[outer_row_start:]).to_csv(checkpoint_arms,index=False,encoding="utf-8-sig")
        pd.DataFrame(pair_rows[pair_row_start:]).to_csv(checkpoint_pairs,index=False,encoding="utf-8-sig")
        pd.DataFrame(alpha_rows[alpha_row_start:]).to_csv(checkpoint_alpha,index=False,encoding="utf-8-sig")
        print(f"outer={outer} "+" ".join(f"{a}:{outer_rows[-len(ARMS)+i]['subject_macro_mae_improvement_deg']:+.3f}@{selected_alpha[a]:.2f}" for i,a in enumerate(ARMS)),flush=True)
    outer_df=pd.DataFrame(outer_rows); pair_df=pd.DataFrame(pair_rows); alpha_df=pd.DataFrame(alpha_rows)
    outer_df.to_csv(out_dir/"tables"/"arm_results.csv",index=False,encoding="utf-8-sig"); pair_df.to_csv(out_dir/"tables"/"paired_modality_gates.csv",index=False,encoding="utf-8-sig"); alpha_df.to_csv(out_dir/"tables"/"alpha_selection.csv",index=False,encoding="utf-8-sig")
    arm_summary=outer_df.groupby("arm",as_index=False).agg(mean_gain_deg=("subject_macro_mae_improvement_deg","mean"),min_gain_deg=("subject_macro_mae_improvement_deg","min"),positive_outer_count=("subject_macro_mae_improvement_deg",lambda x:int((x>0).sum())),gate_pass_count=("base_gate_pass","sum"),mean_ci_lower=("bootstrap_ci_lower_deg","mean"),mean_leave_top=("leave_top_subject_improvement_deg","mean"),mean_event_improved_fraction=("event_improved_fraction","mean"),mean_selected_alpha=("selected_alpha","mean")).sort_values("mean_gain_deg",ascending=False)
    pair_summary=pair_df.groupby("comparison",as_index=False).agg(mean_pair_gain_deg=("subject_macro_mae_improvement_deg","mean"),min_pair_gain_deg=("subject_macro_mae_improvement_deg","min"),positive_outer_count=("subject_macro_mae_improvement_deg",lambda x:int((x>0).sum())),pair_gate_pass_count=("base_gate_pass","sum"),mean_ci_lower=("bootstrap_ci_lower_deg","mean"),mean_leave_top=("leave_top_subject_improvement_deg","mean")).sort_values("mean_pair_gain_deg",ascending=False)
    arm_summary.to_csv(out_dir/"tables"/"arm_summary.csv",index=False,encoding="utf-8-sig"); pair_summary.to_csv(out_dir/"tables"/"paired_modality_gate_summary.csv",index=False,encoding="utf-8-sig")
    ps=pair_summary.set_index("comparison")
    required=["VP_minus_V","VP_minus_shadow","VS_minus_V","VS_minus_shadow","VPS_minus_VP","VPS_minus_VS","KD_minus_VPS","Teacher_minus_no_post","Teacher_minus_post_shadow"]
    advance=all(ps.loc[k,"mean_pair_gain_deg"]>=0.02 and ps.loc[k,"positive_outer_count"]>=4 and ps.loc[k,"mean_leave_top"]>0 for k in required)
    decision={"status":"strict_inner_multiscale_distillation_complete","outer_test_opened":False,"advance_to_outer":bool(advance),"device":str(DEVICE),"gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"arm_summary":arm_summary.to_dict(orient="records"),"paired_gate_summary":pair_summary.to_dict(orient="records"),"required_pair_gates":required,"alpha_candidates":list(ALPHAS),"exact_fallback_available":True,"elapsed_seconds":float(time.time()-started),"boundary":"Strict nested subject-disjoint base predictions. Post-event physiology is teacher-only. Modality claims require paired gains over vehicle and shuffled shadows."}
    (out_dir/"outputs"/"decision.json").write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding="utf-8")
    (out_dir/"outputs"/"RESULT_CN.md").write_text("# Run65 v2 严格训练侧结果\n\n## 模型臂\n\n"+arm_summary.to_markdown(index=False,floatfmt=".4f")+"\n\n## 成对模态门\n\n"+pair_summary.to_markdown(index=False,floatfmt=".4f")+f"\n\n是否进入outer：**{advance}**\n",encoding="utf-8")
    print(json.dumps(decision,ensure_ascii=False,indent=2)); return 0


if __name__=="__main__":
    out=RUN_DIR/"run_1_inner_screen_v2"; raise SystemExit(main(out))
