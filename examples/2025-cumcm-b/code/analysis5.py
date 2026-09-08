# -*- coding: utf-8 -*-
"""v4 修订实验：MC 200次×4档噪声 + Si残余色散敏感性 + 发散角/楔角量化"""
import json, os
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

OUT = "/Users/gongzhen/.zcode/workspace/default/sic_review"
FIG = os.path.join(OUT, "figs")
R4 = {}

code = open(os.path.join(OUT, "analysis.py"), encoding="utf-8").read()
head = code.split("# ============================================================\n# 1) SiC")[0]
exec(head)

# 基线数据（附件1, 10°）
sig1, rr1 = load("附件1.xlsx")
sig1, rr1, sm1, base1 = prep(sig1, rr1, 1200, 3300)

# ---------- 实验 1：蒙特卡洛 200 次 × 噪声四档 ----------
print("[MC] 200 次 × 噪声 {0.1%, 0.3%, 0.5%, 1.0%} ...")
rng = np.random.default_rng(11)
noise_levels = [0.1, 0.3, 0.5, 1.0]
mc_table = []
for nl in noise_levels:
    vals, fails = [], 0
    for k in range(200):
        rr_n = rr1 + rng.normal(0, nl, len(rr1))
        sm_n = savgol_filter(rr_n, 31, 3)
        base_n = savgol_filter(rr_n, 301, 2)
        try:
            pk_n, vy_n = extrema(sig1, sm_n, prom=0.35, dist=12, base=base_n)
            idxn = np.sort(np.concatenate([pk_n, vy_n]))
            if len(idxn) < 6:
                fails += 1; continue
            b = scan_sic_series(sig1[idxn], 10.0)
            if 5.0 < b["d_um"] < 10.0:
                vals.append(b["d_um"])
            else:
                fails += 1
        except Exception:
            fails += 1
    v = np.array(vals)
    med = float(np.median(v)); mad = float(np.median(np.abs(v - med)) * 1.4826)
    mc_table.append(dict(noise=nl, n_ok=int(len(v)), n_fail=int(fails),
                         median=med, mad=mad, mean=float(v.mean()) if len(v) else None))
    print(f"  {nl}%: 成功 {len(v)}/200, median={med:.4f}±{mad:.4f} um, 失败率={fails/2:.1f}%")
R4["mc"] = mc_table

fig, ax = plt.subplots(figsize=(6.8, 3.4))
xs = np.arange(len(noise_levels))
meds = [m["median"] for m in mc_table]
mads = [m["mad"] for m in mc_table]
ax.errorbar(xs, meds, yerr=mads, fmt="o-", color="#1f4e79", lw=1.4, ms=6,
            capsize=4, label="中位数 ± MAD（200 次）")
ax.axhline(7.202, color="#c00000", ls="--", lw=1.2, label="干净数据反演值 7.202 μm")
ax.set_xticks(xs); ax.set_xticklabels([f"{n}%" for n in noise_levels])
ax.set_xlabel("注入反射率噪声标准差"); ax.set_ylabel("反演厚度 (μm)")
ax.legend(fontsize=9, frameon=False); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_mc_levels.png", dpi=200); plt.close(fig)
print("fig_mc_levels.png 已生成")

# ---------- 实验 2：Si 外延层残余色散敏感性 ----------
print("[Si色散] B 项扰动对 Si 厚度的影响")
def si_series_B(sig_e, theta1, B, base_n=3.42):
    n = base_n + B * np.asarray(sig_e, float)**2
    return series_fit_generic_local(sig_e, theta1, n)

def series_fit_generic_local(sig_e, theta1, n_values):
    c2 = np.sqrt(np.clip(1.0 - np.sin(np.radians(theta1))**2 / np.asarray(n_values)**2, 0, None))
    x = 2.0 * np.asarray(n_values) * c2 * sig_e * 1e-4
    N = len(x)
    slope0 = 0.5 * (N - 1) / max(x[-1] - x[0], 1e-12)
    m0 = float(np.round(np.median(0.5 * np.arange(N) - slope0 * x) * 2) / 2)
    m = m0 + 0.5 * np.arange(N)
    A = np.column_stack([x, np.ones(N)])
    coef, *_ = np.linalg.lstsq(A, m, rcond=None)
    return float(coef[0])

si = {}
for tag, fname, th in [("a3", "附件3.xlsx", 10.0), ("a4", "附件4.xlsx", 15.0)]:
    sig, rr = load(fname)
    sig, rr, sm, base = prep(sig, rr, 700, 4000, win=25)
    pk, vy = extrema(sig, sm, prom=0.6, dist=12, base=base)
    idx = np.sort(np.concatenate([pk, vy]))
    se = sig[idx]
    d0 = si_series_B(se, th, 0.0)
    dP = si_series_B(se, th, 2e-7)   # +Δn≈0.32% @4000
    dM = si_series_B(se, th, -2e-7)
    si[tag] = dict(d0=d0, dP=dP, dM=dM, sens_pct=float(100*max(abs(dP-d0), abs(dM-d0))/d0))
    print(f"  {tag}: B=0: {d0:.4f} | +2e-7: {dP:.4f} | -2e-7: {dM:.4f} (最大偏移 {si[tag]['sens_pct']:.3f}%)")
R4["si_dispersion"] = si

# ---------- 实验 3：发散角与楔角的解析量化 ----------
print("[发散角/楔角]")
# 3a. 入射角发散 ±1°（θ1=10°, σ=2500, n 由 TO-LO）
best = scan_sic_series(np.sort(np.concatenate([sic_pk := [], []])), 10.0) if False else None
# 重建 TO-LO n（eps_inf=6.90）
nfun = sic_n_factory(6.90)
d_um, th1, sig_x = 7.202, 10.0, 2500.0
opd = lambda th1x: 2 * d_um * 1e-4 * float(np.asarray(nfun(sig_x))) * np.cos(np.arcsin(np.sin(np.radians(th1x)) / float(np.asarray(nfun(sig_x))))) * sig_x  # 无量纲相位系数
# 相位 φ=2π·OPD·σ? 用 OPD(μm)：Δ=2d√(n²−sin²θ) [cm]
opd_cm = lambda th1x: 2 * d_um * 1e-4 * np.sqrt(float(nfun(sig_x))**2 - np.sin(np.radians(th1x))**2)
dopd = (opd_cm(11.0) - opd_cm(9.0)) / (2 * np.radians(2.0))     # dOPD/dθ1 (cm/rad)
delta_th = np.radians(1.0)
d_opd_1deg = abs(dopd) * delta_th * 1e4                            # μm per ±1°
phase_frac = d_opd_1deg / (opd_cm(10.0) * 1e4)                     # 相对光程差扰动
d_err_pct = 100 * phase_frac                                       # 厚度相对误差近似
R4["divergence"] = dict(d_opd_1deg_um=float(d_opd_1deg), phase_frac=float(phase_frac), d_err_pct=float(d_err_pct))
print(f"  θ1±1° → 光程差变化 {d_opd_1deg:.4f} μm（相对 {phase_frac*100:.3f}%）→ 厚度误差 ≈ {d_err_pct:.3f}%")

# 3b. 楔角（厚度不均匀）导致条纹消失的临界 Δd
for sig_w in [2500.0, 3300.0]:
    n_w = float(nfun(sig_w))
    dd_crit = 1.0 / (4.0 * sig_w * n_w * np.sqrt(1 - np.sin(np.radians(10.0))**2 / n_w**2)) * 1e4
    print(f"  σ={sig_w:.0f}: 条纹洗白临界厚度不均匀 Δd ≈ {dd_crit:.4f} μm（跨光斑）")
    R4[f"wedge_{int(sig_w)}"] = float(dd_crit)
print("完成 results4 部分")

with open(os.path.join(OUT, "results4.json"), "w", encoding="utf-8") as f:
    json.dump(R4, f, ensure_ascii=False, indent=2, default=float)
print("results4.json 已保存")
