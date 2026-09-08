# -*- coding: utf-8 -*-
"""v3 修订实验：D3 折射率通道归因 + 波峰/峰谷对比（薛毅5.1）+ Finesse 精确值"""
import json, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

OUT = "/Users/gongzhen/.zcode/workspace/default/sic_review"
FIG = os.path.join(OUT, "figs")
R3 = {}

code = open(os.path.join(OUT, "analysis.py"), encoding="utf-8").read()
head = code.split("# ============================================================\n# 1) SiC")[0]
exec(head)

sic = {}
for tag, fname, th in [("a1", "附件1.xlsx", 10.0), ("a2", "附件2.xlsx", 15.0)]:
    sig, rr = load(fname)
    sig, rr, sm, base = prep(sig, rr, 1200, 3300)
    pk, vy = extrema(sig, sm, prom=0.35, dist=12, base=base)
    sic[tag] = dict(sig=sig, rr=rr, sm=sm, base=base, pk=pk, vy=vy, th=th)

best = {}
for tag in ["a1", "a2"]:
    d = sic[tag]
    idx = np.sort(np.concatenate([d["pk"], d["vy"]]))
    best[tag] = scan_sic_series(d["sig"][idx], d["th"])
n_olo = {tag: sic_n_factory(best[tag]["eps_inf"]) for tag in ["a1", "a2"]}

# ============ 实验 1：包络法折射率通道（薛毅 式(18)-(23)） ============
def envelope_n(d):
    """由上下包络反演 n(σ)（R 取小数），返回 (σ_grid, n_env, n_sub)"""
    sig_pk, val_pk = d["sig"][d["pk"]], d["sm"][d["pk"]]
    sig_vy, val_vy = d["sig"][d["vy"]], d["sm"][d["vy"]]
    sg = np.linspace(max(sig_pk.min(), sig_vy.min()), min(sig_pk.max(), sig_vy.max()), 300)
    Rmax = np.interp(sg, sig_pk, val_pk) / 100.0
    Rmin = np.interp(sg, sig_vy, val_vy) / 100.0
    A = (np.sqrt(Rmax) + np.sqrt(Rmin)) / 2.0
    B = (np.sqrt(Rmax) - np.sqrt(Rmin)) / 2.0
    n_env = (1 + A) / (1 - A)
    n_sub = n_env * (1 - A**2 - B) / (1 - A**2 + B)
    return sg, n_env, n_sub

env = {}
for tag in ["a1", "a2"]:
    sg, n_env, n_sub = envelope_n(sic[tag])
    env[tag] = dict(sg=sg, n=n_env, n_sub=n_sub)
    # 薛毅表1对照点：1500-3500 每 250
    pts = [1500, 2000, 2500, 3000, 3500]
    vals = [float(n_env[np.argmin(np.abs(sg - p))]) for p in pts if p >= sg.min() and p <= sg.max()]
    print(f"[包络n] {tag}: n(1500..3500) = " + ", ".join(f"{v:.2f}" for v in vals))

# 级数标定 + 包络折射率 → 厚度
def series_fit_generic(sig_e, theta1, n_values):
    """n_values: 与 sig_e 等长的折射率数组"""
    c2 = np.sqrt(np.clip(1.0 - np.sin(np.radians(theta1))**2 / np.asarray(n_values)**2, 0, None))
    x = 2.0 * np.asarray(n_values) * c2 * sig_e * 1e-4
    N = len(x)
    slope0 = 0.5 * (N - 1) / max(x[-1] - x[0], 1e-12)
    m0 = float(np.round(np.median(0.5 * np.arange(N) - slope0 * x) * 2) / 2)
    m = m0 + 0.5 * np.arange(N)
    A = np.column_stack([x, np.ones(N)])
    coef, *_ = np.linalg.lstsq(A, m, rcond=None)
    resid = m - A @ coef
    r2 = 1 - np.sum(resid**2) / np.sum((m - m.mean())**2)
    return float(coef[0]), float(r2)

chan = {}
for tag in ["a1", "a2"]:
    d = sic[tag]
    idx = np.sort(np.concatenate([d["pk"], d["vy"]]))
    se = d["sig"][idx]
    n_olo_e = np.asarray(n_olo[tag](se), float)
    d_olo, r2_olo = series_fit_generic(se, d["th"], n_olo_e)
    n_env_e = np.interp(se, env[tag]["sg"], env[tag]["n"])
    d_env, r2_env = series_fit_generic(se, d["th"], n_env_e)
    n_const_e = np.full_like(se, 2.55)
    d_const, r2_const = series_fit_generic(se, d["th"], n_const_e)
    chan[tag] = dict(d_olo=d_olo, r2_olo=r2_olo, d_env=d_env, r2_env=r2_env,
                     d_const=d_const, r2_const=r2_const)
    print(f"[通道] {tag}: TO-LO={d_olo:.3f}(R2={r2_olo:.5f}) | 包络={d_env:.3f}(R2={r2_env:.5f}) | n=2.55常数={d_const:.3f}(R2={r2_const:.5f})")
R3["channel"] = chan

# ============ 实验 2：波峰法 vs 峰谷法（薛毅 5.1） ============
pv = {}
for tag in ["a1", "a2"]:
    d = sic[tag]
    th = d["th"]
    # 波峰法：仅波峰，级数差为整数
    idx_pk = np.sort(d["pk"])
    se_pk = d["sig"][idx_pk]
    n_pk = np.asarray(n_olo[tag](se_pk), float)
    c2_pk = np.sqrt(1 - np.sin(np.radians(th))**2 / n_pk**2)
    x_pk = 2.0 * n_pk * c2_pk * se_pk * 1e-4
    Np = len(x_pk)
    sl0 = (Np - 1) / max(x_pk[-1] - x_pk[0], 1e-12)
    m0 = float(np.round(np.median(np.arange(Np) - sl0 * x_pk) ))
    m = m0 + np.arange(Np)
    coef, *_ = np.linalg.lstsq(np.column_stack([x_pk, np.ones(Np)]), m, rcond=None)
    d_peak = float(coef[0])
    # 峰谷法（半级，主方法）
    idx_all = np.sort(np.concatenate([d["pk"], d["vy"]]))
    d_pv, r2_pv = series_fit_generic(d["sig"][idx_all], th, np.asarray(n_olo[tag](d["sig"][idx_all]), float))
    pv[tag] = dict(d_peak=d_peak, d_peakvalley=d_pv)
    print(f"[波峰vs峰谷] {tag}: 波峰法={d_peak:.3f} | 峰谷法={d_pv:.3f} | 差={100*abs(d_peak-d_pv)/d_pv:.2f}%")
R3["peak_vs_valley"] = pv

# ============ 实验 3：Finesse 精确值（σ=2000，含角度） ============
def fresnel_r12(n1v, n2v, th_deg):
    s = np.sin(np.radians(th_deg)); c1 = np.cos(np.radians(th_deg))
    c2 = np.sqrt(1 - (s/n1v)**2); c3 = np.sqrt(1 - (s/n2v)**2)
    r01 = abs((c1 - n1v*c2)/(c1 + n1v*c2))
    r12 = abs((n1v*c2 - n2v*c3)/(n1v*c2 + n2v*c3))
    F = 4*r01*r12/(1 - r01*r12)**2
    return float(r01), float(r12), float(F)

fin = {}
# Si：外延 n=3.42；衬底重掺杂（用 v2 拟合参数 σp=4010, γ=404, einf=12.82）
eps_si = lambda s: 12.8163 - 4009.6708**2/(np.asarray(s, float)**2 + 1j*403.9676*np.asarray(s, float))
n2_si = lambda s: np.sqrt(eps_si(s))
r01, r12, F = fresnel_r12(3.42, complex(n2_si(2000.0)), 10.0)
fin["si"] = dict(r01=r01, r12=r12, F=F)
# SiC：外延 TO-LO n(2000)；衬底 δn=-0.5%
n1c = float(n_olo["a1"](2000.0))
r01c, r12c, Fc = fresnel_r12(n1c, n1c*0.995, 10.0)
fin["sic"] = dict(r01=r01c, r12=r12c, F=Fc, n_epi=n1c)
print(f"[Finesse] Si:  r01={r01:.3f}, r12={r12:.4f}, F={F:.3f}")
print(f"[Finesse] SiC: r01={r01c:.3f}, r12={r12c:.5f}, F={Fc:.5f} (n_epi={n1c:.3f})")
R3["finesse"] = fin

# ============ 图：折射率通道对厚度的影响 ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
# 左：两条折射率曲线
sg = env["a1"]["sg"]
ax1.plot(sg, env["a1"]["n"], color="#2e7d32", lw=1.6, label="包络反演通道（薛毅式(22)）")
ax1.plot(sg, np.asarray(n_olo["a1"](sg), float), color="#1f4e79", lw=1.8, label="TO--LO 单振子通道（本文）")
ax1.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax1.set_ylabel(r"外延层折射率 $n_1(\sigma)$")
ax1.legend(fontsize=8.5, frameon=False); ax1.grid(ls="--", alpha=0.35)
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title("两条折射率通道", fontsize=11)
# 右：通道 → 厚度 条形图
labels = ["TO--LO 振子\n（本文主方法）", "包络反演\n（薛毅通道）", "常数 n=2.55\n（对照）"]
d10 = [chan["a1"]["d_olo"], chan["a1"]["d_env"], chan["a1"]["d_const"]]
d15 = [chan["a2"]["d_olo"], chan["a2"]["d_env"], chan["a2"]["d_const"]]
xpos = np.arange(3); w = 0.36
b1 = ax2.bar(xpos - w/2, d10, w, color="#1f4e79", alpha=0.9, label="附件 1（10°）")
b2 = ax2.bar(xpos + w/2, d15, w, color="#c55a11", alpha=0.9, label="附件 2（15°）")
for b, v in list(zip(b1, d10)) + list(zip(b2, d15)):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
ax2.axhspan(7.250, 7.687, color="#888888", alpha=0.12)
ax2.text(2.42, 7.47, "薛毅加权\n7.469", fontsize=8, color="#555555", va="center")
ax2.set_xticks(xpos); ax2.set_xticklabels(labels, fontsize=8.5)
ax2.set_ylabel(r"反演厚度 $d$ (μm)"); ax2.set_ylim(6.8, 8.1)
ax2.legend(fontsize=8.5, frameon=False, loc="lower left")
ax2.grid(ls="--", alpha=0.35, axis="y"); ax2.spines[["top", "right"]].set_visible(False)
ax2.set_title("折射率通道 → 厚度（级数标定不变）", fontsize=11)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_channel_impact.png", dpi=200); plt.close(fig)
print("fig_channel_impact.png 已生成")

with open(os.path.join(OUT, "results3.json"), "w", encoding="utf-8") as f:
    json.dump(R3, f, ensure_ascii=False, indent=2, default=float)
print("完成 results3.json")
