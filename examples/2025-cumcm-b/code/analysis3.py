# -*- coding: utf-8 -*-
"""修复三张图：fig_n_compare（包络反射率单位）、fig_regression（R² 五位小数）、fig_eps_scan（y 量程）"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

OUT = "/Users/gongzhen/.zcode/workspace/default/sic_review"
FIG = os.path.join(OUT, "figs")
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
nfunC = {tag: sic_n_factory(best[tag]["eps_inf"]) for tag in ["a1", "a2"]}

# ---- 图 A'：级数回归（R² 五位小数）----
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
for ax, tag, lab, cl in [(axes[0], "a1", "附件 1（10°）", "#1f4e79"), (axes[1], "a2", "附件 2（15°）", "#c55a11")]:
    d = sic[tag]
    idx = np.sort(np.concatenate([d["pk"], d["vy"]]))
    se = d["sig"][idx]
    x = 2.0 * np.asarray(nfunC[tag](se), float) * cos_t2(se, d["th"], nfunC[tag]) * se * 1e-4
    N = len(x)
    slope0 = 0.5 * (N - 1) / max(x[-1] - x[0], 1e-12)
    m0 = float(np.round(np.median(0.5 * np.arange(N) - slope0 * x) * 2) / 2)
    m = m0 + 0.5 * np.arange(N)
    A = np.column_stack([x, np.ones(N)])
    coef, *_ = np.linalg.lstsq(A, m, rcond=None)
    res = m - A @ coef
    r2 = 1 - np.sum(res**2) / np.sum((m - m.mean())**2)
    ax.plot(x, m, "o", ms=5, color=cl, label="极值点 $(x_j,\\,m_j)$")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, coef[0] * xs + coef[1], "-", color="#c00000", lw=1.4,
            label=f"拟合：$d$={coef[0]:.3f} μm，$R^2$={r2:.5f}")
    ax.set_xlabel(r"光学厚度坐标 $x_j = 2n(\sigma_j)\cos\theta_2\,\sigma_j$")
    ax.set_ylabel(r"干涉级数 $m_j$")
    ax.set_title(lab, fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(ls="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_regression.png", dpi=200); plt.close(fig)
print("fig_regression 重新生成（R² 五位小数）")

# ---- 图 B'：ε∞ 扫描（y 量程自适应 + 水平标注）----
grid = np.arange(6.30, 6.901, 0.01)
scan = {tag: [] for tag in ["a1", "a2"]}
for eps in grid:
    nf = sic_n_factory(float(eps))
    for tag in ["a1", "a2"]:
        d = sic[tag]
        idx = np.sort(np.concatenate([d["pk"], d["vy"]]))
        dd, mm, rr2, _ = series_fit(d["sig"][idx], d["th"], nf)
        scan[tag].append(rr2)
fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.plot(grid, scan["a1"], "-o", ms=3, color="#1f4e79", label="附件 1（10°）")
ax.plot(grid, scan["a2"], "-s", ms=3, color="#c55a11", label="附件 2（15°）")
lo = min(min(scan["a1"]), min(scan["a2"])); hi = max(max(scan["a1"]), max(scan["a2"]))
pad = max((hi - lo) * 0.15, 2e-6)
ax.axvline(6.90, color="gray", ls=":", lw=1.1)
ax.annotate("选定 $\\varepsilon_\\infty=6.90$", xy=(6.90, max(max(scan["a1"]), max(scan["a2"]))),
            xytext=(6.42, hi + pad * 0.35), fontsize=9, color="gray",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", lw=0.6),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
ax.set_ylim(lo - pad, hi + pad)
ax.set_xlabel(r"振子高频介电常数 $\varepsilon_\infty$")
ax.set_ylabel(r"级数回归优度 $R^2$")
ax.legend(fontsize=9, frameon=False, loc="lower right"); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_eps_scan.png", dpi=200); plt.close(fig)
print("fig_eps_scan 重新生成（y 量程自适应）")

# ---- 图 I'：包络反演（反射率转为小数）----
d = sic["a1"]
sig_pk, val_pk = d["sig"][d["pk"]], d["sm"][d["pk"]]
sig_vy, val_vy = d["sig"][d["vy"]], d["sm"][d["vy"]]
sg = np.linspace(sig_pk.min(), sig_pk.max(), 220)
Rmax = np.interp(sg, sig_pk, val_pk) / 100.0     # 百分数 → 小数（振幅公式要求 R∈[0,1]）
Rmin = np.interp(sg, sig_vy, val_vy) / 100.0
A = (np.sqrt(Rmax) + np.sqrt(Rmin)) / 2
n_env = (1 + A) / (1 - A)
sig_g = np.linspace(1200, 3300, 300)
n_ph = nfunC["a1"](sig_g)
print(f"[包络n] 范围: {n_env.min():.3f} - {n_env.max():.3f}")
fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.plot(sg, n_env, color="#2e7d32", lw=1.6, label="包络反演 $n(\\sigma)$（对照方法[15]）")
ax.plot(sig_g, n_ph, color="#1f4e79", lw=1.8, label="TO--LO 单振子（本文）")
ax.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax.set_ylabel("外延层折射率")
ax.legend(fontsize=9, frameon=False); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_n_compare.png", dpi=200); plt.close(fig)
print("fig_n_compare 重新生成（包络反射率已转小数）")
