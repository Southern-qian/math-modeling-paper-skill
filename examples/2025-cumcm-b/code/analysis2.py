# -*- coding: utf-8 -*-
"""扩写用补充计算：9 张科研配图 + 精确对差法 + 谐波判据 + DE 收敛"""
import json, os
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution, least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

OUT = "/Users/gongzhen/.zcode/workspace/default/sic_review"
FIG = os.path.join(OUT, "figs")
R2 = {}

# 复用 analysis.py 的全部模型函数（仅 defs 部分）
code = open(os.path.join(OUT, "analysis.py"), encoding="utf-8").read()
head = code.split("# ============================================================\n# 1) SiC")[0]
exec(head)

# ---------- SiC 基础结果复算 ----------
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

# ---------- 图 A：级数标定线性回归图 ----------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=False)
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
            label=f"拟合：$d$={coef[0]:.3f} μm，$R^2$={r2:.4f}")
    ax.set_xlabel(r"光学厚度坐标 $x_j = 2n(\sigma_j)\cos\theta_2\,\sigma_j$")
    ax.set_ylabel(r"干涉级数 $m_j$")
    ax.set_title(lab, fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(ls="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_regression.png", dpi=200); plt.close(fig)
R2["regression"] = {tag: dict(d=float(best[tag]["d_um"]), R2=float(best[tag]["R2"]), m0=float(best[tag]["m0"])) for tag in ["a1", "a2"]}

# ---------- 图 B：ε∞ 扫描曲线 ----------
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
ax.axvline(6.90, color="gray", ls=":", lw=1)
ax.text(6.905, min(min(scan["a1"]), min(scan["a2"])), "选定点 6.90", fontsize=9, color="gray", rotation=90, va="bottom")
ax.set_xlabel(r"振子高频介电常数 $\varepsilon_\infty$")
ax.set_ylabel(r"级数回归优度 $R^2$")
ax.legend(fontsize=9, frameon=False); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_eps_scan.png", dpi=200); plt.close(fig)

# ---------- 图 C：相速折射率 vs 群指数 ----------
sig_g = np.linspace(1200, 3300, 300)
nf10 = nfunC["a1"]
n_ph = nf10(sig_g)
h = 2.0
n_gr = n_ph + sig_g * (nf10(sig_g + h) - nf10(sig_g - h)) / (2 * h)
fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.plot(sig_g, n_ph, color="#1f4e79", lw=1.8, label=r"相速折射率 $n_1(\sigma)$")
ax.plot(sig_g, n_gr, color="#c00000", lw=1.8, ls="--", label=r"群指数 $n_g = n_1 + \sigma\,\mathrm{d}n_1/\mathrm{d}\sigma$")
ax.fill_between(sig_g, n_ph, n_gr, color="#c00000", alpha=0.08)
ax.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax.set_ylabel("折射率")
ax.legend(fontsize=10, frameon=False); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_group.png", dpi=200); plt.close(fig)

# ---------- 精确对差法（全组合，N≥1，式 11 的波数形式） ----------
def exact_pairs(sig, pk, vy, theta1, nfun):
    idx = np.sort(np.concatenate([pk, vy]))
    se = sig[idx]
    f = se * np.asarray(nfun(se), float) * cos_t2(se, theta1, nfun)  # = σ n cosθ2
    vals = []
    Nall = []
    for i in range(len(se)):
        for j in range(i + 1, len(se)):
            N = (j - i) / 2.0
            if N < 1:
                continue
            vals.append((N) / (2.0 * (f[j] - f[i])) * 1e4)
            Nall.append(N)
    v = np.array(vals)
    med = np.median(v); mad = np.median(np.abs(v - med)) * 1.4826
    keep = np.abs(v - med) <= max(3.0 * mad, 0.03 * med)
    return v[keep], np.array(Nall)[keep]

exact = {}
for tag in ["a1", "a2"]:
    d = sic[tag]
    v, Nk = exact_pairs(d["sig"], d["pk"], d["vy"], d["th"], nfunC[tag])
    exact[tag] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), n=int(len(v)))
    print(f"[精确对差] {tag}: d = {np.mean(v):.4f} ± {np.std(v,ddof=1):.4f} um (n={len(v)}, N∈[{Nk.min()},{Nk.max()}])")
R2["exact_pairs"] = exact

# ---------- 图 D：ξ 域 FFT 与二次谐波判据 ----------
def fft_harmonic(sig, sm, base, nfun, theta1):
    y = sm - base
    xi = 2.0 * np.asarray(nfun(sig), float) * cos_t2(sig, theta1, nfun) * sig * 1e-4  # μm
    xg = np.linspace(xi.min(), xi.max(), 8192)
    yg = np.interp(xg, xi, y) * np.hanning(8192)
    Y = np.abs(np.fft.rfft(yg))
    fr = np.fft.rfftfreq(8192, d=(xg[1] - xg[0]))
    mask = fr > 1.2
    k = int(np.argmax(Y[1:] * mask[1:]) + 1)
    tau_main = fr[k]; H_main = Y[k]
    harm_mask = (fr > 1.85 * tau_main) & (fr < 2.15 * tau_main)
    H_harm = float(Y[harm_mask].max())
    noise_mask = fr > 3.2 * tau_main
    H_noise = float(np.mean(Y[noise_mask]))
    snr = (H_harm - H_noise) / H_noise
    rel = (H_harm - H_noise) / (H_main - H_noise)
    return dict(fr=fr, Y=Y, tau_main=float(tau_main), H_main=float(H_main),
                H_harm=H_harm, snr=float(snr), rel=float(rel))

# SiC 用 a1 的色散；Si 用 n=3.42
si_data = {}
for tag, fname, th in [("a3", "附件3.xlsx", 10.0), ("a4", "附件4.xlsx", 15.0)]:
    sig, rr = load(fname)
    sig, rr, sm, base = prep(sig, rr, 900, 4000, win=25)
    si_data[tag] = dict(sig=sig, rr=rr, sm=sm, base=base, th=th)

harm = {}
harm["sic_a1"] = fft_harmonic(sic["a1"]["sig"], sic["a1"]["sm"], sic["a1"]["base"], nfunC["a1"], 10.0)
harm["sic_a2"] = fft_harmonic(sic["a2"]["sig"], sic["a2"]["sm"], sic["a2"]["base"], nfunC["a2"], 15.0)
harm["si_a3"] = fft_harmonic(si_data["a3"]["sig"], si_data["a3"]["sm"], si_data["a3"]["base"], lambda s: 3.42, 10.0)
harm["si_a4"] = fft_harmonic(si_data["a4"]["sig"], si_data["a4"]["sm"], si_data["a4"]["base"], lambda s: 3.42, 15.0)
for k, v in harm.items():
    print(f"[谐波判据] {k}: τ_main={v['tau_main']:.2f} μm, SNR_harm={v['snr']:.1f}, R_rel={v['rel']:.3f}")
R2["harmonic"] = {k: dict(tau_main=v["tau_main"], snr=v["snr"], rel=v["rel"]) for k, v in harm.items()}

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
for ax, key, lab, cl in [(axes[0], "sic_a1", "碳化硅（附件 1）", "#1f4e79"),
                          (axes[1], "si_a3", "硅（附件 3）", "#c00000")]:
    v = harm[key]
    m = v["fr"] < 3.4 * v["tau_main"]
    ax.plot(v["fr"][m], v["Y"][m], color=cl, lw=1.2)
    ax.axvline(v["tau_main"], color="gray", ls="--", lw=0.9)
    ax.axvline(2 * v["tau_main"], color="green", ls="--", lw=0.9)
    ax.annotate("基频 $\\tau_0$", xy=(v["tau_main"], v["H_main"]), xytext=(v["tau_main"] * 1.15, v["H_main"] * 0.9),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.annotate("二次谐波 $2\\tau_0$", xy=(2 * v["tau_main"], v["H_harm"]), xytext=(2.18 * v["tau_main"], v["H_harm"] + 0.35 * v["H_main"]),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set_xlabel(r"光程差轴 $\tau$ (μm)"); ax.set_ylabel("FFT 幅值")
    ax.set_title(f"{lab}：$R_\\mathrm{{rel}}$={v['rel']:.3f}", fontsize=11)
    ax.grid(ls="--", alpha=0.35); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_fft_harmonic.png", dpi=200); plt.close(fig)

# ---------- 图 E：SiC 全谱 双光束 vs 多光束(Airy, δn=-0.5%) ----------
eps0 = 0.5 * (best["a1"]["eps_inf"] + best["a2"]["eps_inf"])
d0 = float(np.mean([best["a1"]["d_um"], best["a2"]["d_um"]]))
n_ep = sic_n_factory(eps0)
n_sb = lambda s: n_ep(s) * 0.995
def sic_ck(p, datasets, two_beam):
    d, sd = p
    deco = lambda s: np.exp(-(np.asarray(s, float) / sd) ** 2)
    raw = []
    for sigma, rr, th in datasets:
        Rm = R_model(sigma, d, n_ep, n_sb, th, deco=deco(sigma), two_beam=two_beam)
        raw.append(Rm - rr)
    return nuis_residual(np.concatenate(raw), datasets)
ds_c = [(sic["a1"]["sig"][::2], sic["a1"]["sm"][::2], 10.0),
        (sic["a2"]["sig"][::2], sic["a2"]["sm"][::2], 15.0)]
bnd = [(d0 - 0.5, d0 + 0.5), (2000.0, 50000.0)]
solM = least_squares(sic_ck, [d0, 12000.0], args=(ds_c, False), method="trf",
                     bounds=([b[0] for b in bnd], [b[1] for b in bnd]))
solT = least_squares(sic_ck, [d0, 12000.0], args=(ds_c, True), method="trf",
                     bounds=([b[0] for b in bnd], [b[1] for b in bnd]))
R2["sic_fullfit"] = dict(d_multi=float(solM.x[0]), d_2b=float(solT.x[0]),
                         mse_multi=float(np.mean(solM.fun**2)), mse_2b=float(np.mean(solT.fun**2)))
print(f"[SiC全谱] 多光束 d={solM.x[0]:.4f} (MSE {R2['sic_fullfit']['mse_multi']:.3f}), "
      f"双光束 d={solT.x[0]:.4f} (MSE {R2['sic_fullfit']['mse_2b']:.3f})")

sigma, rr, th = ds_c[0]
dcM = np.exp(-(np.asarray(sigma, float) / solM.x[1]) ** 2)
dcT = np.exp(-(np.asarray(sigma, float) / solT.x[1]) ** 2)
RmM = R_model(sigma, solM.x[0], n_ep, n_sb, th, deco=dcM)
RmT = R_model(sigma, solT.x[0], n_ep, n_sb, th, deco=dcT, two_beam=True)
Amat = np.column_stack([RmM, np.ones(len(RmM))]); cM, *_ = np.linalg.lstsq(Amat, rr, rcond=None)
Amat = np.column_stack([RmT, np.ones(len(RmT))]); cT, *_ = np.linalg.lstsq(Amat, rr, rcond=None)
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.plot(sic["a1"]["sig"], sic["a1"]["rr"], color="#bbbbbb", lw=0.6, label="附件 1 实测（10°）")
ax.plot(sigma, cM[0] * RmM + cM[1], color="#c00000", lw=1.3, label=f"多光束 Airy（$d$={solM.x[0]:.3f} μm）")
ax.plot(sigma, cT[0] * RmT + cT[1], color="#1f4e79", lw=1.1, ls="--", label=f"双光束（$d$={solT.x[0]:.3f} μm）")
ax.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax.set_ylabel("反射率 (%)")
ax.legend(fontsize=9, frameon=False, loc="upper left"); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_sic_fit.png", dpi=200); plt.close(fig)

# ---------- 图 F：Si 衬底 Drude 曲线（用拟合参数重算） ----------
si_p = [3.4144, 12.8163, 4009.6708, 403.9676, 0.0, 6386.3362]
d_si, n_epi_si, n_sub_si, deco_si = si_models(si_p)
sig_grid = np.linspace(700, 4000, 500)
eps_v = 12.8163 - 4009.6708**2 / (sig_grid**2 + 1j * 403.9676 * sig_grid)
fig, ax1 = plt.subplots(figsize=(6.8, 3.5))
ax1.plot(sig_grid, np.real(eps_v), color="#1f4e79", lw=1.6, label=r"$\mathrm{Re}\,\varepsilon_s$")
ax1.plot(sig_grid, np.imag(eps_v), color="#c55a11", lw=1.6, ls="--", label=r"$\mathrm{Im}\,\varepsilon_s$")
ax1.axhline(0, color="gray", lw=0.6)
ax1.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax1.set_ylabel(r"衬底介电函数 $\varepsilon_s(\sigma)$")
ax1.legend(fontsize=9, frameon=False, loc="upper right"); ax1.grid(ls="--", alpha=0.35)
ax1.spines[["top", "right"]].set_visible(False)
ax2 = ax1.twinx()
ax2.plot(sig_grid, np.abs(n_sub_si(sig_grid)), color="#2e7d32", lw=1.6, label=r"$|n_2(\sigma)|$")
ax2.set_ylabel(r"衬底复折射率模 $|n_2|$", color="#2e7d32")
ax2.tick_params(axis="y", labelcolor="#2e7d32")
ax2.spines[["top"]].set_visible(False)
ax1.legend(fontsize=9, frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_drude.png", dpi=200); plt.close(fig)

# ---------- 图 G：DE 收敛曲线（记录每代最优） ----------
ds_si = [(si_data["a3"]["sig"][::2], si_data["a3"]["sm"][::2], 10.0),
         (si_data["a4"]["sig"][::2], si_data["a4"]["sm"][::2], 15.0)]
bounds = [(2.5, 5.0), (8.0, 14.5), (200.0, 6000.0), (5.0, 800.0), (-5e-7, 5e-7), (1500.0, 50000.0)]
p0 = [3.5, 11.7, 2600.0, 200.0, 0.0, 6000.0]
def cost_si(p):
    return float(np.mean(nuis_residual(si_residual(p, ds_si, False), ds_si) ** 2))
conv = []
def cb(xk, convergence=None):
    conv.append(float(cost_si(xk)))
de2 = differential_evolution(cost_si, bounds, seed=1, maxiter=60, popsize=24, tol=1e-10,
                             polish=False, workers=1, x0=p0, callback=cb)
conv = np.array(conv)
print(f"[DE] 收敛: 初值 MSE={conv[0]:.2f} → 终值 MSE={conv[-1]:.2f}, {len(conv)} 代")
R2["de"] = dict(n_gen=int(len(conv)), first=float(conv[0]), last=float(conv[-1]))
fig, ax = plt.subplots(figsize=(6.4, 3.3))
ax.semilogy(np.arange(1, len(conv) + 1), conv, "-o", ms=3, color="#1f4e79", lw=1.2)
ax.set_xlabel("差分进化代数"); ax.set_ylabel("目标函数（残差 MSE）")
ax.grid(ls="--", alpha=0.35, which="both"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_de_conv.png", dpi=200); plt.close(fig)

# ---------- 图 H：与开源实现结果对比 ----------
names = ["本文（级数标定）", "本文（FFT）", "文献[15] 命题思路复现", "文献[16] GMM 统计", "文献[19] 柯西+GA"]
vals1 = [7.202, 7.336, 7.564, 7.504, 7.447]
fig, ax = plt.subplots(figsize=(7.4, 3.4))
colors = ["#c00000", "#c00000", "#1f4e79", "#1f4e79", "#1f4e79"]
bars = ax.barh(names[::-1], vals1[::-1], color=colors[::-1], alpha=0.85, height=0.55)
for b, v in zip(bars, vals1[::-1]):
    ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
ax.axvspan(7.20, 7.36, color="#c00000", alpha=0.10)
ax.set_xlim(7.0, 7.75)
ax.set_xlabel("碳化硅外延层厚度 $d$ (μm)")
ax.grid(ls="--", alpha=0.35, axis="x"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_compare.png", dpi=200); plt.close(fig)

# ---------- 图 I：包络反演折射率 vs TO-LO 振子折射率 ----------
d = sic["a1"]
pk_i, vy_i = d["pk"], d["vy"]
sig_pk, val_pk = d["sig"][pk_i], d["sm"][pk_i]
sig_vy, val_vy = d["sig"][vy_i], d["sm"][vy_i]
sg = np.linspace(sig_pk.min(), sig_pk.max(), 200)
Rmax = np.interp(sg, sig_pk, val_pk)
Rmin = np.interp(sg, sig_vy, val_vy)
A = (np.sqrt(np.clip(Rmax, 0, None)) + np.sqrt(np.clip(Rmin, 0, None))) / 2
n_env = (1 + A) / (1 - A)
fig, ax = plt.subplots(figsize=(6.8, 3.4))
ax.plot(sg, n_env, color="#2e7d32", lw=1.6, label="包络反演 $n(\\sigma)$（对照方法[15]）")
ax.plot(sig_g, n_ph, color="#1f4e79", lw=1.8, label="TO--LO 单振子（本文）")
ax.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax.set_ylabel("外延层折射率")
ax.legend(fontsize=9, frameon=False); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_n_compare.png", dpi=200); plt.close(fig)

with open(os.path.join(OUT, "results2.json"), "w", encoding="utf-8") as f:
    json.dump(R2, f, ensure_ascii=False, indent=2)
print("完成：results2.json + 9 张新图")
