# -*- coding: utf-8 -*-
"""2025 CUMCM B 题求解 v2：SiC/Si 外延层厚度反演
方法体系：
  P2(SiC): SG 滤波 → 极值检测(剪枝) → TO-LO 单振子色散 + 级数标定线性回归 →
           相邻极值对 bootstrap → 双角度逆方差合并
  P3(Si) : 必要条件推导 → 定性判定(可见度/不对称) → 极值点解析法 →
           Drude 衬底 + Airy 多光束全谱拟合(DE+TRF, 含去相干) → 双光束对比
"""
import json, os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import differential_evolution, least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "Arial Unicode MS", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12

DATA = "/Users/gongzhen/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_tio3k2023sgm22_d0f8/temp/RWTemp/2026-08/f04edf4511520fe739c29e5d043bf714"
OUT = "/Users/gongzhen/.zcode/workspace/default/sic_review"
FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)
R = {}

def load(fname):
    df = pd.read_excel(os.path.join(DATA, fname), header=0)
    df.columns = ["sigma", "R"]
    return df["sigma"].to_numpy(float), df["R"].to_numpy(float)

def cos_t2(sigma, theta1_deg, n):
    if callable(n):
        n = n(sigma)
    n = np.asarray(n, float)
    s = np.sin(np.radians(theta1_deg))
    return np.sqrt(np.clip(1.0 - s**2 / n**2, 0, None))

def prep(sigma, Rr, smin, smax, win=31):
    m = (sigma >= smin) & (sigma <= smax)
    sig, rr = sigma[m], Rr[m]
    base = savgol_filter(rr, 301, 2)
    sm = savgol_filter(rr, win, 3)
    return sig, rr, sm, base

def extrema(sig, sm, prom=0.4, dist=12, base=None):
    """极值检测 + 同类型相邻合并剪枝（强制峰谷交替）"""
    pk, _ = find_peaks(sm, prominence=prom, distance=dist)
    vy, _ = find_peaks(-sm, prominence=prom, distance=dist)
    ref = np.abs(sm - base) if base is not None else np.abs(sm)
    pts = sorted([(i, 1) for i in pk] + [(i, -1) for i in vy])
    kept = []
    for i, t in pts:
        if kept and kept[-1][1] == t:
            if ref[i] > ref[kept[-1][0]]:
                kept[-1] = (i, t)
        else:
            kept.append((i, t))
    pk2 = np.array([i for i, t in kept if t == 1], dtype=int)
    vy2 = np.array([i for i, t in kept if t == -1], dtype=int)
    return pk2, vy2

# ---------- 色散模型 ----------
# SiC 外延层：TO-LO 单振子（文献 σTO=797.7, σLO=992.1 cm^-1, 轻掺杂 γ 小）
SIC_TO, SIC_LO = 797.7, 992.1
def sic_n_factory(eps_inf, gam=6.0):
    def n(s):
        s = np.asarray(s, float)
        eps = eps_inf * (SIC_LO**2 - s**2 - 1j * gam * s) / (SIC_TO**2 - s**2 - 1j * gam * s)
        return np.real(np.sqrt(eps))
    return n

SI_N = 3.42  # 轻掺杂硅外延层红外折射率（近常数）

def d_pair(sig, pk, vy, theta1, nfun):
    """相邻峰谷交替极值（半级 Δm=0.5）→ 逐对厚度（MAD 剔离群），μm"""
    idx = np.sort(np.concatenate([pk, vy]))
    sig_e = sig[idx]
    d_list, sig_mid = [], []
    h = 2.0
    for a, b in zip(sig_e[:-1], sig_e[1:]):
        sbar = 0.5 * (a + b)
        nbar = float(np.asarray(nfun(sbar)))
        ng = nbar + sbar * (float(np.asarray(nfun(sbar + h))) - float(np.asarray(nfun(sbar - h)))) / (2 * h)
        c2 = float(cos_t2(sbar, theta1, nbar))
        d_list.append(0.25 / (ng * c2 * (b - a)) * 1e4)   # 极值间距由群指数决定
        sig_mid.append(sbar)
    d_arr = np.array(d_list)
    med = np.median(d_arr)
    mad = np.median(np.abs(d_arr - med)) * 1.4826
    keep = np.abs(d_arr - med) <= max(3.0 * mad, 0.04 * med)
    return d_arr[keep], np.array(sig_mid)[keep]

def series_fit(sig_e, theta1, nfun):
    """级数标定线性回归：m_j = m0 + 0.5j = d·x_j, x_j = 2 n cosθ2 σ (×1e-4→μm 尺度)"""
    x = 2.0 * np.asarray(nfun(sig_e), float) * cos_t2(sig_e, theta1, nfun) * sig_e * 1e-4
    N = len(x)
    slope0 = 0.5 * (N - 1) / max(x[-1] - x[0], 1e-12)
    m0_est = 0.5 * np.arange(N) - slope0 * x
    m0 = float(np.round(np.median(m0_est) * 2) / 2)
    m = m0 + 0.5 * np.arange(N)
    A = np.column_stack([x, np.ones(N)])
    coef, *_ = np.linalg.lstsq(A, m, rcond=None)
    d_um, intercept = coef
    resid = m - A @ coef
    r2 = 1 - np.sum(resid**2) / np.sum((m - m.mean())**2)
    return float(d_um), float(m0), float(r2), float(np.sqrt(np.mean(resid**2)))

def scan_sic_series(sig_e, theta1, eps_grid=None):
    """扫描 ε∞，取级数回归 R² 最高者"""
    if eps_grid is None:
        eps_grid = np.arange(6.3, 6.91, 0.02)
    best = None
    for eps in eps_grid:
        nfun = sic_n_factory(float(eps))
        d, m0, r2, rms = series_fit(sig_e, theta1, nfun)
        if best is None or r2 > best["R2"]:
            best = dict(d_um=d, m0=m0, R2=r2, rms=rms, eps_inf=float(eps))
    return best

def d_fft(sig, sm, theta1, nfun, base):
    """色散校正坐标 ξ = 2 n(σ) cosθ2 σ 下的 FFT：谱峰频率 = d [cm]"""
    y = sm - base
    xi_full = 2.0 * np.asarray(nfun(sig), float) * cos_t2(sig, theta1, nfun) * sig
    xi = np.linspace(xi_full.min(), xi_full.max(), 4096)
    yi = np.interp(xi, xi_full, y) * np.hanning(4096)
    Y = np.abs(np.fft.rfft(yi))
    freqs = np.fft.rfftfreq(4096, d=(xi[1] - xi[0]))
    k = int(np.argmax(Y[1:]) + 1)
    return float(freqs[k] * 1e4), (freqs, Y)

# ---------- 光学模型（Airy / 双光束，s+p 平均） ----------
def R_model(sigma, d_um, n_epi, n_sub, theta1_deg, deco=1.0, two_beam=False):
    d = d_um * 1e-4
    n1 = np.asarray(n_epi(sigma), complex)
    n2 = np.asarray(n_sub(sigma), complex)
    s = np.sin(np.radians(theta1_deg))
    c1 = np.cos(np.radians(theta1_deg))
    c2 = np.sqrt(1.0 - (s / n1) ** 2 + 0j)
    c3 = np.sqrt(1.0 - (s / n2) ** 2 + 0j)
    r01s = (c1 - n1 * c2) / (c1 + n1 * c2)
    r01p = (n1 * c1 - c2) / (n1 * c1 + c2)
    r12s = (n1 * c2 - n2 * c3) / (n1 * c2 + n2 * c3)
    r12p = (n2 * c2 - n1 * c3) / (n2 * c2 + n1 * c3)
    delta = 4.0 * np.pi * n1 * d * sigma * c2
    e = np.exp(1j * delta)
    out = []
    for r01, r12 in [(r01s, r12s), (r01p, r12p)]:
        if two_beam:
            r = r01 + deco * r12 * e
        else:
            r = (r01 + deco * r12 * e) / (1.0 + deco * r01 * r12 * e)
        out.append(np.abs(r) ** 2)
    return 50.0 * (out[0] + out[1])

SI_A = 3.42  # 外延层折射率固定为文献值（与极值法一致）
def si_models(p):
    d, einf_s, sp, g_s, B_e, sd = p
    n_epi = lambda s: SI_A + B_e * np.asarray(s, float) ** 2
    eps_s = lambda s: einf_s - sp**2 / (np.asarray(s, float) ** 2 + 1j * g_s * np.asarray(s, float))
    n_sub = lambda s: np.sqrt(eps_s(s))
    deco = lambda s: np.exp(-(np.asarray(s, float) / sd) ** 2)
    return d, n_epi, n_sub, deco

def si_residual(p, datasets, two_beam=False):
    d, n_epi, n_sub, deco = si_models(p)
    res = []
    for sigma, Rr, th in datasets:
        Rm = R_model(sigma, d, n_epi, n_sub, th, deco=deco(sigma), two_beam=two_beam)
        res.append(Rm - Rr)
    return np.concatenate(res)

def nuis_residual(raw, datasets):
    """每角度 profile 线性标定 R_meas ≈ a·R_model + b（raw = model − meas）"""
    out, i = [], 0
    for sigma, rr, th in datasets:
        n = len(sigma)
        model = rr + raw[i:i + n]
        Amat = np.column_stack([model, np.ones(n)])
        coef, *_ = np.linalg.lstsq(Amat, rr, rcond=None)
        out.append(coef[0] * model + coef[1] - rr)
        i += n
    return np.concatenate(out)

# ============================================================
# 1) SiC（问题二）
# ============================================================
print("=" * 60, "\n[SiC] 附件1(10°)、附件2(15°)")
sic = {}
for tag, fname, th in [("a1", "附件1.xlsx", 10.0), ("a2", "附件2.xlsx", 15.0)]:
    sig, rr = load(fname)
    sig, rr, sm, base = prep(sig, rr, 1200, 3300)
    pk, vy = extrema(sig, sm, prom=0.35, dist=12, base=base)
    sic[tag] = dict(sig=sig, rr=rr, sm=sm, base=base, pk=pk, vy=vy, th=th)
    print(f"  {tag}: pts={len(sig)}, peaks={len(pk)}, valleys={len(vy)}")

sic_res = {}
for tag in ["a1", "a2"]:
    d = sic[tag]
    idx = np.sort(np.concatenate([d["pk"], d["vy"]]))
    best = scan_sic_series(d["sig"][idx], d["th"])
    nfun = sic_n_factory(best["eps_inf"])
    d1, smid = d_pair(d["sig"], d["pk"], d["vy"], d["th"], nfun)
    d2, _ = d_fft(d["sig"], d["sm"], d["th"], nfun, d["base"])
    sic_res[tag] = dict(series=best, d_pair_mean=float(np.mean(d1)),
                        d_pair_std=float(np.std(d1, ddof=1)), d_pair_n=int(len(d1)),
                        d_fft=d2, d_pairs=d1, sig_mid=smid)
    print(f"  {tag}: 级数回归 d={best['d_um']:.4f} um (R2={best['R2']:.5f}, eps_inf={best['eps_inf']:.2f},"
          f" n∈[{nfun(1200):.3f},{nfun(3300):.3f}])")
    print(f"        相邻极值对 d={np.mean(d1):.4f}±{np.std(d1, ddof=1):.4f} (n={len(d1)}) | FFT={d2:.4f} um")

d10, d15 = sic_res["a1"]["series"]["d_um"], sic_res["a2"]["series"]["d_um"]
s10, s15 = sic_res["a1"]["d_pair_std"], sic_res["a2"]["d_pair_std"]
w10, w15 = 1 / s10**2, 1 / s15**2
d_comb = (w10 * d10 + w15 * d15) / (w10 + w15)
se_comb = float(np.sqrt(1 / (w10 + w15)))
R["sic"] = {
    "d10": d10, "d15": d15, "d_comb": float(d_comb), "se_comb": se_comb,
    "d10_pair": sic_res["a1"]["d_pair_mean"], "d15_pair": sic_res["a2"]["d_pair_mean"],
    "d10_pair_std": s10, "d15_pair_std": s15,
    "d_pair_n": [sic_res["a1"]["d_pair_n"], sic_res["a2"]["d_pair_n"]],
    "d10_fft": sic_res["a1"]["d_fft"], "d15_fft": sic_res["a2"]["d_fft"],
    "series10": sic_res["a1"]["series"], "series15": sic_res["a2"]["series"],
}
print(f"  SiC 合并(逆方差): d = {d_comb:.4f} ± {se_comb:.4f} um (95% CI [{d_comb-1.96*se_comb:.3f},{d_comb+1.96*se_comb:.3f}])")

rng = np.random.default_rng(42)
boots = {}
for tag in ["a1", "a2"]:
    dp = sic_res[tag]["d_pairs"]
    bs = [float(np.mean(rng.choice(dp, size=len(dp), replace=True))) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    boots[tag] = [float(lo), float(hi)]
    print(f"  {tag} bootstrap 95% CI(逐对均值): [{lo:.3f}, {hi:.3f}]")
R["sic"]["boot10"], R["sic"]["boot15"] = boots["a1"], boots["a2"]

# ============================================================
# 2) Si（问题三）
# ============================================================
print("=" * 60, "\n[Si] 附件3(10°)、附件4(15°)")
si = {}
for tag, fname, th in [("a3", "附件3.xlsx", 10.0), ("a4", "附件4.xlsx", 15.0)]:
    sig, rr = load(fname)
    sig, rr, sm, base = prep(sig, rr, 700, 4000, win=25)
    pk, vy = extrema(sig, sm, prom=0.6, dist=12, base=base)
    si[tag] = dict(sig=sig, rr=rr, sm=sm, base=base, pk=pk, vy=vy, th=th)
    print(f"  {tag}: pts={len(sig)}, peaks={len(pk)}, valleys={len(vy)}")

vis = {}
for tag in ["a3", "a4"]:
    d = si[tag]
    idx = np.sort(np.concatenate([d["pk"], d["vy"]]))
    vals = d["sm"][idx]
    n2 = min(len(vals) // 2, 4)
    vmax, vmin = vals[0:n2 * 2:2], vals[1:n2 * 2:2]
    v = (vmax - vmin) / (vmax + vmin + 1e-9)
    vis[tag] = float(np.mean(np.abs(v)))
    print(f"  {tag} 低波数段平均条纹可见度 V = {vis[tag]:.3f}")
R["si"] = {"vis10": vis["a3"], "vis15": vis["a4"]}

for tag, key in [("a3", "extrema10"), ("a4", "extrema15")]:
    d = si[tag]
    idx = np.sort(np.concatenate([d["pk"], d["vy"]]))
    dfit, m0, r2, rms = series_fit(d["sig"][idx], d["th"], lambda s: SI_N)
    R["si"][key] = dfit
    R["si"][key + "_R2"] = r2
    print(f"  {tag} 极值点解析法: d = {dfit:.4f} um (R2={r2:.5f})")

ds = [(si["a3"]["sig"][::2], si["a3"]["sm"][::2], 10.0),
      (si["a4"]["sig"][::2], si["a4"]["sm"][::2], 15.0)]
bounds = [(2.5, 5.0), (8.0, 14.5), (200.0, 6000.0), (5.0, 800.0),
          (-5e-7, 5e-7), (1500.0, 50000.0)]
p0 = [3.5, 11.7, 2600.0, 200.0, 0.0, 6000.0]
def cost(p):
    return float(np.mean(nuis_residual(si_residual(p, ds, False), ds) ** 2))
print("  DE 全局搜索中（多光束）...")
de = differential_evolution(cost, bounds, seed=1, maxiter=60, popsize=24, tol=1e-10,
                            polish=False, workers=1, x0=p0)
sol = least_squares(lambda p: nuis_residual(si_residual(p, ds, False), ds), de.x,
                    method="trf", bounds=([b[0] for b in bounds], [b[1] for b in bounds]))
p_opt = sol.x
print(f"  多光束最优: x={np.round(p_opt, 4).tolist()}, MSE={np.mean(sol.fun**2):.3f}")
R["si"]["params"] = dict(zip(["d_um", "einf_s", "sigma_p", "gamma_s", "B_epi", "sigma_d"],
                             map(float, p_opt)))
d_si, n_epi_si, n_sub_si, deco_si = si_models(p_opt)
for tag, dsx in zip(["a3", "a4"], ds):
    sigma, rr, th = dsx
    Rm = R_model(sigma, d_si, n_epi_si, n_sub_si, th, deco=deco_si(sigma))
    Amat = np.column_stack([Rm, np.ones(len(Rm))])
    coef, *_ = np.linalg.lstsq(Amat, rr, rcond=None)
    fit = coef[0] * Rm + coef[1]
    ss_res = float(np.sum((fit - rr) ** 2)); ss_tot = float(np.sum((rr - rr.mean()) ** 2))
    R["si"][f"R2_{tag}"] = 1 - ss_res / ss_tot
    print(f"  {tag}: R²(标定后) = {1-ss_res/ss_tot:.5f}")

def cost2(p):
    return float(np.mean(nuis_residual(si_residual(p, ds, True), ds) ** 2))
de2 = differential_evolution(cost2, bounds, seed=2, maxiter=60, popsize=24, tol=1e-10,
                             polish=False, workers=1, x0=p0)
sol2 = least_squares(lambda p: nuis_residual(si_residual(p, ds, True), ds), de2.x,
                     method="trf", bounds=([b[0] for b in bounds], [b[1] for b in bounds]))
p2 = sol2.x
R["si"]["two_beam"] = {"d_um": float(p2[0])}
for tag, dsx in zip(["a3", "a4"], ds):
    sigma, rr, th = dsx
    Rm = R_model(sigma, p2[0], si_models(p2)[1], si_models(p2)[2], th,
                 deco=si_models(p2)[3](sigma), two_beam=True)
    Amat = np.column_stack([Rm, np.ones(len(Rm))])
    coef, *_ = np.linalg.lstsq(Amat, rr, rcond=None)
    fit = coef[0] * Rm + coef[1]
    ss_res = float(np.sum((fit - rr) ** 2)); ss_tot = float(np.sum((rr - rr.mean()) ** 2))
    R["si"]["two_beam"][f"R2_{tag}"] = 1 - ss_res / ss_tot
print(f"  双光束: d={p2[0]:.4f} um, R²={R['si']['two_beam']['R2_a3']:.4f}/{R['si']['two_beam']['R2_a4']:.4f}")
print(f"  多光束: d={p_opt[0]:.4f} um, R²={R['si']['R2_a3']:.4f}/{R['si']['R2_a4']:.4f}")

sig_mid = 2000.0
n1v = complex(n_epi_si(sig_mid)); n2v = complex(n_sub_si(sig_mid))
s_ = np.sin(np.radians(10.0)); c1_ = np.cos(np.radians(10.0)); c2_ = np.sqrt(1 - (s_ / n1v) ** 2)
r01_v = abs((c1_ - n1v * c2_) / (c1_ + n1v * c2_))
r12_v = abs((n1v * c2_ - n2v * np.sqrt(1 - (s_ / n2v) ** 2)) / (n1v * c2_ + n2v * np.sqrt(1 - (s_ / n2v) ** 2)))
F_coef = 4 * r01_v * r12_v / (1 - r01_v * r12_v) ** 2
R["si"].update(r01=float(r01_v), r12=float(r12_v), F_coef=float(F_coef))
print(f"  σ=2000: |r01|={r01_v:.4f}, |r12|={r12_v:.4f}, Airy 精细系数 F={F_coef:.3f}")

# ============================================================
# 3) SiC 多光束复核
# ============================================================
print("=" * 60, "\n[SiC 多光束复核]")
ds_c = [(sic["a1"]["sig"][::2], sic["a1"]["sm"][::2], 10.0),
        (sic["a2"]["sig"][::2], sic["a2"]["sm"][::2], 15.0)]
eps0 = 0.5 * (R["sic"]["series10"]["eps_inf"] + R["sic"]["series15"]["eps_inf"])
d0 = float(d_comb)
n_epi_ck = sic_n_factory(eps0)
DN_SUB = -0.005  # 重掺杂衬底 Drude 位移（物理值：N~1e19 → δn≈−0.5%）
n_sub_ck = lambda s: n_epi_ck(s) * (1.0 + DN_SUB)
def sic_ck_resid(p, datasets, two_beam):
    d, sd = p
    deco = lambda s: np.exp(-(np.asarray(s, float) / sd) ** 2)
    raw = []
    for sigma, rr, th in datasets:
        Rm = R_model(sigma, d, n_epi_ck, n_sub_ck, th, deco=deco(sigma), two_beam=two_beam)
        raw.append(Rm - rr)
    return nuis_residual(np.concatenate(raw), datasets)
bnd_ck = [(d0 - 0.5, d0 + 0.5), (2000.0, 50000.0)]
solC = least_squares(sic_ck_resid, [d0, 12000.0], args=(ds_c, False), method="trf",
                     bounds=([b[0] for b in bnd_ck], [b[1] for b in bnd_ck]))
solC2 = least_squares(sic_ck_resid, [d0, 12000.0], args=(ds_c, True), method="trf",
                      bounds=([b[0] for b in bnd_ck], [b[1] for b in bnd_ck]))
R["sic"]["multi"] = {"d_um": float(solC.x[0]), "mse": float(np.mean(solC.fun**2)),
                     "eps_inf": float(eps0),
                     "d_diff_pct": float(100 * (solC.x[0] - d0) / d0)}
R["sic"]["recheck_2b"] = {"d_um": float(solC2.x[0]), "mse": float(np.mean(solC2.fun**2))}
pc = solC.x
for tag, dsx in zip(["a1", "a2"], ds_c):
    sigma, rr, th = dsx
    dc = lambda s: np.exp(-(np.asarray(s, float) / pc[1]) ** 2)
    Rm = R_model(sigma, pc[0], n_epi_ck, n_sub_ck, th, deco=dc(sigma))
    Amat = np.column_stack([Rm, np.ones(len(Rm))])
    coef, *_ = np.linalg.lstsq(Amat, rr, rcond=None)
    fit = coef[0] * Rm + coef[1]
    ss_res = float(np.sum((fit - rr) ** 2)); ss_tot = float(np.sum((rr - rr.mean()) ** 2))
    R["sic"]["multi"][f"R2_{tag}"] = 1 - ss_res / ss_tot
sigm = 2000.0
n1c = complex(n_epi_ck(sigm)); n2c = complex(n_sub_ck(sigm))
c1c = np.cos(np.radians(10.0)); c2c = np.sqrt(1 - (np.sin(np.radians(10.0)) / n1c) ** 2)
r12_ck = abs((n1c * c2c - n2c * np.sqrt(1 - (np.sin(np.radians(10.0)) / n2c) ** 2)) /
             (n1c * c2c + n2c * np.sqrt(1 - (np.sin(np.radians(10.0)) / n2c) ** 2)))
R["sic"]["multi"]["r12"] = float(r12_ck)
print(f"  复核·多光束(Airy): d={solC.x[0]:.4f} um, MSE={R['sic']['multi']['mse']:.4f}, "
      f"R²={R['sic']['multi']['R2_a1']:.4f}/{R['sic']['multi']['R2_a2']:.4f}")
print(f"  复核·双光束: d={solC2.x[0]:.4f} um, MSE={R['sic']['recheck_2b']['mse']:.4f}")
print(f"  纯多光束效应导致的厚度偏移: {R['sic']['multi']['d_diff_pct']:+.4f}% (|r12|={r12_ck:.5f})")

# ============================================================
# 4) 灵敏度与蒙特卡洛
# ============================================================
print("=" * 60, "\n[灵敏度与稳健性]")
nfun10 = sic_n_factory(R["sic"]["series10"]["eps_inf"])
d1c, _ = d_pair(sic["a1"]["sig"], sic["a1"]["pk"], sic["a1"]["vy"], 10.0, nfun10)
d_base = float(np.mean(d1c))
sens = {}
for label, nf in [("n+1%", lambda s: nfun10(s) * 1.01), ("n-1%", lambda s: nfun10(s) * 0.99)]:
    dp, _ = d_pair(sic["a1"]["sig"], sic["a1"]["pk"], sic["a1"]["vy"], 10.0, nf)
    sens[label] = float(100 * (np.mean(dp) - d_base) / d_base)
for label, thx in [("θ+0.5°", 10.5), ("θ-0.5°", 9.5)]:
    dp, _ = d_pair(sic["a1"]["sig"], sic["a1"]["pk"], sic["a1"]["vy"], thx, nfun10)
    sens[label] = float(100 * (np.mean(dp) - d_base) / d_base)
R["sens"] = sens
print("  灵敏度:", {k: f"{v:+.3f}%" for k, v in sens.items()})

mc = []
rng2 = np.random.default_rng(7)
sig1, rr1 = load("附件1.xlsx")
sig1, rr1, sm1, base1 = prep(sig1, rr1, 1200, 3300)
for k in range(30):
    rr_n = rr1 + rng2.normal(0, 0.3, len(rr1))
    sm_n = savgol_filter(rr_n, 31, 3)
    base_n = savgol_filter(rr_n, 301, 2)
    pk_n, vy_n = extrema(sig1, sm_n, prom=0.35, dist=12, base=base_n)
    try:
        idxn = np.sort(np.concatenate([pk_n, vy_n]))
        best_n = scan_sic_series(sig1[idxn], 10.0)
        mc.append(best_n["d_um"])
    except Exception:
        pass
mc = np.array(mc)
med = float(np.median(mc))
mad = float(np.median(np.abs(mc - med)) * 1.4826)
R["mc"] = {"mean": float(mc.mean()), "std": float(mc.std(ddof=1)),
           "median": med, "mad_sigma": mad, "n": int(len(mc))}
print(f"  蒙特卡洛(30 次, 0.3% 噪声): median = {med:.4f} ± {mad:.4f} um (mean {mc.mean():.4f} ± {mc.std(ddof=1):.4f})")

# ============================================================
# 5) 图表
# ============================================================
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(sic["a1"]["sig"], sic["a1"]["rr"], color="#9db8d9", lw=0.7, label="附件1 原始（10°）")
ax.plot(sic["a1"]["sig"], sic["a1"]["sm"], color="#1f4e79", lw=1.6, label="附件1 SG 滤波")
ax.plot(sic["a2"]["sig"], sic["a2"]["rr"], color="#f4c7a5", lw=0.7, alpha=0.9, label="附件2 原始（15°）")
ax.plot(sic["a2"]["sig"], sic["a2"]["sm"], color="#c55a11", lw=1.6, label="附件2 SG 滤波")
ax.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax.set_ylabel("反射率 (%)")
ax.legend(fontsize=9, ncol=2, frameon=False); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_spectra_sic.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.plot(sic["a1"]["sig"], sic["a1"]["sm"], color="#1f4e79", lw=1.4, label="附件1（10°，SG 滤波）")
ax.plot(sic["a1"]["sig"][sic["a1"]["pk"]], sic["a1"]["sm"][sic["a1"]["pk"]], "v", color="#c00000", ms=6, label="波峰")
ax.plot(sic["a1"]["sig"][sic["a1"]["vy"]], sic["a1"]["sm"][sic["a1"]["vy"]], "^", color="#2e7d32", ms=6, label="波谷")
ax.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax.set_ylabel("反射率 (%)")
ax.legend(fontsize=9, frameon=False); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_extrema_sic.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.hist(sic_res["a1"]["d_pairs"], bins=8, alpha=0.65, color="#1f4e79",
        label=f"10°（均值 {np.mean(sic_res['a1']['d_pairs']):.3f} μm）")
ax.hist(sic_res["a2"]["d_pairs"], bins=8, alpha=0.65, color="#c55a11",
        label=f"15°（均值 {np.mean(sic_res['a2']['d_pairs']):.3f} μm）")
ax.axvline(np.mean(sic_res["a1"]["d_pairs"]), color="#1f4e79", ls="--", lw=1.2)
ax.axvline(np.mean(sic_res["a2"]["d_pairs"]), color="#c55a11", ls="--", lw=1.2)
ax.set_xlabel("由相邻极值对反演的厚度 $d_i$ (μm)"); ax.set_ylabel("频数")
ax.legend(fontsize=9, frameon=False); ax.grid(ls="--", alpha=0.35, axis="y")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_d_hist_sic.png", dpi=200); plt.close(fig)

sigma, rr, th = ds[0]
Rm_airy = R_model(sigma, d_si, n_epi_si, n_sub_si, th, deco=deco_si(sigma))
Rm_2b = R_model(sigma, p2[0], si_models(p2)[1], si_models(p2)[2], th,
                deco=si_models(p2)[3](sigma), two_beam=True)
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(si["a3"]["sig"], si["a3"]["rr"], color="#aaaaaa", lw=0.6, label="附件3 实测（10°）")
ax.plot(sigma, Rm_airy, color="#c00000", lw=1.4, label=f"多光束 Airy 拟合（$d$={d_si:.3f} μm）")
ax.plot(sigma, Rm_2b, color="#1f4e79", lw=1.1, ls="--", label=f"双光束拟合（$d$={p2[0]:.3f} μm）")
ax.set_xlim(700, 4000)
ax.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax.set_ylabel("反射率 (%)")
ax.legend(fontsize=9, frameon=False); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_si_fit.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(sigma, Rm_airy - rr, color="#c00000", lw=0.9, label="多光束残差")
ax.plot(sigma, Rm_2b - rr, color="#1f4e79", lw=0.9, ls="--", label="双光束残差")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlim(700, 4000)
ax.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax.set_ylabel("拟合残差 (%)")
ax.legend(fontsize=9, frameon=False); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_residual_si.png", dpi=200); plt.close(fig)

sig_grid = np.linspace(1200, 3300, 400)
fig, ax = plt.subplots(figsize=(6.6, 3.2))
ax.plot(sig_grid, sic_n_factory(R["sic"]["series10"]["eps_inf"])(sig_grid), color="#1f4e79", lw=1.6,
        label=f"TO–LO 单振子（$\\varepsilon_\\infty$={R['sic']['series10']['eps_inf']:.2f}）")
ax.axhline(2.55, color="#888888", ls=":", lw=1.0)
ax.text(1250, 2.575, "文献平台值 2.55", fontsize=9, color="#666666")
ax.set_xlabel(r"波数 $\sigma$ (cm$^{-1}$)"); ax.set_ylabel("折射率 $n(\\sigma)$")
ax.legend(fontsize=9, frameon=False); ax.grid(ls="--", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_dispersion_sic.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.hist(mc, bins=10, color="#1f4e79", alpha=0.75)
ax.axvline(mc.mean(), color="#c00000", ls="--", lw=1.4, label=f"均值 {mc.mean():.3f} μm")
ax.axvline(mc.mean() + mc.std(), color="#c55a11", ls=":", lw=1.2, label=f"±1σ = {mc.std():.3f} μm")
ax.axvline(mc.mean() - mc.std(), color="#c55a11", ls=":", lw=1.2)
ax.set_xlabel("反演厚度 (μm)"); ax.set_ylabel("频数")
ax.legend(fontsize=9, frameon=False); ax.grid(ls="--", alpha=0.35, axis="y")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_mc.png", dpi=200); plt.close(fig)

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print("=" * 60, "\n完成：results.json + figs/ 已生成")
