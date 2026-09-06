"""
Experiment 6 - Digital line coding and power spectral density
MAIN DRIVER.  Produces every figure and every table used in the report.

Run:  python3 run_experiment6.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linecodes import (A, TB, L, FS, CODES, SHORT, COLOR, ENCODERS, encode,
                       welch_psd, theory_psd, THEORY_TEX, per_bit_level,
                       running_digital_sum, transition_stats, dc_metrics,
                       dc_power_fraction, containment_bw, first_null, db,
                       step_plot_arrays, prbs_bits, parseval_check,
                       VALIDATION_WORD, DEMO_BITS)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
RES = os.path.join(HERE, "..", "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
})

# ==========================================================================
# TASK 19 - one common bit sequence drives all six encoders
# ==========================================================================
N_BITS = 4096
bits = prbs_bits(N_BITS, seed=0x2AB5)   # PRBS-15, x^15 + x^14 + 1
print(f"Bit stream: {N_BITS} bits, ones = {bits.sum()} "
      f"({100*bits.mean():.2f} %)")

wave = {c: encode(bits, c) for c in CODES}
demo = {c: encode(DEMO_BITS, c) for c in CODES}
val = {c: encode(VALIDATION_WORD, c) for c in CODES}

# ==========================================================================
# FIGURE 1 - aligned line-code waveforms (required visualisation #1)
# ==========================================================================
fig, ax = plt.subplots(7, 1, figsize=(9.5, 9.0), sharex=True,
                       gridspec_kw={"height_ratios": [0.45] + [1] * 6})
nb = len(DEMO_BITS)

# top strip: the bit stream itself
ax[0].set_ylim(0, 1)
ax[0].set_yticks([])
for i, b in enumerate(DEMO_BITS):
    ax[0].text(i + 0.5, 0.35, str(b), ha="center", va="center",
               fontsize=10, fontweight="bold")
    ax[0].axvline(i, color="0.85", lw=0.7)
ax[0].set_title("Common bit sequence  b[k] = 1011 0010 1110 0010", fontsize=10,
                fontweight="bold")
ax[0].set_ylabel("bits", rotation=0, ha="right", va="center")
ax[0].grid(False)

for j, c in enumerate(CODES, start=1):
    t, y = step_plot_arrays(demo[c])
    ax[j].step(t, y, where="post", color=COLOR[c], lw=1.7)
    for i in range(nb + 1):
        ax[j].axvline(i, color="0.85", lw=0.7, zorder=0)
    ax[j].axhline(0, color="0.55", lw=0.8, ls=":")
    ax[j].set_ylim(-1.55, 1.55)
    ax[j].set_yticks([-1, 0, 1])
    ax[j].set_ylabel(SHORT[c], rotation=0, ha="right", va="center",
                     fontweight="bold")
ax[-1].set_xlabel("time  t / Tb   (bit intervals)")
ax[-1].set_xlim(0, nb)
ax[-1].set_xticks(range(0, nb + 1))
fig.suptitle("Fig. 1  Six line codes generated from one common bit sequence "
             f"(A = {A:.0f} V, {L} samples/bit)", fontsize=11, y=0.995)
fig.savefig(os.path.join(FIG, "fig1_waveforms.png"))
plt.close(fig)

# ==========================================================================
# TASK 20 - Welch PSD estimate of every code
# ==========================================================================
NPERSEG = 64 * L            # 64-bit segments -> resolution ~ Rb/64
psd = {}
for c in CODES:
    f, P, nseg = welch_psd(wave[c], fs=FS, nperseg=NPERSEG, overlap=0.5)
    psd[c] = (f, P)
print(f"Welch: nperseg = {NPERSEG} samples ({NPERSEG//L} bits), "
      f"50% overlap, Hamming window, {nseg} segments averaged")

# Normalisation reference.  Every curve is plotted as 10*log10(S(f)/(A^2*Tb)),
# i.e. 0 dB = the peak of the polar-NRZ spectrum, A^2*Tb.  Using this exact
# constant (rather than the largest measured bin) keeps the 0 dB line
# deterministic and free of Welch estimation noise.
REF = A ** 2 * TB

# --- FIGURE 2: measured vs theoretical PSD, one panel per code ------------
fig, axs = plt.subplots(2, 3, figsize=(12.5, 6.4), sharex=True, sharey=True)
for k, c in enumerate(CODES):
    a = axs.flat[k]
    f, P = psd[c]
    m = f <= 3.0
    a.plot(f[m], db(P[m], REF), color=COLOR[c], lw=1.2,
           label="Welch estimate")
    ft = np.linspace(1e-4, 3.0, 3000)
    a.plot(ft, db(theory_psd(ft, c), REF), "k--", lw=0.9, alpha=0.8,
           label="theory (continuous part)")
    a.set_title(c, fontsize=10, fontweight="bold")
    a.set_ylim(-62, 14)
    a.set_xlim(0, 3)
    if k >= 3:
        a.set_xlabel("normalised frequency  f / Rb")
    if k % 3 == 0:
        a.set_ylabel("normalised PSD  (dB)")
    a.legend(fontsize=7, loc="upper right", framealpha=0.9)
fig.suptitle("Fig. 2  Normalised power spectral density (Welch, Hamming, "
             f"{NPERSEG//L}-bit segments, 50 % overlap) vs theory",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_psd_grid.png"))
plt.close(fig)

# --- FIGURE 3: all six overlaid -------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.2))
for c in CODES:
    f, P = psd[c]
    m = f <= 2.5
    ax.plot(f[m], db(P[m], REF), color=COLOR[c], lw=1.3, label=c)
ax.axvline(1.0, color="0.4", ls=":", lw=1)
ax.text(1.01, 2, "f = Rb", fontsize=8, color="0.35")
ax.set_xlabel("normalised frequency  f / Rb")
ax.set_ylabel("normalised PSD  (dB)")
ax.set_ylim(-55, 14)
ax.set_xlim(0, 2.5)
ax.legend(fontsize=8, ncol=2)
ax.set_title("Fig. 3  Normalised PSD of all six line codes on a common "
             "0 dB reference", fontsize=11, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig3_psd_overlay.png"))
plt.close(fig)

# --- FIGURE 4 (inset): the DC region, linear frequency, zoomed ------------
fig, ax = plt.subplots(figsize=(9.0, 4.4))
for c in CODES:
    f, P = psd[c]
    m = f <= 0.30
    ax.plot(f[m], db(P[m], REF), color=COLOR[c], lw=1.5, label=c)
ax.set_xlabel("normalised frequency  f / Rb")
ax.set_ylabel("normalised PSD  (dB)")
ax.set_title("Fig. 4  Low-frequency detail: which codes put power at DC",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=2, loc="lower right")
ax.set_xlim(0, 0.30)
fig.savefig(os.path.join(FIG, "fig4_psd_dc_zoom.png"))
plt.close(fig)

# ==========================================================================
# TASK 21 - average level and running digital sum
# ==========================================================================
fig, axs = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
NSHOW = 400
STYLE = {c: dict(lw=1.2, ls="-") for c in CODES}
STYLE["Manchester"].update(lw=2.6, ls="-")
STYLE["Differential Manchester"].update(lw=1.2, ls="--")
for c in CODES:
    rds = running_digital_sum(wave[c])[:NSHOW]
    axs[0].plot(np.arange(1, len(rds) + 1), rds, color=COLOR[c], label=c,
                **STYLE[c])
axs[0].axhline(0, color="0.5", lw=0.8, ls=":")
axs[0].set_ylabel("running digital sum  (V·Tb)")
axs[0].legend(fontsize=8, ncol=2)
axs[0].set_title("Fig. 5  Running digital sum: unbounded wander (unipolar) vs "
                 "bounded RDS (balanced codes)", fontsize=11, fontweight="bold")

for c in CODES:
    if c == "Unipolar NRZ":
        continue
    rds = running_digital_sum(wave[c])[:NSHOW]
    axs[1].plot(np.arange(1, len(rds) + 1), rds, color=COLOR[c], label=c,
                **STYLE[c])
axs[1].axhline(0, color="0.5", lw=0.8, ls=":")
axs[1].set_xlabel("bit index  k")
axs[1].set_ylabel("RDS  (zoom, unipolar removed)")
axs[1].legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_rds.png"))
plt.close(fig)

# ==========================================================================
# TASK 22 - long run of identical bits
# ==========================================================================
RUN = 24
run_ones = np.ones(RUN, dtype=int)
run_zeros = np.zeros(RUN, dtype=int)
long_bits = np.concatenate((np.array([1, 0, 1, 0]), run_ones,
                            np.array([1, 0]), run_zeros,
                            np.array([1, 0, 1, 0])))

longw = {c: encode(long_bits, c) for c in CODES}

fig, ax = plt.subplots(7, 1, figsize=(11.5, 9.2), sharex=True,
                       gridspec_kw={"height_ratios": [0.4] + [1] * 6})
nb = len(long_bits)
ax[0].set_ylim(0, 1)
ax[0].set_yticks([])
ax[0].grid(False)
ax[0].axvspan(4, 4 + RUN, color="#ffe08a", alpha=0.55)
ax[0].axvspan(6 + RUN, 6 + 2 * RUN, color="#a8d8ff", alpha=0.55)
ax[0].text(4 + RUN / 2, 0.5, f"{RUN} consecutive 1s", ha="center", fontsize=9,
           fontweight="bold")
ax[0].text(6 + RUN + RUN / 2, 0.5, f"{RUN} consecutive 0s", ha="center",
           fontsize=9, fontweight="bold")
ax[0].set_ylabel("bits", rotation=0, ha="right", va="center")

for j, c in enumerate(CODES, start=1):
    t, y = step_plot_arrays(longw[c])
    ax[j].axvspan(4, 4 + RUN, color="#ffe08a", alpha=0.35, zorder=0)
    ax[j].axvspan(6 + RUN, 6 + 2 * RUN, color="#a8d8ff", alpha=0.35, zorder=0)
    ax[j].step(t, y, where="post", color=COLOR[c], lw=1.4)
    ax[j].axhline(0, color="0.55", lw=0.8, ls=":")
    ax[j].set_ylim(-1.55, 1.55)
    ax[j].set_yticks([-1, 0, 1])
    ax[j].set_ylabel(SHORT[c], rotation=0, ha="right", va="center",
                     fontweight="bold")
ax[-1].set_xlabel("time  t / Tb   (bit intervals)")
ax[-1].set_xlim(0, nb)
fig.suptitle("Fig. 6  Long-run test: 24 identical 1s followed by 24 identical "
             "0s  (baseline wander and loss of clock)", fontsize=11, y=0.995)
fig.savefig(os.path.join(FIG, "fig6_longrun.png"))
plt.close(fig)

# quantitative long-run table: examine the 1-run and the 0-run separately
def run_metrics(code, runbits):
    x = encode(runbits, code)
    ts = transition_stats(x)
    m = dc_metrics(x)
    return dict(transitions_per_bit=ts["per_bit"],
                longest_flat_bits=ts["longest_flat_bits"],
                mean_level=m["mean_level"],
                rds_drift=m["rds_final"])

longrun_tbl = {c: {"ones": run_metrics(c, run_ones),
                   "zeros": run_metrics(c, run_zeros)} for c in CODES}

# ---- long-run RDS drift figure -------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.9), sharey=True)
for c in CODES:
    axs[0].plot(np.arange(1, RUN + 1), running_digital_sum(encode(run_ones, c)),
                color=COLOR[c], lw=1.5, marker="o", ms=3, label=c)
    axs[1].plot(np.arange(1, RUN + 1), running_digital_sum(encode(run_zeros, c)),
                color=COLOR[c], lw=1.5, marker="o", ms=3, label=c)
axs[0].set_title("24 consecutive 1s", fontsize=10, fontweight="bold")
axs[1].set_title("24 consecutive 0s", fontsize=10, fontweight="bold")
for a in axs:
    a.set_xlabel("bit index within the run")
    a.axhline(0, color="0.5", lw=0.8, ls=":")
axs[0].set_ylabel("running digital sum")
axs[1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle("Fig. 7  Baseline wander during a long run of identical bits",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig7_longrun_rds.png"))
plt.close(fig)

# ==========================================================================
# Power-containment / bandwidth figure
# ==========================================================================
fig, ax = plt.subplots(figsize=(9.0, 4.6))
for c in CODES:
    f, P = psd[c]
    cum = np.concatenate(([0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(f))))
    cum /= cum[-1]
    m = f <= 3.0
    ax.plot(f[m], 100 * cum[m], color=COLOR[c], lw=1.4, label=c)
for lvl in (90, 99):
    ax.axhline(lvl, color="0.55", ls=":", lw=0.9)
    ax.text(2.85, lvl + 1, f"{lvl} %", fontsize=8, color="0.35", ha="right")
ax.set_xlabel("normalised frequency  f / Rb")
ax.set_ylabel("cumulative power  (%)")
ax.set_xlim(0, 3)
ax.set_ylim(0, 102)
ax.legend(fontsize=8, loc="lower right")
ax.set_title("Fig. 8  Fractional power containment vs bandwidth",
             fontsize=11, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig8_containment.png"))
plt.close(fig)

# ==========================================================================
# MANDATORY VALIDATION - the 8-bit word 1011 0010 in every code
# ==========================================================================
fig, ax = plt.subplots(7, 1, figsize=(8.0, 8.6), sharex=True,
                       gridspec_kw={"height_ratios": [0.45] + [1] * 6})
nb = len(VALIDATION_WORD)
ax[0].set_ylim(0, 1)
ax[0].set_yticks([])
ax[0].grid(False)
for i, b in enumerate(VALIDATION_WORD):
    ax[0].text(i + 0.5, 0.35, str(b), ha="center", va="center", fontsize=12,
               fontweight="bold")
    ax[0].axvline(i, color="0.8", lw=0.8)
ax[0].set_title("Validation word  b = 1011 0010  (0xB2)", fontsize=11,
                fontweight="bold")
ax[0].set_ylabel("bits", rotation=0, ha="right", va="center")

for j, c in enumerate(CODES, start=1):
    t, y = step_plot_arrays(val[c])
    ax[j].step(t, y, where="post", color=COLOR[c], lw=2.0)
    for i in range(nb + 1):
        ax[j].axvline(i, color="0.8", lw=0.8, zorder=0)
    for i in range(nb):
        ax[j].axvline(i + 0.5, color="0.9", lw=0.6, ls="--", zorder=0)
    ax[j].axhline(0, color="0.55", lw=0.8, ls=":")
    ax[j].set_ylim(-1.6, 1.6)
    ax[j].set_yticks([-1, 0, 1])
    nt = transition_stats(val[c])["transitions"]
    ax[j].set_ylabel(SHORT[c], rotation=0, ha="right", va="center",
                     fontweight="bold")
    ax[j].text(0.995, 0.92, f"{nt} transitions", transform=ax[j].transAxes,
               ha="right", va="top", fontsize=8, color="0.3")
ax[-1].set_xlabel("time  t / Tb")
ax[-1].set_xlim(0, nb)
ax[-1].set_xticks(range(nb + 1))
fig.suptitle("Fig. 9  Mandatory validation: 1011 0010 encoded in all six "
             "line codes", fontsize=11, y=0.996)
fig.savefig(os.path.join(FIG, "fig9_validation.png"))
plt.close(fig)

# half-bit symbol table for the hand-check
half = {}
for c in CODES:
    x = val[c]
    h = L // 2
    pairs = []
    for k in range(nb):
        f1 = x[k * L]
        f2 = x[k * L + h]
        pairs.append((float(f1), float(f2)))
    half[c] = pairs

# ==========================================================================
# NUMERICAL RESULTS TABLES
# ==========================================================================
summary = {}
for c in CODES:
    f, P = psd[c]
    x = wave[c]
    m = dc_metrics(x)
    ts = transition_stats(x)
    summary[c] = dict(
        levels=int(len(np.unique(np.round(x, 6)))),
        mean_level=m["mean_level"],
        power=m["power"],
        rds_final=m["rds_final"],
        dsv=m["dsv"],
        rds_abs_max=m["rds_abs_max"],
        transitions_per_bit=ts["per_bit"],
        longest_flat_bits=ts["longest_flat_bits"],
        dc_frac=dc_power_fraction(f, P, band=0.02),
        psd_at_dc_db=float(db(P, REF)[0]),
        first_null=first_null(f, P),
        bw90=containment_bw(f, P, 0.90),
        bw99=containment_bw(f, P, 0.99),
        theory=THEORY_TEX[c],
    )

# --------------------------------------------------------------------------
# DIAGNOSTIC TEST 1 - Parseval / scaling check on the Welch estimate
#   mean(x^2) must equal 2 * integral of the two-sided density over f >= 0
# --------------------------------------------------------------------------
parseval = {}
for c in CODES:
    f, P = psd[c]
    t_pow, f_pow = parseval_check(wave[c], f, P)
    parseval[c] = dict(time=t_pow, freq=f_pow,
                       err_pct=100 * (f_pow - t_pow) / t_pow)

# --------------------------------------------------------------------------
# DIAGNOSTIC TEST 2 - is the peak at f = 0 an impulse or continuous power?
#   A Dirac delta is spread over one window main lobe, so its *density*
#   estimate grows in proportion to the segment length; a continuous
#   spectrum does not move.  Sweeping nperseg separates the two.
# --------------------------------------------------------------------------
SEG_BITS = [16, 32, 64, 128, 256]
delta_test = {}
for c in ("Unipolar NRZ", "Polar NRZ", "Manchester", "AMI (bipolar NRZ)"):
    row = []
    for nb_ in SEG_BITS:
        f_, P_, _ = welch_psd(wave[c], fs=FS, nperseg=nb_ * L, overlap=0.5)
        row.append(float(db(P_, REF)[0]))
    delta_test[c] = row

fig, ax = plt.subplots(figsize=(8.6, 4.3))
for c, row in delta_test.items():
    ax.plot(SEG_BITS, row, "o-", color=COLOR[c], lw=1.5, ms=5, label=c)
ideal = np.array(delta_test["Unipolar NRZ"][0]) + \
    10 * np.log10(np.array(SEG_BITS) / SEG_BITS[0])
ax.plot(SEG_BITS, ideal, "k--", lw=1.0,
        label="+3 dB per doubling (ideal impulse)")
ax.set_xscale("log", base=2)
ax.set_xticks(SEG_BITS)
ax.set_xticklabels([str(s) for s in SEG_BITS])
ax.set_xlabel("Welch segment length  (bits)")
ax.set_ylabel("estimated PSD at f = 0  (dB)")
ax.legend(fontsize=8)
ax.set_title("Fig. 10  Diagnostic: the DC peak of unipolar NRZ grows with the\n"
             "segment length (it is an impulse); continuous spectra do not",
             fontsize=10.5, fontweight="bold")
fig.savefig(os.path.join(FIG, "fig10_delta_test.png"))
plt.close(fig)

# theoretical cross-check of the PSD shape (RMS error in dB over 0.05..2.5 Rb)
xcheck = {}
for c in CODES:
    f, P = psd[c]
    m = (f >= 0.05) & (f <= 2.5)
    meas = db(P[m], REF)
    th = db(theory_psd(f[m], c), REF)
    good = (meas > -45) & (th > -45)      # ignore the deep nulls (leakage floor)
    xcheck[c] = float(np.sqrt(np.mean((meas[good] - th[good]) ** 2)))

# ---- write CSVs ----------------------------------------------------------
with open(os.path.join(RES, "table_summary.csv"), "w") as fh:
    fh.write("Line code,Levels,Mean level (V),Power (V^2),DC power frac "
             "(|f|<0.02Rb),PSD at DC (dB),First null (f/Rb),B90 (f/Rb),"
             "B99 (f/Rb),Transitions per bit,Longest flat run (bits),"
             "RDS final,DSV\n")
    for c in CODES:
        s = summary[c]
        fh.write(f"{c},{s['levels']},{s['mean_level']:.4f},{s['power']:.4f},"
                 f"{s['dc_frac']:.4f},{s['psd_at_dc_db']:.1f},"
                 f"{s['first_null']:.3f},{s['bw90']:.3f},{s['bw99']:.3f},"
                 f"{s['transitions_per_bit']:.3f},{s['longest_flat_bits']:.1f},"
                 f"{s['rds_final']:.1f},{s['dsv']:.1f}\n")

with open(os.path.join(RES, "table_longrun.csv"), "w") as fh:
    fh.write("Line code,Run of 1s: transitions/bit,Run of 1s: longest flat "
             "(bits),Run of 1s: mean level,Run of 1s: RDS drift,"
             "Run of 0s: transitions/bit,Run of 0s: longest flat (bits),"
             "Run of 0s: mean level,Run of 0s: RDS drift\n")
    for c in CODES:
        o, z = longrun_tbl[c]["ones"], longrun_tbl[c]["zeros"]
        fh.write(f"{c},{o['transitions_per_bit']:.2f},"
                 f"{o['longest_flat_bits']:.1f},{o['mean_level']:.3f},"
                 f"{o['rds_drift']:.1f},{z['transitions_per_bit']:.2f},"
                 f"{z['longest_flat_bits']:.1f},{z['mean_level']:.3f},"
                 f"{z['rds_drift']:.1f}\n")

with open(os.path.join(RES, "table_validation.csv"), "w") as fh:
    fh.write("Line code," + ",".join(f"b{k}={b}" for k, b in
                                     enumerate(VALIDATION_WORD, 1)) +
             ",Transitions\n")
    for c in CODES:
        cells = []
        for (f1, f2) in half[c]:
            cells.append(f"{f1:+.0f}/{f2:+.0f}" if f1 != f2
                         else f"{f1:+.0f}")
        cells = [s.replace("+0", "0") for s in cells]
        nt = transition_stats(val[c])["transitions"]
        fh.write(f"{c}," + ",".join(cells) + f",{nt}\n")

payload = dict(bits_n=N_BITS, ones=int(bits.sum()), L=L, fs=FS,
               nperseg=NPERSEG, nseg=int(nseg), ref=float(REF),
               summary=summary, longrun=longrun_tbl, xcheck=xcheck,
               parseval=parseval, delta_test=delta_test, seg_bits=SEG_BITS,
               half={c: half[c] for c in CODES},
               validation_word=[int(b) for b in VALIDATION_WORD],
               demo_bits=[int(b) for b in DEMO_BITS],
               long_bits=[int(b) for b in long_bits], run_len=RUN)
with open(os.path.join(RES, "results.json"), "w") as fh:
    json.dump(payload, fh, indent=2)

# ---- console report ------------------------------------------------------
print("\n" + "=" * 108)
print(f"{'Line code':26s} {'mean':>7s} {'DCfrac':>7s} {'PSD(0)':>8s} "
      f"{'null':>6s} {'B90':>6s} {'B99':>6s} {'tr/bit':>7s} {'flat':>6s} "
      f"{'DSV':>7s} {'dB err':>7s}")
print("-" * 108)
for c in CODES:
    s = summary[c]
    print(f"{c:26s} {s['mean_level']:7.4f} {s['dc_frac']:7.4f} "
          f"{s['psd_at_dc_db']:8.1f} {s['first_null']:6.2f} {s['bw90']:6.2f} "
          f"{s['bw99']:6.2f} {s['transitions_per_bit']:7.3f} "
          f"{s['longest_flat_bits']:6.1f} {s['dsv']:7.1f} {xcheck[c]:7.2f}")
print("=" * 108)
print("\nLong-run test (24 identical bits):")
for c in CODES:
    o, z = longrun_tbl[c]["ones"], longrun_tbl[c]["zeros"]
    print(f"  {c:26s} 1s: {o['transitions_per_bit']:5.2f} tr/bit, "
          f"flat {o['longest_flat_bits']:4.1f} Tb, RDS {o['rds_drift']:+6.1f} "
          f"| 0s: {z['transitions_per_bit']:5.2f} tr/bit, "
          f"flat {z['longest_flat_bits']:4.1f} Tb, RDS {z['rds_drift']:+6.1f}")
print("\nParseval check (mean square: time domain vs 2*integral of PSD):")
for c in CODES:
    p = parseval[c]
    print(f"  {c:26s} time {p['time']:.4f}  freq {p['freq']:.4f}  "
          f"error {p['err_pct']:+.2f} %")
print("\nPSD at f=0 (dB) vs Welch segment length (bits):",
      "  ".join(str(s) for s in SEG_BITS))
for c, row in delta_test.items():
    print(f"  {c:26s} " + "  ".join(f"{v:6.1f}" for v in row))
print("\nFigures ->", os.path.abspath(FIG))
print("Tables  ->", os.path.abspath(RES))
