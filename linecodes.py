"""
Experiment 6 - Digital line coding and power spectral density
------------------------------------------------------------
Core library: hand-written line-code generators, a from-scratch Welch
periodogram, and the DC / clocking / bandwidth metrics used by the report.

Deliberately NO toolbox / library encoders are used:
  * every line code is built sample-by-sample from the bit stream,
  * the PSD is a manual Welch estimate (segmenting + windowing + averaging
    of |FFT|^2), not scipy.signal.welch.

Conventions used throughout (state these in the viva):
  A   = 1 V  peak level
  Tb  = 1 s  bit period  (so bit rate Rb = 1 baud; every frequency axis is
             therefore already normalised as f/Rb)
  L   = samples per bit (must be even so that half-bit codes are exact)

Author: Arkaprava Dutta
"""

import numpy as np

# --------------------------------------------------------------------------
# 0.  Global experiment constants
# --------------------------------------------------------------------------
A = 1.0            # peak amplitude, volts
TB = 1.0           # bit period, seconds  -> Rb = 1 bit/s
L = 20             # samples per bit (even: RZ / Manchester need half bits)
FS = L / TB        # sampling rate = 20 x Rb

CODES = [
    "Unipolar NRZ",
    "Polar NRZ",
    "Polar RZ",
    "Manchester",
    "Differential Manchester",
    "AMI (bipolar NRZ)",
]

SHORT = {
    "Unipolar NRZ": "UNI-NRZ",
    "Polar NRZ": "POL-NRZ",
    "Polar RZ": "POL-RZ",
    "Manchester": "MAN",
    "Differential Manchester": "D-MAN",
    "AMI (bipolar NRZ)": "AMI",
}

COLOR = {
    "Unipolar NRZ": "#1f77b4",
    "Polar NRZ": "#d62728",
    "Polar RZ": "#2ca02c",
    "Manchester": "#9467bd",
    "Differential Manchester": "#ff7f0e",
    "AMI (bipolar NRZ)": "#17becf",
}


# --------------------------------------------------------------------------
# 1.  The six encoders  (each returns the waveform, L samples per bit)
# --------------------------------------------------------------------------
def enc_unipolar_nrz(bits, L=L, A=A):
    """1 -> +A for the full bit, 0 -> 0 V."""
    x = np.zeros(len(bits) * L)
    for k, b in enumerate(bits):
        x[k * L:(k + 1) * L] = A if b else 0.0
    return x


def enc_polar_nrz(bits, L=L, A=A):
    """1 -> +A, 0 -> -A, held for the full bit."""
    x = np.zeros(len(bits) * L)
    for k, b in enumerate(bits):
        x[k * L:(k + 1) * L] = A if b else -A
    return x


def enc_polar_rz(bits, L=L, A=A):
    """50 % return-to-zero: 1 -> (+A, 0), 0 -> (-A, 0)."""
    h = L // 2
    x = np.zeros(len(bits) * L)
    for k, b in enumerate(bits):
        x[k * L:k * L + h] = A if b else -A
        x[k * L + h:(k + 1) * L] = 0.0
    return x


def enc_manchester(bits, L=L, A=A):
    """
    Bi-phase-L (Haykin / G. E. Thomas convention):
        1 -> +A for the first half, -A for the second half
        0 -> -A for the first half, +A for the second half
    (IEEE 802.3 uses the opposite mapping; only the sign of the data changes,
     the spectrum and every metric in this experiment are identical.)
    """
    h = L // 2
    x = np.zeros(len(bits) * L)
    for k, b in enumerate(bits):
        first = A if b else -A
        x[k * L:k * L + h] = first
        x[k * L + h:(k + 1) * L] = -first
    return x


def enc_diff_manchester(bits, L=L, A=A, start_level=-A):
    """
    Differential Manchester (conditioned bi-phase):
        * a mid-bit transition is ALWAYS present  -> guaranteed clock,
        * data 0 -> an EXTRA transition at the start of the bit interval,
          data 1 -> NO transition at the start of the bit interval.
    Only transitions carry information, so the waveform is immune to an
    overall polarity inversion. `start_level` is the assumed level just
    before bit 1 (reference level, taken as -A here).
    """
    h = L // 2
    x = np.zeros(len(bits) * L)
    prev_end = start_level          # level at the end of the previous bit
    for k, b in enumerate(bits):
        first = prev_end if b == 1 else -prev_end   # 0 => boundary transition
        x[k * L:k * L + h] = first
        x[k * L + h:(k + 1) * L] = -first           # mandatory mid-bit flip
        prev_end = -first
    return x


def enc_ami(bits, L=L, A=A):
    """
    Alternate Mark Inversion (bipolar NRZ):
        0 -> 0 V,
        1 -> alternately +A and -A (first mark is +A).
    """
    x = np.zeros(len(bits) * L)
    last = -A                       # so the first mark comes out +A
    for k, b in enumerate(bits):
        if b:
            last = -last
            x[k * L:(k + 1) * L] = last
        else:
            x[k * L:(k + 1) * L] = 0.0
    return x


ENCODERS = {
    "Unipolar NRZ": enc_unipolar_nrz,
    "Polar NRZ": enc_polar_nrz,
    "Polar RZ": enc_polar_rz,
    "Manchester": enc_manchester,
    "Differential Manchester": enc_diff_manchester,
    "AMI (bipolar NRZ)": enc_ami,
}


def encode(bits, code, L=L, A=A):
    return ENCODERS[code](np.asarray(bits, dtype=int), L=L, A=A)


# --------------------------------------------------------------------------
# 2.  Manual Welch power spectral density estimate
# --------------------------------------------------------------------------
def hamming(N):
    """Hamming window, written out rather than imported."""
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))


def welch_psd(x, fs=FS, nperseg=None, overlap=0.5, detrend=False,
              onesided=False):
    """
    Welch PSD, implemented from first principles.

      1. split x into overlapping segments of length `nperseg`
      2. multiply each by a Hamming window
      3. take |FFT|^2, scale by 1/(fs * sum(w^2))   (density scaling)
      4. average the segment periodograms

    onesided=False (default) returns the TWO-SIDED density sampled at the
    non-negative frequencies, which is what the textbook expressions
    S(f) = A^2*Tb*sinc^2(f*Tb) etc. give, so measurement and theory can be
    overlaid without a hidden factor of two.  onesided=True doubles the
    interior bins if a single-sided density is wanted instead.

    detrend=False on purpose: the mean must be kept, because the DC content
    of the line code is exactly what this experiment is measuring.
    """
    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = min(len(x), 1024)
    nperseg = int(nperseg)
    step = max(1, int(nperseg * (1 - overlap)))
    w = hamming(nperseg)
    scale = 1.0 / (fs * np.sum(w ** 2))

    starts = range(0, len(x) - nperseg + 1, step)
    nseg = 0
    Pxx = np.zeros(nperseg // 2 + 1)
    for s in starts:
        seg = x[s:s + nperseg]
        if detrend:
            seg = seg - seg.mean()
        X = np.fft.rfft(seg * w)
        P = scale * np.abs(X) ** 2
        if onesided:
            P[1:-1] *= 2.0                   # fold the negative frequencies
        Pxx += P
        nseg += 1
    Pxx /= nseg
    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return f, Pxx, nseg


def parseval_check(x, f, P):
    """
    Sanity test on the density scaling: for a two-sided estimate sampled at
    f >= 0 the mean square of x must equal 2 * integral_0^(fs/2) S(f) df.
    Returns (mean square from time domain, from frequency domain).
    """
    return float(np.mean(np.asarray(x) ** 2)), float(2 * np.trapezoid(P, f))


# --------------------------------------------------------------------------
# 3.  Theoretical PSDs (A = 1, Tb = 1), continuous part only
# --------------------------------------------------------------------------
def _sinc(x):
    """Normalised sinc: sin(pi x)/(pi x)."""
    return np.sinc(x)


def theory_psd(f, code, A=A, Tb=TB):
    f = np.asarray(f, dtype=float)
    if code == "Unipolar NRZ":
        # (A^2 Tb/4) sinc^2(f Tb)  + (A^2/4) delta(f)
        return (A ** 2 * Tb / 4) * _sinc(f * Tb) ** 2
    if code == "Polar NRZ":
        return (A ** 2 * Tb) * _sinc(f * Tb) ** 2
    if code == "Polar RZ":
        return (A ** 2 * Tb / 4) * _sinc(f * Tb / 2) ** 2
    if code == "Manchester":
        return (A ** 2 * Tb) * _sinc(f * Tb / 2) ** 2 * np.sin(np.pi * f * Tb / 2) ** 2
    if code == "Differential Manchester":
        # same spectral envelope as Manchester (differential pre-coding of an
        # i.i.d. equiprobable stream leaves the power spectrum unchanged)
        return (A ** 2 * Tb) * _sinc(f * Tb / 2) ** 2 * np.sin(np.pi * f * Tb / 2) ** 2
    if code == "AMI (bipolar NRZ)":
        return (A ** 2 * Tb) * _sinc(f * Tb) ** 2 * np.sin(np.pi * f * Tb) ** 2
    raise ValueError(code)


THEORY_TEX = {
    "Unipolar NRZ": "S(f) = (A^2*Tb/4)*sinc^2(f*Tb) + (A^2/4)*delta(f)",
    "Polar NRZ": "S(f) = A^2*Tb*sinc^2(f*Tb)",
    "Polar RZ": "S(f) = (A^2*Tb/4)*sinc^2(f*Tb/2)",
    "Manchester": "S(f) = A^2*Tb*sinc^2(f*Tb/2)*sin^2(pi*f*Tb/2)",
    "Differential Manchester": "S(f) = A^2*Tb*sinc^2(f*Tb/2)*sin^2(pi*f*Tb/2)",
    "AMI (bipolar NRZ)": "S(f) = A^2*Tb*sinc^2(f*Tb)*sin^2(pi*f*Tb)",
}


# --------------------------------------------------------------------------
# 4.  Time-domain metrics
# --------------------------------------------------------------------------
def per_bit_level(x, L=L):
    """Average level inside each bit interval (the per-bit 'digital sum')."""
    return x.reshape(-1, L).mean(axis=1)


def running_digital_sum(x, L=L):
    """RDS_n = sum of the per-bit average levels up to bit n."""
    return np.cumsum(per_bit_level(x, L))


def transition_stats(x, L=L, tol=1e-9):
    """
    Transition count / density and the longest interval (in bit periods)
    during which the waveform never changes level - the key self-clocking
    indicator.
    """
    d = np.diff(x)
    idx = np.where(np.abs(d) > tol)[0]
    nbits = len(x) // L
    ntr = len(idx)
    if ntr == 0:
        longest = nbits
    else:
        gaps = np.diff(np.concatenate(([-1], idx, [len(x) - 1])))
        longest = gaps.max() / L
    return dict(transitions=ntr,
                per_bit=ntr / nbits,
                longest_flat_bits=longest)


def dc_metrics(x, L=L):
    rds = running_digital_sum(x, L)
    return dict(mean_level=float(x.mean()),
                power=float(np.mean(x ** 2)),
                rds_final=float(rds[-1]),
                rds_max=float(rds.max()),
                rds_min=float(rds.min()),
                dsv=float(rds.max() - rds.min()),
                rds_abs_max=float(np.abs(rds).max()))


# --------------------------------------------------------------------------
# 5.  Spectral metrics
# --------------------------------------------------------------------------
def dc_power_fraction(f, P, band=0.02):
    """Fraction of the total power that sits in |f| < band*Rb (the 'DC lump')."""
    tot = np.trapezoid(P, f)
    m = f <= band
    return float(np.trapezoid(P[m], f[m]) / tot)


def containment_bw(f, P, frac=0.9):
    """Smallest B such that the power in [0, B] is `frac` of the total."""
    c = np.concatenate(([0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(f))))
    c /= c[-1]
    i = int(np.searchsorted(c, frac))
    if i == 0:
        return float(f[0])
    if i >= len(f):
        return float(f[-1])
    # linear interpolation between the bracketing bins
    f0, f1, c0, c1 = f[i - 1], f[i], c[i - 1], c[i]
    return float(f0 + (frac - c0) * (f1 - f0) / (c1 - c0))


def _movavg(y, w):
    w = int(w) | 1                                    # force odd
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def first_null(f, P, fmin=0.15, fmax=3.0, depth_db=15.0, span=4):
    """
    Frequency of the first spectral null above `fmin`.

    The estimate is smoothed, then a null is accepted only if it is a local
    minimum over +/- `span` bins AND sits at least `depth_db` below the
    largest CONTINUOUS-spectrum value in the search band (the DC line of
    unipolar NRZ is excluded from that reference, otherwise the impulse at
    f = 0 would make every ordinary lobe look like a null).
    """
    Ps = _movavg(P, 3)
    band = (f >= fmin) & (f <= fmax)
    ref = Ps[band].max()
    thr = ref * 10 ** (-depth_db / 10)
    idx = np.where(band)[0]
    for i in idx[span:-span]:
        seg = Ps[i - span:i + span + 1]
        if Ps[i] <= seg.min() and Ps[i] < thr:
            return float(f[i])
    return float("nan")


def db(P, ref=None):
    ref = P.max() if ref is None else ref
    return 10 * np.log10(np.maximum(P, 1e-20) / ref)


# --------------------------------------------------------------------------
# 6.  Helpers
# --------------------------------------------------------------------------
def time_axis(x, L=L, Tb=TB):
    return np.arange(len(x)) * (Tb / L)


def step_plot_arrays(x, L=L, Tb=TB):
    """Return (t, y) that draw the waveform with true vertical edges."""
    t = np.arange(len(x) + 1) * (Tb / L)
    y = np.concatenate((x, [x[-1]]))
    return t, y


def prbs15(n, seed=0x2AB5):
    """
    Maximal-length PRBS-15 (generator polynomial x^15 + x^14 + 1), written
    out as a Fibonacci LFSR.  Period 2^15 - 1 = 32767, so the stream is
    very nearly equiprobable and - crucially - EXACTLY reproducible, which
    lets the MATLAB and Python versions of this experiment be compared
    bit for bit instead of only statistically.
    """
    reg = seed & 0x7FFF or 1
    out = np.empty(n, dtype=int)
    for k in range(n):
        b = reg & 1
        out[k] = b
        fb = b ^ ((reg >> 1) & 1)
        reg = (reg >> 1) | (fb << 14)
    return out


def prbs_bits(n, seed=0x2AB5):
    """Repeatable pseudo-random equiprobable bit stream (PRBS-15)."""
    return prbs15(n, seed)


VALIDATION_WORD = np.array([1, 0, 1, 1, 0, 0, 1, 0])   # 0xB2 = 178
DEMO_BITS = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0])
