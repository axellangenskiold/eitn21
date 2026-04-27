"""Clean Task 3 receiver, v3: length-symbol channel est + phase-slope correction."""
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, savgol_filter
from numpy.lib.stride_tricks import sliding_window_view
from commpy.channelcoding import Trellis, viterbi_decode

data = scipy.io.loadmat('Signals_task3/signal3.mat')
r = data['R'].flatten()

fs, fc = 44100, 10000
Nsc, Ncp = 128, 20
Tsym = 58e-3
RUN_EXHAUSTIVE = False

n = np.arange(len(r))
r_I = 2 * r * np.cos(2 * np.pi * fc * n / fs)
r_Q = -2 * r * np.sin(2 * np.pi * fc * n / fs)
b, a = butter(8, 0.05)
r_I_f = filtfilt(b, a, r_I)
r_Q_f = filtfilt(b, a, r_Q)

D = fs * Tsym / Nsc
idx = np.arange(0, len(r_I_f), D)
r_bb = (np.interp(idx, np.arange(len(r_I_f)), r_I_f)
        + 1j * np.interp(idx, np.arange(len(r_Q_f)), r_Q_f))

T = Nsc // 2
wins = sliding_window_view(r_bb, 2 * T)
gamma = np.sum(wins[:, :T] * np.conj(wins[:, T:]), axis=1)
phi1 = np.sum(np.abs(wins[:, :T])**2, axis=1)
phi2 = np.sum(np.abs(wins[:, T:])**2, axis=1)
mu = np.abs(gamma) / np.sqrt(phi1 * phi2 + 1e-10)
peak = int(np.argmax(mu))
print(f"Sync peak at baseband sample {peak}")

# matlab uses deprecated method to generate this
# thus it needs to be hardcoded to work in python
def get_P():
    return np.array([
        1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1
    ])


def estimate_channel(X_length):
    """Estimate H[k] from the length OFDM symbol only.

    In practice this is more stable on this recording than averaging in the
    pilot estimate, which injects errors on the later payload symbols.
    """
    known_idx = np.arange(15, Nsc)
    H_known = X_length[known_idx] / (1 + 1j)
    H = (np.interp(np.arange(Nsc), known_idx, H_known.real)
         + 1j * np.interp(np.arange(Nsc), known_idx, H_known.imag))
    return savgol_filter(H.real, 7, 3) + 1j * savgol_filter(H.imag, 7, 3)


def nearest_qpsk(symbols):
    return np.where(symbols.real >= 0, 1.0, -1.0) + 1j * np.where(symbols.imag >= 0, 1.0, -1.0)


def correct_phase_slope(X_data, used_qpsk, n_iter=2):
    """Correct a per-symbol linear phase ramp across subcarriers.

    A small residual timing mismatch shows up as a phase slope over subcarrier
    index rather than as a common phase offset. This is estimated from hard
    QPSK decisions and removed iteratively.
    """
    corrected = X_data.copy()
    k_full = np.arange(Nsc)
    active_counts = [Nsc] * corrected.shape[0]
    active_counts[-1] = used_qpsk - Nsc * (corrected.shape[0] - 1)

    for _ in range(n_iter):
        for i, n_active in enumerate(active_counts):
            z = corrected[i, :n_active]
            q = nearest_qpsk(z)
            phase = np.unwrap(np.angle(z * np.conj(q)))
            if len(phase) < 8:
                continue
            slope, intercept = np.polyfit(np.arange(n_active), phase, 1)
            corrected[i] *= np.exp(-1j * (slope * k_full + intercept))

    return corrected


def try_decode(t0, g2_poly, search_length=None):
    trellis = Trellis(memory=np.array([5]), g_matrix=np.array([[0o77, g2_poly]]))
    if t0 < 0 or len(r_bb) - t0 < 3 * (Nsc + Ncp):
        return None
    n_sym = (len(r_bb) - t0) // (Nsc + Ncp)
    blocks = r_bb[t0 : t0 + n_sym * (Nsc + Ncp)].reshape(n_sym, Nsc + Ncp)
    X = np.fft.fft(blocks[:, Ncp:], axis=1)

    H_full = estimate_channel(X[1])

    # Decode length
    X_len_eq = X[1, :15] / H_full[:15]
    bits_len = np.empty(30, dtype=int)
    bits_len[0::2] = (X_len_eq.real < 0).astype(int)
    bits_len[1::2] = (X_len_eq.imag < 0).astype(int)
    dec_len = viterbi_decode(bits_len.astype(float), trellis, tb_depth=15, decoding_type='hard')
    l_m_decoded = int(''.join(dec_len[:10].astype(int).astype(str)), 2)

    candidates = [l_m_decoded] if search_length is None else search_length

    best_local = None
    for l_m in candidates:
        if l_m < 1 or l_m > 400:
            continue
        n_data = int(np.ceil((7 * l_m + 5) / Nsc))
        if n_sym < 2 + n_data:
            continue
        used_qpsk = 7 * l_m + 5
        X_data = X[2 : 2 + n_data] / H_full
        X_data = correct_phase_slope(X_data, used_qpsk)
        bits_data = np.empty(X_data.size * 2, dtype=int)
        bits_data[0::2] = (X_data.real < 0).flatten().astype(int)
        bits_data[1::2] = (X_data.imag < 0).flatten().astype(int)
        needed = 14 * l_m + 10
        dec = viterbi_decode(bits_data[:needed].astype(float), trellis, decoding_type='hard')
        msg_bits = dec[:7 * l_m].astype(int)
        chars = []
        for i in range(l_m):
            b7 = msg_bits[7 * i : 7 * i + 7]
            if len(b7) == 7:
                chars.append(chr(int(''.join(b7.astype(str)), 2)))
        msg = ''.join(chars)
        printable = sum(32 <= ord(c) <= 126 or c in '\n\r\t' for c in msg)
        # reward alphanumeric/space ratio more
        letters = sum(c.isalpha() or c == ' ' for c in msg)
        score = (printable + letters) / max(1, 2 * len(msg))
        if best_local is None or score > best_local['score']:
            best_local = {'t0': t0, 'l_m': l_m, 'l_m_dec': l_m_decoded,
                          'msg': msg, 'score': score, 'H': H_full, 'X': X}
    return best_local


best = None
# Try both poly conventions and search sync offset
for g2 in (0o45, 0o51):
    for offset in range(-Ncp - 5, 5):
        t0 = peak + offset
        res = try_decode(t0, g2)
        if res is None:
            continue
        if best is None or res['score'] > best['score']:
            best = {**res, 'g2': g2}

print(f"\nBest single decode: g2=0o{best['g2']:o}, t0={best['t0']}, "
      f"l_m(raw)={best['l_m_dec']}, l_m={best['l_m']}, score={best['score']:.3f}")
print(f"Message: {best['msg']}")

# Try length search too (assume length decode may be wrong, search over plausible values)
if RUN_EXHAUSTIVE:
    print("\n--- Exhaustive length search ---")
    best2 = None
    for g2 in (0o45, 0o51):
        for offset in range(-Ncp - 5, 5):
            t0 = peak + offset
            res = try_decode(t0, g2, search_length=range(50, 200))
            if res is None:
                continue
            if best2 is None or res['score'] > best2['score']:
                best2 = {**res, 'g2': g2}

    print(f"Best: g2=0o{best2['g2']:o}, t0={best2['t0']}, l_m={best2['l_m']}, score={best2['score']:.3f}")
    print(f"Message: {best2['msg']}")
else:
    print("\n--- Exhaustive length search skipped ---")
