"""Clean Task 3 receiver, v2: combine pilot + length symbol for channel est."""
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


def estimate_channel(X_pilot, X_length):
    """H estimate combining length-symbol (known 1+1j) and pilot-symbol (known |value|=2)."""
    # Length symbol: subcarriers 15..127 are 1+1j
    known_idx = np.arange(15, Nsc)
    H_from_length = X_length[known_idx] / (1 + 1j)

    # Pilot symbol: even subcarriers have 2*P (P = +-1). Only |H| is easily recovered
    # per subcarrier; the sign of P is unknown. Use the length-based H (after smoothing)
    # to resolve the sign, then take the pilot measurement / (2P) for even subcarriers.
    even_idx = np.arange(0, Nsc, 2)

    # Initial smooth H from length symbol (interpolate to all subcarriers)
    H_init = (np.interp(np.arange(Nsc), known_idx, H_from_length.real)
              + 1j * np.interp(np.arange(Nsc), known_idx, H_from_length.imag))

    # Recover P[k] for even subcarriers
    #   X_pilot[2k] = 2 * P[k] * H[2k]  ->  P[k] = sign(Re(X_pilot[2k] / H[2k] / 2))
    raw = X_pilot[even_idx] / (2 * H_init[even_idx])
    # Use whichever of real/imag has larger magnitude (more reliable)
    P = get_P()
    H_from_pilot = X_pilot[even_idx] / (2 * P)

    # Combine: pilot gives H at even subcarriers, length gives H at 15..127
    H = np.empty(Nsc, dtype=complex)
    for k in range(Nsc):
        if k in known_idx and k in even_idx:
            # average the two estimates
            j_len = np.where(known_idx == k)[0][0]
            j_pil = np.where(even_idx == k)[0][0]
            H[k] = 0.5 * (H_from_length[j_len] + H_from_pilot[j_pil])
        elif k in known_idx:
            j_len = np.where(known_idx == k)[0][0]
            H[k] = H_from_length[j_len]
        elif k in even_idx:
            j_pil = np.where(even_idx == k)[0][0]
            H[k] = H_from_pilot[j_pil]
        else:
            H[k] = 0  # odd subcarrier below index 15, need interpolation
    # Fill odd subcarriers below 15 by linear interpolation from nearest known
    for k in range(1, 15, 2):
        H[k] = 0.5 * (H[k - 1] + H[k + 1])

    # Light smoothing to reduce noise
    if len(H) >= 7:
        H = (savgol_filter(H.real, 7, 3) + 1j * savgol_filter(H.imag, 7, 3))
    return H


def try_decode(t0, g2_poly, search_length=None):
    trellis = Trellis(memory=np.array([5]), g_matrix=np.array([[0o77, g2_poly]]))
    if t0 < 0 or len(r_bb) - t0 < 3 * (Nsc + Ncp):
        return None
    n_sym = (len(r_bb) - t0) // (Nsc + Ncp)
    blocks = r_bb[t0 : t0 + n_sym * (Nsc + Ncp)].reshape(n_sym, Nsc + Ncp)
    X = np.fft.fft(blocks[:, Ncp:], axis=1)

    H_full = estimate_channel(X[0], X[1])

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
        X_data = X[2 : 2 + n_data] / H_full
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
