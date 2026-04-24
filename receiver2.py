import numpy as np
import scipy.io
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt, savgol_filter
from commpy.channelcoding import Trellis, viterbi_decode

mat_path = 'Signals_task3/signal3.mat'
data = scipy.io.loadmat(mat_path)
r = data['R'].flatten()
fs, fc, Nsc, Ncp, Tsym = 44100, 10000, 128, 20, 58e-3

n = np.arange(len(r))
r_I = 2 * r * np.cos(2 * np.pi * fc * n / fs)
r_Q = -2 * r * np.sin(2 * np.pi * fc * n / fs)
b, a = butter(8, 0.05)
r_I_filt = filtfilt(b, a, r_I)
r_Q_filt = filtfilt(b, a, r_Q)

D = fs * Tsym / Nsc
idx = np.arange(0, len(r_I_filt), D)
r_I_down = np.interp(idx, np.arange(len(r_I_filt)), r_I_filt)
r_Q_down = np.interp(idx, np.arange(len(r_Q_filt)), r_Q_filt)
r_bb = r_I_down + 1j * r_Q_down

Tper = Nsc // 2
wins = sliding_window_view(r_bb, 2 * Tper)
gamma = np.sum(wins[:, :Tper] * np.conj(wins[:, Tper:]), axis=1)
phi1 = np.sum(np.abs(wins[:, :Tper])**2, axis=1)
phi2 = np.sum(np.abs(wins[:, Tper:])**2, axis=1)
mu = np.abs(gamma) / np.sqrt(phi1 * phi2 + 1e-10)
T0 = np.argmax(mu)
Nofdm = Nsc + Ncp

# Try multiple offsets and both polynomials
best = None
for offset in range(-Ncp - 5, 6):
    t0 = (T0 - Ncp) + offset
    if t0 < 0: continue
    n_sym = (len(r_bb) - t0) // Nofdm
    if n_sym < 8: continue
    blocks = r_bb[t0 : t0 + n_sym*Nofdm].reshape(n_sym, Nofdm)
    X = np.fft.fft(blocks[:, Ncp:], axis=1)

    # Channel estimation
    H_from_len = X[1, 15:] / (1+1j)
    len_pos = np.arange(15, 128)
    all_pos = np.arange(Nsc)
    H_init = (np.interp(all_pos, len_pos, np.abs(H_from_len))
            * np.exp(1j * np.interp(all_pos, len_pos, np.unwrap(np.angle(H_from_len)))))

    # Use ALL even subcarriers from pilot to improve H estimate
    raw_all = X[0, 0::2] # all 64 even subcarrier pilot values
    ratio_all = raw_all / (2 * H_init[::2])
    P_all = np.where(np.abs(ratio_all.real) >= np.abs(ratio_all.imag),
                    np.sign(ratio_all.real), np.sign(ratio_all.imag))
    P_all = np.where(P_all == 0, 1.0, P_all)
    H_from_pilot = raw_all / (2 * P_all) # H at all 64 even subcarriers

    # Build combined H
    H_full = np.empty(Nsc, dtype=complex)
    for k in range(Nsc):
        if k >= 15 and k % 2 == 0:
            # both estimates available: average them
            H_full[k] = 0.5 * (H_from_len[k-15] + H_from_pilot[k//2])
        elif k >= 15:
            H_full[k] = H_from_len[k-15]
        elif k % 2 == 0:
            H_full[k] = H_from_pilot[k//2]
        else:
            H_full[k] = 0 # fill below
    for k in range(1, 15, 2):
        H_full[k] = 0.5 * (H_full[k-1] + H_full[k+1])
    H_full = savgol_filter(H_full.real, 7, 3) + 1j * savgol_filter(H_full.imag, 7, 3)

    X_data = X[1:]
    X_eq = X_data / H_full
    bits = []
    for sym in X_eq.flatten():
        bits.append(1 if sym.real < 0 else 0)
        bits.append(1 if sym.imag < 0 else 0)
    bits_arr = np.array(bits).astype(float)
    bits_len = bits_arr[:256]
    bits_data = bits_arr[256:]

    for g2 in (0o45, 0o51):
        trellis = Trellis(memory=np.array([5]), g_matrix=np.array([[0o77, g2]]))
        for l_m in [87, 88, 86, 85, 90]:   # try known length and nearby
            needed = 14 * l_m + 10
            if needed > len(bits_data): continue
            decoded = viterbi_decode(bits_data[:needed], trellis, decoding_type='hard')
            msg_bits = decoded[:l_m * 7].astype(int).astype(str)
            message = ''.join(
                chr(int(''.join(msg_bits[i:i+7]), 2))
                for i in range(0, l_m*7, 7) if len(msg_bits[i:i+7]) == 7
            )
            letters = sum(c.isalpha() or c == ' ' for c in message)
            score = letters / max(1, len(message))
            if best is None or score > best['score']:
                best = {'t0': t0, 'offset': offset, 'g2': g2, 'l_m': l_m, 'msg': message, 'score': score}

print(f"Best: offset={best['offset']}, g2=0o{best['g2']:o}, l_m={best['l_m']}, score={best['score']:.3f}")
print(best['msg'])