#!/usr/bin/env python3
"""Diagnostics for Task 3 receiver blocks.

This does not try to "fix" the receiver. It tests the main blocks and prints
which one looks unstable:
 - synchronization / CP start
 - pilot + length channel estimate
 - length-symbol decoding
 - payload stability across OFDM symbols
 - sensitivity to fractional resampling origin
"""

import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, savgol_filter
from numpy.lib.stride_tricks import sliding_window_view
from commpy.channelcoding import Trellis, viterbi_decode


FS = 44100
FC = 10000
NSC = 128
NCP = 20
TSYM = 58e-3
TRELLIS_45 = Trellis(memory=np.array([5]), g_matrix=np.array([[0o77, 0o45]]))
TRELLIS_51 = Trellis(memory=np.array([5]), g_matrix=np.array([[0o77, 0o51]]))
SIGNAL_PATH = "Signals_task3/signal3.mat"
MIN_PLAUSIBLE_LENGTH = 50


def get_pilot_signs():
    # Matlab legacy randn('state', 100) pilot used in the project.
    return np.array(
        [
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
        ],
        dtype=float,
    )


def load_signal():
    data = scipy.io.loadmat(SIGNAL_PATH)
    return data["R"].ravel()


def preprocess(phi=0.0):
    r = load_signal()
    n = np.arange(len(r))
    r_i = 2 * r * np.cos(2 * np.pi * FC * n / FS)
    r_q = -2 * r * np.sin(2 * np.pi * FC * n / FS)

    b, a = butter(8, 0.05)
    r_i_filt = filtfilt(b, a, r_i)
    r_q_filt = filtfilt(b, a, r_q)

    d = FS * TSYM / NSC
    idx = phi + np.arange(0, len(r_i_filt) - phi, d)
    r_bb = np.interp(idx, np.arange(len(r_i_filt)), r_i_filt) + 1j * np.interp(
        idx, np.arange(len(r_q_filt)), r_q_filt
    )
    return r_bb, d


def compute_sync_metric(r_bb):
    t_half = NSC // 2
    wins = sliding_window_view(r_bb, 2 * t_half)
    gamma = np.sum(wins[:, :t_half] * np.conj(wins[:, t_half:]), axis=1)
    phi1 = np.sum(np.abs(wins[:, :t_half]) ** 2, axis=1)
    phi2 = np.sum(np.abs(wins[:, t_half:]) ** 2, axis=1)
    mu = np.abs(gamma) / np.sqrt(phi1 * phi2 + 1e-10)
    return mu, int(np.argmax(mu))


def cp_similarity(r_bb, t0, num_symbols=6):
    n_sym = (len(r_bb) - t0) // (NSC + NCP)
    if t0 < 0 or n_sym <= 0:
        return np.nan
    blocks = r_bb[t0 : t0 + n_sym * (NSC + NCP)].reshape(n_sym, NSC + NCP)
    vals = []
    for blk in blocks[: min(num_symbols, len(blocks))]:
        cp = blk[:NCP]
        tail = blk[-NCP:]
        num = np.abs(np.vdot(cp, tail))
        den = np.linalg.norm(cp) * np.linalg.norm(tail) + 1e-12
        vals.append(num / den)
    return float(np.mean(vals))


def extract_fft_blocks(r_bb, t0):
    if t0 < 0:
        return None
    n_sym = (len(r_bb) - t0) // (NSC + NCP)
    if n_sym < 3:
        return None
    blocks = r_bb[t0 : t0 + n_sym * (NSC + NCP)].reshape(n_sym, NSC + NCP)
    return np.fft.fft(blocks[:, NCP:], axis=1)


def estimate_channel(x_pilot, x_length):
    known_idx = np.arange(15, NSC)
    even_idx = np.arange(0, NSC, 2)

    h_from_length = x_length[known_idx] / (1 + 1j)
    h_from_pilot = x_pilot[even_idx] / (2 * get_pilot_signs())

    h = np.empty(NSC, dtype=complex)
    for k in range(NSC):
        if k in known_idx and k in even_idx:
            j_len = np.where(known_idx == k)[0][0]
            j_pil = np.where(even_idx == k)[0][0]
            h[k] = 0.5 * (h_from_length[j_len] + h_from_pilot[j_pil])
        elif k in known_idx:
            j_len = np.where(known_idx == k)[0][0]
            h[k] = h_from_length[j_len]
        elif k in even_idx:
            j_pil = np.where(even_idx == k)[0][0]
            h[k] = h_from_pilot[j_pil]
        else:
            h[k] = 0.0

    for k in range(1, 15, 2):
        h[k] = 0.5 * (h[k - 1] + h[k + 1])

    return savgol_filter(h.real, 7, 3) + 1j * savgol_filter(h.imag, 7, 3)


def channel_agreement_nmse(x_pilot, x_length):
    even_idx = np.arange(16, NSC, 2)
    h_pilot = x_pilot[::2] / (2 * get_pilot_signs())
    h_length = x_length[even_idx] / (1 + 1j)
    ref = np.mean(np.abs(h_length) ** 2) + 1e-12
    return float(np.mean(np.abs(h_pilot[even_idx // 2] - h_length) ** 2) / ref)


def decode_length(x, h_full, trellis):
    x_len_eq = x[1, :15] / h_full[:15]
    bits_len = np.empty(30, dtype=int)
    bits_len[0::2] = (x_len_eq.real < 0).astype(int)
    bits_len[1::2] = (x_len_eq.imag < 0).astype(int)
    decoded = viterbi_decode(bits_len.astype(float), trellis, tb_depth=15, decoding_type="hard").astype(int)
    length = int("".join(decoded[:10].astype(str)), 2)

    tail_eq = x[1, 15:] / h_full[15:]
    tail_target = np.full_like(tail_eq, 1 + 1j)
    tail_evm = float(np.mean(np.abs(tail_eq - tail_target) ** 2))

    return {
        "length": length,
        "tail_evm": tail_evm,
    }


def nearest_qpsk(symbols):
    real = np.where(symbols.real >= 0, 1.0, -1.0)
    imag = np.where(symbols.imag >= 0, 1.0, -1.0)
    return real + 1j * imag


def bits_to_ascii(bits, length):
    bits = np.asarray(bits, dtype=int)[: 7 * length]
    chars = []
    for i in range(0, len(bits) - 6, 7):
        chars.append(chr(int("".join(bits[i : i + 7].astype(str)), 2)))
    return "".join(chars)


def message_score(msg):
    if not msg:
        return -1e9
    printable = sum((32 <= ord(c) <= 126) or c in "\n\r\t" for c in msg) / len(msg)
    letters = sum(c.isalpha() or c == " " for c in msg) / len(msg)
    spaces = msg.count(" ") / len(msg)
    return printable + 0.4 * letters + 0.2 * spaces


def apply_phase_correction(x_data, mode):
    if mode == "none":
        return x_data.copy()
    if mode != "per_symbol":
        raise ValueError(mode)

    corrected = x_data.copy()
    for i, row in enumerate(corrected):
        q = nearest_qpsk(row)
        phase = np.angle(np.sum(row * np.conj(q)))
        corrected[i] *= np.exp(-1j * phase)
    return corrected


def decode_payload(x, h_full, length, trellis, phase_mode="none"):
    n_data = int(np.ceil((7 * length + 5) / NSC))
    if x.shape[0] < 2 + n_data:
        return None

    x_data = x[2 : 2 + n_data] / h_full
    x_data = apply_phase_correction(x_data, phase_mode)

    bits = np.empty(x_data.size * 2, dtype=int)
    bits[0::2] = (x_data.real < 0).ravel().astype(int)
    bits[1::2] = (x_data.imag < 0).ravel().astype(int)
    needed = 14 * length + 10
    decoded = viterbi_decode(bits[:needed].astype(float), trellis, decoding_type="hard").astype(int)
    msg = bits_to_ascii(decoded, length)

    q = nearest_qpsk(x_data)
    evm = np.mean(np.abs(x_data - q) ** 2, axis=1)
    phase = np.angle(np.sum(x_data * np.conj(q), axis=1))

    return {
        "message": msg,
        "score": message_score(msg),
        "evm_by_symbol": evm,
        "phase_by_symbol": phase,
        "n_data": n_data,
    }


def status(label, ok, warn=None):
    if ok:
        return f"{label}: PASS"
    if warn:
        return f"{label}: WARN"
    return f"{label}: FAIL"


def print_sync_table(rows):
    print("\nSync / Length Test")
    print("t0      cp_corr   chan_nmse  len(45)  len(51)  tail_evm")
    print("-" * 58)
    for row in rows:
        print(
            f"{row['t0']:4d}    {row['cp_corr']:.4f}    {row['chan_nmse']:.4f}    "
            f"{row['len45']:7d}  {row['len51']:7d}  {row['tail_evm']:.4f}"
        )


def best_candidate_for_origin(phi):
    r_bb, _ = preprocess(phi=phi)
    mu, peak = compute_sync_metric(r_bb)
    best = None
    for t0 in range(peak - NCP - 5, peak + 5):
        x = extract_fft_blocks(r_bb, t0)
        if x is None:
            continue
        h = estimate_channel(x[0], x[1])
        length = decode_length(x, h, TRELLIS_51)["length"]
        if not (MIN_PLAUSIBLE_LENGTH <= length <= 400):
            continue
        payload = decode_payload(x, h, length, TRELLIS_51)
        if payload is None:
            continue
        cand = (payload["score"], phi, t0, length, payload["message"])
        if best is None or cand[0] > best[0]:
            best = cand
    return best


def main():
    r_bb, d = preprocess()
    mu, peak = compute_sync_metric(r_bb)

    print("Task 3 Diagnostics")
    print(f"Signal path: {SIGNAL_PATH}")
    print(f"Downsampling factor D: {d:.6f}")
    print(f"Baseband samples: {len(r_bb)}")
    print(f"Approx OFDM symbols available: {len(r_bb) // (NSC + NCP)}")
    print(f"Sync peak index: {peak}")

    rows = []
    for t0 in range(peak - NCP - 5, peak + 5):
        x = extract_fft_blocks(r_bb, t0)
        if x is None:
            continue
        h = estimate_channel(x[0], x[1])
        len45 = decode_length(x, h, TRELLIS_45)
        len51 = decode_length(x, h, TRELLIS_51)
        rows.append(
            {
                "t0": t0,
                "cp_corr": cp_similarity(r_bb, t0),
                "chan_nmse": channel_agreement_nmse(x[0], x[1]),
                "len45": len45["length"],
                "len51": len51["length"],
                "tail_evm": len51["tail_evm"],
                "x": x,
                "h": h,
            }
        )

    rows.sort(key=lambda r: (-r["cp_corr"], r["tail_evm"]))
    print_sync_table(rows[:10])

    best = None
    all_candidates = []
    for row in rows:
        for name, trellis in (("0o45", TRELLIS_45), ("0o51", TRELLIS_51)):
            length = decode_length(row["x"], row["h"], trellis)["length"]
            if not (MIN_PLAUSIBLE_LENGTH <= length <= 400):
                continue
            payload = decode_payload(row["x"], row["h"], length, trellis)
            if payload is None:
                continue
            cand = {
                "t0": row["t0"],
                "g2": name,
                "length": length,
                "cp_corr": row["cp_corr"],
                "chan_nmse": row["chan_nmse"],
                "tail_evm": row["tail_evm"],
                **payload,
            }
            all_candidates.append(cand)
            if best is None or cand["score"] > best["score"]:
                best = cand

    if best is None:
        raise RuntimeError("No decodable candidate found.")

    print("\nBest First-Pass Candidate")
    print(
        f"t0={best['t0']}, g2={best['g2']}, length={best['length']}, "
        f"score={best['score']:.4f}"
    )
    print(best["message"])

    best_phase = decode_payload(
        next(row["x"] for row in rows if row["t0"] == best["t0"]),
        next(row["h"] for row in rows if row["t0"] == best["t0"]),
        best["length"],
        TRELLIS_51 if best["g2"] == "0o51" else TRELLIS_45,
        phase_mode="per_symbol",
    )

    early = float(np.mean(best["evm_by_symbol"][: max(1, len(best["evm_by_symbol"]) // 2)]))
    late = float(np.mean(best["evm_by_symbol"][len(best["evm_by_symbol"]) // 2 :]))
    ratio = late / (early + 1e-12)
    phase_span = float(np.ptp(best["phase_by_symbol"])) if len(best["phase_by_symbol"]) > 1 else 0.0

    print("\nPayload Stability Test")
    print(f"EVM by OFDM symbol: {np.array2string(best['evm_by_symbol'], precision=3)}")
    print(f"Common phase by OFDM symbol [rad]: {np.array2string(best['phase_by_symbol'], precision=3)}")
    print(f"Early/late EVM ratio: {ratio:.3f}")
    print(f"Phase span across payload symbols: {phase_span:.3f} rad")
    print(status("Late symbols degrade", ratio < 1.25, warn=ratio < 1.8))
    print(status("Per-symbol phase drift small", phase_span < 0.25, warn=phase_span < 0.6))

    print("\nPhase-Correction Test")
    print(f"Baseline score: {best['score']:.4f}")
    print(f"Per-symbol phase corrected score: {best_phase['score']:.4f}")
    print(best_phase["message"])

    print("\nTiming Sensitivity Test")
    neighbor_scores = []
    trellis_best = TRELLIS_51 if best["g2"] == "0o51" else TRELLIS_45
    for t0 in range(best["t0"] - 2, best["t0"] + 3):
        x = extract_fft_blocks(r_bb, t0)
        if x is None:
            continue
        h = estimate_channel(x[0], x[1])
        payload = decode_payload(x, h, best["length"], trellis_best)
        if payload is None:
            continue
        neighbor_scores.append((t0, cp_similarity(r_bb, t0), payload["score"], payload["message"][:60]))
    for t0, cp_corr, score, preview in neighbor_scores:
        print(f"t0={t0:4d}  cp_corr={cp_corr:.4f}  score={score:.4f}  preview={preview!r}")

    print("\nResampling-Origin Sensitivity Test")
    origin_results = []
    for phi in (0.0, 1.0, 2.0, 3.0):
        best_phi = best_candidate_for_origin(phi)
        if best_phi is not None:
            origin_results.append(best_phi)
    for score, phi, t0, length, preview in origin_results:
        print(f"phi={phi:.1f}  score={score:.4f}  t0={t0}  length={length}  preview={preview[:60]!r}")

    print("\nLikely Failure Point")
    if ratio > 1.5 or phase_span > 0.5:
        print("Payload quality gets worse later in the frame. This points more to timing drift / resampling mismatch than a single fixed delay.")
    elif best_phase["score"] > best["score"] + 0.05:
        print("A per-symbol phase fix helps. Residual phase drift looks like the main problem.")
    elif best["tail_evm"] > 0.75 or best["chan_nmse"] > 0.5:
        print("Channel estimation from pilot/length is inconsistent. The pilot handling is the next place to inspect.")
    else:
        print("Synchronization and channel tests look acceptable. The remaining issue is likely in the decoding convention or a subtle timing offset.")


if __name__ == "__main__":
    main()
