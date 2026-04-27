%% Task 3 OFDM Receiver
% Based on: ProjectDescription_26_Part1.pdf and Guide Book Part I
% Receiver chain (from guide Fig. 7):
%   Demodulation -> LPF -> A/D -> Sync -> CP Removal -> FFT
%   -> Channel Estimation -> Equalization -> QPSK Decode -> Viterbi -> ASCII
clear; clc; close all;

%% Parameters (from assignment description)
fs   = 44100;    % sampling frequency [Hz]
fc   = 10000;    % carrier frequency [Hz]
Nsc  = 128;      % number of subcarriers
Ncp  = 20;       % cyclic prefix length [samples]
Tsym = 58e-3;    % OFDM symbol duration (data portion) [s]
Nofdm = Nsc + Ncp;

%% 1. Load signal
data = load('Signals_task3/signal3.mat');
r = double(data.R(:));

%% 2. Demodulation — down-convert from fc to baseband
n    = (0:length(r)-1)';
r_I  = 2 * r .* cos(2*pi*fc*n/fs);
r_Q  = -2 * r .* sin(2*pi*fc*n/fs);

%% 3. Low-pass filter (Butterworth, order 8, normalised cutoff 0.05)
[b, a] = butter(8, 0.05);
r_I_f = filtfilt(b, a, r_I);
r_Q_f = filtfilt(b, a, r_Q);

%% 4. Fractional downsampling — D = fs*Tsym/Nsc (not integer; use interpolation)
D     = fs * Tsym / Nsc;         % ≈ 19.9828
t_out = (0 : D : length(r_I_f)-1)';
r_bb  = interp1((0:length(r_I_f)-1)', r_I_f + 1j*r_Q_f, t_out, 'linear');
fprintf('D = %.4f,  length(r_bb) = %d,  ~%d OFDM symbols\n', ...
        D, length(r_bb), floor(length(r_bb)/Nofdm));

%% 5. Synchronization — periodic pilot correlation (Lecture 4)
% Pilot symbol has two identical halves of length Tper = Nsc/2 = 64
% mu(k) = |P(k)| / sqrt(E1(k)*E2(k)), start = argmax(mu)
Tper = Nsc / 2;
Nbb  = length(r_bb);
mu   = zeros(Nbb - 2*Tper, 1);
for k = 1:length(mu)
    s1 = r_bb(k         : k+Tper-1);
    s2 = r_bb(k+Tper    : k+2*Tper-1);
    gamma = sum(s1 .* conj(s2));
    E1    = sum(abs(s1).^2);
    E2    = sum(abs(s2).^2);
    mu(k) = abs(gamma) / sqrt(E1*E2 + 1e-10);
end
[~, T0] = max(mu);          % T0: 1-indexed sample where sync metric peaks
t0 = T0 - Ncp - 1;          % back up by CP to reach start of pilot symbol
fprintf('Sync peak T0 = %d,  frame start t0 = %d\n', T0, t0);

%% 6. Frame extraction and CP removal
n_sym  = floor((Nbb - t0 + 1) / Nofdm);
frame  = r_bb(t0 : t0 + n_sym*Nofdm - 1);
blocks = reshape(frame, Nofdm, n_sym)';   % n_sym x Nofdm
ofdm_no_cp = blocks(:, Ncp+1:end);        % n_sym x Nsc  (CP removed)
fprintf('Total OFDM symbols: %d\n', n_sym);
% Layout: row 1 = pilot, row 2 = length, rows 3..end = data

%% 7. FFT — frequency-domain representation
X = fft(ofdm_no_cp, Nsc, 2);
% X(1,:) = pilot symbol
% X(2,:) = length symbol
% X(3:end,:) = data symbols

%% 8. Reconstruct pilot sequence (identical to transmitter)
% Transmitter: randn('state',100); P = sign(randn(1,Nsc/2))
randn('state', 100);
P = sign(randn(1, Nsc/2));   % 64-element sequence of +1/-1

%% 9. Channel estimation
% Pilot symbol: x_pilot(1:2:Nsc) = 2*P  (odd MATLAB indices carry pilots, even = 0)
%   => X(1, 2k-1) = H(2k-1) * 2 * P(k)  for k = 1..64
pilot_sc = 1:2:Nsc;                        % MATLAB 1-indexed: 1,3,5,...,127
H_pil    = X(1, pilot_sc) ./ (2 * P);     % H at all 64 pilot subcarriers

% Length symbol: subcarriers 16..128 carry known 1+j
%   (10-bit length → conv encode → zero-pad to 2*Nsc bits → QPSK; first 15 sc carry
%    length bits, remaining 113 carry zero-bits which QPSK-map to 1+j)
len_sc = 16:Nsc;
H_len  = X(2, len_sc) ./ (1 + 1j);       % H at subcarriers 16..128 (113 values)

% Combine into full channel estimate for all Nsc subcarriers
H_full = zeros(1, Nsc);
for k = 1:Nsc
    is_pil = (mod(k, 2) == 1);   % odd MATLAB index → pilot measurement available
    is_len = (k >= 16);           % in length-symbol known region
    if is_pil && is_len
        % Both estimates available — average them
        H_full(k) = 0.5 * (H_pil((k+1)/2) + H_len(k-15));
    elseif is_pil
        H_full(k) = H_pil((k+1)/2);
    elseif is_len
        H_full(k) = H_len(k-15);
    % else: even subcarrier below 16 — filled by interpolation below
    end
end
% Interpolate even subcarriers 2,4,...,14 from their odd neighbours
for k = 2:2:14
    H_full(k) = 0.5 * (H_full(k-1) + H_full(k+1));
end
% Smooth real and imaginary parts to reduce per-subcarrier noise
H_full = sgolayfilt(real(H_full), 3, 7) + 1j * sgolayfilt(imag(H_full), 3, 7);

%% 10. Equalization (divide by channel estimate)
X_len_eq  = X(2,:) ./ H_full;         % length symbol equalized  (1 x Nsc)
X_data_eq = X(3:end,:) ./ H_full;     % data symbols equalized   (n_data x Nsc)

%% 11. QPSK demodulation
% Mapping: 00->1+j, 10->-1+j, 11->-1-j, 01->1-j
% First bit = sign(real):  0 if real>0,  1 if real<0
% Second bit = sign(imag): 0 if imag>0,  1 if imag<0
% Bit stream is serialised symbol-by-symbol, subcarrier-by-subcarrier within each symbol

% Length symbol bits (1 x Nsc -> 1 x 2*Nsc bit vector)
bits_len = zeros(1, Nsc*2);
bits_len(1:2:end) = double(real(X_len_eq) < 0);
bits_len(2:2:end) = double(imag(X_len_eq) < 0);

% Data symbol bits — row-major flatten (symbol 0 all scs, then symbol 1, ...)
% In MATLAB (column-major): transpose first, then column-flatten = row-major of original
X_data_flat = reshape(X_data_eq.', 1, []);
bits_data = zeros(1, length(X_data_flat)*2);
bits_data(1:2:end) = double(real(X_data_flat) < 0);
bits_data(2:2:end) = double(imag(X_data_flat) < 0);

%% 12. Viterbi decoding — convolutional code (77,45) octal, rate 1/2, K=6
% Encoder terminates with K-1=5 zero bits, so use 'term' mode
trellis = poly2trellis(6, [77 45]);

% --- Decode message length ---
% Length: 10 data bits + 5 tail bits -> 15 input bits -> 30 encoded bits
dec_len = vitdec(bits_len(1:30), trellis, 35, 'term', 'hard');
l_m = sum(dec_len(1:10)' .* (2.^(9:-1:0))');
fprintf('Decoded message length: %d characters\n', l_m);

% --- Decode message (search around decoded length for best result) ---
best_msg = ''; best_score = -1; best_lm = -1;
for lm_try = max(1, l_m-30) : min(400, l_m+30)
    needed = 14*lm_try + 10;   % l_m*7 data bits + 5 tail bits, encoded at rate 1/2
    if needed > length(bits_data), continue; end
    dec = vitdec(bits_data(1:needed), trellis, 35, 'term', 'hard');
    msg_bits = dec(1 : 7*lm_try);   % vitdec 'term' output = needed/2 - (K-1) = 7*lm_try
    chars = zeros(1, lm_try);
    for i = 1:lm_try
        b7 = msg_bits((i-1)*7+1 : i*7);
        chars(i) = sum(b7(:)' .* (2.^(6:-1:0)));
    end
    msg   = char(chars);
    score = sum(isletter(msg) | msg == ' ') / max(1, lm_try);
    if score > best_score || (score == best_score && lm_try > best_lm)
        best_score = score;
        best_msg   = msg;
        best_lm    = lm_try;
    end
end

fprintf('\nBest decode  (length=%d, alpha-rate=%.3f):\n%s\n', best_lm, best_score, best_msg);
