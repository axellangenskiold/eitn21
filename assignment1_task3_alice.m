% assignment1_task3_alice.m
% MATLAB translation of assignment1_task3_alice.ipynb
% Implements: load signal, demod, filtering, downsample, sync, FFT,
% channel estimation (length & pilots), equalization, masking, QPSK demod,
% and (optional) Viterbi decode — closely following the notebook flow.

clear; close all; clc;

% --- Load signal (same file as notebook) ---
matPath = '/Users/axellangenskiold/eitn21/Signals_task3/signal3.mat';
S = load(matPath);
if isfield(S,'R')
    r = S.R(:);
else
    fn = fieldnames(S);
    r = S.(fn{1}); r = r(:);
end

% --- System parameters (copy from notebook) ---
fs = 44100;
fc = 10000;
Nsc = 128;
Ncp = 20;
Tsym = 58 * 10^-3;

% --- Demodulation to baseband I/Q ---
n = (0:numel(r)-1).';
r_I = 2 * r .* cos(2*pi*fc.*n./fs);
r_Q = -2 * r .* sin(2*pi*fc.*n./fs);

% --- Low-pass filter (Butterworth) ---
[b,a] = butter(8, 0.05);
r_I_filt = filter(b,a,r_I);
r_Q_filt = filter(b,a,r_Q);

% --- A/D downsampling ---
D = round(fs * Tsym / Nsc); % expected 20
r_I_down = r_I_filt(1:D:end);
r_Q_down = r_Q_filt(1:D:end);
r_bb = r_I_down + 1i * r_Q_down;
fprintf('D=%d, r_bb length=%d, ~%d OFDM symbols\n', D, numel(r_bb), floor(numel(r_bb)/(Nsc+Ncp)));

% --- Synchronization and CP removal ---
Tper = Nsc/2; % 64
winlen = 2 * Tper;
Nwin = numel(r_bb) - winlen + 1;
gamma = zeros(Nwin,1); phi1 = zeros(Nwin,1); phi2 = zeros(Nwin,1);
for ii = 1:Nwin
    w = r_bb(ii:ii+winlen-1);
    x1 = w(1:Tper); x2 = w(Tper+1:end);
    gamma(ii) = sum(x1 .* conj(x2));
    phi1(ii) = sum(abs(x1).^2);
    phi2(ii) = sum(abs(x2).^2);
end
mu = abs(gamma) ./ sqrt(phi1 .* phi2 + 1e-10);
[~, T0] = max(mu);
t0 = T0 - Ncp;
fprintf('Pilot starts at: %d, OFDM data at: %d\n', t0, T0);

Nofdm = Nsc + Ncp;
n_sym = floor((numel(r_bb) - t0) / Nofdm);
ofdm_blocks = reshape(r_bb(t0 + (0:n_sym*Nofdm-1) + 1), Nofdm, n_sym).';
ofdm_data = ofdm_blocks(:, Ncp+1:end);
fprintf('Total OFDM symbols: %d\n', n_sym);

% --- FFT ---
X = fft(ofdm_data, [], 2); % shape: (n_sym, Nsc)

% --- Channel estimation (from notebook) ---
% H_from_len = X[2,16:128] / (1+1j)  (MATLAB 1-based)
H_from_len = X(2, 16:128) ./ (1 + 1i);     % shape (1,113)
len_pos = 15:127;                          % zero-based style positions mapping

% Extrapolate back to even subcarriers 0-14 (MATLAB indices 1..15 -> positions 0..14)
fit_pos = len_pos(1:7);                    % 15..21
mag_p = polyfit(fit_pos, abs(H_from_len(1:7)), 1);
pha_p = polyfit(fit_pos, unwrap(angle(H_from_len(1:7))), 1);
extrap_pos = 0:2:14;                       % even: 0,2,..,14
H_extrap = polyval(mag_p, extrap_pos) .* exp(1i * polyval(pha_p, extrap_pos));

% Raw pilots: X(1,1:2:128) covers subcarriers 0,2,...,126
raw = X(1, 1:2:128);
P_low = sign(real(raw(1:8) .* conj(H_extrap)));
P_low(P_low == 0) = 1;
H_pilot_low = raw(1:8) ./ (2 .* P_low);

% Combine pilot-based H at even subcarriers 0..14 with H_from_len (15..127)
pos_known = [extrap_pos, len_pos];
H_known = [H_pilot_low, H_from_len];
[~, idx] = sort(pos_known);
all_pos = 0:(Nsc-1);
mag_k = abs(H_known(idx));
phase_k = unwrap(angle(H_known(idx)));
mag_interp = interp1(pos_known(idx), mag_k, all_pos, 'linear', 'extrap');
phase_interp = interp1(pos_known(idx), phase_k, all_pos, 'linear', 'extrap');
H_full = mag_interp .* exp(1i * phase_interp);

% Plot raw constellation (length/payload)
figure; scatter(real(X(2:end).'), imag(X(2:end).'), 1); title(' constellation'); xlabel('Real'); ylabel('Imaginary'); grid on;

% --- Remove pilots / prepare data ---
X_data = X(2:end-1, :);   % same as notebook (exclude first and last symbol rows)
figure; plot(abs(X_data).'); title('OFDM symbols magnitude per subcarrier'); xlabel('Subcarrier'); grid on;

% --- Equalization ---
X_eq = X_data ./ (H_full + 1e-6);
figure; pts = X_eq.'; scatter(real(pts(:)), imag(pts(:)), 1); xlim([-3 3]); ylim([-3 3]); hold on; plot([0 0], ylim, 'k-'); plot(xlim, [0 0], 'k-'); title('Equalized constellation'); grid on;

% --- Mask near-zero symbols and QPSK demod ---
thresh = 0.5; % tuneable
mask = abs(X_eq) > thresh; % same shape as X_eq

figure; pts = X_eq(mask); scatter(real(pts(:)), imag(pts(:)), 4, '.'); axis equal; grid on; title('Masked equalized constellation');

% Build bits from kept symbols
bits = []; % char array
for i = 1:numel(X_eq)
    sym = X_eq(i);
    if ~mask(i)
        continue;
    end
    if real(sym) < 0
        bits = [bits '1'];
    else
        bits = [bits '0'];
    end
    if imag(sym) < 0
        bits = [bits '1'];
    else
        bits = [bits '0'];
    end
end
fprintf('Total demod bits (after masking): %d\n', numel(bits));

% --- Decoding and ASCII (Viterbi) ---
if exist('poly2trellis','file') && exist('vitdec','file')
    trellis = poly2trellis(6, [77 45]);
    bits_num = bits - '0';
    if numel(bits_num) < 256
        warning('Not enough bits for length decoding: %d', numel(bits_num));
    else
        bits_len = bits_num(1:256);
        tblen = 5 * 6; % approximate trace-back length
        decoded_len = vitdec(bits_len, trellis, tblen, 'trunc', 'hard');
        l_m = bin2dec(sprintf('%d', decoded_len(1:10)));
        fprintf('Message length: %d characters\n', l_m);

        bits_data = bits_num(257:end);
        decoded_msg = vitdec(bits_data, trellis, tblen, 'trunc', 'hard');
        if l_m * 7 > numel(decoded_msg)
            warning('Not enough decoded bits for message payload');
        else
            msg = '';
            for k = 1:l_m
                start = (k-1)*7 + 1;
                chunk = decoded_msg(start:start+6);
                ch = char(bin2dec(sprintf('%d', chunk)));
                msg = [msg ch];
            end
            fprintf('Decoded message:\n%s\n', msg);
        end
    end
else
    warning('Viterbi decode not available; saved bits to bits_assignment1_task3.mat');
    save('bits_assignment1_task3.mat','bits');
end

% End of MATLAB translation
