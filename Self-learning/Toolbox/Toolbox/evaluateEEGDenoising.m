function results = evaluateEEGDenoising(X_clean, X_denoised, fs, method, plot_flag)
% evaluateEEGDenoising
% -------------------------------------------------------------------------
% Evaluate EEG denoising performance using time- and frequency-domain metrics
% plus a time–frequency (scalogram) similarity measure.
%
% INPUTS
%   X_clean     : Clean (reference) EEG signals, size [T × N]
%   X_denoised  : Denoised EEG signals,        size [T × N]
%   fs          : Sampling frequency (Hz)
%   method      : Method label (char/str), e.g., 'VAE', 'AE' (used in titles/prints)
%   plot_flag   : Logical flag to enable plots (true/false)
%
% OUTPUT
%   results     : Struct with mean and std of all metrics across the N samples
%
% NOTES / ASSUMPTIONS
%   - Columns correspond to independent epochs/samples; rows correspond to time.
%   - A zero-phase IIR band-pass (0.5–50 Hz) is applied (via filtfilt) before
%     time-domain metrics to reduce drift and high-frequency noise without
%     phase distortion.
%   - Frequency-domain metrics are computed on the raw sequences via Welch PSD,
%     then restricted to 0.5–50 Hz.
%   - Scalogram similarity is computed using CWT (Morlet/‘amor’); normxcorr2
%     is applied on |CWT| to obtain a single similarity score per sample.
%   - Requires Signal Processing Toolbox (designfilt, pwelch) and Wavelet
%     Toolbox (cwt). normxcorr2 requires Image Processing Toolbox.
% -------------------------------------------------------------------------

% ----- Time-domain preprocessing: band-pass (0.5–50 Hz), zero-phase -----
bpFilt = designfilt('bandpassiir','FilterOrder',4, ...
    'HalfPowerFrequency1',0.5,'HalfPowerFrequency2',50, ...
    'SampleRate',fs);

% ----- Frequency-domain preprocessing parameters (Welch PSD) -------------
% Use segment length equal to the epoch length to avoid zero-padding.
n_fft          = size(X_clean,1);     % FFT length = number of time samples (no zero pad)
epoch_duration = n_fft / fs;          % Epoch duration (s)
window_length  = fs * epoch_duration; % Welch window length = epoch length (samples)
overlap_point  = floor(0.5 * window_length); % 50% overlap
numSamples     = size(X_clean, 2);    % Number of epochs
freq_lower     = 0.5;
freq_upper     = 50;

% ----- Preallocate storage for per-epoch metrics -------------------------
all_rmse_t = zeros(numSamples, 1);    % Time-domain RMSE
all_rrms_t = zeros(numSamples, 1);    % Time-domain RRMSE
all_rmse_f = zeros(numSamples, 1);    % Frequency-domain RMSE (on PSD)
all_rrms_f = zeros(numSamples, 1);    % Frequency-domain RRMSE (on PSD)
all_cc_t   = zeros(numSamples, 1);    % Time-domain Pearson CC
all_cc_f   = zeros(numSamples, 1);    % Frequency-domain Pearson CC (on PSD)
all_scalo  = zeros(numSamples, 1);    % Scalogram similarity (max of normxcorr2)

% ======================== Per-sample evaluation loop =====================
for i = 1:numSamples
    x_ref = X_clean(:, i);
    x_out = X_denoised(:, i);

    % ----- Time-domain metrics (after 0.5–50 Hz band-pass) ---------------
    x_ref_filt = filtfilt(bpFilt, x_ref);
    x_out_filt = filtfilt(bpFilt, x_out);

    % RMSE / RRMSE / CC in time domain
    all_rmse_t(i) = rms(x_out_filt - x_ref_filt);
    all_rrms_t(i) = all_rmse_t(i) / rms(x_ref_filt);
    cc_t          = corrcoef(x_out_filt, x_ref_filt);
    all_cc_t(i)   = cc_t(1,2);

    % ----- Frequency-domain metrics (Welch PSD, restricted to 0.5–50 Hz) --
    [P_ref_full, F] = pwelch(x_ref, window_length, overlap_point, n_fft, fs);
    [P_out_full, ~] = pwelch(x_out, window_length, overlap_point, n_fft, fs);
    freq_mask       = (F >= freq_lower) & (F <= freq_upper);
    P_ref           = P_ref_full(freq_mask);
    P_out           = P_out_full(freq_mask);

    all_rmse_f(i)   = rms(P_out - P_ref);
    all_rrms_f(i)   = all_rmse_f(i) / rms(P_ref);
    cc_f            = corrcoef(P_ref, P_out);
    all_cc_f(i)     = cc_f(1,2);

    % ----- Time–frequency similarity (CWT amplitude maps) -----------------
    % Compare |CWT| maps using normalized 2D cross-correlation; take maximum.
    [cfs_ref, ~] = cwt(x_ref, 'amor', fs);
    [cfs_out, ~] = cwt(x_out, 'amor', fs);
    corrMat      = normxcorr2(abs(cfs_ref), abs(cfs_out));
    all_scalo(i) = max(corrMat(:));
end

% ============================== Reporting ================================
fprintf('\n[%s Evaluation]:\n', method);
fprintf('RMSE_t:  %.4f ± %.4f\n', mean(all_rmse_t), std(all_rmse_t));
fprintf('RMSE_f:  %.4f ± %.4f\n', mean(all_rmse_f), std(all_rmse_f));
fprintf('RRMSE_t: %.4f ± %.4f\n', mean(all_rrms_t), std(all_rrms_t));
fprintf('RRMSE_f: %.4f ± %.4f\n', mean(all_rrms_f), std(all_rrms_f));
fprintf('CC_t:    %.4f ± %.4f\n', mean(all_cc_t), std(all_cc_t));
fprintf('CC_f:    %.4f ± %.4f\n', mean(all_cc_f), std(all_cc_f));
fprintf('Scalogram Similarity: %.4f ± %.4f\n', mean(all_scalo), std(all_scalo));

% Package results
results = struct( ...
    'RMSE_t', mean(all_rmse_t),     'RMSE_t_std', std(all_rmse_t), ...
    'RMSE_f', mean(all_rmse_f),     'RMSE_f_std', std(all_rmse_f), ...
    'RRMSE_t', mean(all_rrms_t),    'RRMSE_t_std', std(all_rrms_t), ...
    'RRMSE_f', mean(all_rrms_f),    'RRMSE_f_std', std(all_rrms_f), ...
    'CC_t', mean(all_cc_t),         'CC_t_std', std(all_cc_t), ...
    'CC_f', mean(all_cc_f),         'CC_f_std', std(all_cc_f), ...
    'ScalogramSim', mean(all_scalo), 'ScalogramSim_std', std(all_scalo));

% ============================ Visualization =============================
if plot_flag
    % Histograms of core metrics
    figure('Name', [method ' Evaluation Metrics'], 'Position', [100 100 1000 300]);
    subplot(1,4,1); histogram(all_rmse_t); title('RMSE (Time)'); xlabel('Value'); ylabel('Count');
    subplot(1,4,2); histogram(all_rmse_f); title('RMSE (Freq)'); xlabel('Value'); ylabel('Count');
    subplot(1,4,3); histogram(all_cc_t);   title('CC (Time)');  xlabel('r');     ylabel('Count');
    subplot(1,4,4); histogram(all_cc_f);   title('CC (Freq)');  xlabel('r');     ylabel('Count');

    figure('Name','Scalogram Similarity');
    histogram(all_scalo); title('Scalogram Similarity');
    xlabel('Similarity'); ylabel('Count');
end
end
