function plotPSDComparison(X_clean, X_denoised, fs, example_indices)
    epoch_duration = 2;  % seconds
    n_fft = 800;
    window_length = fs * epoch_duration;
    overlap = 0.5;
    overlap_point = floor(overlap * window_length);

    figure('Name', 'PSD Comparison (Clean vs. Denoised)', 'Position', [100, 100, 1200, 400]);

    for idx = 1:length(example_indices)
        i = example_indices(idx);
        x_ref = X_clean(:, i);
        x_out = X_denoised(:, i);

        [P_ref, F] = pwelch(x_ref, window_length, overlap_point, n_fft, fs);
        [P_out, ~] = pwelch(x_out, window_length, overlap_point, n_fft, fs);

        subplot(1, length(example_indices), idx);
        semilogx(F, 10*log10(P_ref + eps), 'b', 'LineWidth', 1.2); hold on;
        semilogx(F, 10*log10(P_out + eps), 'r--', 'LineWidth', 1.2);
        xlim([0.5, 50]);
        xlabel('Frequency (Hz, log scale)');
        ylabel('Power/Frequency (dB/Hz)');
        legend('Clean', 'Denoised');
        title(['Sample ' num2str(i)]);
        grid on;
    end
end
