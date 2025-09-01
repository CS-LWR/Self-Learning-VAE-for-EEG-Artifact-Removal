function plotScalogramComparison(X_clean, X_denoised, fs, sampleID)
    T = size(X_clean, 1);
    x_ref = X_clean(:, sampleID);
    x_out = X_denoised(:, sampleID);
    [cfs_ref, f] = cwt(x_ref, 'amor', fs);
    [cfs_out, ~] = cwt(x_out, 'amor', fs);

    % 获取统一 color scale
    vmin = min([abs(cfs_ref(:)); abs(cfs_out(:))]);
    vmax = max([abs(cfs_ref(:)); abs(cfs_out(:))]);

    % 绘制 clean 信号 scalogram
    figure('Name', 'Scalogram - Clean Signal');
    imagesc(1:length(x_ref), f, abs(cfs_ref)); axis xy;
    clim([vmin vmax]);   % 设置 color scale 一致
    xlabel('Time'); ylabel('Frequency (Hz)');
    title(['Scalogram of Clean Sample ' num2str(sampleID)]); colorbar;

    % 绘制 denoised 信号 scalogram
    figure('Name', 'Scalogram - Denoised Signal');
    imagesc(1:length(x_out), f, abs(cfs_out)); axis xy;
    clim([vmin vmax]);   % 设置 color scale 一致
    xlabel('Time'); ylabel('Frequency (Hz)');
    title(['Scalogram of Denoised Sample ' num2str(sampleID)]); colorbar;
end
