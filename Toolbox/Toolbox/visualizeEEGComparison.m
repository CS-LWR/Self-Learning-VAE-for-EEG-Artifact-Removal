function visualizeEEGComparison(X_denoised, X_clean, X_noisy, fs, sampleID1, sampleID2, sampleID3)
%VISUALIZEEEGCOMPARISON Plot PSD and scalogram comparison of EEG signals.
%
%   visualizeEEGComparison(X_clean, X_denoised, fs)
%   visualizeEEGComparison(X_clean, X_denoised, fs, example_indices, sampleID)
    if nargin < 5 || isempty(sampleID1)
        sampleID1 = 1;
    end

    if nargin < 6 || isempty(sampleID2)
        sampleID2 = 1;
    end
    if nargin < 7
        sampleID3 = 1;
    end

    compare_waveform(X_denoised, X_clean, X_noisy, sampleID1)

    % 调用子函数：绘制 PSD 比较图
    plotPSDComparison(X_clean, X_denoised, fs, sampleID2);

    % 调用子函数：绘制 Scalogram 比较图
    plotScalogramComparison(X_clean, X_denoised, fs, sampleID3);
end

