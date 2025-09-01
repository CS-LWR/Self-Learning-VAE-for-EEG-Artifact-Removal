function compare_waveform(X, Y, Z, sampleIdx)
%COMPARE Compare a specific EEG sample (Clean vs Predicted, optional Noisy).
%
%   compare(X, Y)               - Plot Predicted vs Clean for sample 1
%   compare(X, Y, Z)            - Plot Predicted vs Clean vs Noisy for sample 1
%   compare(X, Y, Z, sampleIdx) - Plot for a specified sample index
%
%   Inputs:
%     X - Predicted signals, size [T × N]
%     Y - Clean (ground truth), size [T × N]
%     Z - (Optional) Noisy signals, size [T × N]
%     sampleIdx - (Optional) Sample column to plot (default = 1)
%
%   T = Time points, N = Number of samples

    % Default to sample 1
    if nargin < 4
        sampleIdx = 1;
    end

    % Validate input dimensions
    if ndims(X) ~= 2 || ndims(Y) ~= 2 || ~isequal(size(X), size(Y))
        error('X and Y must both be 2D matrices of size [T × N]');
    end
    if nargin >= 3 && ~isempty(Z)
        if ~isequal(size(Z), size(X))
            error('Z must also be of size [T × N]');
        end
    end
    if sampleIdx > size(X, 2) || sampleIdx < 1
        error('Sample index (%d) is out of range. Total samples: %d', sampleIdx, size(X, 2));
    end

    % Extract the specified sample (1 column)
    predicted = X(:, sampleIdx);
    clean     = Y(:, sampleIdx);

    if nargin >= 3 && ~isempty(Z)
        noisy = Z(:, sampleIdx);
    end

    % Plot
    t = 1:length(predicted);
    figure;

    if nargin >= 3 && ~isempty(Z)
        plot(t, noisy, 'r--', 'LineWidth', 1); hold on;
    end
    plot(t, clean, 'g:', 'LineWidth', 1.2); hold on;
    plot(t, predicted, 'b', 'LineWidth', 1.2);

    xlabel('Time'); ylabel('Amplitude');
    title(sprintf('Signal Comparison (Sample #%d)', sampleIdx));
    grid on;

    if nargin >= 3 && ~isempty(Z)
        legend('Noisy', 'Clean', 'Predicted');
    else
        legend('Clean', 'Predicted');
    end

    xlim([1, length(predicted)]);
end
