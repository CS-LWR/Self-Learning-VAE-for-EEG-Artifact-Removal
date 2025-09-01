function standardizedList = standardizeEpochLengthWithFs(dataList, fsList, targetLength, targetFs, method)
%STANDARDIZEEPOCHLENGTHWITHFS Standardize EEG matrices to same time/frequency basis
%
%   Inputs:
%     dataList   - Cell array of matrices {X1, X2, ..., Xn}, each size [T × N]
%     fsList     - Array of sampling rates for each matrix, e.g., [200, 256]
%     targetLength - Desired length in time points after resampling/cropping
%     targetFs     - Target sampling rate (Hz)
%     method       - 'center_crop', 'pad', 'head_crop', 'tail_crop'
%
%   Output:
%     standardizedList - Cell array of matrices, each size [targetLength × N], all at targetFs
%
%   Notes:
%     1. If fs_i ≠ targetFs, signal will be resampled before trimming/padding.
%     2. Signals are assumed to be in [T × N] format.

    if nargin < 5
        method = 'center_crop';
    end

    nSignals = length(dataList);
    standardizedList = cell(1, nSignals);

    for k = 1:nSignals
        data = dataList{k};     % [T × N]
        fs_in = fsList(k);      % 当前数据采样率
        [T, N] = size(data);

        % Step 1: Resample with dynamic output length
        if fs_in ~= targetFs
            fprintf('Resampling signal %d from %d Hz to %d Hz...\n', k, fs_in, targetFs);
            example = resample(data(:,1), targetFs, fs_in);
            T_target = length(example);  % Use actual resampled length
            resampled = zeros(T_target, N);
            resampled(:,1) = example;
            for i = 2:N
                resampled(:,i) = resample(data(:,i), targetFs, fs_in);
            end
        else
            resampled = data;
        end

        % Step 2: Crop or pad to targetLength
        [T_now, ~] = size(resampled);

        if T_now == targetLength
            standardizedList{k} = resampled;
        else
            switch method
                case 'center_crop'
                    if T_now > targetLength
                        startIdx = floor((T_now - targetLength)/2) + 1;
                        endIdx = startIdx + targetLength - 1;
                        standardizedList{k} = resampled(startIdx:endIdx, :);
                    else
                        padLeft = floor((targetLength - T_now) / 2);
                        padRight = targetLength - T_now - padLeft;
                        standardizedList{k} = [zeros(padLeft, N); resampled; zeros(padRight, N)];
                    end

                case 'pad'
                    if T_now > targetLength
                        standardizedList{k} = resampled(1:targetLength, :);
                    else
                        padRight = targetLength - T_now;
                        standardizedList{k} = [resampled; zeros(padRight, N)];
                    end

                case 'head_crop'
                    if T_now > targetLength
                        standardizedList{k} = resampled(1:targetLength, :);
                    else
                        padRight = targetLength - T_now;
                        standardizedList{k} = [resampled; zeros(padRight, N)];
                    end

                case 'tail_crop'
                    if T_now > targetLength
                        standardizedList{k} = resampled(end - targetLength + 1:end, :);
                    else
                        padLeft = targetLength - T_now;
                        standardizedList{k} = [zeros(padLeft, N); resampled];
                    end

                otherwise
                    error('Unknown method. Choose center_crop, pad, head_crop, or tail_crop');
            end
        end
    end
end
