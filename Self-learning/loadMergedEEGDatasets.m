function [x_IT,x_OT,x_IV,x_OV,x_IE,x_OE] = loadMergedEEGDatasets(presetList, baseDir)
% loadMergedEEGDatasets
% -------------------------------------------------------------------------
% Reads and merges .mat datasets from one or more presets (e.g., 'EOG','EMG'),
% reshapes them using reshape_batches, and returns six concatenated 4D tensors
% for train/validation/evaluation inputs and targets.
%
% INPUTS
%   presetList : char/string (e.g., 'EOG') or cell array of strings (e.g., {'EOG','EMG'})
%   baseDir    : dataset root directory; defaults to 'Dataset' when omitted or empty
%
% OUTPUTS  (each returned as a 4D tensor after squeeze + gather)
%   x_IT, x_OT : training inputs / targets
%   x_IV, x_OV : validation inputs / targets
%   x_IE, x_OE : evaluation/test inputs / targets
%
% ASSUMPTIONS
% - Each preset subdirectory contains .mat files defining variables:
%       x_IT, x_OT, x_IV, x_OV, x_IE, x_OE
% - When two or more presets are merged, the following additional
%   concatenations are applied to reproduce the original pipeline behavior:
%       x_IT_Overall = cat(4, x_IT_Overall, x_OT_Overall);
%       x_OT_Overall = cat(4, x_OT_Overall, x_OT_Overall);
%
% DEPENDENCIES
%   reshape_batches (must be available on the MATLAB path)
% -------------------------------------------------------------------------

    if nargin < 2 || isempty(baseDir)
        baseDir = 'Dataset';
    end

    % Normalize to cell array for uniform iteration.
    if ischar(presetList) || isstring(presetList)
        presetList = cellstr(presetList);
    end

    % Accumulators for merged tensors.
    x_IT_Overall = [];
    x_OT_Overall = [];
    x_IV_Overall = [];
    x_OV_Overall = [];
    x_IE_Overall = [];
    x_OE_Overall = [];

    for p = 1:numel(presetList)
        preset = char(presetList{p});
        folderPath = fullfile(baseDir, preset);
        files = dir(fullfile(folderPath,'*.mat'));
        if isempty(files)
            warning('No .mat files found in: %s', folderPath);
        end

        % Load .mat files that define required variables.
        for i = 1:length(files)
            filePath = fullfile(folderPath, files(i).name);
            S = load(filePath);

            % Mirror variables locally if present.
            if isfield(S,'x_IT'), x_IT = S.x_IT; end 
            if isfield(S,'x_OT'), x_OT = S.x_OT; end 
            if isfield(S,'x_IV'), x_IV = S.x_IV; end 
            if isfield(S,'x_OV'), x_OV = S.x_OV; end 
            if isfield(S,'x_IE'), x_IE = S.x_IE; end 
            if isfield(S,'x_OE'), x_OE = S.x_OE; end 
        end

        % Reshape into the expected 4D layout (SSCB convention).
        XiT = reshape_batches(x_IT);
        XoT = reshape_batches(x_OT);
        XiV = reshape_batches(x_IV);
        XoV = reshape_batches(x_OV);
        XiE = reshape_batches(x_IE);
        XoE = reshape_batches(x_OE);

        % Concatenate along the 4th dimension.
        if isempty(x_IT_Overall)
            x_IT_Overall = XiT;
            x_OT_Overall = XoT;
            x_IV_Overall = XiV;
            x_OV_Overall = XoV;
            x_IE_Overall = XiE;
            x_OE_Overall = XoE;
        else
            x_IT_Overall = cat(4, x_IT_Overall, XiT);
            x_OT_Overall = cat(4, x_OT_Overall, XoT);
            x_IV_Overall = cat(4, x_IV_Overall, XiV);
            x_OV_Overall = cat(4, x_OV_Overall, XoV);
            x_IE_Overall = cat(4, x_IE_Overall, XiE);
            x_OE_Overall = cat(4, x_OE_Overall, XoE);
        end
    end

    % Apply additional concatenations when merging >= 2 presets (pipeline parity).
    if numel(presetList) >= 2
        x_IT_Overall = cat(4, x_IT_Overall, x_OT_Overall);
        x_OT_Overall = cat(4, x_OT_Overall, x_OT_Overall);
    end

    % Finalize outputs with squeeze + gather (gather is a no-op on CPU arrays).
    x_IT = safeGatherSqueeze(x_IT_Overall);
    x_OT = safeGatherSqueeze(x_OT_Overall);
    x_IV = safeGatherSqueeze(x_IV_Overall);
    x_OV = safeGatherSqueeze(x_OV_Overall);
    x_IE = safeGatherSqueeze(x_IE_Overall);
    x_OE = safeGatherSqueeze(x_OE_Overall);
end

% Helper: gather + squeeze with GPU/CPU compatibility.
function X = safeGatherSqueeze(Xin)
    try
        X = gather(squeeze(Xin));
    catch
        X = squeeze(Xin);
    end
end
