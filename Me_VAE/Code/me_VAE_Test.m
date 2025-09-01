% clear
% clc

%% Load Data
% -------------------------------------------------------------------------
% Experiment controls:
%   model       : tag for method/filenames
%   dataset     : dataset index as string (e.g., '1', '2', ...)
%   preset      : domain/artifact preset ('EMG' | 'EOG' | 'EEG' | 'EEG_motion')
%   preset_save : not used in this script (kept for consistency)
%   fs          : sampling frequency (Hz)
%   plot_flag   : passed through to evaluation/visualization utilities
model='me_VAE';
dataset = '1';
preset = 'EOG';      % Change to: 'EMG' | 'EOG' | 'EEG' | 'EEG_motion'
preset_save = 0;     % 0 or 1 (unused here)
fs = 256;
plot_flag = 0;

% -------------------------------------------------------------------------
% Load ground-truth/inputs for the selected dataset & preset
folderPath = fullfile('..', ['Dataset' dataset], preset);
files = dir(fullfile(folderPath,'*.mat'));
for i = 1:length(files)
    filePath = fullfile(folderPath, files(i).name);
    load(filePath);
    disp(['Loaded: ' files(i).name]);
end

% -------------------------------------------------------------------------
% Load predictions/outputs produced by model inference
% Expected folder naming: 'P_<preset>_<model>'
folderPath = fullfile('..', ['Dataset' dataset], ['P_' preset '_' model]);
files = dir(fullfile(folderPath,'*.mat'));
for i = 1:length(files)
    filePath = fullfile(folderPath, files(i).name);
    load(filePath);
    disp(['Loaded: ' files(i).name]);
end

%%
fs = 256;
plot_flag = 0;

% -------------------------------------------------------------------------
% Add local Toolbox path to access evaluation/visualization helpers
toolboxPath = fullfile(pwd, 'Toolbox');
addpath(toolboxPath);

% Evaluate denoising performance and visualize comparisons
%   evaluateEEGDenoising(clean, denoised, fs, methodLabel, plotFlag)
%   visualizeEEGComparison(denoised, clean, input, fs)
results = evaluateEEGDenoising(x_OE, x_PE, fs, [model '_' preset], plot_flag);
visualizeEEGComparison(x_PE, x_OE, x_IE, fs)

% Optional cleanup: remove Toolbox path
rmpath(toolboxPath);
