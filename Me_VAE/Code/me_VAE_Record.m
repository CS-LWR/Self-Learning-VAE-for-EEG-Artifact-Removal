% clear
% clc

%% Load Model Here
% -------------------------------------------------------------------------
% NOTE:
%   - Load your trained networks here (netE1, netE2, netD), e.g.:
%       load('trainedMe_VAE_Dataset2_New.mat','netE1','netE2','netD');
%   - Keep this section above data loading to ensure the nets exist
%     before running inference below.


%% Load Data
% -------------------------------------------------------------------------
% Experiment/pipeline controls
%   model       : label used in evaluation/filenames
%   dataset     : dataset index (string)
%   preset      : data preset ('EMG' | 'EOG' | 'EEG' | 'EEG_motion')
%   preset_save : whether to save outputs from this script (0/1) — unused here
%   fs          : sampling frequency (Hz)
%   plot_flag   : passed to evaluation functions if used elsewhere
model = 'me_VAE';
dataset = '1';
preset = 'EOG';     % Change to: 'EMG' | 'EOG' | 'EEG' | 'EEG_motion'
preset_save = 0;    % 0 or 1
fs = 256;
plot_flag = 0;

% Define the relative path to the preset folder containing .mat files
folderPath = fullfile('..', ['Dataset' dataset], preset);

% Enumerate all .mat files in the target folder
files = dir(fullfile(folderPath,'*.mat'));

% Batch load each .mat file (expects compatible variables inside)
for i = 1:length(files)
    filePath = fullfile(folderPath, files(i).name);
    load(filePath);
    disp(['Loaded: ' files(i).name]);
end


%% Deduction (Inference / Prediction)
% -------------------------------------------------------------------------
% Reshape raw arrays into 4-D (SSCB) batches as expected by the network
[x_IT] = reshape_batches(x_IT);   % Train inputs (if present)
[x_OT] = reshape_batches(x_OT);   % Train targets (if present)
[x_IV] = reshape_batches(x_IV);   % Validation inputs
[x_OV] = reshape_batches(x_OV);   % Validation targets
[x_IE] = reshape_batches(x_IE);   % Evaluation/Test inputs
[x_OE] = reshape_batches(x_OE);   % Evaluation/Test targets

% ---------- Evaluate set E ----------
ds_IE = arrayDatastore(x_IE, IterationDimension=4);
numOutputs = 1;

mbq_IE = minibatchqueue(ds_IE, numOutputs, ...
    MiniBatchSize = 1, ...
    MiniBatchFcn = @preprocessMiniBatch, ...
    MiniBatchFormat = "SSCB", ...
    PartialMiniBatch = "discard");

% Forward pass using the loaded/trained networks
x_PE = modelPredictions(netE1, netE2, netD, mbq_IE);

% Gather to CPU and squeeze to [T × N], then save
x_PE = gather(squeeze(x_PE));     % [T × N]
save('PE.mat','x_PE')


% ---------- Evaluate set T ----------
ds_IT = arrayDatastore(x_IT, IterationDimension=4);
numOutputs = 1;

mbq_IT = minibatchqueue(ds_IT, numOutputs, ...
    MiniBatchSize = 1, ...
    MiniBatchFcn = @preprocessMiniBatch, ...
    MiniBatchFormat = "SSCB", ...
    PartialMiniBatch = "discard");

x_PT = modelPredictions(netE1, netE2, netD, mbq_IT);

x_PT = gather(squeeze(x_PT));     % [T × N]
save('PT.mat','x_PT')


% ---------- Evaluate set V ----------
ds_IV = arrayDatastore(x_IV, IterationDimension=4);
numOutputs = 1;

mbq_IV = minibatchqueue(ds_IV, numOutputs, ...
    MiniBatchSize = 1, ...
    MiniBatchFcn = @preprocessMiniBatch, ...
    MiniBatchFormat = "SSCB", ...
    PartialMiniBatch = "discard");

x_PV = modelPredictions(netE1, netE2, netD, mbq_IV);

x_PV = gather(squeeze(x_PV));     % [T × N]
save('PV.mat','x_PV')
