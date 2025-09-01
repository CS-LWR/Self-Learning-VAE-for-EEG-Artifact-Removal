clear
clc

%% Load Data
% -------------------------------------------------------------------------
% Experiment presets:
%   model       : label for logging/saving
%   dataset     : dataset index as a string (e.g., '1', '2', ...)
%   preset      : artifact/domain preset ('EMG' | 'EOG' | 'EEG' | 'EEG_motion')
%   preset_save : whether to save trained networks (1=yes, 0=no)
%   fs          : sampling frequency (Hz)
%   plot_flag   : pass-through flag for evaluation visuals (0/1)
model='me_VAE';
dataset = '1';
preset = 'EOG';          % Change to: 'EMG' | 'EOG' | 'EEG' | 'EEG_motion'
preset_save = 1;         % 0 or 1
fs = 256;
plot_flag = 0;

% Target folder containing .mat files for this dataset/preset
folderPath = fullfile('..', ['Dataset' dataset], preset);

% Enumerate all .mat files within the folder
files = dir(fullfile(folderPath,'*.mat'));

% Batch load each file (assumes compatible variables are defined within)
for i = 1:length(files)
    filePath = fullfile(folderPath, files(i).name);
    load(filePath);
    disp(['Loaded: ' files(i).name]);
end


%% Model
% -------------------------------------------------------------------------
% Variational autoencoder with two encoder branches (E1, E2) and one decoder (D).
%   - E1: conv stack ending in FC(2*K1) + samplingLayer  -> produces latent Z1
%   - E2: 1x1 conv stack ending in FC(K2)                -> produces latent Z2
%   - D : projects [Z1; Z2] to spatial tensor via custom project layer, then
%         transposed conv stack reconstructs the 1x512x1 signal.
numLatentChannel1 = 512;
numLatentChannel2 = 512;
numLatentChannels = numLatentChannel1 + numLatentChannel2;
imageSize = [1 512 1];  % [H W C]

layersE1 = [
    imageInputLayer(imageSize,Normalization="none")   % inputs are assumed pre-normalized
    convolution2dLayer([1 3], 64, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 3], 32, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 3], 16, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 3], 4, Stride=1, Padding='same')
    reluLayer
    fullyConnectedLayer(2*numLatentChannel1)          % [mu, logVar] for K1 channels
    samplingLayer                                      % reparameterization: Z1
];

layersE2 = [
    imageInputLayer(imageSize,Normalization="none")
    convolution2dLayer([1 1], 64, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 1], 32, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 1], 16, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 1], 4, Stride=1, Padding='same')
    reluLayer
    fullyConnectedLayer(numLatentChannel2)            % direct latent projection Z2
];

% Decoder: feature input -> project+reshape -> deconvolutional stack
projectionSize   = [1 512 4];                         % spatial target for projection
numInputChannels = imageSize(3);

layersD = [
    featureInputLayer(numLatentChannels)
    projectAndReshapeLayer(projectionSize)             % maps latent to [1x512x4]
    transposedConv2dLayer([1 3],16,Cropping="same",Stride=1)
    reluLayer
    transposedConv2dLayer([1 3],32,Cropping="same",Stride=1)
    reluLayer
    transposedConv2dLayer([1 3],64,Cropping="same",Stride=1)
    reluLayer
    transposedConv2dLayer([1 3],numInputChannels,Cropping="same") % reconstruct 1x512x1
    ];

% Wrap layers as dlnetwork modules
netE1 = dlnetwork(layersE1);
netE2 = dlnetwork(layersE2);
netD = dlnetwork(layersD);

%% Training
% -------------------------------------------------------------------------
% Reshape raw arrays into 4-D batches (SSCB) expected by the network.
[xInputCell_Train]  = reshape_batches(x_IT);
[xOutputCell_Train] = reshape_batches(x_OT);
[xInputCell_Val]    = reshape_batches(x_IV);
[xOutputCell_Val]   = reshape_batches(x_OV);

% Training hyperparameters
numEpochs     = 50;
miniBatchSize = 128;
learnRate     = 1e-3;

% Datastores for input/output training data
dsIT = arrayDatastore(xInputCell_Train,IterationDimension=4);
dsOT = arrayDatastore(xOutputCell_Train,IterationDimension=4);

% Combine to preserve alignment between inputs and targets
dsCombined = combine(dsIT, dsOT);

% Minibatch queue producing two outputs: (I, O)
mbq_IOT = minibatchqueue(dsCombined, 2, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn = @(x, y) preprocessMiniBatchPair(x, y), ...
    MiniBatchFormat = ["SSCB", "SSCB"], ...
    PartialMiniBatch = "discard");

% Validation queue (inputs only)
ds_IV = arrayDatastore(xInputCell_Val,IterationDimension=4);
numOutputs = 1;
mbq_IV = minibatchqueue(ds_IV,numOutputs, ...
    MiniBatchSize = size(x_OV,2), ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat="SSCB", ...
    PartialMiniBatch="return");

% Add local Toolbox path for evaluation/visualization helpers
toolboxPath = fullfile(pwd, 'Toolbox');
addpath(toolboxPath);

% Adam moment buffers for each subnetwork
trailingAvgE1 = [];
trailingAvgSqE1 = [];
trailingAvgE2 = [];
trailingAvgSqE2 = [];
trailingAvgD = [];
trailingAvgSqD = [];

% Training bookkeeping
numObservationsTrain  = size(xInputCell_Train,4);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations         = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor( ...
    Metrics=["Loss","Loss_Validation"], ...
    Info="Epoch", ...
    XLabel="Iteration");

epoch = 0;
iteration = 0;
history = inf;    % best validation metric seen so far
times = 0;        % plateau counter

% ============================== Epoch loop ===============================
while epoch < numEpochs 
    epoch = epoch + 1;

    shuffle(mbq_IOT);

    % ------------------------- Minibatch loop ---------------------------
    while hasdata(mbq_IOT) 
        iteration = iteration + 1;

        % Read a minibatch (I: input, O: target)
        [I,O] = next(mbq_IOT);

        % Compute loss and gradients via custom objective
        [loss,gradientsE1,gradientsE2,gradientsD] = dlfeval(@modelLoss,netE1,netE2,netD,I,O);

        % Adam updates for E1, E2, D
        [netE1,trailingAvgE1,trailingAvgSqE1] = adamupdate(netE1, ...
            gradientsE1,trailingAvgE1,trailingAvgSqE1,iteration,learnRate);

        [netE2,trailingAvgE2,trailingAvgSqE2] = adamupdate(netE2, ...
            gradientsE2,trailingAvgE2,trailingAvgSqE2,iteration,learnRate);

        [netD, trailingAvgD, trailingAvgSqD] = adamupdate(netD, ...
            gradientsD,trailingAvgD,trailingAvgSqD,iteration,learnRate);

        % Progress monitor (training)
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100*iteration/numIterations;
    end

    % ----------------------- Validation per epoch -----------------------
    x_PV = modelPredictions(netE1,netE2,netD,mbq_IV);
    x_PV = gather(squeeze(x_PV)); 

    % Evaluate denoising performance (uses Toolbox function)
    results = evaluateEEGDenoising(x_OV, x_PV, fs, [model '_' preset], plot_flag);
    loss_v = results.RMSE_t;   % use time-domain RMSE as validation proxy

    % Simple LR scheduling / early stop based on plateau count
    if loss_v >= history
        times = times + 1;
        if times >= 2 && epoch > 10
            learnRate = 1e-5;  % reduce LR after repeated stalls
        end
        if times >= 3 && epoch > 10
            break;             % early stop after prolonged stagnation
        end
    else
        history = loss_v;
        times = 0;
    end
    
    % Progress monitor (validation)
    recordMetrics(monitor,iteration,Loss_Validation=loss_v);
    updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
    monitor.Progress = 100*iteration/numIterations;
    reset(mbq_IV);
end

% Remove Toolbox path (optional cleanup)
rmpath(toolboxPath);

% Save trained networks (optional)
if preset_save
    % Construct filename (adjust as needed for your naming convention)
    filename = ['trainedMe_VAE_' 'Dataset2_New' '.mat'];
    save(filename, 'netE1', 'netE2','netD');
    disp(['Network saved as ' filename]);
end


%% Verification - Waveform (optional)
% -------------------------------------------------------------------------
% Example block for single-batch prediction and visualization after training.
% Disabled by default; uncomment and adapt as needed.
%
% ds_IV = arrayDatastore(xInputCell_Val,IterationDimension=4);
% numOutputs = 1;
% mbq_IV = minibatchqueue(ds_IV,numOutputs, ...
%     MiniBatchSize = 1, ...
%     MiniBatchFcn=@preprocessMiniBatch, ...
%     MiniBatchFormat="SSCB", ...
%     PartialMiniBatch="discard");
%
% x_Predicted = modelPredictions(netE,netD,mbq_IV);
%
% toolboxPath = fullfile(pwd, 'Toolbox'); addpath(toolboxPath);
% results = evaluateEEGDenoising(x_PT, x_OT, fs, [model '_' preset], plot_flag);
% visualizeEEGComparison(x_OT, x_OT, x_IT, fs)
% rmpath(toolboxPath);
