clear
clc

%% Setup
% I load and merge multiple presets (here EOG + EMG) from the 'Dataset' root.
[x_IT,x_OT,x_IV,x_OV,x_IE,x_OE] = loadMergedEEGDatasets({'EOG','EMG'}, 'Dataset');

%% Data_Preprocessing
% I split each sample into front/middle/behind segments and rebuild my
% inputs/targets accordingly. This mirrors the exact slicing logic I use
% elsewhere, then returns tensors in the shapes my model expects.
[x_IT,x_IV,x_IE,x_OT,x_OV,x_OE,x_AT,x_AV,x_AE] = makeFrontBehindMiddleSets(x_IT,x_IV,x_IE,x_OT,x_OV,x_OE);
% I keep this gather+squeeze on x_AV to validate metrics change across the training process.
% This is to evaluate the denosing effectiveness on validation set by
% evaluateEEGDenoising function after each epoch (approximately 20 h data in one epoch)
x_AV = gather(squeeze(x_AV)); 

%% ----------------------- Model -----------------------
% I use a two-branch VAE encoder (E1 stochastic via samplingLayer, E2 deterministic)
% and a decoder that projects back to a 1×256×4 feature map and upsamples.
numLatentChannel1 = 512;
numLatentChannel2 = 512;
numLatentChannels = numLatentChannel1 + numLatentChannel2;
imageSize = [1 128 2];

layersE1 = [
    imageInputLayer(imageSize,Normalization="rescale-symmetric")
    convolution2dLayer([1 3], 64, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 3], 32, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 3], 16, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 3], 4, Stride=1, Padding='same')
    reluLayer
    fullyConnectedLayer(2*numLatentChannel1)
    samplingLayer
];

layersE2 = [
    imageInputLayer(imageSize,Normalization="rescale-symmetric")
    convolution2dLayer([1 1], 64, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 1], 32, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 1], 16, Stride=1, Padding='same')
    reluLayer
    convolution2dLayer([1 1], 4, Stride=1, Padding='same')
    reluLayer
    fullyConnectedLayer(numLatentChannel2)
];

projectionSize = [1 256 4];
numInputChannels = 1;

layersD = [
    featureInputLayer(numLatentChannels)
    projectAndReshapeLayer(projectionSize)
    transposedConv2dLayer([1 3],16,Cropping="same",Stride=1)
    reluLayer
    transposedConv2dLayer([1 3],32,Cropping="same",Stride=1)
    reluLayer
    transposedConv2dLayer([1 3],64,Cropping="same",Stride=1)
    reluLayer
    transposedConv2dLayer([1 3],numInputChannels,Cropping="same")
    ];

% I wrap the layers as dlnetwork modules.
netE1 = dlnetwork(layersE1);
netE2 = dlnetwork(layersE2);
netD  = dlnetwork(layersD);

%% Training
% I add my local Toolbox for evaluation/visualization helpers.
toolboxPath = fullfile(pwd, 'Toolbox');
addpath(toolboxPath);

% I set my run label and flags here.
model        = 'me_VAE+CS';
preset       = 'MIX';      % 'EMG' | 'EOG' | 'EEG' | 'EEG_motion'
preset_save  = 1;          % 0 | 1
fs           = 256;
plot_flag    = 0;

% I train for a large budget but rely on validation to break early.
numEpochs     = 10000;
miniBatchSize = 128;
learnRate     = 1e-4;

% I build datastores and minibatchqueues for inputs/targets/validation.
dsIT = arrayDatastore(x_IT,  IterationDimension=4);
dsOT = arrayDatastore(x_OT,  IterationDimension=4);
dsIV = arrayDatastore(x_IV,  IterationDimension=4);

% I keep my combined two-output pipeline here; the second stream goes to
% the loss function that decides what to use.
dsCombined = combine(dsIT, dsOT);
mbq_IOT = minibatchqueue(dsCombined, 2, ...
    MiniBatchSize = miniBatchSize, ...
    MiniBatchFcn = @(x, y) preprocessMiniBatchPair(x, y), ...
    MiniBatchFormat = ["SSCB", "SSCB"], ...
    PartialMiniBatch = "discard");

mbq_IV = minibatchqueue(dsIV, 1, ...
    MiniBatchSize = size(x_IV,4), ...
    MiniBatchFcn = @(x) preprocessMiniBatch(x), ...
    MiniBatchFormat = "SSCB", ...
    PartialMiniBatch = "return");

% I initialize Adam states for all three networks.
trailingAvgE1  = [];
trailingAvgSqE1= [];
trailingAvgE2  = [];
trailingAvgSqE2= [];
trailingAvgD   = [];
trailingAvgSqD = [];

% I set up the progress monitor and basic early-stop bookkeeping.
numObservationsTrain  = size(x_IT,4);
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations         = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor( ...
    Metrics=["Loss","Loss_Validation"], ...
    Info="Epoch", ...
    XLabel="Iteration");

epoch     = 0;
iteration = 0;
history   = inf;
times     = 0;

% I iterate over epochs and minibatches, updating the nets with adamupdate.
while epoch < numEpochs 
    epoch = epoch + 1;

    shuffle(mbq_IOT);

    while hasdata(mbq_IOT) 
        iteration = iteration + 1;

        % I read a minibatch and compute loss + gradients.
        [I,O] = next(mbq_IOT);
        [loss,gradientsE1,gradientsE2,gradientsD] = dlfeval(@modelLoss_N2V,netE1,netE2,netD,I,O);

        % I apply Adam updates to E1/E2/D.
        [netE1,trailingAvgE1,trailingAvgSqE1] = adamupdate(netE1, ...
            gradientsE1,trailingAvgE1,trailingAvgSqE1,iteration,learnRate);

        [netE2,trailingAvgE2,trailingAvgSqE2] = adamupdate(netE2, ...
            gradientsE2,trailingAvgE2,trailingAvgSqE2,iteration,learnRate);

        [netD, trailingAvgD, trailingAvgSqD] = adamupdate(netD, ...
            gradientsD,trailingAvgD,trailingAvgSqD,iteration,learnRate);

        % I log training loss to the monitor.
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100*iteration/numIterations;
    end

    % I run a full-batch validation pass and compute metrics on x_AV vs x_PV.
    x_PV = modelPredictions(netE1,netE2,netD,mbq_IV);
    x_PV = gather(squeeze(x_PV)); 

    results = evaluateEEGDenoising(x_AV, x_PV, fs, [model '_' preset], plot_flag);
    loss_v  = results.RMSE_t;

    % I keep a simple early-stop counter: if it gets worse repeatedly, I stop.
    if loss_v >= history
        times = times + 1;
        if times > 3
            break;
        end
    else
        history = loss_v;
        times   = 0;
    end

    % I log validation loss to the monitor and reset the val queue.
    recordMetrics(monitor,iteration,Loss_Validation=loss_v);
    updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
    monitor.Progress = 100*iteration/numIterations;
    reset(mbq_IV);
end

% If I want to visualize afterwards:
% x_OV = gather(squeeze(x_OV));
% visualizeEEGComparison(x_PV, x_AV, x_OV, fs)

% I clean up the Toolbox path when I'm done here.
rmpath(toolboxPath);

% I optionally save the trained networks with a name tied to my run label.
if preset_save
    filename = [model '_CS' '.mat'];
    save(filename, 'netE1', 'netE2', 'netD');
    disp(['Network saved as ' filename]);
end
