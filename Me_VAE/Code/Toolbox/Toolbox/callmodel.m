function X_predicted = callmodel(X_input, model)
% CALLMODEL  Perform denoising inference using a trained VAE model.
%
%   X_predicted = callmodel(X_input, model)
%
%   Inputs:
%     - X_input : EEG input matrix of size [T × N], where T is the number of
%                 time points per sample, and N is the number of samples.
%                 Must be a 2D matrix. If input is a cell array, it will be
%                 converted to a matrix.
%     - model   : Model type as a string. Only 'VAE' is supported currently.
%
%   Output:
%     - X_predicted : Reconstructed/denoised output in the format [1 × T × 1 × N]

% Get current script folder (Toolbox/)
toolboxDir = fileparts(mfilename('fullpath'));

    if strcmp(model, 'VAE')

        modelPath = fullfile(toolboxDir, '\', 'Model', 'trainedVAE_New4.mat');
        % Load trained encoder and decoder networks
        load(modelPath);

        % Convert cell input to matrix if needed
        if iscell(X_input)
            X_input = cell2mat(X_input);   
        end

        % Check and reshape input if it's in [T × N] format
        sz = size(X_input);
        if ismatrix(X_input) && sz(1) > 1 && sz(2) > 1
            X_input = reshape_batches(X_input);  % Reshape to [1 × T × 1 × N]
        elseif ndims(X_input) == 4 && isequal(sz(1:3), [1, sz(2), 1])
            % Already in the expected format [1 × T × 1 × N], do nothing
        else
            error("Unsupported input shape: expected [T×N] or [1×T×1×N], got [%s]", ...
                num2str(sz));
        end

        % Create a datastore for inference
        ds_I = arrayDatastore(X_input, 'IterationDimension', 4);

        % Create minibatch queue
        mbq_I = minibatchqueue(ds_I, 1, ...
            'MiniBatchSize', 1, ...
            'MiniBatchFcn', @preprocessMiniBatch, ...
            'MiniBatchFormat', "SSCB", ...
            'PartialMiniBatch', "discard");

        % Run predictions
        X_predict = modelPredictions(netE, netD, mbq_I);
        X_predicted = reshape_batches(X_predict, 'backward');
    else
        error("Unsupported model type: %s", model);
    end
end


function Y = modelPredictions(netE, netD, mbq)
% MODELPREDICTIONS  Perform forward pass through encoder and decoder networks.
%
%   Inputs:
%     - netE : Trained encoder network
%     - netD : Trained decoder network
%     - mbq  : Minibatch queue containing input data
%
%   Output:
%     - Y    : Concatenated predicted outputs of size [1 × T × 1 × N]

    Y = [];

    % Loop over minibatches
    while hasdata(mbq)
        X = next(mbq);  % Get one batch

        % Pass through encoder to get latent variables
        Z = predict(netE, X);

        % Pass through decoder to reconstruct signal
        XGenerated = predict(netD, Z);

        % Concatenate results along the 4th dimension
        Y = cat(4, Y, extractdata(XGenerated));
    end
end
