function [loss, gE1,gE2,gD] = modelLoss_N2V(netE1,netE2,netD, I, O)
% modelLoss_N2V
% -------------------------------------------------------------------------
% Noise2Void-style reconstruction objective with total variation (TV) smoothing.
%
% INPUTS
%   netE1, netE2, netD : dlnetwork objects (two encoders and one decoder)
%   I                  : dlarray input, expected shape [1 128 2 B]
%                        (left/right context across the second spatial axis)
%   O                  : dlarray target, expected shape [1 256 1 B]
%                        (middle segment target; can be the noisy signal itself)
%
% OUTPUTS
%   loss               : scalar dlarray, total objective (MSE + λ·TV)
%   gE1, gE2, gD       : gradients w.r.t. learnables of netE1, netE2, netD
%
% NOTES
%   - Decoder output Y must match O in size.
%   - TV is computed along the temporal dimension (dimension 2).
% -------------------------------------------------------------------------

    % Hyperparameters
    lamTV = 1e-3;

    % --- Forward pass ---
    Z1 = forward(netE1, I);
    Z2 = forward(netE2, I);
    Z  = cat(1, Z1, Z2);
    Y  = forward(netD, Z);     % Expected size: [1 256 1 B]

    % --- Reconstruction loss (scalar MSE) ---
    diff    = Y - O;
    mse_all = mean(diff.^2, 'all');       % Scalar

    % --- Total variation smoothing (temporal dimension = 2) ---
    dY = Y(:,2:end,:,:) - Y(:,1:end-1,:,:);
    tv = mean(dY.^2, 'all');              % Scalar

    % Total loss
    loss = mse_all + lamTV*tv;            % Scalar

    % --- Backpropagation ---
    [gE1,gE2,gD] = dlgradient(loss, netE1.Learnables, netE2.Learnables, netD.Learnables);
end
