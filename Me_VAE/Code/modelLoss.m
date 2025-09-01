function [loss,gradientsE1,gradientsE2,gradientsD,Z,mu,logSigmaSq] = modelLoss(netE1,netE2,netD,N,C)
% modelLoss
% -------------------------------------------------------------------------
% Compute ELBO-style loss and gradients for a two-branch VAE:
%   - Forward pass through encoders E1 (stochastic) and E2 (deterministic)
%   - Concatenate latents -> decode with D
%   - ELBO loss = MSE reconstruction + beta * KL (see elboLoss)
%   - Backprop to obtain gradients for E1, E2, D
%
% INPUTS
%   netE1, netE2, netD : dlnetwork objects (encoders E1/E2 and decoder D)
%   N                  : input batch (dlarray, e.g., "SSCB" as in pipeline)
%   C                  : reconstruction target batch (same size as decoder output)
%
% OUTPUTS
%   loss        : scalar dlarray, total ELBO-style loss
%   gradients*  : gradients for learnable parameters of E1, E2, D
%   Z           : concatenated latent codes [Z1; Z2]
%   mu          : posterior means from E1
%   logSigmaSq  : posterior log-variances from E1
% -------------------------------------------------------------------------

% ----- Encoder forward passes --------------------------------------------
% E1 returns sampled latent Z1 along with its parameters (mu, logSigmaSq).
[Z1, mu, logSigmaSq] = forward(netE1, N);

% E2 returns a deterministic latent Z2.
Z2 = forward(netE2, N);

% Concatenate latents from both branches.
Z = cat(1, Z1, Z2);

% ----- Decoder forward pass ----------------------------------------------
Y = forward(netD, Z);

% ----- Loss and gradients -------------------------------------------------
% ELBO-style objective: MSE recon + beta * KL (beta and KL handling in elboLoss).
loss = elboLoss(Y, C, mu, logSigmaSq);

% Backpropagate to encoder/decoder learnables.
[gradientsE1, gradientsE2, gradientsD] = dlgradient( ...
    loss, netE1.Learnables, netE2.Learnables, netD.Learnables);

% Debug helpers (kept commented):
% disp("Mean of μ: " + mean(extractdata(mu), 'all'));
% disp("Mean of σ²: " + mean(extractdata(exp(logSigmaSq)), 'all'));
end
