function loss = elboLoss(Y,T,mu,logSigmaSq)
% elboLoss
% -------------------------------------------------------------------------
% ELBO-style objective combining reconstruction fidelity and KL regularization.
%
% INPUTS
%   Y           : network reconstruction (same size as T)
%   T           : ground-truth target
%   mu          : latent means, size [K × B]
%   logSigmaSq  : latent log-variances, size [K × B]
%
% OUTPUT
%   loss        : scalar objective value (MSE + β·KL)
%
% DETAILS
%   - Reconstruction term uses mean squared error over all elements.
%   - KL term corresponds to KL(N(mu, diag(sigma^2)) || N(0, I)),
%     summed across latent dimensions and reduced by max over the batch.
%   - β (beta) weights the KL contribution (β-VAE style).
% -------------------------------------------------------------------------

% Reconstruction loss.
% reconstructionLoss = computePCCLoss(Y, T);
reconstructionLoss = mse(Y, T); 

% KL divergence.
KL = -0.5 * sum(1 + logSigmaSq - mu.^2 - exp(logSigmaSq),1);
KL = max(KL);
beta = 0.1;

% Combined loss.
loss = reconstructionLoss + beta*KL;
