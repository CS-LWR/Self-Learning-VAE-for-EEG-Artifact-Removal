function loss = elboLoss(Y,T,mu,logSigmaSq)
% elboLoss
% -------------------------------------------------------------------------
% Evidence Lower BOund (ELBO)-style objective for VAEs:
%   loss = reconstructionLoss + beta * KL
%
% INPUTS
%   Y           : network output (reconstruction), same size as T
%   T           : ground-truth target
%   mu          : latent means, size [K × B] (K=latent dims, B=batch)
%   logSigmaSq  : latent log-variances, size [K × B]
%
% OUTPUT
%   loss        : scalar loss value (reconstruction + weighted KL)
%
% NOTES
%   - Reconstruction term uses MSE across all elements.
%   - KL term is computed per sample by summing over latent dims, then
%     the maximum over the batch is taken (KL = max(KL)) rather than the
%     mean. This emphasizes the "hardest" example in the batch.
%   - beta controls the KL weight (β-VAE style).

% ------------------------- Reconstruction loss ---------------------------
% Alternative PCC-based loss is retained for reference:
% reconstructionLoss = computePCCLoss(Y, T);
reconstructionLoss = mse(Y, T); 

% ----------------------------- KL divergence -----------------------------
% For a diagonal Gaussian posterior q(z|x)=N(mu, diag(sigma^2)) vs. prior
% p(z)=N(0, I), the closed-form KL is:
%   KL = -0.5 * sum(1 + log(sigma^2) - mu.^2 - sigma.^2)
% Here logSigmaSq = log(sigma^2).
KL = -0.5 * sum(1 + logSigmaSq - mu.^2 - exp(logSigmaSq), 1);  % sum over K
KL = max(KL);                                                   % pick hardest sample
beta = 0.1;                                                     % KL weight

% ------------------------------ Total loss -------------------------------
loss = reconstructionLoss + beta*KL;
