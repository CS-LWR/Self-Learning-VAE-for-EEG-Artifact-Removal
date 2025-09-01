function X_new = reshape_batches(X)
% reshape_batches
% -------------------------------------------------------------------------
% Reshapes a 2-D sequence matrix into a 4-D tensor in SSCB layout.
%
% INPUT
%   X     : [L × N]
%           L = sequence length, N = number of samples (batch size)
%
% OUTPUT
%   X_new : [1 × L × 1 × N]
%           First spatial dim fixed to 1; temporal length kept as L on the
%           second dim; channel dim fixed to 1; batch preserved as N.
%
% NOTES
%   - Performs a pure reshape without transposition.
%   - Intended for models expecting SSCB ordering.
% -------------------------------------------------------------------------

[seqLength, numSamples] = size(X);
X_new = reshape(X, [1, seqLength, 1, numSamples]);
end
