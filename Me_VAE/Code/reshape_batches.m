function X_new = reshape_batches(X)
% reshape_batches
% -------------------------------------------------------------------------
% Reshape a 2-D array of sequences into a 4-D tensor expected by the model.
%
% INPUT
%   X     : [L × N]
%           L = sequence length (rows), N = number of samples (columns)
%
% OUTPUT
%   X_new : [1 × L × 1 × N]
%           Formatted as (spatial, spatial, channel, batch)
%
% NOTE
%   This is a pure reshape (no transpose). It expands dimensions to match
%   SSCB-style inputs used by convolutional layers.

    [seqLength, numSamples] = size(X);

    % Expand to [1 × L × 1 × N]
    X_new = reshape(X, [1, seqLength, 1, numSamples]);
end
