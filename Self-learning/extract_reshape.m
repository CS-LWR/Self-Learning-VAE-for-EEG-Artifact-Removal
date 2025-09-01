function X_new = extract_reshape(X)
% extract_reshape
% -------------------------------------------------------------------------
% Reshapes a 2-D sequence matrix into a 4-D tensor by splitting the time
% dimension into two channels.
%
% INPUT
%   X     : [L × N]
%           L = sequence length, N = number of samples
%
% OUTPUT
%   X_new : [1 × (L/2) × 2 × N]
%           First spatial dimension fixed to 1; temporal length halved and
%           placed along the second dimension; two channels created along
%           the third dimension; batch preserved as the fourth dimension.
%
% NOTE
%   Assumes L is even (L mod 2 == 0). No validation is performed.
% -------------------------------------------------------------------------

[seqLength, numSamples] = size(X);
X_new = reshape(X, [1, seqLength/2, 2, numSamples]);
end
