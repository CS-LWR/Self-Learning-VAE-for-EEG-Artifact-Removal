function Y = modelPredictions(netE1,netE2,netD,mbq)
% modelPredictions
% -------------------------------------------------------------------------
% Run inference over a minibatchqueue and return concatenated predictions.
%
% INPUTS
%   netE1, netE2 : encoder dlnetworks
%   netD         : decoder dlnetwork
%   mbq          : minibatchqueue yielding input batches (e.g., "SSCB")
%
% OUTPUT
%   Y            : predictions concatenated along the 4th dimension
%                  (batch axis), i.e., size [..., N_total]
%
% PIPELINE (per batch)
%   1) Read a batch X from mbq.
%   2) Encode with E1 and E2 to obtain latent codes Z1 and Z2 (predict mode).
%   3) Concatenate latents (cat along dim 1) -> Z.
%   4) Decode Z with netD (predict) to get XGenerated.
%   5) Accumulate outputs along the 4th dimension.

Y = [];

% Iterate over all minibatches from the queue
while hasdata(mbq)
    X = next(mbq);

    % Encoders (inference)
    Z1 = predict(netE1, X);
    Z2 = predict(netE2, X);

    % Concatenate latent vectors
    Z = cat(1, Z1, Z2);

    % Decoder (inference)
    XGenerated = predict(netD, Z);

    % Collect predictions along the batch dimension
    Y = cat(4, Y, extractdata(XGenerated));
end
