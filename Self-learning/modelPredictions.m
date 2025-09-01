function Y = modelPredictions(netE1,netE2,netD, mbq)
% modelPredictions
% -------------------------------------------------------------------------
% Runs inference over a minibatchqueue and concatenates decoder outputs.
%
% INPUTS
%   netE1, netE2 : encoder dlnetworks (E1 may be stochastic; E2 deterministic)
%   netD         : decoder dlnetwork
%   mbq          : minibatchqueue yielding input batches (expected format: SSCB)
%
% OUTPUT
%   Y            : 4-D array of predictions concatenated along dimension 4
%                  (batch aggregation across all minibatches)
%
% DETAILS
%   - Resets the minibatchqueue at the start to ensure full traversal.
%   - For each batch X, encodes to Z1 and Z2, concatenates along dim 1,
%     decodes to Xg, extracts numeric data, and concatenates along dim 4.
% -------------------------------------------------------------------------

Y = [];
reset(mbq);
while hasdata(mbq)
    X = next(mbq);              % Expected SSCB format
    Z1 = predict(netE1, X);
    Z2 = predict(netE2, X);
    Z  = cat(1, Z1, Z2);        % Concatenate latents along channel-like dim
    Xg = predict(netD, Z);      % Decoder output
    Y  = cat(4, Y, extractdata(Xg));  % Accumulate along batch dimension
end
end
