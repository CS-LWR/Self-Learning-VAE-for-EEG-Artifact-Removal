function X_out = reshape_batches(X, direction)
%RESHAPE_PRESERVE Reshape [T×N] <--> [1×T×1×N] with strict order preservation.
%
%   X_out = reshape_preserve(X, direction)
%   direction = 'forward'  : [T × N] --> [1 × T × 1 × N]
%   direction = 'backward' : [1 × T × 1 × N] --> [T × N]
%
%   Ensures that the i-th sample remains the i-th sample after reshape.
if nargin < 2
        direction = 'forward';
end

direction = lower(direction);

switch direction
    case 'forward'
        if ismatrix(X)
            [T, N] = size(X);
            % Transpose for correct sample ordering (each column becomes [1 × T × 1 × 1])
            X_out = reshape(permute(X', [2, 1]), 1, T, 1, N);  % [1 × T × 1 × N]
        elseif ndims(X) == 4 && size(X,1)==1 && size(X,3)==1
            X_out = X;  % Already in desired shape
        else
            error('Unsupported input shape for forward direction.');
        end

    case 'backward'
        if ndims(X) == 4 && size(X,1)==1 && size(X,3)==1
            [~, T, ~, N] = size(X);
            X_out = permute(reshape(X, T, N), [1, 2]);  % [T × N]
        elseif ismatrix(X)
            X_out = X;  % Already [T × N]
        else
            error('Unsupported input shape for backward direction.');
        end

    otherwise
        error('Invalid direction. Use ''forward'' or ''backward''.');
end
end
