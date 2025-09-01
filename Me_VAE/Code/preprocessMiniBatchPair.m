function [X, Y] = preprocessMiniBatchPair(dataX, dataY)
    % Convert to dlarray if needed
    X = cat(4, dataX{:});
    Y = cat(4, dataY{:});

end
