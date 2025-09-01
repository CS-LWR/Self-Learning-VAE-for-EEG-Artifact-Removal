function [X, Y] = preprocessMiniBatchPair(dataX, dataY)
X = cat(4, dataX{:});
Y = cat(4, dataY{:});
X = dlarray(single(X), 'SSCB');
Y = dlarray(single(Y), 'SSCB');
end
