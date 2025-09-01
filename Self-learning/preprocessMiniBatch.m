function X = preprocessMiniBatch(dataX)
X = cat(4, dataX{:});
X = dlarray(single(X), 'SSCB');
end