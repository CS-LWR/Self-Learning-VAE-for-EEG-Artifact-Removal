classdef projectAndReshapeLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    % projectAndReshapeLayer
    % ---------------------------------------------------------------------
    % Custom layer that linearly projects a feature vector and reshapes it
    % to a specified spatial tensor size. The output is formatted as SSCB.
    %
    % PURPOSE
    %   - Apply a fully connected projection (Weights, Bias).
    %   - Reshape the projected vector to [H × W × C × B].
    %
    % EXPECTED INPUT/OUTPUT
    %   Input : dlarray with channel ("C") and optional batch ("B") dims.
    %   Output: dlarray with format "SSCB" and size given by OutputSize.
    %
    % LEARNABLES
    %   Weights : [prod(OutputSize) × Cin]
    %   Bias    : [prod(OutputSize) × 1]
    % ---------------------------------------------------------------------

    properties
        % Target output spatial size as a 1×3 vector [H W C].
        OutputSize
    end
    properties (Learnable)
        % Fully connected projection parameters.
        Weights
        Bias
    end
    methods
        function layer = projectAndReshapeLayer(outputSize,NameValueArgs)
            % Constructor: sets name, description, type, and OutputSize.
            arguments
                outputSize
                NameValueArgs.Name = ""
            end
            name = NameValueArgs.Name; layer.Name = name;
            layer.Description = "Project and reshape to size " + join(string(outputSize));
            layer.Type = "Project and Reshape";
            layer.OutputSize = outputSize;
        end
        function layer = initialize(layer,layout)
            % Parameter initialization using Glorot for Weights and zeros for Bias.
            % The number of input channels Cin is read from the "C" dimension.
            outputSize = layer.OutputSize;
            if isempty(layer.Weights)
                idx = finddim(layout,"C");
                numChannels = layout.Size(idx);
                sz = [prod(outputSize) numChannels];
                numOut = prod(outputSize); numIn = numChannels;
                layer.Weights = initializeGlorot(sz,numOut,numIn);
            end
            if isempty(layer.Bias)
                layer.Bias = initializeZeros([prod(outputSize) 1]);
            end
        end
        function Z = predict(layer, X)
            % Forward pass: fully connected projection followed by reshape
            % to [H × W × C × B] and labeling as "SSCB".
            weights = layer.Weights; bias = layer.Bias;
            X = fullyconnect(X,weights,bias);
            outputSize = layer.OutputSize;
            Z = reshape(X,outputSize(1),outputSize(2),outputSize(3),[]);
            Z = dlarray(Z,"SSCB");
        end
    end
end
