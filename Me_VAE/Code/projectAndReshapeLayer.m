classdef projectAndReshapeLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    % projectAndReshapeLayer
    % ---------------------------------------------------------------------
    % Custom layer that applies a fully connected projection from a
    % channel-wise vector to a target spatial tensor, then reshapes.
    %
    % Expected I/O (dlarray formats):
    %   - Input  X : formatted dlarray with channel dimension "C"
    %                and optional batch dimension "B"  --> ("CB" or "C")
    %   - Output Z : formatted dlarray with format "SSCB"
    %                (spatial, spatial, channel, batch)
    %
    % OutputSize should be a 1×3 vector: [S1, S2, C_out], where
    %   S1, S2 : spatial dimensions after reshape
    %   C_out  : channel dimension after reshape
    %
    % Learnable parameters:
    %   Weights : [prod(OutputSize) × C_in]
    %   Bias    : [prod(OutputSize) × 1]
    % where C_in is inferred from the input layout ("C" dimension).
    % ---------------------------------------------------------------------

    properties
        % Target spatial-plus-channel size after projection and reshape.
        OutputSize
    end

    properties (Learnable)
        % Fully connected weights and bias applied before reshaping.
        Weights
        Bias
    end

    methods
        function layer = projectAndReshapeLayer(outputSize,NameValueArgs)
            % projectAndReshapeLayer Constructor
            % -------------------------------------------------------------
            % layer = projectAndReshapeLayer(outputSize) creates the layer
            % that projects the input and reshapes it to OutputSize.
            %
            % layer = projectAndReshapeLayer(outputSize, Name=name) also
            % sets a custom layer name.
            %
            % INPUTS
            %   outputSize : [S1 S2 C_out]
            %   Name       : (optional) string scalar for layer.Name

            % Parse name-value inputs.
            arguments
                outputSize
                NameValueArgs.Name = "";
            end

            % Name and basic metadata.
            name = NameValueArgs.Name;
            layer.Name = name;
            layer.Description = "Project and reshape to size " + join(string(outputSize));
            layer.Type = "Project and Reshape";

            % Store target output size.
            layer.OutputSize = outputSize;
        end

        function layer = initialize(layer,layout)
            % initialize Initialize learnable parameters based on input layout
            % -------------------------------------------------------------
            % layer = initialize(layer, layout) allocates and initializes
            % Weights and Bias given the input networkDataLayout.
            %
            % INPUTS
            %   layer  : this layer
            %   layout : networkDataLayout describing input (must include "C")
            %
            % OUTPUT
            %   layer  : layer with initialized Weights and Bias

            % Cache target size info.
            outputSize = layer.OutputSize;

            % Initialize fully connected weights if empty.
            if isempty(layer.Weights)
                % Determine input channel count from the "C" dimension.
                idx = finddim(layout,"C");
                numChannels = layout.Size(idx);

                % Glorot initialization for [prod(OutputSize) × C_in].
                sz     = [prod(outputSize) numChannels];
                numOut = prod(outputSize);
                numIn  = numChannels;
                layer.Weights = initializeGlorot(sz, numOut, numIn);
            end

            % Initialize bias if empty (zeros).
            if isempty(layer.Bias)
                layer.Bias = initializeZeros([prod(outputSize) 1]);
            end
        end

        function Z = predict(layer, X)
            % predict Forward pass (projection + reshape)
            % -------------------------------------------------------------
            % Z = predict(layer, X) applies a fully connected mapping from
            % the input channels to a flattened target tensor, then reshapes
            % to [S1 S2 C_out B] and returns a formatted dlarray "SSCB".
            %
            % INPUTS
            %   layer : this layer
            %   X     : formatted dlarray with "C" (and optional "B")
            %
            % OUTPUT
            %   Z     : formatted dlarray of size [S1 S2 C_out B] with "SSCB"

            % Fully connected projection to a flat vector of length prod(OutputSize).
            weights = layer.Weights;
            bias    = layer.Bias;
            X       = fullyconnect(X, weights, bias);

            % Reshape flat vector to spatial + channel tensor, keep batch dim.
            outputSize = layer.OutputSize;
            Z = reshape(X, outputSize(1), outputSize(2), outputSize(3), []);
            Z = dlarray(Z, "SSCB");
        end
    end
end
