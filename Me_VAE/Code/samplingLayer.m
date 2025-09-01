classdef samplingLayer < nnet.layer.Layer
    % samplingLayer
    % ---------------------------------------------------------------------
    % Variational Autoencoder (VAE) reparameterization layer.
    % Splits a concatenated vector of means and log-variances, then
    % samples latent variables via:
    %       Z = mu + exp(0.5 * logSigmaSq) .* epsilon,   epsilon ~ N(0, I)
    %
    % Intended tensor layout:
    %   - Inputs X are sized [2K × B], where:
    %       * K = number of latent channels
    %       * B = mini-batch size
    %       * X(1:K,   :) are the means (mu)
    %       * X(K+1:2K,:) are the log-variances (logSigmaSq)
    %   - Outputs:
    %       * Z           : [K × B] sampled latent vectors
    %       * mu          : [K × B] means
    %       * logSigmaSq  : [K × B] log-variances
    %
    % Notes on numerical stability:
    %   - mu is clipped to [-1, 1] to avoid extreme activations.
    %   - logSigmaSq is capped (<= 2) to limit exp(0.5 * logSigmaSq).
    % ---------------------------------------------------------------------

    methods
        function layer = samplingLayer(args)
            % layer = samplingLayer() creates a VAE sampling layer.
            %
            % layer = samplingLayer(Name=name) also sets the layer name.
            %
            % Parameters (Name-Value):
            %   Name : (string) layer name

            arguments
                args.Name = "";
            end

            % Layer metadata
            layer.Name        = args.Name;
            layer.Type        = "Sampling";
            layer.Description = "Mean and log-variance sampling";
            layer.OutputNames = ["out" "mean" "log-variance"];
        end

        function [Z, mu, logSigmaSq] = predict(~, X)
            % [Z, mu, logSigmaSq] = predict(~, X)
            % Forward pass for both training and inference.
            %
            % INPUT
            %   X : [2K × B] concatenated statistics, where the first K rows
            %       are means (mu) and the next K rows are log-variances.
            %
            % OUTPUTS
            %   Z           : [K × B] reparameterized samples
            %   mu          : [K × B] means
            %   logSigmaSq  : [K × B] log-variances

            % Infer dimensions
            numLatentChannels = size(X, 1) / 2;
            miniBatchSize     = size(X, 2);

            % Split mu and log-variance
            mu = X(1:numLatentChannels, :);
            % Clip mu to a reasonable range for stability
            mu = max(min(mu, 1), -1);

            logSigmaSq = X(numLatentChannels+1:end, :);
            % Cap log-variance to avoid exponential overflow
            logSigmaSq = min(logSigmaSq, 2);

            % Reparameterization trick: Z = mu + sigma .* epsilon
            epsilon = randn(numLatentChannels, miniBatchSize, "like", X);
            sigma   = exp(0.5 * logSigmaSq);
            Z       = epsilon .* sigma + mu;
        end
    end
end
