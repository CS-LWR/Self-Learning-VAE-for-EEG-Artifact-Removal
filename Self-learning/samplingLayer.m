classdef samplingLayer < nnet.layer.Layer
    % samplingLayer
    % ---------------------------------------------------------------------
    % Custom VAE sampling layer that interprets the input as concatenated
    % mean and log-variance vectors, performs reparameterization, and
    % returns the sampled latent along with the mean and log-variance.
    %
    % BEHAVIOR
    %   - Input X is split along the first dimension into:
    %       mu         = X(1:K, :)
    %       logSigmaSq = X(K+1:end, :)
    %   - Mean is clipped to [-1, 1] to limit magnitude.
    %   - log-variance is upper-bounded (≤ 2) to avoid numerical overflow
    %     when exponentiated.
    %   - Reparameterization: Z = mu + sigma .* epsilon,
    %     where sigma = exp(0.5 * logSigmaSq) and epsilon ~ N(0, I).
    %
    % OUTPUT NAMES
    %   "out" (sampled Z), "mean" (mu), "log-variance" (logSigmaSq)
    %
    % NOTES
    %   - Dimensions: if size(X,1) = 2K, then each of mu/logSigmaSq has K
    %     rows; size(X,2) is treated as mini-batch size.
    %   - predict is used for both training and inference in dlnetwork.
    % ---------------------------------------------------------------------

    methods
        function layer = samplingLayer(args)
            arguments; args.Name = ""; end
            layer.Name = args.Name;
            layer.Type = "Sampling";
            layer.Description = "Mean and log-variance sampling";
            layer.OutputNames = ["out" "mean" "log-variance"];
        end

        function [Z,mu,logSigmaSq] = predict(~,X)
            % Split statistics
            K = size(X,1)/2; B = size(X,2);
            mu = X(1:K,:);             mu = max(min(mu, 1), -1);
            logSigmaSq = X(K+1:end,:); logSigmaSq = min(logSigmaSq, 2);

            % Reparameterization trick
            epsilon = randn(K,B,"like",X);
            sigma   = exp(.5*logSigmaSq);
            Z = epsilon .* sigma + mu;
        end
    end
end
