function kappa = checkCovCondition(classifierEpochs, trainLabels)
% checkCovCondition  Compute condition number of pooled within-class covariance.
%
%   kappa = checkCovCondition(classifierEpochs, trainLabels)
%
%   Inputs:
%     classifierEpochs : [d x N] feature matrix (features x trials)
%     trainLabels      : [N x 1] class labels
%
%   Output:
%     kappa : condition number of pooled within-class covariance matrix
%
%   Prints diagnostic message about stability.

    % Ensure correct shapes
    X = classifierEpochs;   % d x N
    y = trainLabels(:);     % N x 1
    [d, N] = size(X);

    if numel(y) ~= N
        error('Number of labels (%d) must match number of trials (%d).', numel(y), N);
    end

    classes = unique(y);
    K = numel(classes);

    Sw = zeros(d, d);
    Nk_total = 0;

    % Compute within-class covariance for each class
    for k = 1:K
        idx = (y == classes(k));    % logical mask
        Xk  = X(:, idx).';          % [n_k x d]
        nk  = size(Xk, 1);

        if nk <= 1
            error('Class %d has <= 1 sample, cannot compute covariance.', k);
        end

        Ck = cov(Xk);               % [d x d] covariance of class k
        Sw = Sw + (nk - 1) * Ck;    % accumulate unnormalized
        Nk_total = Nk_total + nk;
    end

    % Pooled covariance
    Sw = Sw / (Nk_total - K);

    % Condition number
    kappa = cond(Sw);

    % Print summary
    fprintf('Features (d): %d, Trials (N): %d, Classes (K): %d\n', d, N, K);
    fprintf('Condition number (kappa): %.3e\n', kappa);

    if kappa < 1e3
        verdict = 'Well-conditioned (no shrinkage likely needed).';
    elseif kappa < 1e6
        verdict = 'Moderately conditioned (consider small Gamma, e.g., 0.05–0.2).';
    else
        verdict = 'Ill-conditioned (use shrinkage: Gamma ~ 0.1–0.3).';
    end

    fprintf('Diagnosis: %s\n', verdict);
end
