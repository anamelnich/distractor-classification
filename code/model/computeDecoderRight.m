function [decoder, classifierEpochs] = computeDecoderRight(trainEpochs, trainLabels, params)
% computeDecoderLeft trains a classifier to detect left-distractor trials 
% using lateralized EEG features from left vs. right posterior electrodes.
%
% Output:
%   decoder           - trained model structure with metadata
%   classifierEpochs  - feature matrix used to train the model

%% ==================== Baseline Correction ==================== %%
baseline_window = [-0.2, 0];
baseline_idx = find(params.epochTime >= baseline_window(1) & params.epochTime <= baseline_window(2));
baseline = mean(trainEpochs(baseline_idx, :, :), 1);
trainEpochs = trainEpochs - baseline;

%% ==================== ROI Selection ==================== %%
LeftElectrodes  = {'P1', 'P3', 'P5', 'P7', 'PO3', 'PO5', 'PO7'};
RightElectrodes = {'P2', 'P4', 'P6', 'P8', 'PO4', 'PO6', 'PO8'};
leftIdx  = find(ismember(params.chanLabels, LeftElectrodes));
rightIdx = find(ismember(params.chanLabels, RightElectrodes));

erpEpochs   = trainEpochs(:, rightIdx, :);
diffEpochs  = trainEpochs(:, rightIdx, :) - trainEpochs(:, leftIdx, :);

% Distractor class: distractor left (label 2)
% No distractor class: distractor right, no distractor (labels 1 or 0)
trainLabels(trainLabels == 1) = 0;
trainLabels(trainLabels == 2) = 1;

%% ==================== Feature Extraction ==================== %%
% ERP features
if params.features.erp_iscompute
    [ERP_feats, ERPfilter] = processFeatures(erpEpochs, trainLabels, params);
else
    ERP_feats = [];
    ERPfilter = 'na';
end

% Difference wave features
if params.features.diffwave_iscompute
    [Diff_feats, DiffFilter] = processFeatures(diffEpochs, trainLabels, params);
else
    Diff_feats = [];
    DiffFilter = 'na';
end

% Combine features
if params.features.erp_iscompute && params.features.diffwave_iscompute
    classifierEpochs = [ERP_feats; Diff_feats];
elseif params.features.erp_iscompute
    classifierEpochs = ERP_feats;
elseif params.features.diffwave_iscompute
    classifierEpochs = Diff_feats;
else
    error('No features selected. Set params.features.erp_iscompute or diffwave_iscompute to true.');
end

%% ==================== Dimensionality Reduction ==================== %%
if isequal(params.classify.reduction.type, 'pca')
    [coeff, ~, ~, ~, explained, mu] = pca(classifierEpochs');
    numKeep = find(cumsum(explained) > 95, 1);
    coeff = coeff(:, 1:numKeep);
    applyPCA = @(x) bsxfun(@minus, x', mu) * coeff;
    classifierEpochs = applyPCA(classifierEpochs)';
end

% Normalization
if params.classify.is_normalize
    maxVal = max(classifierEpochs, [], 2);
    minVal = min(classifierEpochs, [], 2);
    normalize = @(x) (x - minVal) ./ (maxVal - minVal);
    classifierEpochs = normalize(classifierEpochs);
end

% LASSO
if isequal(params.classify.reduction.type, 'lasso')
    lambdaMax = 0.1;
    Lambda = logspace(log10(0.001 * lambdaMax), log10(lambdaMax), 100);
    cvmodel = fitrlinear(classifierEpochs, trainLabels, 'ObservationsIn', 'columns', ...
        'Lambda', Lambda, 'KFold', 5, 'Learner', 'leastsquares', ...
        'Solver', 'sparsa', 'Regularization', 'lasso');
    mse = kfoldLoss(cvmodel);
    [~, idx] = min(mse);
    selectedLambda = Lambda(idx);

    modelLasso = fitrlinear(classifierEpochs, trainLabels, 'ObservationsIn', 'columns', ...
        'Lambda', selectedLambda, 'Learner', 'leastsquares', ...
        'Solver', 'sparsa', 'Regularization', 'lasso');
    
    keepIdx = modelLasso.Beta ~= 0;
    classifierEpochs = classifierEpochs(keepIdx, :);
    disp(['Number of features selected: ', num2str(sum(keepIdx))]);

elseif isequal(params.classify.reduction.type, 'r2')
    power = compute_r2(permute(classifierEpochs, [1 3 2]), trainLabels); 
    [~, keepIdx] = sort(power, 'descend');
    keepIdx = keepIdx(1:30);
    classifierEpochs = classifierEpochs(keepIdx, :);
end

%% ==================== Model Training ==================== %%
modelRaw = fitcdiscr(classifierEpochs', trainLabels, ...
    'Prior', 'uniform', 'DiscrimType', params.classify.type);

% Transform LDA output to probability using a sigmoid fit
w = modelRaw.Coeffs(2,1).Linear;
mu_coef = modelRaw.Coeffs(2,1).Const;
distance = classifierEpochs' * w + mu_coef;
p1 = 0.025; p2 = 1 - p1;
b1 = -log((1 - p1) / p1) / prctile(distance, 100 * p1);
b2 = -log((1 - p2) / p2) / prctile(distance, 100 * p2);
b = (b1 + b2) / 2;
model = @(x) 1 ./ (1 + exp(-b * (x * w + mu_coef)));

%% ==================== Store Decoder ==================== %%
decoder = struct();
decoder.Classes = modelRaw.ClassNames;
decoder.fsamp = params.fsamp;
decoder.epochOnset = params.epochOnset;
decoder.numFeatures = size(classifierEpochs, 1);
decoder.classify = struct( ...
    'type', params.classify.type, ...
    'is_normalize', params.classify.is_normalize, ...
    'reduction', struct('type', params.classify.reduction.type) ...
);
if params.classify.is_normalize
    decoder.classify.funNormalize = normalize;
end
if ismember(params.classify.reduction.type, {'lasso', 'r2'})
    decoder.classify.keepIdx = keepIdx;
elseif isequal(params.classify.reduction.type, 'pca')
    decoder.classify.applyPCA = applyPCA;
end
decoder.classify.model = model;
decoder.resample = params.resample;
decoder.features = struct( ...
    'erp_iscompute', params.features.erp_iscompute, ...
    'diffwave_iscompute', params.features.diffwave_iscompute ...
);
decoder.spatialFilter = struct( ...
    'erp', ERPfilter, ...
    'diff', DiffFilter ...
);
decoder.leftElectrodeIndices = leftIdx;
decoder.rightElectrodeIndices = rightIdx;
decoder.baseline_indices = baseline_idx;
if exist('selectedLambda', 'var')
    decoder.lassoLambda = selectedLambda;
end

end

%% ==================== Helper Functions ==================== %%
function [features, filterMatrix] = processFeatures(epochData, trainLabels, params)
% Applies spatial filtering and resamples data for feature extraction

% Apply CCA spatial filter
eeg = epochData(params.spatialFilter.time, :, :);
filterMatrix = get_cca_spatialfilter(eeg, trainLabels);
filterMatrix = filterMatrix(:, 1:params.spatialFilter.nComp);
epochData = apply_spatialFilter(epochData, filterMatrix);

% Resample
if params.resample.is_compute
    resamps = epochData(params.resample.time(1:params.resample.ratio:end), :, :);
    features = reshape(resamps, [], size(epochData, 3));
else
    features = [];
end
end

function output = apply_spatialFilter(data, filter)
% Applies spatial filter to each trial of data
[nSamples, ~, nTrials] = size(data);
nComp = size(filter, 2);
output = nan(nSamples, nComp, nTrials);
for i = 1:nTrials
    output(:, :, i) = data(:, :, i) * filter;
end
end
