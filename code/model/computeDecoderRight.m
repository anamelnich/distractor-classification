function [decoder, classifierEpochs] = computeDecoderRight(trainEpochs, trainLabels, params)
% computeDecoderLeft trains a classifier to detect left-distractor trials 
% using lateralized EEG features from left vs. right posterior electrodes.
%
% Output:
%   decoder           - trained model structure with metadata
%   classifierEpochs  - feature matrix used to train the model

%% ==================== Baseline Correction ==================== %%
if params.baseline_iscompute
    baseline_window = params.baseline_window;
    baseline_idx = find(params.epochTime >= baseline_window(1) & params.epochTime <= baseline_window(2));
    baseline = mean(trainEpochs(baseline_idx, :, :), 1);
    trainEpochs = trainEpochs - baseline;
end 

%% ==================== ROI Selection ==================== %%

if isequal(params.roi, 'P/PO')

    LeftElectrodes  = {'P1', 'P3', 'P5', 'P7', 'PO3', 'PO5', 'PO7'};
    RightElectrodes = {'P2', 'P4', 'P6', 'P8', 'PO4', 'PO6', 'PO8'};
    leftIdx  = find(ismember(params.chanLabels, LeftElectrodes));
    rightIdx = find(ismember(params.chanLabels, RightElectrodes));
    
    erpEpochs   = trainEpochs(:, rightIdx, :);
    diffEpochs  = trainEpochs(:, rightIdx, :) - trainEpochs(:, leftIdx, :);
elseif isequal(params.roi, 'None')
    erpEpochs   = trainEpochs(:, :, :);
end

% Distractor class: distractor left (label 2)
% No distractor class: distractor right, no distractor (labels 1 or 0)
trainLabels(trainLabels == 1) = 0;
trainLabels(trainLabels == 2) = 1;

%% ==================== Power Spectral Density ==================== %%
if (params.psd.is_compute)
    midfrontElectrodes = {'AF3','AFZ','AF4','F5','F3','F1','FZ','F2',...
    'F4','F6','FC3','FC1','FCZ','FC2','FC4'};
    midfrontIdx = find(ismember(params.chanLabels, midfrontElectrodes));
    psdEpochs = trainEpochs(:, midfrontIdx, :);
    [psds, params] = compute_stockwell(psdEpochs, params);
    psds = squeeze(mean(abs(psds).^2, 1));
    psds = psds(params.resample.time(1:params.resample.ratio:end),:,:);
    psds = reshape(psds, [size(psds,1)*size(psds,2) size(psds,3)]);
    
    if any(isnan(psds))
        logicalIdx  = not(isnan(psds(:,1)));
        psds = psds(logicalIdx,:);
    end
end

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

% TFR wave features
if params.psd.is_compute
    tfr_feats = psds;
else
    tfr_feats = [];
end


classifierEpochs = cat(1, ERP_feats, Diff_feats, tfr_feats);

if isempty(classifierEpochs)
    error('No features selected. Set at least one of the params.features flags to true.');
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
decoder.roi = params.roi;
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
if isequal(params.roi, 'P/PO')
    decoder.leftElectrodeIndices = leftIdx;
    decoder.rightElectrodeIndices = rightIdx;
end 
decoder.psd = params.psd;
if params.psd.is_compute
    decoder.midfrontIdx = midfrontIdx;
end
decoder.baseline_iscompute = params.baseline_iscompute;
if params.baseline_iscompute
    decoder.baseline_idx = baseline_idx;
end 
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
