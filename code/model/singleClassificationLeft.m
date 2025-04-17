function [posterior, epoch] = singleClassificationLeft(decoder, eeg)
% singleClassificationLeft applies a trained decoder to classify EEG data
% from a single trial using left vs. right posterior electrode features.
%
% Inputs:
%   decoder        - Struct containing trained model and feature parameters
%   eeg            - [time x channels x trials] EEG data
%   leftElectrodes - Indices of left hemisphere electrodes
%   rightElectrodes - Indices of right hemisphere electrodes
%
% Outputs:
%   posterior - Classifier output (probability or score)
%   epoch     - Feature vector used for classification

%% ---------------- Baseline Correction ---------------- %%
baselineStart = decoder.epochOnset - round(0.2 * decoder.fsamp);
baseline = mean(eeg(baselineStart:decoder.epochOnset, :, :), 1);
eeg = eeg - baseline;

%% --------- ROI Extraction & Difference Wave ---------- %%
erpEpochs  = eeg(:, decoder.leftElectrodeIndices, :);
diffEpochs = eeg(:, decoder.leftElectrodeIndices, :) - eeg(:, decoder.rightElectrodeIndices, :);

%% ---------------- Feature Processing ---------------- %%
if decoder.features.erp_iscompute
    ERP_feats = processFeatures(erpEpochs, decoder, decoder.spatialFilter.erp);
else
    ERP_feats = [];
end

if decoder.features.diffwave_iscompute
    Diff_feats = processFeatures(diffEpochs, decoder, decoder.spatialFilter.diff);
else
    Diff_feats = [];
end

% Combine features
if ~isempty(ERP_feats) && ~isempty(Diff_feats)
    epoch = [ERP_feats; Diff_feats];
elseif ~isempty(ERP_feats)
    epoch = ERP_feats;
elseif ~isempty(Diff_feats)
    epoch = Diff_feats;
else
    error('No features selected in decoder.features.');
end

%% ----------- Apply Dimensionality Reduction ----------- %%
if isequal(decoder.classify.reduction.type, 'pca')
    epoch = decoder.classify.applyPCA(epoch)';
end

if decoder.classify.is_normalize
    epoch = decoder.classify.funNormalize(epoch);
end

if ismember(decoder.classify.reduction.type, {'lasso', 'r2'})
    epoch = epoch(decoder.classify.keepIdx, :);
end

%% ------------------- Classification ------------------- %%
posterior = decoder.classify.model(epoch');

end

%% ===================================================== %%
%% ================= Helper Functions ================== %%
%% ===================================================== %%

function features = processFeatures(eeg, decoder, filterMatrix)
% Applies spatial filtering and resampling to extract features

[nSamples, ~, nTrials] = size(eeg);
filtered = nan(nSamples, size(filterMatrix, 2), nTrials);

for i = 1:nTrials
    filtered(:, :, i) = eeg(:, :, i) * filterMatrix;
end

% Resample and reshape
if decoder.resample.is_compute
    resamp = filtered(decoder.resample.time(1:decoder.resample.ratio:end), :, :);
    features = reshape(resamp, [], nTrials);
else
    features = [];
end

end
