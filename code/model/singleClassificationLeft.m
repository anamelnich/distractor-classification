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
if decoder.baseline_iscompute
    baseline = mean(eeg(decoder.baseline_idx, :, :), 1);
    eeg = eeg - baseline;
end 
%% --------- ROI Extraction & Difference Wave ---------- %%
if isequal(decoder.roi, 'P/PO')
    erpEpochs  = eeg(:, decoder.leftElectrodeIndices, :);
    diffEpochs = eeg(:, decoder.leftElectrodeIndices, :) - eeg(:, decoder.rightElectrodeIndices, :);
elseif isequal(decoder.roi, 'None')
    erpEpochs  = eeg(:, :, :);
end

%% --------- Power Spectral Density ---------- %%
if (decoder.psd.is_compute)
    psd_epoch = eeg(:, decoder.midfrontIdx,:);
    [psd, decoder] = compute_stockwell(psd_epoch,decoder);
    psd = squeeze(mean(abs(psd).^2, 1));
    psd = psd(decoder.resample.time(1:decoder.resample.ratio:end),:,:);
    [~, ~, n_trials] = size(psd);
    psd = reshape(psd, [size(psd,1)*size(psd,2) n_trials]);
end

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

if decoder.psd.is_compute
    tfr_feats = psd;
else 
    tfr_feats = [];
end

epoch = cat(1, ERP_feats, Diff_feats, tfr_feats);

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
