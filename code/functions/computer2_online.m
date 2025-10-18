function totalr2 = computer2_online(decoder, eeg,labels)

%% ---------------- Baseline Correction ---------------- %%
if decoder.baseline_iscompute
    baseline = mean(eeg(decoder.baseline_idx, :, :), 1);
    eeg = eeg - baseline; % e.g. online 721 x 64 so eeg ~ 1.4 sec with baseline
end 
%% --------- ROI Extraction & Difference Wave ---------- %%

    diffEpochs = eeg(:, decoder.rightElectrodeIndices, :) - eeg(:, decoder.leftElectrodeIndices, :); % online 721x7
   
%% ---------------- Feature Processing ---------------- %%


    epoch = processFeatures(diffEpochs, decoder, decoder.spatialFilter.diff,...
        decoder.classify.applyPCA.diff);


%% ----------- Apply Dimensionality Reduction ----------- %%

    epoch = decoder.classify.funNormalize(epoch);
    
power = compute_r2(permute(epoch, [1 3 2]), labels);
totalr2 = sum(power(decoder.classify.keepIdx));


end

%% ===================================================== %%
%% ================= Helper Functions ================== %%
%% ===================================================== %%

function features = processFeatures(eeg, decoder, filterMatrix,applyPCA)
% Applies spatial filtering and resampling to extract features

[nSamples, ~, nTrials] = size(eeg); % if no 3rd dimension, set to 1
if isequal(decoder.spatialFilter.type, 'None')
    filtered = eeg;
else
    filtered = nan(nSamples, size(filterMatrix, 2), nTrials); %768 x 2 x n trials
    
    for i = 1:nTrials
        filtered(:, :, i) = eeg(:, :, i) * filterMatrix; % 768 x 2 x n of trials
    end
end

% Resample
if decoder.resample.is_compute
    resamp = filtered(decoder.resample.time(1:decoder.resample.ratio:end), :, :);
    features = reshape(resamp, [], nTrials);
elseif decoder.statsfeatures.is_compute
    filtered = filtered(decoder.resample.time, :, :); % 179 x 2 x 480
    avg = mean(filtered,1); % 1 x 2 comp x 480 trials
    variance = var(filtered,0,1); % 1 x 2 comp x 480 trials
    [peak_amp,peak_latency] = max(filtered,[],1); % 1 x 2 comp x 480 trials
    pos_area = sum(max(filtered, 0), 1); % 1 x 2 comp x 480 trials
    features = cat(1, avg, variance, peak_amp, peak_latency, pos_area); % 5 feat x 2 comp x 480 trial
    features = reshape(features, [], size(features, 3)); % 10 x 480
else
    features = [];
end
if isequal(decoder.classify.reduction.type, 'pca')
    features = applyPCA(features)';
end

end