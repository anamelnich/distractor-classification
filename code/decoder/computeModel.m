% function [trainingData, bestItrData] = computeModel(subjectID)
function computeModel(subjectID)

%% ====================== Initialization ====================== %%
clearvars -except subjectID cfg;
close all; rng('default');
addpath(genpath('../functions'));

%% ======================== Load Data ========================= %%

dataPath = [pwd '/../../data/'];
data = loadData(dataPath, subjectID);
delete sopen.mat

%% ============== Set Params and Preprocess Data ============== %%
cfg = setParams(data.training1.header);
cfg.fsamp = data.training1.header.SampleRate;  

subjNum = sscanf(subjectID, 'e%d');
%%
if subjNum <= 8   
    cfg.eegChannels = 1:64; 
    cfg.eogChannels = 65:68;
    cfg.triggerChannel = 69;
    
    cfg.chanLabels = data.training1.header.Label;
    cfg.chanLabels(65:69)=[];
else 
    cfg.eegChannels = 1:64; 
    cfg.eogChannels = 65:66;
    cfg.triggerChannel = 67;
    
    cfg.chanLabels = data.training1.header.Label;
    cfg.chanLabels(65:67)=[];
end 
%%
fields = fieldnames(data);
for i = 1:numel(fields)
    fname = fields{i};
    if isempty(data.(fname))
        data = rmfield(data, fname);
        continue;
    end
    if subjNum <= 7
        % trigtype = 0;
        trigtype = 3;
    elseif subjNum <=15
        % trigtype = 1;
        trigtype = 4;
    else 
        trigtype = 2;
    end
    data.(fname) = preprocessDataset(data.(fname), cfg, fname, trigtype);
    % trigType = 3 left distractor only on subjects 1-8
    % trigType =  2 for new left distractor only sibjects
end

fields = fieldnames(data);

%% =============== Remove Non-EEG Channels ==================== %%
% chanRemove = {'M1','M2','EOG','FP1','FP2','FPZ'};
% removeIdx = find(ismember(cfg.chanLabels, chanRemove));
% cfg.chanLabels(removeIdx) = [];
% 
% for i = 1:numel(fields)
%     fname = fields{i};
%     data.(fname).data(:, removeIdx) = [];
% end

%% ==================== Bandpass Filter ======================= %%
[b, a] = butter(cfg.spectralFilter.order, cfg.spectralFilter.freqs./(cfg.fsamp/2), 'bandpass');
cfg.spectralFilter.b = b;
cfg.spectralFilter.a = a;

for i = 1:numel(fields)
    fname = fields{i};
    data.(fname).data = filter(b, a, data.(fname).data);
end

%% ======================== Epoching ========================== %%
for i = 1:numel(fields)
    
    fname = fields{i};
    d = data.(fname);
    
    epochs.data = nan(length(cfg.epochSamples), length(cfg.chanLabels), length(d.index.pos));
    epochs.labels = d.index.typ;
    epochs.file_id = nan(length(d.index.typ), 1);

    for t = 1:length(d.index.pos)
        epochs.data(:, :, t) = d.data(d.index.pos(t) + cfg.epochSamples, :);
        epochs.file_id(t) = find(d.index.pos(t) <= d.eof, 1, 'first');
    end
    
    data.(fname).epochs = epochs;
    data.(fname).epochs.eof = d.eof;
    if isfield(d, 'beh') && isfield(d.beh, 'RT')
        % data.(fname).epochs.RT = d.beh.RT;
        try
            data.(fname).epochs.RT = d.beh.RT(d.beh.dpos~=2);
        catch
            nTrials = size(data.(fname).epochs.data, 3);
            data.(fname).epochs.RT = nan(nTrials, 1);
        end
    end
    if isfield(d, 'beh') && isfield(d.beh, 'tpos')
        % data.(fname).epochs.RT = d.beh.RT;
        try
            data.(fname).epochs.tpos = d.beh.tpos(d.beh.dpos~=2);
            data.(fname).epochs.dpos = d.beh.dpos(d.beh.dpos~=2);
        catch
            nTrials = size(data.(fname).epochs.data, 3);
            data.(fname).epochs.tpos = nan(nTrials, 1);
            data.(fname).epochs.dpos = nan(nTrials, 1);
        end
    end
end

%% ================== Classification Setup ==================== %%
trainingData = combineEpochs({data.training1.epochs});
nFiles = length(trainingData.eof);
trainingData.posteriors = nan(length(trainingData.labels), 1);
master = true(numel(trainingData.labels),1);
nIter  = 20;
TPR   = nan(nIter,1);
TNR   = nan(nIter,1);
ACC   = nan(nIter,1);
AUPRC = nan(nIter,1);
thr   = nan(nIter,1);
Ntr   = nan(nIter,1);
maskMatrix = nan(numel(trainingData.labels),nIter);

for i=1:nIter
    fprintf('--- Iteration %d ---\n', i);

  prunedData.data    = trainingData.data(:,:,master);
  prunedData.labels  = trainingData.labels(master);
  prunedData.file_id = trainingData.file_id(master);

    if cfg.balance_iscompute
        mask = balanceRuns(prunedData); % left distractor only 
        balancedData.data   = prunedData.data(:,:,mask);
        balancedData.labels = prunedData.labels(mask);
        balancedData.file_id = prunedData.file_id(mask);
    end

% =================== Cross-Validation ======================= %%
    disp('Performing cross-validation')
    for iFile = 1:nFiles
        trainIdx = balancedData.file_id ~= iFile;
        testIdx  = trainingData.file_id == iFile;
        % Right decoder: classify left distractor
        [decoderRight, featuresR] = computeDecoderRight(balancedData.data(:, :, trainIdx), balancedData.labels(trainIdx), cfg);
        trainingData.posteriors(testIdx) = singleClassificationRight(decoderRight, ...
            trainingData.data(:, :, testIdx));
    end

% ============ Evaluate Right Distractor Decoder ============ %%

    [~, ~, ~, aucRight, ~] = perfcurve(trainingData.labels, ...
        trainingData.posteriors, 1, 'Prior', 'uniform','xCrit', 'reca','yCrit', 'prec');
    
    range = linspace(0.35, 0.65, 61); %no extreme thresholds
    [x, y, t, ~, opt] = perfcurve(trainingData.labels, ...
        trainingData.posteriors, 1, 'Prior', 'uniform','TVals', range);
    threshold = t(x == opt(1) & y == opt(2));
    % threshold = 0.5;
    [tpr,tnr,acc]=printConfusionMatrix(trainingData.labels, trainingData.posteriors >= threshold);
    fprintf('AUPRC: %.2f\n', aucRight);

      TPR(i)   = tpr;
      TNR(i)   = tnr;
      ACC(i)   = acc;
      AUPRC(i) = aucRight;
      thr(i)   = threshold;
      Ntr(i)   = sum(master);
      maskMatrix(:,i)=master;

    pruneMask = pruneTrialsMask(trainingData.labels, trainingData.posteriors, 0.5, 0.05,master);
    master = master & pruneMask;
    

end

[~, bestItr] = max(AUPRC);
bestMask = logical(maskMatrix(:,bestItr));
bestItrData = struct();
bestItrData.data = trainingData.data(:,:,bestMask);
bestItrData.labels = trainingData.labels(bestMask);
bestItrData.file_id = trainingData.file_id(bestMask);
bestItrData.RT = trainingData.RT(bestMask);
bestItrData.posteriors = trainingData.posteriors(bestMask);
bestItrData.tpos = trainingData.tpos(bestMask);


performance.auprc = AUPRC(bestItr);
performance.threshold = thr(bestItr);
performance.accuracy = ACC(bestItr);
performance.tpr = TPR(bestItr);
performance.tnr = TNR(bestItr);
performance.posteriors = bestItrData.posteriors;
performance.labels = bestItrData.labels;
performance.file_id = bestItrData.file_id;


%% ================== Pruning tests ==================== %%

iters = 1:nIter;
plotPruningMetrics(iters, ACC, AUPRC, TPR, TNR, Ntr, subjectID);
plotERPpruned(trainingData,bestItrData,cfg)
plotTargetPosDistribution(trainingData, bestItrData, cfg)
plotRunPruning(trainingData, bestItrData)

%% ===================== Build Model ======================== %%

[decoder, ~] = computeDecoderRight(bestItrData.data, bestItrData.labels, cfg);
posteriors = singleClassificationRight(decoder, bestItrData.data);
disp('Overall Model Performance')
[tpr,tnr,acc]=printConfusionMatrix(bestItrData.labels, posteriors >= performance.threshold);

%%
decoder.eegChannels = cfg.eegChannels; 
decoder.eogChannels = cfg.eogChannels;
decoder.spectralFilter = cfg.spectralFilter;
decoder.threshold = performance.threshold;
decoder.thresholdMargin = 0.1;
decoder.onlinePosteriors = [];
% decoder.chantoremove = removeIdx;

% decoder.resample.time = decoder.resample.time - decoder.epochOnset;

decoder.performance = performance;
decoder.subjectID = subjectID;
decoder.datetime = datetime;
disp(' ');
disp('Decoder Updated at');
disp(decoder.datetime);

save(sprintf('./decoders/%s_decoder.mat', subjectID), 'decoder');
save('../cnbiLoop/decoder.mat', 'decoder');

%% ================== Riemannian Classifier ==================== %%
% trainingData = combineEpochs({data.training1.epochs});
% nFiles = length(trainingData.eof);
% trainingData.posteriors = nan(length(trainingData.labels), 2);
% trainingData.pred_class = nan(length(trainingData.labels), 1);
% 
% if cfg.balance_iscompute
%     mask = balanceRuns(trainingData); % left distractor only 
%     balancedData.data   = trainingData.data(:,:,mask);
%     balancedData.labels = trainingData.labels(mask);
%     balancedData.file_id = trainingData.file_id(mask);
% end
% 
% disp('Performing cross-validation')
% for iFile = 1:nFiles
%     trainIdx = balancedData.file_id ~= iFile;
%     testIdx  = trainingData.file_id == iFile;
%     decoderRieman = computeDecoderRieman(balancedData.data(:, :, trainIdx), balancedData.labels(trainIdx), cfg);
%     [Ytest, post] = singleClassificationRieman(decoderRieman, ...
%         trainingData.data(:, :, testIdx));
%     trainingData.pred_class(testIdx)     = Ytest;
%     trainingData.posteriors(testIdx, :) = post;
% end
% %
% isCorrect = trainingData.pred_class == trainingData.labels;
% accuracy = mean(isCorrect);
% fprintf('Accuracy = %.2f%%\n', accuracy*100);
% TP = sum( trainingData.pred_class==1 & trainingData.labels==1 );
% FN = sum( trainingData.pred_class==0 & trainingData.labels==1 );
% TN = sum( trainingData.pred_class==0 & trainingData.labels==0 );
% FP = sum( trainingData.pred_class==1 & trainingData.labels==0 );
% TPR = TP / (TP + FN);   % true positive rate (recall/sensitivity)
% TNR = TN / (TN + FP);   % true negative rate (specificity)
% 
% fprintf('TPR (sensitivity) = %.2f%%\n', TPR*100);
% fprintf('TNR (specificity) = %.2f%%\n', TNR*100);

end
% 
% ================= Helper Function ================= %%
function [tpr,tnr,acc] = printConfusionMatrix(trueLabels, predictedLabels)
cm = confusionmat(logical(trueLabels), predictedLabels);
disp('Confusion Matrix (with labels):');
disp('--------------------------------');
disp('            Pred=0    Pred=1');
fprintf('True=0:       %3d       %3d\n', cm(1,1), cm(1,2));
fprintf('True=1:       %3d       %3d\n', cm(2,1), cm(2,2));
tnr = cm(1,1) / sum(cm(1,:));
tpr = cm(2,2) / sum(cm(2,:));
acc = sum(diag(cm)) / sum(cm(:));
fprintf('TNR: %.2f | TPR: %.2f | Accuracy: %.2f\n\n', tnr, tpr, acc);
end

