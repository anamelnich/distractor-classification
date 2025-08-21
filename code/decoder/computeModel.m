% function [trainingData, bestItrData] = computeModel(subjectID)
function [performanceR, performanceL] = computeModel(subjectID)

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
        trigtype = 0;
        % trigtype = 3;
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
rightMask = trainingData.labels ~=2 ; % distractor right trials --> left side decoder
rightDdata.data = trainingData.data(:,:,rightMask);
rightDdata.labels = trainingData.labels(rightMask);
rightDdata.file_id = trainingData.file_id(rightMask) ; 
rightDdata.eof = trainingData.eof ; 

[~, bestItrDataR] = iterativePrune(rightDdata, cfg, 20)

leftMask = trainingData.labels ~=1 ; % distractor left trials --> right side decoder
leftDdata.data = trainingData.data(:,:,leftMask);
leftDdata.labels = trainingData.labels(leftMask);
leftDdata.labels(leftDdata.labels == 2) = 1;
leftDdata.file_id = trainingData.file_id(leftMask) ; 
leftDdata.eof = trainingData.eof ; 

[~, bestItrDataL] = iterativePrune(leftDdata, cfg, 20)


%% ================== Pruning tests ==================== %%

% iters = 1:nIter;
% plotPruningMetrics(iters, ACC, AUPRC, TPR, TNR, Ntr, subjectID);
% plotERPpruned(trainingData,bestItrData,cfg)
% plotTargetPosDistribution(trainingData, bestItrData, cfg)
% plotRunPruning(trainingData, bestItrData)

%% ===================== Build Model ======================== %%
nFiles = numel(trainingData.eof);
fprintf('Performing cross-validation...\n');
posteriorR  = nan(numel(trainingData.labels),1);
posteriorL  = nan(numel(trainingData.labels),1);
for fileIdx = 1:nFiles
    trainIdx = bestItrDataR.file_id ~= fileIdx;
    testIdx  = trainingData.file_id == fileIdx;
    % right-distractor classification
    [decoderR, ~] = computeDecoderRight(bestItrDataR.data(:,:,trainIdx), bestItrDataR.labels(trainIdx), cfg);
    posteriorR(testIdx) = singleClassificationRight(decoderR, trainingData.data(:,:,testIdx));
end
trainingData.posteriorsR = posteriorR;

for fileIdx = 1:nFiles
    trainIdx = bestItrDataL.file_id ~= fileIdx;
    testIdx  = trainingData.file_id == fileIdx;
    % left-distractor classification
    [decoderL, ~] = computeDecoderRight(bestItrDataL.data(:,:,trainIdx), bestItrDataL.labels(trainIdx), cfg);
    posteriorL(testIdx) = singleClassificationRight(decoderL, trainingData.data(:,:,testIdx));
end
trainingData.posteriorsL = posteriorL;

%% Performance evaluation %%

%% Right dsitractor
allLabels = trainingData.labels;
allLabels(trainingData.labels==2) = 0;
% Compute AUPRC
[~, ~, ~, aucRight] = perfcurve(allLabels, ...
    trainingData.posteriorsR, 1, 'Prior','uniform', 'xCrit','reca','yCrit','prec');

% Find optimal threshold over a limited range
range = linspace(0.35,0.65,61);
[x,y,t,~,opt] = perfcurve(allLabels, ...
    trainingData.posteriorsR, 1, 'Prior','uniform','TVals',range);
thresholdR = t(x==opt(1) & y==opt(2));

% Compute confusion metrics
[tprR, tnrR, accR] = printConfusionMatrix(allLabels, ...
    trainingData.posteriorsR >= thresholdR);
fprintf('AUPRC = %.3f, TPR = %.3f, TNR = %.3f, ACC = %.3f, thr = %.3f\n', ...
    aucRight, tprR, tnrR, accR, thresholdR);
performanceR.posteriors = posteriorR;
performanceR.labels = trainingData.labels;
performanceR.thr = thresholdR;
performanceR.tpr = tprR;
performanceR.tnr = tnrR;
performanceR.acc = accR;
performanceR.auprc = aucRight;

%% Left dsitractor
allLabels = trainingData.labels;
allLabels(trainingData.labels==1) = 0;
allLabels(trainingData.labels==2) = 1;
% Compute AUPRC
[~, ~, ~, aucLeft] = perfcurve(allLabels, ...
    trainingData.posteriorsL, 1, 'Prior','uniform', 'xCrit','reca','yCrit','prec');

% Find optimal threshold over a limited range
range = linspace(0.35,0.65,61);
[x,y,t,~,opt] = perfcurve(allLabels, ...
    trainingData.posteriorsL, 1, 'Prior','uniform','TVals',range);
thresholdL = t(x==opt(1) & y==opt(2));

% Compute confusion metrics
[tprL, tnrL, accL] = printConfusionMatrix(allLabels, ...
    trainingData.posteriorsL >= thresholdL);
fprintf('AUPRC = %.3f, TPR = %.3f, TNR = %.3f, ACC = %.3f, thr = %.3f\n', ...
    aucLeft, tprL, tnrL, accL, thresholdL);

performanceL.posteriors = posteriorL;
performanceL.labels = trainingData.labels;
performanceL.thr = thresholdL;
performanceL.tpr = tprL;
performanceL.tnr = tnrL;
performanceL.acc = accL;
performanceL.auprc = aucLeft;
%%
decoderR.eegChannels = cfg.eegChannels; 
decoderR.eogChannels = cfg.eogChannels;
decoderR.spectralFilter = cfg.spectralFilter;
decoderR.threshold = performanceR.thr;
decoderR.thresholdMargin = 0.1;
decoderR.performance = performanceR;
decoderR.subjectID = subjectID;
decoderR.onlinePosteriors = [];
decoderR.datetime = datetime;
disp(' ');
disp('DecoderR Updated at');
disp(decoderR.datetime);


decoderL.eegChannels = cfg.eegChannels; 
decoderL.eogChannels = cfg.eogChannels;
decoderL.spectralFilter = cfg.spectralFilter;
decoderL.threshold = performanceL.thr;
decoderL.thresholdMargin = 0.1;
decoderL.performance = performanceL;
decoderL.subjectID = subjectID;
decoderR.onlinePosteriors = [];
decoderL.datetime = datetime;
disp(' ');
disp('DecoderL Updated at');
disp(decoderL.datetime);

save(sprintf('./decoders/%s_decoderR.mat', subjectID), 'decoderR');
save('../cnbiLoop/decoderR.mat', 'decoderR');

save(sprintf('./decoders/%s_decoderL.mat', subjectID), 'decoderL');
save('../cnbiLoop/decoderL.mat', 'decoderL');

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

