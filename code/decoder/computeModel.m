% function [trainingData, bestItrData] = computeModel(subjectID)
function [performanceR, performanceL] = computeModel(subjectID)

%% ====================== Initialization ====================== %%
clearvars -except subjectID cfg;
close all; rng('default');
addpath(genpath('../functions'));

%% Load Data 

dataPath = [pwd '/../../data/'];
data = loadData(dataPath, subjectID);
delete sopen.mat

%% Set Parameters andPreprocess
cfg = setParams(data.training1.header);
cfg.fsamp = data.training1.header.SampleRate;  

cfg.eegChannels = 1:64; 
cfg.eogChannels = 65:66;
cfg.triggerChannel = 67;

cfg.chanLabels = data.training1.header.Label;
cfg.chanLabels(65:67)=[];
 

fields = fieldnames(data);
for i = 1:numel(fields)
    fname = fields{i};
    if isempty(data.(fname))
        data = rmfield(data, fname);
        continue;
    end

    trigtype = 2;
    data.(fname) = preprocessDataset(data.(fname), cfg, fname, trigtype);
    
end

fields = fieldnames(data);

%% Remove Non-EEG Channels 
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
        data.(fname).epochs.RT = d.beh.RT;
    end
    if isfield(d, 'beh') && isfield(d.beh, 'tpos')
        data.(fname).epochs.RT = d.beh.RT;
    end
end

%% ================== Classification Setup ==================== %%
nIter=1;
% trainingData = combineEpochs({data.training1.epochs, data.decoding3.epochs});
trainingData = combineEpochs({data.training1.epochs});
rightMask = trainingData.labels ~=2 ; % distractor right trials --> left side decoder
rightDdata.data = trainingData.data(:,:,rightMask);
rightDdata.labels = trainingData.labels(rightMask);
rightDdata.file_id = trainingData.file_id(rightMask) ; 
rightDdata.eof = trainingData.eof ; 

[performanceR, bestItrDataR] = iterativePrune(rightDdata, cfg, nIter);

leftMask = trainingData.labels ~=1 ; % distractor left trials --> right side decoder
leftDdata.data = trainingData.data(:,:,leftMask);
leftDdata.labels = trainingData.labels(leftMask);
leftDdata.labels(leftDdata.labels == 2) = 1;
leftDdata.file_id = trainingData.file_id(leftMask) ; 
leftDdata.eof = trainingData.eof ; 

[performanceL, bestItrDataL] = iterativePrune(leftDdata, cfg, nIter);


%% ================== Pruning tests ==================== %%

iters = 1:nIter;
historyR = performanceR.history;
historyL = performanceL.history;

outDir = './figures';
ts = datestr(now,'yyyymmdd_HHMM_SS');
base = fullfile(outDir, sprintf('%s_%s', subjectID, ts));

% -------- 1) Pruning results: RIGHT distractor --------
plotPruningMetrics(iters, historyR.ACC, historyR.AUPRC, historyR.TPR, historyR.TNR, historyR.nTrials, subjectID);
h1 = gcf; set(h1,'PaperPositionMode','auto');
print(h1, [base '_rightDistractor_pruning.pdf'], '-dpdf','-painters');

% -------- 2) ERP: RIGHT distractor --------
plotERPpruned(rightDdata, bestItrDataR, cfg,"right");
h2 = gcf; set(h2,'PaperPositionMode','auto');
print(h2, [base '_rightDistractor_ERP.pdf'], '-dpdf','-painters');

% -------- 3) Pruning results: LEFT distractor --------
plotPruningMetrics(iters, historyL.ACC, historyL.AUPRC, historyL.TPR, historyL.TNR, historyL.nTrials, subjectID);
h3 = gcf; set(h3,'PaperPositionMode','auto');
print(h3, [base '_leftDistractor_pruning.pdf'], '-dpdf','-painters');

% -------- 4) ERP: LEFT distractor --------
plotERPpruned(leftDdata, bestItrDataL, cfg,"left");
h4 = gcf; set(h4,'PaperPositionMode','auto');
print(h4, [base '_leftDistractor_ERP.pdf'], '-dpdf','-painters');


%% ===================== Build Model ======================== %%
nFiles = numel(trainingData.eof);
fprintf('Performing cross-validation...\n');
posteriorR  = nan(numel(rightDdata.labels),1);
posteriorL  = nan(numel(leftDdata.labels),1);
for fileIdx = 1:nFiles
    trainIdx = bestItrDataR.file_id ~= fileIdx;
    testIdx  = rightDdata.file_id == fileIdx;
    % right-distractor classification
    [decoderR, ~] = computeDecoderRight(bestItrDataR.data(:,:,trainIdx), bestItrDataR.labels(trainIdx), cfg);
    posteriorR(testIdx) = singleClassificationRight(decoderR, rightDdata.data(:,:,testIdx));
end
trainingData.posteriorsR = posteriorR;

for fileIdx = 1:nFiles
    trainIdx = bestItrDataL.file_id ~= fileIdx;
    testIdx  = leftDdata.file_id == fileIdx;
    % left-distractor classification
    [decoderL, ~] = computeDecoderRight(bestItrDataL.data(:,:,trainIdx), bestItrDataL.labels(trainIdx), cfg);
    posteriorL(testIdx) = singleClassificationRight(decoderL, leftDdata.data(:,:,testIdx));
end
trainingData.posteriorsL = posteriorL;

%% Performance evaluation %%

%% Right dsitractor
allLabels = rightDdata.labels;
% Compute AUPRC
[~, ~, ~, aucRight] = perfcurve(allLabels, ...
    trainingData.posteriorsR, 1, 'Prior','uniform', 'xCrit','reca','yCrit','prec');

% Find optimal threshold over a limited range
range = linspace(0.1,0.9,161);
[x,y,t,~,opt] = perfcurve(allLabels, ...
    trainingData.posteriorsR, 1, 'Prior','uniform','TVals',range);
thresholdR = findThreshold(y,x,t);

% Compute confusion metrics
disp(' ');
disp('RIGHT-DISTRACTOR CLASSIFICATION PERFORMANCE (deocderR)')
[tprR, tnrR, accR] = printConfusionMatrix(allLabels, ...
    trainingData.posteriorsR >= thresholdR);
fprintf('AUPRC = %.3f, TPR = %.3f, TNR = %.3f, ACC = %.3f, thr = %.3f\n', ...
    aucRight, tprR, tnrR, accR, thresholdR);
performanceR.posteriors = posteriorR;
performanceR.labels = rightDdata.labels;
performanceR.thr = thresholdR;
performanceR.tpr = tprR;
performanceR.tnr = tnrR;
performanceR.acc = accR;
performanceR.auprc = aucRight;

%% Left dsitractor
allLabels = leftDdata.labels;
% Compute AUPRC
[~, ~, ~, aucLeft] = perfcurve(allLabels, ...
    trainingData.posteriorsL, 1, 'Prior','uniform', 'xCrit','reca','yCrit','prec');

% Find optimal threshold over a limited range
range = linspace(0.2,0.8,121);
[x,y,t,~,opt] = perfcurve(allLabels, ...
    trainingData.posteriorsL, 1, 'Prior','uniform','TVals',range);
thresholdL = findThreshold(y,x,t);

% Compute confusion metrics
disp(' ');
disp('LEFT-DISTRACTOR CLASSIFICATION PERFORMANCE (deocderL)') 
[tprL, tnrL, accL] = printConfusionMatrix(allLabels, ...
    trainingData.posteriorsL >= thresholdL);
fprintf('AUPRC = %.3f, TPR = %.3f, TNR = %.3f, ACC = %.3f, thr = %.3f\n', ...
    aucLeft, tprL, tnrL, accL, thresholdL);

performanceL.posteriors = posteriorL;
performanceL.labels = leftDdata.labels;
performanceL.thr = thresholdL;
performanceL.tpr = tprL;
performanceL.tnr = tnrL;
performanceL.acc = accL;
performanceL.auprc = aucLeft;

%% Train final decoders
[decoderR, ~] = computeDecoderRight(bestItrDataR.data, bestItrDataR.labels, cfg);    
[decoderL, ~] = computeDecoderRight(bestItrDataL.data, bestItrDataL.labels, cfg);
   
%%
decoderR.eegChannels = cfg.eegChannels; 
decoderR.eogChannels = cfg.eogChannels;
decoderR.spectralFilter = cfg.spectralFilter;
decoderR.threshold = 0.2;
decoderR.thresholdMargin = 0.1;
decoderR.performance = performanceR;
decoderR.subjectID = subjectID;
decoderR.onlinePosteriors = [];
decoderR.datetime = datetime;
decoderR.params = cfg;
disp(' ');
disp('DecoderR Updated at');
disp(decoderR.datetime);


decoderL.eegChannels = cfg.eegChannels; 
decoderL.eogChannels = cfg.eogChannels;
decoderL.spectralFilter = cfg.spectralFilter;
decoderL.threshold = 0.2;
decoderL.thresholdMargin = 0.1;
decoderL.performance = performanceL;
decoderL.subjectID = subjectID;
decoderL.onlinePosteriors = [];
decoderL.datetime = datetime;
decoderL.params = cfg;
disp(' ');
disp('DecoderL Updated at');
disp(decoderL.datetime);

if decoderR.performance.tnr > decoderL.performance.tnr
    decoderN = decoderR;
else
    decoderN = decoderL;
end
decoderN.threshold = 0.8;
%%
save(sprintf('./decoders/%s_decoderR.mat', subjectID), 'decoderR');
% save('../cnbiLoop/decoderR.mat', 'decoderR');

save(sprintf('./decoders/%s_decoderL.mat', subjectID), 'decoderL');
% save('../cnbiLoop/decoderL.mat', 'decoderL');

save(sprintf('./decoders/%s_decoderN.mat', subjectID), 'decoderN');
% save('../cnbiLoop/decoderN.mat', 'decoderN');

%% make threshold logging struct
thrLog = struct( ...
    'subjectID',      subjectID, ...
    'timestamp',      datestr(now,'yyyy-mm-dd HH:MM:SS'), ...
    'margin'   ,      decoderR.thresholdMargin, ...
    'thrR'     ,      decoderR.threshold, ...
    'thrL'     ,      decoderL.threshold, ...
    'thrN'     ,      decoderN.threshold ...
);
log_path = sprintf('../cnbiLoop/online_info/%s_thrlog.mat',subjectID);
save(log_path, 'thrLog');

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

