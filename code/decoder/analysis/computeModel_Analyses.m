function performance = computeModel(subjectID)
% function [performance, modulation1, modulation2] = computeModel(subjectID)
% function [corr,peak,RT] = computeModel(subjectID,cfg)
% function [meanD, meanND, rtD_mean, rtND_mean] = computeModel(subjectID)
% computeModel runs preprocessing and classification on EEG data for a given subject.
% It trains and evaluates left and right distractor decoders using cross-validation.

%% ====================== Initialization ====================== %%
clearvars -except subjectID cfg;
% close all; rng('default');
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
    else
        trigtype = 1;
    end
    data.(fname) = preprocessDataset(data.(fname), cfg, fname, trigtype);
end

fields = fieldnames(data);

%% =============== Remove Non-EEG Channels ==================== %%
chanRemove = {'M1','M2','EOG','FP1','FP2','FPZ'};
removeIdx = find(ismember(cfg.chanLabels, chanRemove));
cfg.chanLabels(removeIdx) = [];

for i = 1:numel(fields)
    fname = fields{i};
    data.(fname).data(:, removeIdx) = [];
end

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
end

%% ================== Classification Setup ==================== %%
trainingData = combineEpochs({data.training1.epochs});
% trainingData = combineEpochs({data.decoding2.epochs});
% testingData1 = combineEpochs({data.decoding1.epochs});
% testingData2 = combineEpochs({data.decoding2.epochs});

if isfield(trainingData, 'RT')
    RT = trainingData.RT;
else
    RT = [];
end

[meanD, meanND, rtD_mean, rtND_mean] = computeDiffWave(trainingData.data,...
    trainingData.labels, RT, cfg);
% 
% % [corr, ~,peak,RT] = corrAUCvsRT(trainingData,cfg);
% 
%% 
nFiles = length(trainingData.eof);
trainingData_unbalanced = trainingData;

if cfg.balance_iscompute
    trainingData = balanceRuns(trainingData); %equal number of trials per class
    % testingData1 = balanceRuns(testingData1);
    % testingData2 = balanceRuns(testingData2);
end

% trainingData.posteriors.Bilateral = nan(length(trainingData.labels), 1);
% trainingData.posteriors.Left  = nan(length(trainingData.labels), 1);
% trainingData.posteriors.Right = nan(length(trainingData.labels), 1);
trainingData.posteriors.Bilateral = nan(length(trainingData_unbalanced.labels), 1);
trainingData.posteriors.Left  = nan(length(trainingData_unbalanced.labels), 1);
trainingData.posteriors.Right = nan(length(trainingData_unbalanced.labels), 1);

%% =================== Cross-Validation ======================= %%
disp('Performing cross-validation')
for iFile = 1:nFiles
    trainIdx = trainingData.file_id ~= iFile;
    % testIdx  = trainingData.file_id == iFile;
    testIdx  = trainingData_unbalanced.file_id == iFile;
    fprintf('Fold %d\n',iFile)
    % Bilateral decoder
    % [decoder,featuresB] = computeDecoder(trainingData.data(:, :, trainIdx), trainingData.labels(trainIdx), cfg);
    % trainingData.posteriors.Bilateral(testIdx) = singleClassification(decoder, ...
    %     trainingData.data(:, :, testIdx), trainingData.labels(testIdx));
    % trainingData.posteriors.Bilateral(testIdx) = singleClassification(decoder, ...
    %     trainingData_unbalanced.data(:, :, testIdx), trainingData_unbalanced.labels(testIdx));

    % Left decoder: classify right distractor
    [decoderLeft, featuresL] = computeDecoderLeft(trainingData.data(:, :, trainIdx), trainingData.labels(trainIdx), cfg);
    % trainingData.posteriors.Left(testIdx) = singleClassificationLeft(decoderLeft, ...
    %     trainingData.data(:, :, testIdx), trainingData.labels(testIdx));
    trainingData.posteriors.Left(testIdx) = singleClassificationLeft(decoderLeft, ...
        trainingData_unbalanced.data(:, :, testIdx));

    % Right decoder: classify left distractor
    [decoderRight, featuresR] = computeDecoderRight(trainingData.data(:, :, trainIdx), trainingData.labels(trainIdx), cfg);
    % trainingData.posteriors.Right(testIdx) = singleClassificationRight(decoderRight, ...
    %     trainingData.data(:, :, testIdx),trainingData.labels(testIdx));
    trainingData.posteriors.Right(testIdx) = singleClassificationRight(decoderRight, ...
        trainingData_unbalanced.data(:, :, testIdx));
end
%% ============ Evaluate Bilateral Decoder ============ %%
% 
% trainingData.labelsforBilateralDecoder = trainingData.labels;
% trainingData.labelsforBilateralDecoder(trainingData.labels == 2) = 1;
% 
% [~, ~, ~, aucBilateral, ~] = perfcurve(trainingData.labelsforBilateralDecoder, ...
%     trainingData.posteriors.Bilateral, 1, 'Prior', 'uniform','xCrit', 'reca','yCrit', 'prec');
% 
% range = linspace(0.2, 0.8, 61); %no extreme thresholds
% [x, y, t, ~, opt] = perfcurve(trainingData.labelsforBilateralDecoder, ...
%     trainingData.posteriors.Bilateral, 1, 'Prior', 'uniform','TVals', range);
% threshold = t(x == opt(1) & y == opt(2));
% [kappa, chance] = calckappa(trainingData.labelsforBilateralDecoder, trainingData.posteriors.Bilateral >= threshold);
% % fprintf('\n[Bilateral Decoder] AUPRC: %.2f | Threshold: %.2f\n | Kappa: %.2f\n', aucBilateral, threshold, kappa);
% [tpr,tnr,acc] = printConfusionMatrix(trainingData.labelsforBilateralDecoder, trainingData.posteriors.Bilateral >= threshold);
% performance.bilateral.auprc = aucBilateral;
% performance.bilateral.threshold = threshold;
% performance.bilateral.kappa = kappa;
% performance.bilateral.accuracy = acc;
% performance.bilateral.tpr = tpr;
% performance.bilateral.tnr = tnr;
% 
%% ============ Evaluate Right Distractor Decoder ============ %%
notNan = ~isnan(trainingData.posteriors.Right);
trainingData.posteriors.Right = trainingData.posteriors.Right(notNan);

trainingData.labelsforRightDecoder = trainingData_unbalanced.labels;
trainingData.labelsforRightDecoder(trainingData_unbalanced.labels == 2) = 1;
trainingData.labelsforRightDecoder(trainingData_unbalanced.labels == 1) = 0;
trainingData.labelsforRightDecoder = trainingData.labelsforRightDecoder(notNan);

[~, ~, ~, aucRight, ~] = perfcurve(trainingData.labelsforRightDecoder, ...
    trainingData.posteriors.Right, 1, 'Prior', 'uniform','xCrit', 'reca','yCrit', 'prec');

range = linspace(0.2, 0.8, 61); %no extreme thresholds
[x, y, t, ~, opt] = perfcurve(trainingData.labelsforRightDecoder, ...
    trainingData.posteriors.Right, 1, 'Prior', 'uniform','TVals', range);
threshold = t(x == opt(1) & y == opt(2));
% [kappa, chance] = calckappa(trainingData.labelsforRightDecoder, trainingData.posteriors.Right >= threshold);
% fprintf('\n[Right Decoder] AUPRC: %.2f | Threshold: %.2f\n | Kappa: %.2f\n', aucRight, threshold, kappa);
%%
[tpr,tnr,acc]=printConfusionMatrix(trainingData.labelsforRightDecoder, trainingData.posteriors.Right >= threshold);
performance.right.auprc = aucRight;
performance.right.threshold = threshold;
% performance.right.kappa = kappa;
performance.right.accuracy = acc;
performance.right.tpr = tpr;
performance.right.tnr = tnr;
% performance.right.chance = chance;
performance.right.posteriors = trainingData.posteriors.Right;
performance.right.labels = trainingData.labelsforRightDecoder;

%% ============ Evaluate Left Distractor Decoder ============= %%
notNan = ~isnan(trainingData.posteriors.Left);
trainingData.posteriors.Left = trainingData.posteriors.Left(notNan);
trainingData.labelsforLeftDecoder = trainingData_unbalanced.labels;
trainingData.labelsforLeftDecoder(trainingData_unbalanced.labels == 2) = 0;
trainingData.labelsforLeftDecoder = trainingData.labelsforLeftDecoder(notNan);

[~, ~, ~, aucLeft, ~] = perfcurve(trainingData.labelsforLeftDecoder, ...
    trainingData.posteriors.Left, 1, 'Prior', 'uniform','xCrit', 'reca','yCrit', 'prec');

range = linspace(0.2, 0.8, 61); %no extreme thresholds
[x, y, t, ~, opt] = perfcurve(trainingData.labelsforLeftDecoder, ...
    trainingData.posteriors.Left, 1, 'Prior', 'uniform','TVals', range);
threshold = t(x == opt(1) & y == opt(2));
% [kappa, chance] = calckappa(trainingData.labelsforLeftDecoder, trainingData.posteriors.Left >= threshold);
% fprintf('\n[Left Decoder] AUPRC: %.2f | Threshold: %.2f\n | Kappa: %.2f\n', aucLeft, threshold,kappa);
[tpr,tnr,acc]=printConfusionMatrix(trainingData.labelsforLeftDecoder, trainingData.posteriors.Left >= threshold);
performance.left.auprc = aucLeft;
performance.left.threshold = threshold;
% performance.left.kappa = kappa;
performance.left.accuracy = acc;
performance.left.tpr = tpr;
performance.left.tnr = tnr;
% performance.left.chance = chance;
performance.left.posteriors = trainingData.posteriors.Left;
performance.left.labels = trainingData.labelsforLeftDecoder;

performance.tpos = data.training1.beh.tpos;
performance.dpos = data.training1.beh.dpos;


% %% ============ Evaluate Right Online Decoder ============= %%
% [decoderRight, modelOutRight]   = computeDecoderRight(trainingData.data, trainingData.labels, cfg);
% 
% [testingData1.posteriors.Right, classOutRight1] = singleClassificationRight(decoderRight, ...
%     testingData1.data,testingData1.labels);
% 
% notNan = ~isnan(testingData1.posteriors.Right);
% testingData1.posteriors.Right = testingData1.posteriors.Right(notNan);
% testingData1.labelsforRightDecoder = testingData1.labels;
% testingData1.labelsforRightDecoder(testingData1.labels == 2) = 1;
% testingData1.labelsforRightDecoder(testingData1.labels == 1) = 0;
% testingData1.labelsforRightDecoder = testingData1.labelsforRightDecoder(notNan);
% 
% [~, ~, ~, aucRight, ~] = perfcurve(testingData1.labelsforRightDecoder, ...
%     testingData1.posteriors.Right, 1, 'Prior', 'uniform','xCrit', 'reca','yCrit', 'prec');
% [tpr,tnr,acc]=printConfusionMatrix(testingData1.labelsforRightDecoder, testingData1.posteriors.Right >= ...
%     performance.right.threshold);
% [kappa, chance] = calckappa(testingData1.labelsforRightDecoder, testingData1.posteriors.Right >= ...
%     performance.right.threshold);
% modulation1.right.auprc = aucRight;
% modulation1.right.threshold = performance.right.threshold;
% modulation1.right.kappa = kappa;
% modulation1.right.accuracy = acc;
% modulation1.right.tpr = tpr;
% modulation1.right.tnr = tnr;
% modulation1.right.chance = chance;
% modulation1.right.posteriors = testingData1.posteriors.Right;
% modulation1.right.labels = testingData1.labelsforRightDecoder;
% modulation1.right.fileid = testingData1.file_id(notNan);
% 
% % Modulation 2
% trainingData2 = combineEpochs({data.training1.epochs,data.decoding1.epochs});
% [decoderRight, modelOutRight]   = computeDecoderRight(trainingData2.data, trainingData2.labels, cfg);
% [testingData2.posteriors.Right, classOutRight2] = singleClassificationRight(decoderRight, ...
%     testingData2.data,testingData2.labels);
% 
% notNan = ~isnan(testingData2.posteriors.Right);
% testingData2.posteriors.Right = testingData2.posteriors.Right(notNan);
% testingData2.labelsforRightDecoder = testingData2.labels;
% testingData2.labelsforRightDecoder(testingData2.labels == 2) = 1;
% testingData2.labelsforRightDecoder(testingData2.labels == 1) = 0;
% testingData2.labelsforRightDecoder = testingData2.labelsforRightDecoder(notNan);
% 
% [~, ~, ~, aucRight, ~] = perfcurve(testingData2.labelsforRightDecoder, ...
%     testingData2.posteriors.Right, 1, 'Prior', 'uniform','xCrit', 'reca','yCrit', 'prec');
% 
% [tpr,tnr,acc]=printConfusionMatrix(testingData2.labelsforRightDecoder, testingData2.posteriors.Right >= ...
%     performance.right.threshold);
% [kappa, chance] = calckappa(testingData2.labelsforRightDecoder, testingData2.posteriors.Right >= ...
%     performance.right.threshold);
% modulation2.right.auprc = aucRight;
% modulation2.right.threshold = performance.right.threshold;
% modulation2.right.kappa = kappa;
% modulation2.right.accuracy = acc;
% modulation2.right.tpr = tpr;
% modulation2.right.tnr = tnr;
% modulation2.right.chance = chance;
% modulation2.right.posteriors = testingData2.posteriors.Right;
% modulation2.right.labels = testingData2.labelsforRightDecoder;
% modulation2.right.fileid = testingData2.file_id(notNan);
% 
% %% ============ Evaluate Left Online Decoder ============= %%
% [decoderLeft, modelOutLeft]   = computeDecoderLeft(trainingData.data, trainingData.labels, cfg);
% 
% [testingData1.posteriors.Left, classOutLeft1] = singleClassificationLeft(decoderLeft, ...
%     testingData1.data,testingData1.labels);
% 
% notNan = ~isnan(testingData1.posteriors.Left);
% testingData1.posteriors.Left = testingData1.posteriors.Left(notNan);
% testingData1.labelsforLeftDecoder = testingData1.labels;
% testingData1.labelsforLeftDecoder(testingData1.labels == 2) = 0;
% testingData1.labelsforLeftDecoder = testingData1.labelsforLeftDecoder(notNan);
% 
% [~, ~, ~, aucLeft, ~] = perfcurve(testingData1.labelsforLeftDecoder, ...
%     testingData1.posteriors.Left, 1, 'Prior', 'uniform','xCrit', 'reca','yCrit', 'prec');
% [tpr,tnr,acc]=printConfusionMatrix(testingData1.labelsforLeftDecoder, testingData1.posteriors.Left >= ...
%     performance.left.threshold);
% [kappa, chance] = calckappa(testingData1.labelsforLeftDecoder, testingData1.posteriors.Left >= ...
%     performance.left.threshold);
% modulation1.left.auprc = aucLeft;
% modulation1.left.threshold = performance.left.threshold;
% modulation1.left.kappa = kappa;
% modulation1.left.accuracy = acc;
% modulation1.left.tpr = tpr;
% modulation1.left.tnr = tnr;
% modulation1.left.chance = chance;
% modulation1.left.posteriors = testingData1.posteriors.Left;
% modulation1.left.labels = testingData1.labelsforLeftDecoder;
% modulation1.left.fileid = testingData1.file_id(notNan);
% 
% % Modulation 2
% trainingData2 = combineEpochs({data.training1.epochs,data.decoding1.epochs});
% [decoderLeft, modelOutLeft]   = computeDecoderLeft(trainingData2.data, trainingData2.labels, cfg);
% [testingData2.posteriors.Left, classOutLeft2] = singleClassificationLeft(decoderLeft, ...
%     testingData2.data,testingData2.labels);
% 
% notNan = ~isnan(testingData2.posteriors.Left);
% testingData2.posteriors.Left = testingData2.posteriors.Left(notNan);
% testingData2.labelsforLeftDecoder = testingData2.labels;
% testingData2.labelsforLeftDecoder(testingData2.labels == 2) = 0;
% testingData2.labelsforLeftDecoder = testingData2.labelsforLeftDecoder(notNan);
% 
% [~, ~, ~, aucLeft, ~] = perfcurve(testingData2.labelsforLeftDecoder, ...
%     testingData2.posteriors.Left, 1, 'Prior', 'uniform','xCrit', 'reca','yCrit', 'prec');
% 
% [tpr,tnr,acc]=printConfusionMatrix(testingData2.labelsforLeftDecoder, testingData2.posteriors.Left >= ...
%     performance.left.threshold);
% [kappa, chance] = calckappa(testingData2.labelsforLeftDecoder, testingData2.posteriors.Left >= ...
%     performance.left.threshold);
% modulation2.left.auprc = aucLeft;
% modulation2.left.threshold = performance.left.threshold;
% modulation2.left.kappa = kappa;
% modulation2.left.accuracy = acc;
% modulation2.left.tpr = tpr;
% modulation2.left.tnr = tnr;
% modulation2.left.chance = chance;
% modulation2.left.posteriors = testingData2.posteriors.Left;
% modulation2.left.labels = testingData2.labelsforLeftDecoder;
% modulation2.left.fileid = testingData2.file_id(notNan);
% 
% 
% 
% % %% ===================== Sanity Check ======================== %%
% % [decoder, modelOutBilateral]   = computeDecoder(trainingData.data, trainingData.labels, cfg);
% % [trainingData.posteriors.Bilateral, classOutBilateral] = singleClassification(decoder, trainingData.data);
% % if cfg.fisher_iscompute
% %     performance.bilateral.fisherchan = decoder.fisher;
% % end
% % if isequal(modelOutBilateral, classOutBilateral)
% %     disp('Bilateral preprocessing consistent.');
% % else
% %     disp('Bilateral preprocessing inconsistent.');
% % end
% % % % % printConfusionMatrix(trainingData.labelsforBilateralDecoder, trainingData.posteriors.Bilateral >= threshold);
% % % % 
% % [decoderLeft, modelOutLeft]   = computeDecoderLeft(trainingData.data, trainingData.labels, cfg);
% % [trainingData.posteriors.Left, classOutLeft] = singleClassificationLeft(decoderLeft, ...
% %     trainingData.data,trainingData.labels);
% % if cfg.fisher_iscompute
% %     performance.left.fisherchan = decoderLeft.fisher;
% % end
% % if isequal(modelOutLeft, classOutLeft)
% %     disp('Left-side preprocessing consistent.');
% % else
% %     disp('Left-side preprocessing inconsistent.');
% % end
% % % % % printConfusionMatrix(trainingData.labelsforLeftDecoder, trainingData.posteriors.Left >= threshold);
% % % % 
% % [decoderRight, modelOutRight] = computeDecoderRight(trainingData.data, trainingData.labels, cfg);
% % [trainingData.posteriors.Right, classOutRight] = singleClassificationRight(decoderRight,...
% %     trainingData.data, trainingData.labels);
% % if cfg.fisher_iscompute
% %     performance.right.fisherchan = decoderRight.fisher;
% % end
% % if isequal(modelOutRight, classOutRight)
% %     disp('Right-side preprocessing consistent.');
% % else
% %     disp('Right-side preprocessing inconsistent.');
% % end
% % % % printConfusionMatrix(trainingData.labelsforRightDecoder, trainingData.posteriors.Right >= threshold);
% 
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

