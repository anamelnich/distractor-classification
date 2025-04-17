function computeModel(subjectID)
% computeModel runs preprocessing and classification on EEG data for a given subject.
% It trains and evaluates left and right distractor decoders using cross-validation.

%% ====================== Initialization ====================== %%
clearvars -except subjectID;
close all; clc; rng('default');
addpath(genpath('../functions'));

%% ======================== Load Data ========================= %%
dataPath = [pwd '/../../data/'];
data = loadData(dataPath, subjectID);
delete sopen.mat

%% ============== Set Params and Preprocess Data ============== %%
cfg = setParams(data.training1.header);

fields = fieldnames(data);
for i = 1:numel(fields)
    fname = fields{i};
    if isempty(data.(fname))
        data = rmfield(data, fname);
        continue;
    end
    data.(fname) = preprocessDataset(data.(fname), cfg, fname, 1);
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
end

%% ================== Classification Setup ==================== %%
trainingData = combineEpochs({data.training1.epochs});
nFiles = length(trainingData.eof);
trainingData.posteriors.Left  = nan(length(trainingData.labels), 1);
trainingData.posteriors.Right = nan(length(trainingData.labels), 1);

%% =================== Cross-Validation ======================= %%
for iFile = 1:nFiles
    trainIdx = trainingData.file_id ~= iFile;
    testIdx  = trainingData.file_id == iFile;

    % Left decoder: classify right distractor
    decoderLeft = computeDecoderLeft(trainingData.data(:, :, trainIdx), trainingData.labels(trainIdx), cfg);
    trainingData.posteriors.Left(testIdx) = singleClassificationLeft(decoderLeft, trainingData.data(:, :, testIdx));

    % Right decoder: classify left distractor
    decoderRight = computeDecoderRight(trainingData.data(:, :, trainIdx), trainingData.labels(trainIdx), cfg);
    trainingData.posteriors.Right(testIdx) = singleClassificationRight(decoderRight, trainingData.data(:, :, testIdx));
end

%% ============ Evaluate Right Distractor Decoder ============ %%
trainingData.labelsforRightDecoder = trainingData.labels;
trainingData.labelsforRightDecoder(trainingData.labels == 2) = 1;
trainingData.labelsforRightDecoder(trainingData.labels == 1) = 0;

[x, y, t, aucRight, opt] = perfcurve(~trainingData.labelsforRightDecoder, ...
    1 - trainingData.posteriors.Right, 1, 'Prior', 'uniform');
threshold = t(x == opt(1) & y == opt(2));

fprintf('\n[Right Decoder] AUC: %.2f | Threshold: %.2f\n', aucRight, threshold);
printConfusionMatrix(trainingData.labelsforRightDecoder, trainingData.posteriors.Right >= threshold);

%% ============ Evaluate Left Distractor Decoder ============= %%
trainingData.labelsforLeftDecoder = trainingData.labels;
trainingData.labelsforLeftDecoder(trainingData.labels == 2) = 0;

[x, y, t, aucLeft, opt] = perfcurve(~trainingData.labelsforLeftDecoder, ...
    1 - trainingData.posteriors.Left, 1, 'Prior', 'uniform');
threshold = t(x == opt(1) & y == opt(2));

fprintf('\n[Left Decoder] AUC: %.2f | Threshold: %.2f\n', aucLeft, threshold);
printConfusionMatrix(trainingData.labelsforLeftDecoder, trainingData.posteriors.Left >= threshold);

%% ===================== Sanity Check ======================== %%
[decoderLeft, modelOutLeft]   = computeDecoderLeft(trainingData.data, trainingData.labels, cfg);
[trainingData.posteriors.Left, classOutLeft] = singleClassificationLeft(decoderLeft, trainingData.data);

if isequal(modelOutLeft, classOutLeft)
    disp('Left-side preprocessing consistent.');
else
    disp('Left-side preprocessing inconsistent.');
end
% printConfusionMatrix(trainingData.labelsforLeftDecoder, trainingData.posteriors.Left >= threshold);

[decoderRight, modelOutRight] = computeDecoderRight(trainingData.data, trainingData.labels, cfg);
[trainingData.posteriors.Right, classOutRight] = singleClassificationRight(decoderRight, trainingData.data);

if isequal(modelOutRight, classOutRight)
    disp('Right-side preprocessing consistent.');
else
    disp('Right-side preprocessing inconsistent.');
end
% printConfusionMatrix(trainingData.labelsforRightDecoder, trainingData.posteriors.Right >= threshold);

end

%% ================= Helper Function ================= %%
function printConfusionMatrix(trueLabels, predictedLabels)
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

