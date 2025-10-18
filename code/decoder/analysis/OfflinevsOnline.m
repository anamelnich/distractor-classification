
% load data via computeModel

calibData = safeCombine(data, 'training1');
finalData   = safeCombine(data, 'training2');
on1Data   = safeCombine(data, 'decoding1');
on2Data   = safeCombine(data, 'decoding2');
on3Data   = safeCombine(data, 'decoding3');
on4Data   = safeCombine(data, 'decoding4');
on5Data   = safeCombine(data, 'decoding5');

%% %%%%%%%%% Evaluate online performance %%%%%%%%%%%%%

ND_decoder = "left";

names = {'calib','on1','on2','on3','on4','on5'};
datasets = {calibData, on1Data, on2Data, on3Data, on4Data, on5Data};

resultsR = struct();
resultsL = struct();

for k = 1:numel(names)
    dset = datasets{k};
    if isempty(dset) || ~isfield(dset,'data') || isempty(dset.data)
        warning('Dataset %s is empty, skipping.', names{k});
        continue;
    end
    [acc, tpr, tnr, auprc, posteriors, labels] = ...
        evaluateDecoder(dset, ND_decoder, decoderR, decoderR);

    % Save results in struct
    resultsR.(names{k}).acc        = acc;
    resultsR.(names{k}).tpr        = tpr;
    resultsR.(names{k}).tnr        = tnr;
    resultsR.(names{k}).auprc      = auprc;
    resultsR.(names{k}).posteriors = posteriors;
    resultsR.(names{k}).labels     = labels;
end

% pruned results
resultsR.pruned.acc        = decoderR.performance.acc;
resultsR.pruned.tpr        = decoderR.performance.tpr;
resultsR.pruned.tnr        = decoderR.performance.tnr;
resultsR.pruned.auprc      = decoderR.performance.auprc;
resultsR.pruned.posteriors = decoderR.performance.posteriors;
resultsR.pruned.labels     = decoderR.performance.labels;

for k = 1:numel(names)
    dset = datasets{k};
    if isempty(dset) || ~isfield(dset,'data') || isempty(dset.data)
        warning('Dataset %s is empty, skipping.', names{k});
        continue;
    end
    [acc, tpr, tnr, auprc, posteriors, labels] = ...
        evaluateDecoder(dset, ND_decoder, decoderL, decoderL);

    % Save results in struct
    resultsL.(names{k}).acc        = acc;
    resultsL.(names{k}).tpr        = tpr;
    resultsL.(names{k}).tnr        = tnr;
    resultsL.(names{k}).auprc      = auprc;
    resultsL.(names{k}).posteriors = posteriors;
    resultsL.(names{k}).labels     = labels;
end

resultsL.pruned.acc        = decoderL.performance.acc;
resultsL.pruned.tpr        = decoderL.performance.tpr;
resultsL.pruned.tnr        = decoderL.performance.tnr;
resultsL.pruned.auprc      = decoderL.performance.auprc;
resultsL.pruned.posteriors = decoderL.performance.posteriors;
resultsL.pruned.labels     = decoderL.performance.labels;

plotDecoderComparison(resultsL, resultsR);

%% %%%%%%%%%%%%%% Plot waveform over session %%%%%%%%%%%
bestItrDataL.labels(bestItrDataL.labels==1)=2;
bestItrDataR.eof = [];
bestItrDataL.eof = [];
prunedData = combineEpochs({bestItrDataL, bestItrDataR});
%%
panelNames = {'Calibration M1','Calibration M1'};
plotERPOffvsOnlineAllD(calibData, calibData, cfg, panelNames);
%% R and L distractor combined: avg of 7 electrodes
panelNames = {'Calibration','Pruned'};
plotERPOffvsOnlineAllD(calibData, prunedData, cfg, panelNames);
%%
panelNames = {'Online1','Online2'};
plotERPOffvsOnlineAllD(on1Data, on2Data, cfg, panelNames)
%%
panelNames = {'Online3','Online4'};
plotERPOffvsOnlineAllD(on3Data, on4Data, cfg, panelNames)
%%
panelNames = {'Online5','Online5'};
plotERPOffvsOnlineAllD(on5Data, on5Data, cfg, panelNames)
%%
panelNames = {'Pre BCI','Post BCI'};
plotERPOffvsOnlineAllD(calibData, finalData, cfg, panelNames)
%%
panelNames = {'Calibration','Online5'};
plotERPOffvsOnlineAllD(calibData, on5Data, cfg, panelNames)

%% R and L distractor combined: avg of 2 xDAWN components
panelNames = {'Calibration','Pruned'};
plotERPOffvsOnlineAllD_xDAWN(calibData, prunedData, cfg, decoderL, decoderR, panelNames)

panelNames = {'Online1','Online2'};
plotERPOffvsOnlineAllD_xDAWN(on1Data, on2Data, cfg, decoderL, decoderR, panelNames,1)

panelNames = {'Online3','Online4'};
plotERPOffvsOnlineAllD_xDAWN(on3Data, on4Data, cfg, decoderL, decoderR, panelNames,1)

panelNames = {'Online5','Online5'};
plotERPOffvsOnlineAllD_xDAWN(on5Data, on5Data, cfg, decoderL, decoderR, panelNames,1)
%%
panelNames = {'Pre BCI','Post BCI'};
plotERPOffvsOnlineAllD_xDAWN(calibData, finalData, cfg, decoderL, decoderR, panelNames,1)

%% R and L distractor separate: avg of 7 electrodes
panelNames = {'Calibration','Pruned'};
plotSidesFromOfflineOnline(calibData, prunedData, cfg, panelNames)

panelNames = {'Online1','Online2'};
plotSidesFromOfflineOnline(on1Data, on2Data, cfg, panelNames)

panelNames = {'Online3','Online4'};
plotSidesFromOfflineOnline(on3Data, on3Data, cfg, panelNames)

%% R and L distractor separate: avg of xDAWN comps
panelNames = {'Calibration','Pruned'};
plotSidesFromOfflineOnline(calibData, prunedData, cfg, panelNames, decoderL, decoderR)
%%
panelNames = {'Online1','Online2'};
plotSidesFromOfflineOnline(on1Data, on2Data, cfg, panelNames,decoderL, decoderR)
%%
panelNames = {'Online3','Online4'};
plotSidesFromOfflineOnline(on3Data, on4Data, cfg, panelNames,decoderL, decoderR)

%%
panelNames = {'Online5','Online5'};
plotSidesFromOfflineOnline(on5Data, on5Data, cfg, panelNames,decoderL, decoderR)

%% %%%%%%%%%%%%%% Plot waveform offline vs online %%%%%%%%%%%
offlineData = combineEpochs({data.training1.epochs});
rightMask = offlineData.labels ~=2 ; % distractor right trials --> left side decoder
rightDOfflinedata.data = offlineData.data(:,:,rightMask);
rightDOfflinedata.labels = offlineData.labels(rightMask);
leftMask = offlineData.labels ~=1 ; % distractor left trials --> right side decoder
leftDOfflinedata.data = offlineData.data(:,:,leftMask);
leftDOfflinedata.labels = offlineData.labels(leftMask);
leftDOfflinedata.labels(leftDOfflinedata.labels == 2) = 1;
%%
onlineData = combineEpochs({data.decoding1.epochs});
rightMask = onlineData.labels ~=2 ; % distractor right trials --> left side decoder
rightDonlinedata.data = onlineData.data(:,:,rightMask);
rightDonlinedata.labels = onlineData.labels(rightMask);
leftMask = onlineData.labels ~=1 ; % distractor left trials --> right side decoder
leftDonlinedata.data = onlineData.data(:,:,leftMask);
leftDonlinedata.labels = onlineData.labels(leftMask);
leftDonlinedata.labels(leftDonlinedata.labels == 2) = 1;
%%

plotERPOffvsOnline(rightDOfflinedata,rightDOfflinedata,cfg,"right")
plotERPOffvsOnline(leftDOfflinedata,leftDOfflinedata,cfg,"left")
%%
plotERPOffvsOnline(rightDOfflinedata,rightDonlinedata,cfg,"right")

plotERPOffvsOnline(leftDOfflinedata,leftDonlinedata,cfg,"left")



%% %%%%%%%%%%%% Compare offline and online posteriors
trainingData = combineEpochs({data.decoding1.epochs});
posteriors = nan(numel(trainingData.labels),1);

testIdxR = trainingData.labels == 1 ; 
posteriors(testIdxR) = singleClassificationRight(decoderR, trainingData.data(:,:,testIdxR));

testIdxL = trainingData.labels ~=1 ; 
posteriors(testIdxL) = singleClassificationRight(decoderL, trainingData.data(:,:,testIdxL));

online_posteriors = decoderR.onlinePosteriors';
% online_posteriors = online_posteriors(541:end);

testR = posteriors(trainingData.labels == 1) - online_posteriors(trainingData.labels == 1);
testL = posteriors(trainingData.labels == 2) - online_posteriors(trainingData.labels == 2);
testN = posteriors(trainingData.labels == 0) - online_posteriors(trainingData.labels == 0);

figure;

% Right trials
subplot(3,1,1);
histogram(testR, 'FaceColor', 'r', 'EdgeColor', 'none', ...
    'Normalization', 'probability','BinWidth', 0.0001);
xlabel('Posterior Difference (Offline - Online)');
ylabel('Proportion');
title('Right Trials');
xlim([-0.01 0.01])

% Left trials
subplot(3,1,2);
histogram(testL, 'FaceColor', 'b', 'EdgeColor', 'none', ...
    'Normalization', 'probability','BinWidth', 0.0001);
xlabel('Posterior Difference (Offline - Online)');
ylabel('Proportion');
title('Left Trials');
xlim([-0.01 0.01])

% None trials
subplot(3,1,3);
histogram(testN, 'FaceColor', 'g', 'EdgeColor', 'none', ...
    'Normalization', 'probability','BinWidth', 0.0001);
xlabel('Posterior Difference (Offline - Online)');
ylabel('Proportion');
title('None Trials');
xlim([-0.01 0.01])
%% ================= Helper Function ================= %%
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

function out = safeCombine(data, topField)
% out = [] unless data.(topField).epochs exists and is nonempty
    out = [];
    if isfield(data, topField) && isfield(data.(topField), 'epochs') ...
            && ~isempty(data.(topField).epochs)
        out = combineEpochs({data.(topField).epochs});
    end
end

function [acc, tpr, tnr, auprc, posteriors, labels] = ...
    evaluateDecoder(calibData, ND_decoder, decoderL, decoderR)
% evaluateDecoder evaluates left/right decoders on calibration data
%
% Inputs:
%   calibData - struct with fields:
%                  .data   [time x channels x trials]
%                  .labels [trials x 1] (1=right, 2=left)
%   ND_decoder - string, "left" or "right"
%   decoderL   - trained decoder for left trials
%   decoderR   - trained decoder for right trials
%
% Outputs:
%   acc        - overall accuracy
%   tpr        - true positive rate
%   tnr        - true negative rate
%   posteriors - posterior probabilities [trials x 1]
%   labels     - binarized labels [trials x 1], with left=1, right=0

    % init posteriors
    posteriors = nan(numel(calibData.labels), 1);

    % assign based on ND_decoder
    if strcmp(ND_decoder, "left")
        % true right trials
        testIdxR = calibData.labels == 1;
        posteriors(testIdxR) = singleClassificationRight( ...
            decoderR, calibData.data(:,:,testIdxR));
        % non-right (so left)
        testIdxL = calibData.labels ~= 1;
        posteriors(testIdxL) = singleClassificationRight( ...
            decoderL, calibData.data(:,:,testIdxL));

    elseif strcmp(ND_decoder, "right")
        % true left trials
        testIdxL = calibData.labels == 2;
        posteriors(testIdxL) = singleClassificationRight( ...
            decoderL, calibData.data(:,:,testIdxL));
        % non-left (so right)
        testIdxR = calibData.labels ~= 2;
        posteriors(testIdxR) = singleClassificationRight( ...
            decoderR, calibData.data(:,:,testIdxR));

    else
        error('ND_decoder must be "left" or "right"');
    end

    % thresholding
    threshold = 0.5;
    labels = calibData.labels;
    labels(labels == 2) = 1;  % collapse left/right → binary (1 vs 0)

    % performance curve (optional, returns auprc if needed)
    [~, ~, ~, auprc] = perfcurve(labels, posteriors, 1, ...
        'Prior','uniform', 'xCrit','reca','yCrit','prec');

    % confusion-matrix-based metrics
    [tpr, tnr, acc] = printConfusionMatrix(labels, posteriors >= threshold);

    % fprintf('AUPRC = %.3f, TPR = %.3f, TNR = %.3f, ACC = %.3f\n', ...
    %     auprc, tpr, tnr, acc);

end

function plotDecoderComparison(resultsL, resultsR)
% plotDecoderComparison
%   Plots Accuracy, AUPRC, TNR, and TPR across sessions.
%   Left (resultsL) and Right (resultsR) shown on same plots.

    sessions = {'calib','pruned','on1','on2','on3','on4','on5'};
    metrics  = {'acc','auprc','tnr','tpr'};
    ylabels  = {'Accuracy','AUPRC','True Negative Rate','True Positive Rate'};
    titles   = {'Accuracy','AUPRC','TNR','TPR'};
    colors   = lines(2); % two distinct colors (left/right)

    for m = 1:numel(metrics)
        % collect Left values
        yL = nan(1, numel(sessions));
        for s = 1:numel(sessions)
            if isfield(resultsL, sessions{s}) && isfield(resultsL.(sessions{s}), metrics{m})
                yL(s) = resultsL.(sessions{s}).(metrics{m});
            end
        end
        % collect Right values
        yR = nan(1, numel(sessions));
        for s = 1:numel(sessions)
            if isfield(resultsR, sessions{s}) && isfield(resultsR.(sessions{s}), metrics{m})
                yR(s) = resultsR.(sessions{s}).(metrics{m});
            end
        end

        % make plot
        figure('Color','w');
        plot(1:numel(sessions), yL, '-o', 'LineWidth', 1.8, ...
             'MarkerSize', 7, 'Color', colors(1,:));
        hold on;
        plot(1:numel(sessions), yR, '-s', 'LineWidth', 1.8, ...
             'MarkerSize', 7, 'Color', colors(2,:));
        grid on;
        xlim([1 numel(sessions)]);
        xticks(1:numel(sessions));
        xticklabels(sessions);
        ylim([0.35 0.7]); % metrics are rates
        xlabel('Session');
        ylabel(ylabels{m});
        title(sprintf('%s over sessions', titles{m}));
        legend({'Left','Right'}, 'Location','best');
    end
end

function plotSidesFromOfflineOnline(offData, onData, params, panelNames, decoder_for_rightD, decoder_for_leftD)
% plotSidesFromOfflineOnline
% Build side-specific datasets and call plotERPOffvsOnline for:
%   1) RIGHT-side distractor subset   (labels ~= 2)
%   2) LEFT-side  distractor subset   (labels ~= 1, with 2 -> 1 remap)
%
% Inputs:
%   offData    - struct with .data [t x ch x n], .labels [n x 1 or 1 x n]
%   onData     - same fields as offData
%   params     - plotting params passed through to plotERPOffvsOnline
%   panelNames - 1x2 cellstr for subplot titles inside plotERPOffvsOnline
%                (e.g., {'Offline','Online'}). If omitted/empty, defaults used.

    if nargin < 4 || isempty(panelNames)
        panelNames = {'Offline','Online'};
    end

    % Ensure column-vector labels
    offLabels = offData.labels(:);
    onLabels  = onData.labels(:);

    % ---------------- RIGHT-side subset ----------------
    % Keep trials that are NOT left-distractor (i.e., include ND and Right)
    rightMaskOff = (offLabels ~= 2);
    rightMaskOn  = (onLabels  ~= 2);

    rightOff.data   = offData.data(:,:, rightMaskOff);
    rightOff.labels = offLabels(rightMaskOff);

    rightOn.data    = onData.data(:,:,  rightMaskOn);
    rightOn.labels  = onLabels(rightMaskOn);

    % Plot if both have data
    if ~isempty(rightOff.data) && ~isempty(rightOn.data)
%         plotERPOffvsOnline(rightOff, rightOn, params, 'right', panelNames);
        plotERPOffvsOnline_xDAWN(rightOff, rightOn, params, 'right', decoder_for_rightD, panelNames);
        try
            sgtitle('Right-side subset'); % optional, requires R2018b+
        catch, end
    else
        warning('Right-side subset is empty in offline or online; skipping right plot.');
    end

    % ---------------- LEFT-side subset -----------------
    % Keep trials that are NOT right-distractor (i.e., include ND and Left)
    leftMaskOff = (offLabels ~= 1);
    leftMaskOn  = (onLabels  ~= 1);

    leftOff.data   = offData.data(:,:, leftMaskOff);
    leftOff.labels = offLabels(leftMaskOff);
    % Remap 2 -> 1 so left subset is binary {0,1}
    leftOff.labels(leftOff.labels == 2) = 1;

    leftOn.data    = onData.data(:,:,  leftMaskOn);
    leftOn.labels  = onLabels(leftMaskOn);
    leftOn.labels(leftOn.labels == 2) = 1;

    % Plot if both have data
    if ~isempty(leftOff.data) && ~isempty(leftOn.data)
        plotERPOffvsOnline_xDAWN(rightOff, rightOn, params, 'right', decoder_for_leftD, panelNames);
        try
            sgtitle('Left-side subset');
        catch, end
    else
        warning('Left-side subset is empty in offline or online; skipping left plot.');
    end
end

