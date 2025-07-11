% subjects = {'e1','e2','e3','e4','e5','e6','e8','e13','e14','e15'};
subjects = {'e1','e2','e3','e4','e5','e6','e8','e11','e10','e12','e13','e14','e15'};
% subjects = {'e1','e2','e3','e4','e5','e6','e13'};
nSub     = numel(subjects);

% Preallocate a struct array with the right fields
% results = struct( ...
%     'subject',      cell(nSub,1), ...
%     'performance',  cell(nSub,1), ...
%     'modulation1',  cell(nSub,1), ...
%     'modulation2',  cell(nSub,1)  ...
% );
results = struct( ...
    'subject',      cell(nSub,1), ...
    'performance',  cell(nSub,1) ...
);

% Loop through and fill it
for i = 1:nSub
    subj = subjects{i};
    results(i).subject = subj;
    try
        % [perf, mod1, mod2] = computeModel(subj);
        perf = computeModel(subj);
        results(i).performance = perf;
        % results(i).modulation1 = mod1;
        % results(i).modulation2 = mod2;
    catch ME
        warning('Failed on %s: %s', subj, ME.message);
        % leave those cells empty if it crashes
    end
end

% Save everything in one .mat
save('decodingResults0623.mat','results');

%% Plot AUPRC and Accuracy
% Parameters
% Load results
S       = load('decodingResultsTestTrain.mat','results');
results = S.results;
nSub    = numel(results);
subjectLabels = {results.subject};

% Prompt user to select which subjects to plot
[sel, ok] = listdlg( ...
    'ListString',    subjectLabels, ...
    'SelectionMode', 'multiple', ...
    'Name',          'Select Subjects', ...
    'PromptString',  'Choose subjects to plot:', ...
    'ListSize',      [200 300] ...
);
if ~ok || isempty(sel)
    sel = 1:nSub;  % if user cancels or picks none, plot all
end

% Only two sessions: Mod1 & Mod2
sessionNames = {'Mod1','Mod2'};
nSess        = numel(sessionNames);

% Preallocate metric matrices (subjects × sessions)
auprc_left  = nan(nSub, nSess);
auprc_right = nan(nSub, nSess);
acc_left    = nan(nSub, nSess);
acc_right   = nan(nSub, nSess);

% Extract modulation metrics
for i = 1:nSub
    m1 = results(i).modulation1;
    m2 = results(i).modulation2;
    
    auprc_left(i,:)  = [ m1.left.auprc,  m2.left.auprc ];
    auprc_right(i,:) = [ m1.right.auprc, m2.right.auprc ];
    
    acc_left(i,:)    = [ m1.left.accuracy,  m2.left.accuracy ];
    acc_right(i,:)   = [ m1.right.accuracy, m2.right.accuracy ];
end

% Filter to selected subjects
selLabels   = subjectLabels(sel);
auprcL_sel  = auprc_left(sel, :);
auprcR_sel  = auprc_right(sel, :);
accL_sel    = acc_left(sel,   :);
accR_sel    = acc_right(sel,  :);

% Generate colors & markers for selected subjects
nPlot   = numel(sel);
colors  = lines(nPlot);
markers = {'o','s','d','^','v','>','<','p','h','x'};
markers = markers(1:nPlot);

% Plot
figure('Position',[100 100 900 650])
tiledlayout(2,2,'Padding','compact','TileSpacing','compact')

% 1) AUPRC — Left decoder
nexttile; hold on
for k = 1:nPlot
    plot(1:nSess, auprcL_sel(k,:), '-o', ...
         'Color', colors(k,:), ...
         'Marker', markers{k}, ...
         'MarkerFaceColor', colors(k,:), ...
         'LineWidth', 1.2);
end
plot(1:nSess, mean(auprcL_sel,1), '-k', 'LineWidth', 2.5);
set(gca,'XTick',1:nSess,'XTickLabel',sessionNames)
xlabel('Session'), ylabel('AUPRC')
title('Left Decoder — AUPRC')
legend([selLabels, {'Mean'}],'Location','eastoutside','FontSize',8)
hold off

% 2) AUPRC — Right decoder
nexttile; hold on
for k = 1:nPlot
    plot(1:nSess, auprcR_sel(k,:), '-o', ...
         'Color', colors(k,:), ...
         'Marker', markers{k}, ...
         'MarkerFaceColor', colors(k,:), ...
         'LineWidth', 1.2);
end
plot(1:nSess, mean(auprcR_sel,1), '-k', 'LineWidth', 2.5);
set(gca,'XTick',1:nSess,'XTickLabel',sessionNames)
xlabel('Session'), title('Right Decoder — AUPRC')
legend([selLabels, {'Mean'}],'Location','eastoutside','FontSize',8)
hold off

% 3) Accuracy — Left decoder
nexttile; hold on
for k = 1:nPlot
    plot(1:nSess, accL_sel(k,:), '-o', ...
         'Color', colors(k,:), ...
         'Marker', markers{k}, ...
         'MarkerFaceColor', colors(k,:), ...
         'LineWidth', 1.2);
end
plot(1:nSess, mean(accL_sel,1), '-k', 'LineWidth', 2.5);
set(gca,'XTick',1:nSess,'XTickLabel',sessionNames)
xlabel('Session'), ylabel('Accuracy')
title('Left Decoder — Accuracy')
legend([selLabels, {'Mean'}],'Location','eastoutside','FontSize',8)
hold off

% 4) Accuracy — Right decoder
nexttile; hold on
for k = 1:nPlot
    plot(1:nSess, accR_sel(k,:), '-o', ...
         'Color', colors(k,:), ...
         'Marker', markers{k}, ...
         'MarkerFaceColor', colors(k,:), ...
         'LineWidth', 1.2);
end
plot(1:nSess, mean(accR_sel,1), '-k', 'LineWidth', 2.5);
set(gca,'XTick',1:nSess,'XTickLabel',sessionNames)
xlabel('Session'), title('Right Decoder — Accuracy')
legend([selLabels, {'Mean'}],'Location','eastoutside','FontSize',8)
hold off

sgtitle('Decoder AUPRC & Accuracy — Selected Subjects','FontWeight','Bold')



%% Plot TPR & TNR
% Load results
S            = load('decodingResultsRetrainedTestTrain.mat','results');
results      = S.results;
nSub         = numel(results);
subjectLabels = {results.subject};

% Prompt user to select subjects
[sel, ok] = listdlg( ...
    'ListString',    subjectLabels, ...
    'SelectionMode', 'multiple', ...
    'Name',          'Select Subjects', ...
    'PromptString',  'Choose subjects to plot:', ...
    'ListSize',      [200 300] ...
);
if ~ok || isempty(sel)
    sel = 1:nSub;
end

% Sessions
sessionNames = {'Online 1','Online 2'};
nSess        = numel(sessionNames);

% Preallocate TPR/TNR matrices (subjects × sessions)
tpr_left   = nan(nSub, nSess);
tpr_right  = nan(nSub, nSess);
tnr_left   = nan(nSub, nSess);
tnr_right  = nan(nSub, nSess);

% Extract TPR & TNR
for i = 1:nSub
    m1 = results(i).modulation1;
    m2 = results(i).modulation2;
    
    % assume fields .tpr and .tnr exist under left/right
    tpr_left(i,:)  = [ m1.left.tpr,  m2.left.tpr  ];
    tpr_right(i,:) = [ m1.right.tpr, m2.right.tpr ];
    tnr_left(i,:)  = [ m1.left.tnr,  m2.left.tnr  ];
    tnr_right(i,:) = [ m1.right.tnr, m2.right.tnr ];
end

% Filter to selected subjects and convert to percentages
tprL = tpr_left(sel,:)*100;
tprR = tpr_right(sel,:)*100;
tnrL = tnr_left(sel,:)*100;
tnrR = tnr_right(sel,:)*100;
nPlot = numel(sel);

% Colors & markers
colors  = lines(nPlot);
markers = {'o','s','d','^','v','>','<','p','h','x'};
markers = markers(1:nPlot);

% Plot
figure('Position',[100 100 900 650]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

% 1) TPR — Left decoder
nexttile; hold on
for k = 1:nPlot
    plot(1:nSess, tprL(k,:), '-o', ...
         'Color', colors(k,:), ...
         'Marker', markers{k}, ...
         'MarkerFaceColor', colors(k,:), ...
         'LineWidth', 1.2);
end
plot(1:nSess, mean(tprL,1), '-k', 'LineWidth', 2.5);
set(gca,'XTick',1:nSess,'XTickLabel',sessionNames);
xlabel('Session'); ylabel('TPR (%)');
title('Left Decoder — TPR');
hold off

% 2) TPR — Right decoder
nexttile; hold on
for k = 1:nPlot
    plot(1:nSess, tprR(k,:), '-s', ...
         'Color', colors(k,:), ...
         'Marker', markers{k}, ...
         'MarkerFaceColor', colors(k,:), ...
         'LineWidth', 1.2);
end
plot(1:nSess, mean(tprR,1), '-k', 'LineWidth', 2.5);
set(gca,'XTick',1:nSess,'XTickLabel',sessionNames);
xlabel('Session'); title('Right Decoder — TPR');
hold off

% 3) TNR — Left decoder
nexttile; hold on
for k = 1:nPlot
    plot(1:nSess, tnrL(k,:), '-d', ...
         'Color', colors(k,:), ...
         'Marker', markers{k}, ...
         'MarkerFaceColor', colors(k,:), ...
         'LineWidth', 1.2);
end
plot(1:nSess, mean(tnrL,1), '-k', 'LineWidth', 2.5);
set(gca,'XTick',1:nSess,'XTickLabel',sessionNames);
xlabel('Session'); ylabel('TNR (%)');
title('Left Decoder — TNR');
hold off

% 4) TNR — Right decoder
nexttile; hold on
for k = 1:nPlot
    plot(1:nSess, tnrR(k,:), '-^', ...
         'Color', colors(k,:), ...
         'Marker', markers{k}, ...
         'MarkerFaceColor', colors(k,:), ...
         'LineWidth', 1.2);
end
plot(1:nSess, mean(tnrR,1), '-k', 'LineWidth', 2.5);
set(gca,'XTick',1:nSess,'XTickLabel',sessionNames);
xlabel('Session'); title('Right Decoder — TNR');
hold off

sgtitle('Decoder TPR & TNR — Selected Subjects','FontWeight','Bold');

%% Performance by target and distractor location
% === Define your condition and distractor settings ===
condsT = {'Up','Down','Right','Left'};
tCodes = [1, 3, 2, 4];  % mapping target names → codes in tpos
dconds = {'No Distractor','Contralateral (d←)','Ipsilateral (d→)'};
dMasks = {@(d) d==0, @(d) d==4, @(d) d==2};

nSubs    = numel(results);
nTargets = numel(condsT);
nDC      = numel(dconds);

% === Preallocate a 4-D array to hold proportions ===
% dims: subject × target × class(0/1) × distractor condition
propsAll = zeros(nSubs, nTargets, 2, nDC);

% === Loop subjects to fill propsAll ===
for s = 1:nSubs
    perf   = results(s).performance;
    tpos   = perf.tpos;
    dpos   = perf.dpos;
    % posts  = perf.right.posteriors;
    posts  = perf.left.posteriors;
    % thr    = perf.right.threshold;
    thr    = 0.5;
    
    for dc = 1:nDC
        maskD = dMasks{dc}(dpos);
        for ti = 1:nTargets
            mask = (tpos == tCodes(ti)) & maskD;
            scores = posts(mask);
            n1 = sum(scores >= thr);
            n0 = sum(scores <  thr);
            total = n0 + n1;
            if total > 0
                propsAll(s, ti, 1, dc) = n0/total;
                propsAll(s, ti, 2, dc) = n1/total;
            else
                propsAll(s, ti, :, dc) = NaN;  % no trials in this condition
            end
        end
    end
end

% === Compute group mean and SEM across subjects ===
meanProps = squeeze(nanmean(propsAll, 1));  
% size: [nTargets × 2 × nDC]
semProps  = squeeze(nanstd(propsAll, 0, 1) ./ sqrt(sum(~isnan(propsAll),1)));

% === Plot group‐average proportions ===
figure('Name','Group Average – Right Decoder','NumberTitle','off');
sgtitle('Group Average Proportions by Target Position (based on individualized thresholds)');

tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

for dc = 1:nDC
    nexttile;
    % Bar plot of mean proportions for this distractor condition
    hb = bar(meanProps(:,:,dc), 'grouped');
    hold on;
    
    % Add SEM error bars
    [ngroups, nbars] = size(meanProps(:,:,dc));
    groupwidth = min(0.8, nbars/(nbars+1.5));
    for j = 1:nbars
        % x positions for each target group
        x = (1:ngroups) - groupwidth/2 + (2*j-1) * groupwidth/(2*nbars);
        errorbar(x, meanProps(:,j,dc), semProps(:,j,dc), 'k', 'linestyle','none', 'LineWidth',1);
    end
    
    hold off;
    xticks(1:nTargets);
    xticklabels(condsT);
    xtickangle(30);
    xlabel('Target Position');
    ylim([0.2 0.8]);
    ylabel('Proportion of trials');
    if dc==1
        legend({'Class 0 (<thr)','Class 1 (≥thr)'}, 'Location','best');
    end
    title(dconds{dc});
    grid on;
end


