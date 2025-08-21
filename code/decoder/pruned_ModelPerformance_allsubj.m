% ------------------------- USER SETTINGS -------------------------
subjectList = {'e1','e2','e3','e4','e5','e6'};
resultsDir  = 'results';
% -----------------------------------------------------------------

% ensure results folder exists
if ~exist(resultsDir,'dir')
    mkdir(resultsDir);
end

nSubs = numel(subjectList);

% build an “empty” template with exactly the fields computeModel returns:
template = struct( ...
    'auprc',      [], ...
    'threshold',  [], ...
    'accuracy',   [], ...
    'tpr',        [], ...
    'tnr',        [], ...
    'posteriors', {{}}, ...   % cell for arrays
    'labels',     {{}}, ...
    'file_id',    {{}}  ...
);

% replicate it into an nSubs×1 array
perfArray = repmat(template, nSubs, 1);

% now loop and fill it
for iSub = 1:nSubs
    sid = subjectList{iSub};
    fprintf('Computing performance for %s...\n', sid);
    perfArray(iSub) = computeModel(sid);  % now they’re compatible
    % save(fullfile(resultsDir, [sid '_perf.mat']), 'perfArray');
end
%

% collate numeric metrics into a matrix
metrics = {'auprc','accuracy','tpr','tnr'};
nM = numel(metrics);
data = nan(nSubs, nM);
for i = 1:nM
    data(:,i) = [perfArray.(metrics{i})].';
end

% compute means
means = mean(data,1,'omitnan');

%% assume you still have:
% assume you still have:
%   data (nSubs × 4), means (1 × 4), nSubs, and burntOrange defined

% select only cols 2–4
data2   = data(:,2:4);
means2  = means(2:4);
std2    = std(data2, 0, 1);   % standard deviation across subjects
nM2     = numel(means2);

% --- Define color ---
burntOrange = [0.8, 0.25, 0];  
lightOrange = burntOrange + 0.3*(1-burntOrange);

% --- Create figure ---
fig = figure('Units','inches','Position',[1 1 4 3], ...
             'PaperPositionMode','auto','Renderer','painters');
         
% Bar plot of means (Accuracy, TPR, TNR)
hb = bar(means2, 'FaceColor', burntOrange, 'EdgeColor', 'none');
hold on;

% Add error bars (±1 SD)
xpos = 1:nM2;
errorbar(xpos, means2, std2, 'k', 'LineStyle', 'none', 'LineWidth', 1.2);

% Scatter individual points (white fill, black border)
for i = 1:nM2
    x = xpos(i) + (rand(nSubs,1)-0.5)*0.08;  
    scatter(x, data2(:,i), 36, ...
            'MarkerFaceColor', 'w', ...
            'MarkerEdgeColor','k', ...
            'LineWidth',0.5);
end

% Axes styling
ax = gca;
ax.FontName   = 'Arial';
ax.FontSize   = 12;
ax.LineWidth  = 1;
ax.TickDir    = 'out';
ax.Box        = 'off';
ax.XLim       = [0.5, nM2+0.5];
ax.YLim       = [0.5, 0.8];    % adjust as needed
ax.YGrid      = 'on';
ax.GridColor  = [0.7 0.7 0.7];
ax.GridAlpha  = 0.3;

% Labels and title
xticks(xpos);
xticklabels({'Accuracy','TPR','TNR'});
ylabel('Performance (%)','FontSize',13);
title('Across-Subject Performance','FontWeight','normal','FontSize',14);

hold off;


%%
% --- Colors ---
burntOrange = [0.8, 0.25, 0];
lightOrange = burntOrange + 0.3*(1-burntOrange);
grayDiag    = [0.6, 0.6, 0.6];

% --- Prepare FPR grid & storage ---
fprGrid   = linspace(0,1,200);
tprMatrix = nan(nSubs, numel(fprGrid));
AUCs      = nan(nSubs,1);

% --- New figure for ROC curves ---
figROC = figure('Units','inches','Position',[1 1 4 4], ...
                'PaperPositionMode','auto','Renderer','painters');
hold on;

% Plot individual curves and grab one handle for the legend
hSub = gobjects(nSubs,1);
for i = 1:nSubs
    lbl  = perfArray(i).labels;
    post = perfArray(i).posteriors;
    
    [FPR, TPR, ~, AUCs(i)] = perfcurve(lbl, post, 1);
    [FPRu, ia]             = unique(FPR);
    TPRu                   = TPR(ia);
    tprMatrix(i,:)         = interp1(FPRu, TPRu, fprGrid, 'linear', 0);
    
    hSub(i) = plot(FPRu, TPRu, 'LineWidth',0.8, 'Color', lightOrange);
end

% chance diagonal and grab its handle
hChance = plot([0 1],[0 1], '--', 'LineWidth',1, 'Color',grayDiag);

% mean ROC and grab its handle
meanTPR = mean(tprMatrix,1,'omitnan');
hMean   = plot(fprGrid, meanTPR, 'LineWidth',2.5, 'Color', burntOrange);

% styling
ax = gca;
ax.FontName  = 'Arial';
ax.FontSize  = 12;
ax.LineWidth = 1;
ax.Box       = 'off';
ax.TickDir   = 'out';
xlabel('False Positive Rate','FontSize',12);
ylabel('True Positive Rate','FontSize',12);
title(sprintf('ROC Curves (AUC = %.3f \\pm %.3f)', mean(AUCs), std(AUCs)), ...
      'FontSize',14,'FontWeight','normal');

% legend: use one representative subject‐handle (hSub(1)), then the others
legend([hSub(1), hChance, hMean], ...
       {'Individual subjects','Chance','Mean ROC'}, ...
       'Location','SouthEast','Box','off');

hold off;
