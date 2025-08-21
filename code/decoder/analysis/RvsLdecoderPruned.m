subjectList = {'e1','e2','e3','e4','e5','e6'};  % <-- fill in your subject IDs
resultsDir  = 'results';           % directory to save per-subject .mat files


% ----- Loop: compute and save per-subject data -----
nSubs = numel(subjectList);
bestPerfRAll = cell(nSubs,1);
bestPerfLAll = cell(nSubs,1);
for iSub = 1:nSubs
    subjID = subjectList{iSub};
    fprintf('Running computeModel for %s...\n', subjID);
    [perfR, perfL] = computeModel(subjID);
    bestPerfRAll{iSub} = perfR;
    bestPerfLAll{iSub} = perfL;
end

%% Accuracy plot
% --- Prepare data ---
nSubs = numel(bestPerfRAll);
accR = cellfun(@(p) p.acc, bestPerfRAll);
accL = cellfun(@(p) p.acc, bestPerfLAll);
meanR = mean(accR);  semR = std(accR)/sqrt(nSubs);
meanL = mean(accL);  semL = std(accL)/sqrt(nSubs);

% --- Define burnt‑orange palette ---
burntOrange = [0.8, 0.25, 0];
lightOrange = burntOrange + 0.3*(1 - burntOrange);

% --- Create figure ---
fig = figure('Units','centimeters','Position',[2 2 8 6], ...
             'Color','w', 'PaperPositionMode','auto');
set(fig,'Renderer','painters');

% --- Bar + error bars in burnt orange ---
hold on;
barW = 0.6;
bar([1 2],[meanR meanL], ...
    'FaceColor',burntOrange, 'EdgeColor','none', 'BarWidth',barW);
errorbar([1 2],[meanR meanL],[semR semL], ...
         'Color','k', 'LineStyle','none', ...
         'LineWidth',1.2, 'CapSize',8);

yOffset = semR * 9;  % offset so text sits above the error bar
text(1, meanR + yOffset, sprintf('%.2f', meanR), ...
     'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
     'FontName','Helvetica', 'FontSize',10, 'Color','k');
text(2, meanL + yOffset, sprintf('%.2f', meanL), ...
     'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
     'FontName','Helvetica', 'FontSize',10, 'Color','k');

% --- Overlay individual dots with jitter in light orange ---
rng(0); % for reproducible jitter
jitAmt = 0.07;
xR = 1 + (rand(nSubs,1)-0.5)*jitAmt;
xL = 2 + (rand(nSubs,1)-0.5)*jitAmt;
scatter(xR, accR, 36, ...
        'MarkerEdgeColor','k', ...
        'MarkerFaceColor',lightOrange, ...
        'MarkerFaceAlpha',0.8);
scatter(xL, accL, 36, ...
        'MarkerEdgeColor','k', ...
        'MarkerFaceColor',lightOrange, ...
        'MarkerFaceAlpha',0.8);

% --- Axis formatting ---
xlim([0.5 2.5]);
xticks([1 2]);
xticklabels({'Right Distractor','Left Distractor'});
ylabel('Accuracy','FontWeight','normal');
title('Decoder Accuracy across Subjects','FontWeight','normal');

ax = gca;
ax.FontName   = 'Helvetica';
ax.FontSize   = 10;
ax.LineWidth  = 0.8;
ax.TickDir    = 'out';
ax.Box        = 'off';
% ax.YLim       = [min([accR;accL]) - 0.05, 0.8];
ax.YLim       = [0.55, 0.8];

hold off;

%% AUPRC plot
% --- Prepare data ---
nSubs = numel(bestPerfRAll);
accR = cellfun(@(p) p.auprc, bestPerfRAll);
accL = cellfun(@(p) p.auprc, bestPerfLAll);
meanR = mean(accR);  semR = std(accR)/sqrt(nSubs);
meanL = mean(accL);  semL = std(accL)/sqrt(nSubs);

% --- Define burnt‑orange palette ---
burntOrange = [0.8, 0.25, 0];
lightOrange = burntOrange + 0.3*(1 - burntOrange);

% --- Create figure ---
fig = figure('Units','centimeters','Position',[2 2 8 6], ...
             'Color','w', 'PaperPositionMode','auto');
set(fig,'Renderer','painters');

% --- Bar + error bars in burnt orange ---
hold on;
barW = 0.6;
bar([1 2],[meanR meanL], ...
    'FaceColor',burntOrange, 'EdgeColor','none', 'BarWidth',barW);
errorbar([1 2],[meanR meanL],[semR semL], ...
         'Color','k', 'LineStyle','none', ...
         'LineWidth',1.2, 'CapSize',8);

yOffset = semR * 7;  % offset so text sits above the error bar
text(1, meanR + yOffset, sprintf('%.2f', meanR), ...
     'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
     'FontName','Helvetica', 'FontSize',10, 'Color','k');
text(2, meanL + yOffset, sprintf('%.2f', meanL), ...
     'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
     'FontName','Helvetica', 'FontSize',10, 'Color','k');

% --- Overlay individual dots with jitter in light orange ---
rng(0); % for reproducible jitter
jitAmt = 0.07;
xR = 1 + (rand(nSubs,1)-0.5)*jitAmt;
xL = 2 + (rand(nSubs,1)-0.5)*jitAmt;
scatter(xR, accR, 36, ...
        'MarkerEdgeColor','k', ...
        'MarkerFaceColor',lightOrange, ...
        'MarkerFaceAlpha',0.8);
scatter(xL, accL, 36, ...
        'MarkerEdgeColor','k', ...
        'MarkerFaceColor',lightOrange, ...
        'MarkerFaceAlpha',0.8);

% --- Axis formatting ---
xlim([0.5 2.5]);
xticks([1 2]);
xticklabels({'Right Distractor','Left Distractor'});
ylabel('AUPRC','FontWeight','normal');
title('AUPRC across Subjects','FontWeight','normal');

ax = gca;
ax.FontName   = 'Helvetica';
ax.FontSize   = 10;
ax.LineWidth  = 0.8;
ax.TickDir    = 'out';
ax.Box        = 'off';
% ax.YLim       = [min([accR;accL]) - 0.05, 0.8];
ax.YLim       = [0.55, 0.8];

hold off;