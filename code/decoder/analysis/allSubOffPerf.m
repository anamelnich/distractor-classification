performance = load('gridSearchResults_parfor.mat');
performance = performance.T;
%%
nParams = 36;
nSubs   = 13;

% Which variable holds the structs?
perfVar = performance.Properties.VariableNames{3};
perfCol = performance.(perfVar);  % 468×1 cell or struct array

% preallocate all metrics
leftAUPRC   = nan(nSubs,1);
rightAUPRC  = nan(nSubs,1);
avgAUPRC    = nan(nSubs,1);

leftAcc     = nan(nSubs,1);
rightAcc    = nan(nSubs,1);
avgAcc      = nan(nSubs,1);

leftTPR     = nan(nSubs,1);
rightTPR    = nan(nSubs,1);
avgTPR      = nan(nSubs,1);

leftTNR     = nan(nSubs,1);
rightTNR    = nan(nSubs,1);
avgTNR      = nan(nSubs,1);

for s = 1:nSubs
    % pick the “best” row for subject s
    rowIdx = nParams*(s-1) + 2;  % adjust if your “best” row index differs

    % extract the struct from the table column
    if iscell(perfCol)
        ps = perfCol{rowIdx};
    else
        ps = perfCol(rowIdx);
    end

    %— AUPRC
    leftAUPRC(s)   = ps.left.auprc;
    rightAUPRC(s)  = ps.right.auprc;
    avgAUPRC(s)    = mean([leftAUPRC(s), rightAUPRC(s)]);

    %— Accuracy
    leftAcc(s)     = ps.left.accuracy;
    rightAcc(s)    = ps.right.accuracy;
    avgAcc(s)      = mean([leftAcc(s), rightAcc(s)]);

    %— True Positive Rate (sensitivity)
    leftTPR(s)     = ps.left.tpr;
    rightTPR(s)    = ps.right.tpr;
    avgTPR(s)      = mean([leftTPR(s), rightTPR(s)]);

    %— True Negative Rate (specificity)
    leftTNR(s)     = ps.left.tnr;
    rightTNR(s)    = ps.right.tnr;
    avgTNR(s)      = mean([leftTNR(s), rightTNR(s)]);
end
%%
% User settings
excludeSubs = [];    % e.g. [2,5] to skip subjects 2 and 5
colors.burntDark  = [0.85, 0.33, 0.10];   % darker burnt orange
colors.burntLight = [0.93, 0.55, 0.38];   % lighter burnt orange

% Collect metrics
nSub = numel(leftAUPRC);
% pack only AUPRC & Accuracy
leftMetrics  = [ leftAUPRC,  leftAcc ];
rightMetrics = [ rightAUPRC, rightAcc ];
metricNames  = {'AUPRC','Accuracy'};

% exclude subjects if requested
keep = true(nSub,1);
keep(excludeSubs) = false;
leftMetrics  = leftMetrics(keep,:);
rightMetrics = rightMetrics(keep,:);
nKeep = sum(keep);

% Compute means & SEMs
meanL = mean(leftMetrics,1);
meanR = mean(rightMetrics,1);
semL  = std(leftMetrics,0,1)/sqrt(nKeep);
semR  = std(rightMetrics,0,1)/sqrt(nKeep);

% Plot
figure('Units','inches','Position',[1 1 6 4],'Color','w');
hold on;

x    = 1:2;          % two metrics
barW = 0.3;          % bar width
jitt = 0.04;         % horizontal jitter for dots

% Bars
bL = bar(x - barW/2, meanL, barW, 'FaceColor',colors.burntDark,'EdgeColor','none');
bR = bar(x + barW/2, meanR, barW, 'FaceColor',colors.burntLight,'EdgeColor','none');

% Error bars
errorbar(x - barW/2, meanL, semL, 'k','LineStyle','none','LineWidth',1.2);
errorbar(x + barW/2, meanR, semR, 'k','LineStyle','none','LineWidth',1.2);

% Subject dots
for m = 1:2
    scatter( x(m)-barW/2 + (rand(nKeep,1)-.5)*jitt, ...
             leftMetrics(:,m), ...
             24, 'k','filled','MarkerFaceAlpha',0.7,'MarkerEdgeColor','w');
    scatter( x(m)+barW/2 + (rand(nKeep,1)-.5)*jitt, ...
             rightMetrics(:,m), ...
             24, 'k','filled','MarkerFaceAlpha',0.7,'MarkerEdgeColor','w');
end
% ---- new: 50% chance line ----
hC = yline(0.5, '--', ...
           'Color','k', ...
           'LineWidth',1.5, ...
           'FontName','Times New Roman', ...
           'FontSize',12, ...
           'LabelHorizontalAlignment','right', ...
           'LabelVerticalAlignment','bottom');
% Styling
xlim([0.5 2.5]);
ylim([0.4 0.75])
set(gca, ...
    'XTick', x, ...
    'XTickLabel', metricNames, ...
    'FontName','Times New Roman', ...
    'FontSize',12, ...
    'LineWidth',1, ...
    'TickDir','out', ...
    'Box','off');
ylabel('Performance', 'FontSize',14, 'FontName','Times New Roman');
title('Decoder Performance (Left vs Right)', 'FontSize',16, 'FontName','Times New Roman');

legend([bL,bR], {'Left Decoder','Right Decoder'}, ...
       'Location','northwest','FontSize',12);

hold off;

