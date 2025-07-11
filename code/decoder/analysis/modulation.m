%% poster_decoder_performance.m
% 6-panel: Left, Right & Combined Decoder metrics
% Thicker subject lines; AUPRC=orange, Accuracy=blue
clearvars; close all

% --- 1) LOAD & PREPARE METRICS -----------------------------------------
S       = load('decodingResultsRetrainedTestTrain.mat','results');
results = S.results;
nSub    = numel(results);
%%
% Select subjects
subjectLabels = {results.subject};
[sel, ok] = listdlg( ...
    'ListString',    subjectLabels, ...
    'SelectionMode', 'multiple', ...
    'Name',          'Select subjects to plot:', ...
    'ListSize',      [200 300] ...
);
%
if ~ok || isempty(sel)
    sel = 1:nSub;
end

% Sessions → Online labels
sessionNames = {'Online 1','Online 2'};
nSess        = numel(sessionNames);

% Preallocate & extract (0–1 range)
auprcL = nan(nSub,nSess);  auprcR = nan(nSub,nSess);
accL   = nan(nSub,nSess);  accR   = nan(nSub,nSess);
for i = 1:nSub
    m1 = results(i).modulation1;
    m2 = results(i).modulation2;
    auprcL(i,:) = [m1.left.auprc,  m2.left.auprc];
    auprcR(i,:) = [m1.right.auprc, m2.right.auprc];
    accL(i,:)   = [m1.left.accuracy,  m2.left.accuracy];
    accR(i,:)   = [m1.right.accuracy, m2.right.accuracy];
end

% Filter to chosen subjects & convert to %
auprcL = auprcL(sel,:)*100;
auprcR = auprcR(sel,:)*100;
accL   = accL(sel,:)*100;
accR   = accR(sel,:)*100;
nPlot  = numel(sel);

% Combined decoder metrics
auprcM = (auprcL + auprcR)/2;
accM   = (accL   + accR  )/2;

% --- 1.5) STATISTICAL TESTS: paired t‐test Online 1 vs Online 2 ----------
[~, p_auprcL] = ttest(auprcL(:,1), auprcL(:,2));
[~, p_auprcR] = ttest(auprcR(:,1), auprcR(:,2));
[~, p_auprcM] = ttest(auprcM(:,1), auprcM(:,2));
[~, p_accL]   = ttest(accL(:,1),   accL(:,2));
[~, p_accR]   = ttest(accR(:,1),   accR(:,2));
[~, p_accM]   = ttest(accM(:,1),   accM(:,2));
fprintf('\nPaired t‐test p‐values (Online 1 vs Online 2):\n');
fprintf('  Left AUPRC:      p = %.3g\n', p_auprcL);
fprintf('  Right AUPRC:     p = %.3g\n', p_auprcR);
fprintf('  Average AUPRC:   p = %.3g\n', p_auprcM);
fprintf('  Left Accuracy:   p = %.3g\n', p_accL);
fprintf('  Right Accuracy:  p = %.3g\n', p_accR);
fprintf('  Average Accuracy:p = %.3g\n\n', p_accM);

% --- 2) STYLE SETTINGS ------------------------------------------------
% AUPRC colors (orange)
burntOrange = [204,85,0]/255;
lightOrange = [1,0.8,0.6];
% Accuracy colors (green)
accColor    = [0, 0.502, 0];
lightAcc    = accColor + (1-accColor)*0.7;

% Font & styling
tickFS    = 12;   % tick labels
labelFS   = 14;   % axis labels
titleFS   = 16;   % subplot titles
superFS   = 18;   % overall title
mkSize    = 10;   % marker size
lineW_sub = 1.5;  % individual subject line width (thicker)
lineW_mn  = 3;    % mean line width

%% --- 3) CREATE FIGURE -------------------------------------------------
figure('Units','inches','Position',[0 0 48 36],'Color','w');
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

% Panels 1–3: AUPRC
nexttile; hold on
doPlot(auprcL, sessionNames, nPlot, ...
       burntOrange, lightOrange, ...
       tickFS, labelFS, titleFS, mkSize, ...
       lineW_sub, lineW_mn, ...
       'AUPRC (%)','Left Decoder — AUPRC',[45 80]);
annotateSig(p_auprcL,[45 75],labelFS);
hold off

nexttile; hold on
doPlot(auprcR, sessionNames, nPlot, ...
       burntOrange, lightOrange, ...
       tickFS, labelFS, titleFS, mkSize, ...
       lineW_sub, lineW_mn, ...
       'AUPRC (%)','Right Decoder — AUPRC',[45 80]);
annotateSig(p_auprcR,[45 75],labelFS);
hold off

nexttile; hold on
doPlot(auprcM, sessionNames, nPlot, ...
       burntOrange, lightOrange, ...
       tickFS, labelFS, titleFS, mkSize, ...
       lineW_sub, lineW_mn, ...
       'AUPRC (%)','Average Decoder — AUPRC',[45 80]);
annotateSig(p_auprcM,[45 75],labelFS);
hold off

% Panels 4–6: Accuracy
nexttile; hold on
doPlot(accL,  sessionNames, nPlot, ...
       accColor, lightAcc, ...
       tickFS, labelFS, titleFS, mkSize, ...
       lineW_sub, lineW_mn, ...
       'Accuracy (%)','Left Decoder — Accuracy',[45 70]);
annotateSig(p_accL,[45 65],labelFS);
hold off

nexttile; hold on
doPlot(accR,  sessionNames, nPlot, ...
       accColor, lightAcc, ...
       tickFS, labelFS, titleFS, mkSize, ...
       lineW_sub, lineW_mn, ...
       'Accuracy (%)','Right Decoder — Accuracy',[45 70]);
annotateSig(p_accR,[45 65],labelFS);
hold off

nexttile; hold on
doPlot(accM,  sessionNames, nPlot, ...
       accColor, lightAcc, ...
       tickFS, labelFS, titleFS, mkSize, ...
       lineW_sub, lineW_mn, ...
       'Accuracy (%)','Average Decoder — Accuracy',[45 70]);
annotateSig(p_accM,[45 65],labelFS);
hold off

% Super‐title
sgtitle('Decoder Performance', ...
    'FontName','Times New Roman', ...
    'FontSize',superFS, ...
    'FontWeight','Bold');

% --- 4) EXPORT --------------------------------------------------------
exportgraphics(gcf,'poster_decoder_performance.tiff','Resolution',300);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function doPlot(metric, sessionNames, nPlot, ...
                primaryCol, lightCol, ...
                tickFS, labelFS, titleFS, mkSize, ...
                lineW_sub, lineW_mn, ...
                ylbl, ttl, ylims)
    % Compute mean & SEM
    m   = mean(metric,1);
    sem = std(metric,0,1)/sqrt(nPlot);
    x   = [ones(nPlot,1), 2*ones(nPlot,1)];

    % 1) Subject trajectories
    for k = 1:nPlot
        plot(x(k,:), metric(k,:), '-', ...
             'Color', lightCol, ...
             'LineWidth', lineW_sub, ...
             'HandleVisibility','off');
    end

    % 2) Mean ± SEM
    errorbar(1:2, m, sem, sem, 's-', ...
        'Color', primaryCol, ...
        'LineWidth', lineW_mn, ...
        'MarkerSize', mkSize, ...
        'MarkerFaceColor', primaryCol, ...
        'HandleVisibility','off');

    % 3) 50% chance line
    hC = yline(50, '--', ...
        'Color','k', ...
        'LineWidth',2, ...
        'FontName','Times New Roman', ...
        'FontSize', labelFS, ...
        'DisplayName','Chance');

    % 4) Axes formatting
    set(gca, ...
        'XTick',[1 2], ...
        'XTickLabel',sessionNames, ...
        'FontName','Times New Roman', ...
        'FontSize',16, ...
        'LineWidth',1.5, ...
        'Box','on');
    xlabel('Session',      'FontName','Times New Roman','FontSize',16);
    ylabel(ylbl,           'FontName','Times New Roman','FontSize',labelFS);
    title(ttl,             'FontName','Times New Roman','FontSize',titleFS);
    xlim([0.8 2.2]);
    ylim(ylims);

    % 5) Legend (chance only)
    legend(hC, 'Location','northwest', ...
           'FontName','Times New Roman','FontSize',labelFS);
end

function annotateSig(p, ylims, labelFS)
    if p >= 0.05
        return;
    end
    y_max = ylims(2); y_min = ylims(1);
    yb    = y_max - (y_max - y_min)*0.05;
    yoff  = (y_max - y_min)*0.02;
    % bracket
    plot([1,1,2,2], [yb, yb+yoff, yb+yoff, yb], 'k-', 'LineWidth',1.5, ...
         'HandleVisibility','off');
    % stars
    if p < 0.01
        stars = '**';
    else
        stars = '*';
    end
    text(1.5, yb + yoff + (y_max - y_min)*0.01, stars, ...
         'FontName','Times New Roman','FontSize',labelFS, ...
         'HorizontalAlignment','center', 'HandleVisibility','off');
end
