function comparePd_rawPO78(D1,D2,params)


LeftElec  = {'PO7'};
RightElec = {'PO8'};

% ---- helper: compute Pd = mean(contra-ipsi for left & right distractor trials)
computePd = @(D, params, lIdx, rIdx) local_compute_pd(D, params, lIdx, rIdx);

% channel indices (assumes D1/D2 share chanLabels in params)
lIdx = find(ismember(params.chanLabels, LeftElec));
rIdx = find(ismember(params.chanLabels, RightElec));
assert(~isempty(lIdx) && ~isempty(rIdx), 'PO7/PO8 not found in params.chanLabels');

% compute Pd waveforms
Pd1 = computePd(D1.epochs, params, lIdx, rIdx);
Pd2 = computePd(D2.epochs, params, lIdx, rIdx);

%% Plot overlay
fig = figure('units','normalized','Position',[0.1 0.1 0.3 0.4]);
ax  = axes(fig); hold(ax,'on');

% y-limits from both traces (so both visible; you can override with your own yL)
yL = [min([Pd1(:); Pd2(:)]) max([Pd1(:); Pd2(:)])];
pad = 0.05 * range(yL + eps);
yL = [yL(1)-pad, yL(2)+pad];

% gray shading (e.g., Pd window 0.2–0.5 s like your snippet)
patch(ax, [0.2 0.5 0.5 0.2], [-5 -5 5 5], ...
    [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.5, 'HandleVisibility','off');

% plot waveforms (use your color scheme if you want)
h1 = plot(ax, params.epochTime, Pd1, 'LineWidth', 2, 'Color', params.plotColor{1});
h2 = plot(ax, params.epochTime, Pd2, 'LineWidth', 2, 'Color', params.plotColor{3});

% reference lines
xline(ax, 0, '-', 'LineWidth', 1.5, 'HandleVisibility','off');
yline(ax, 0, '-', 'LineWidth', 1.5, 'HandleVisibility','off');

meanRTd1  = mean(D1.beh.RT(D1.epochs.labels~=0))/1000;
meanRTd2  = mean(D2.beh.RT(D2.epochs.labels~=0))/1000;
xline(ax, meanRTd1,  '--', 'LineWidth',1.5, 'Color', params.plotColor{1},'HandleVisibility','off');
xline(ax, meanRTd2,  '--', 'LineWidth',1.5, 'Color', params.plotColor{3},'HandleVisibility','off');


% axes limits and ticks
xlim(ax, [-0.2 0.75]);
ylim(ax, [-4 4]);
xticks(ax, 0:0.1:max(params.epochTime));

% labels/title
xlabel(ax, 'Time (s)', 'FontName','Arial', 'FontSize',10);
ylabel(ax, 'Amplitude (\muV)', 'FontName','Arial', 'FontSize',10);
title(ax, 'Pd (contra–ipsi) at PO7/PO8', ...
    'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

% legend
legend(ax, [h1 h2], {'Pre', 'Post'}, 'Box','on', 'FontSize',10, 'Location','northeast');

% aesthetics
set(ax, 'FontName','Arial', 'FontSize',10, 'LineWidth',1);
box(ax,'off');
hold(ax,'off');
end
%% ---- local function (kept at end of script or in its own file) ----
function Pd = local_compute_pd(D, params, lIdx, rIdx)
    % Baseline correction
    bw = params.baseline_window;
    bIdx = (params.epochTime >= bw(1)) & (params.epochTime <= bw(2));
    baseline = mean(D.data(bIdx,:,:), 1);          % 1 x nChan x nTrials
    data = D.data - baseline;                      % time x chan x trial

    % Trial masks (assumes: 1=right distractor, 2=left distractor, 0=no distractor)
    dTrialsR = (D.labels == 1);
    dTrialsL = (D.labels == 2);

    % Average waveforms at PO7/PO8 for each distractor side
    avgL_PO7 = squeeze(mean(mean(data(:, lIdx, dTrialsL), 2), 3)); % time x 1
    avgR_PO7 = squeeze(mean(mean(data(:, lIdx, dTrialsR), 2), 3));
    avgL_PO8 = squeeze(mean(mean(data(:, rIdx, dTrialsL), 2), 3));
    avgR_PO8 = squeeze(mean(mean(data(:, rIdx, dTrialsR), 2), 3));

    % Contra - ipsi per side
    diffL = avgL_PO8 - avgL_PO7;   % left distractor: contra(PO8) - ipsi(PO7)
    diffR = avgR_PO7 - avgR_PO8;   % right distractor: contra(PO7) - ipsi(PO8)

    % Collapse across side (common Pd estimate)
    Pd = (diffL + diffR) / 2;
end

