function plotERPOffvsOnline(origData, bestData, params, dSide, panelNames)

% ---- Defaults for panel names ----
if nargin < 5 || isempty(panelNames)
    panelNames = {'Offline','Online'};
end
if numel(panelNames) ~= 2
    error('panelNames must be a 1x2 cell array of strings, e.g., {''Calibration'',''Online''}.');
end

% Define electrode ROIs
LeftElec  = {'P1','P3','P5','P7','PO3','PO5','PO7'};
RightElec = {'P2','P4','P6','P8','PO4','PO6','PO8'};
lIdx = find( ismember(params.chanLabels, LeftElec) );
rIdx = find( ismember(params.chanLabels, RightElec) );

% Prepare figure
figure('Color','w', 'Units','inches', 'Position',[1 1 4 6]);
tiledlayout(2,1, 'TileSpacing','compact', 'Padding','compact');
panelLetters = {'A','B'};
datasets = {origData, bestData};
yL = [-10 15]; % consistent y-limits

for p = 1:2
    ax = nexttile;
    hold(ax,'on');
    D = datasets{p};

    % Ensure labels are column
    D.labels = D.labels(:);

    % trial indices (your convention: 1 = distractor, 0 = no distractor)
    dTrials  = (D.labels == 1);
    ndTrials = (D.labels == 0);

    % baseline correction
    baseline_window = params.baseline_window;
    baseline_idx = find(params.epochTime >= baseline_window(1) & params.epochTime <= baseline_window(2));
    baseline = mean(D.data(baseline_idx, :, :), 1);
    D.data = D.data - baseline;

    % compute grand-averages
    avgDl  = squeeze(mean(mean(D.data(:, lIdx, dTrials ),2),3)); % distractor L-ROI
    avgDr  = squeeze(mean(mean(D.data(:, rIdx, dTrials ),2),3)); % distractor R-ROI
    avgNdl = squeeze(mean(mean(D.data(:, lIdx, ndTrials),2),3)); % ND L-ROI
    avgNdr = squeeze(mean(mean(D.data(:, rIdx, ndTrials),2),3)); % ND R-ROI

    % default ND diff (will overwrite below based on dSide)
    diffND = avgNdr - avgNdl;

    if strcmpi(dSide, "left")
        % For left-side decoder: R - L for distractor; ND same orientation
        diffD  = avgDr - avgDl;
        diffND = avgNdr - avgNdl;
    elseif strcmpi(dSide, "right")
        % For right-side decoder: L - R for distractor; ND mirrored
        diffD  = avgDl - avgDr;
        diffND = avgNdl - avgNdr;
    else
        error('dSide must be "left" or "right".');
    end

    % gray shading (example: 0.2–0.5 s)
    patch([0.2 0.5 0.5 0.2], [yL(1) yL(1) yL(2) yL(2)], ...
        [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.5, 'HandleVisibility','off');

    % plot waveforms
    h1 = plot(ax, params.epochTime, diffD,  'LineWidth',2, 'Color', params.plotColor{1});
    h2 = plot(ax, params.epochTime, diffND, 'LineWidth',2, 'Color', params.plotColor{5});

    % zero reference lines
    xline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');
    yline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');

    % axes limits and ticks
    xlim(ax,[-0.5 0.65]);
    ylim(ax,yL);
    xticks(ax,0:0.1:max(params.epochTime));

    % labels and title (use panelNames for titles)
    xlabel(ax,'Time (s)', 'FontName','Arial', 'FontSize',10);
    ylabel(ax,'Amplitude (\muV)', 'FontName','Arial', 'FontSize',10);
    title(ax, panelNames{p}, 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    % legend 
    legend(ax, [h1 h2], {'Distractor','No distractor'}, ...
        'Box','on', 'FontSize',10, 'Location','northeast');

    % panel letter (A/B)
    text(ax, -0.08, 1.02, panelLetters{p}, ...
        'Units','normalized', 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    % aesthetic tweaks
    set(ax, 'FontName','Arial', 'FontSize',10, 'LineWidth',1);
    box(ax,'off');
    hold(ax,'off');
end

end
