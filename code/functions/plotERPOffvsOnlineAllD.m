function plotERPOffvsOnlineAllD(origData, bestData, params, panelNames)
% plotERPOffvsOnlineAllD
% - D.labels: 0 = no distractor, 1 = distractor RIGHT, 2 = distractor LEFT
% - For label==1 (R):   use L - R
% - For label==2 (L):   use R - L
% - For label==0 (ND):  randomly choose (R - L) or (L - R) per trial
% - Combine L/R distractor trials into one "Distractor" average
%
% Inputs:
%   origData, bestData each with fields:
%       .data   [time x channels x trials]
%       .labels [1 x trials] or [trials x 1] with values in {0,1,2}
%       (optional) .RT     [trials x 1]
%   params with fields:
%       .chanLabels       cellstr of channel names (len=channels)
%       .epochTime        [time x 1] vector (seconds)
%       .baseline_window  [t1 t2] seconds
%       .plotColor        cell array of colors, e.g. params.plotColor{1}, {5}
%   panelNames (optional) 1x2 cellstr for subplot titles, e.g. {'Calibration','Online'}
%
% Example:
%   plotERPOffvsOnlineAllD(calibData, onData, params, {'Calibration','Online S1'})

% ---- Defaults for panel names ----
if nargin < 4 || isempty(panelNames)
    panelNames = {'Offline','Online'};
end
if numel(panelNames) ~= 2
    error('panelNames must be a 1x2 cell array of strings, e.g., {''Calibration'',''Online''}.');
end

% ---- Electrode ROIs ----
LeftElec  = {'P1','P3','P5','P7','PO3','PO5','PO7'};
RightElec = {'P2','P4','P6','P8','PO4','PO6','PO8'};
lIdx = find( ismember(params.chanLabels, LeftElec) );
rIdx = find( ismember(params.chanLabels, RightElec) );

% ---- Figure layout ----
figure('Color','w', 'Units','inches', 'Position',[1 1 4 6]);
T = tiledlayout(2,1, 'TileSpacing','compact', 'Padding','compact'); %#ok<NASGU>
annotations = {'A','B'};
datasets = {origData, bestData};
yL = [-5 5]; % y-limits (µV) — adjust as needed

% Optional reproducibility for ND random flipping:
if isfield(params,'rng_seed') && ~isempty(params.rng_seed)
    rng(params.rng_seed);
end

for p = 1:2
    ax = nexttile;
    hold(ax,'on');
    D = datasets{p};

    % ---- Ensure labels are row vector ----
    if size(D.labels,1) > 1
        D.labels = D.labels(:)'; 
    end

    % ---- Trial masks ----
    dTrials  = (D.labels == 1) | (D.labels == 2); % any distractor
    ndTrials = (D.labels == 0);

    % ---- Baseline correction ----
    baseline_window = params.baseline_window;
    baseline_idx = find(params.epochTime >= baseline_window(1) & params.epochTime <= baseline_window(2));
    baseline = mean(D.data(baseline_idx, :, :), 1);
    D.data = D.data - baseline;

    % ---- ROI averages per trial ----
    % L, R: [time x trials]
    L = squeeze(mean(D.data(:, lIdx, :), 2));
    R = squeeze(mean(D.data(:, rIdx, :), 2));
    if isvector(L), L = L(:); end
    if isvector(R), R = R(:); end

    % ---- Build per-trial difference according to labels ----
    % diffAll: [time x trials]
    nT = size(D.data, 3);
    diffAll = zeros(size(D.data,1), nT);

    % Distractor RIGHT (label==1): L - R
    idxR = (D.labels == 1);
    if any(idxR)
        diffAll(:, idxR) = L(:, idxR) - R(:, idxR);
    end

    % Distractor LEFT (label==2): R - L
    idxL = (D.labels == 2);
    if any(idxL)
        diffAll(:, idxL) = R(:, idxL) - L(:, idxL);
    end

    % No Distractor (label==0): random per trial choose (R-L) or (L-R)
    idxN = (D.labels == 0);
    if any(idxN)
        nN = sum(idxN);
        s = randi([0 1], [1 nN])*2 - 1; % +1 or -1
        % Base difference = (R - L); if s=-1, it becomes (L - R)
        baseND = R(:, idxN) - L(:, idxN);
        diffAll(:, idxN) = baseND .* s; % implicit expansion over columns
    end

    % ---- Grand averages ----
    % Combine Left/Right distractor trials (after per-trial direction applied)
    if any(dTrials)
        diffD  = mean(diffAll(:, dTrials), 2);
    else
        diffD  = zeros(size(D.data,1),1);
    end

    if any(ndTrials)
        diffND = mean(diffAll(:, ndTrials), 2);
    else
        diffND = zeros(size(D.data,1),1);
    end

    % ---- Gray analysis window shading (example: 0.2–0.5 s) ----
    patch([0.2 0.5 0.5 0.2], [yL(1) yL(1) yL(2) yL(2)], ...
        [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.5, 'HandleVisibility','off');

    % ---- Plot waveforms ----
    h1 = plot(ax, params.epochTime, diffD,  'LineWidth',2, 'Color', params.plotColor{1});
    h2 = plot(ax, params.epochTime, diffND, 'LineWidth',2, 'Color', params.plotColor{5});

    % ---- Zero lines ----
    xline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');
    yline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');

    % ---- Axes/labels ----
    xlim(ax,[-0.1 0.65]);
    ylim(ax,yL);
    xticks(ax,0:0.1:max(params.epochTime));
    xlabel(ax,'Time (s)', 'FontName','Arial', 'FontSize',10);
    ylabel(ax,'Amplitude (\muV)', 'FontName','Arial', 'FontSize',10);

    % ---- Titles using provided panel names ----
    title(ax, sprintf('%s', panelNames{p}), 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    % ---- Legend ----
    legend(ax, [h1 h2], {'Distractor','No distractor'}, ...
        'Box','on', 'FontSize',10, 'Location','northeast');

    % ---- Panel label (A/B) ----
    text(ax, -0.08, 1.02, annotations{p}, ...
        'Units','normalized', 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    % ---- Aesthetics ----
    set(ax, 'FontName','Arial', 'FontSize',10, 'LineWidth',1);
    box(ax,'off');
    hold(ax,'off');
end
end

