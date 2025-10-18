function plotERPOffvsOnline_xDAWN(origData, bestData, params, dSide, decoder, panelNames)
% plotERPOffvsOnline with xDAWN projection (single decoder)
% 
% Inputs:
%   origData, bestData: structs with
%       .data   [T x C x N]
%       .labels [N x 1] or [1 x N], where 1 = distractor, 0 = no distractor
%   params: struct with
%       .chanLabels, .epochTime, .baseline_window, .plotColor
%   dSide: "left" or "right"
%       - "left"  : use (R - L) orientation
%       - "right" : use (L - R) orientation
%   decoder: struct with .spatialFilter.diff (7 x 2) xDAWN weights (top-2 comps)
%   panelNames: optional 1x2 cellstr (e.g., {'Calibration','Online'})
%
% Behavior:
%   1) For each trial, form a 7-channel difference vector (paired L/R ROIs)
%      using the orientation specified by dSide (R-L or L-R).
%   2) Project [time x 7] difference through xDAWN weights (7x2) → [time x 2]
%   3) Average the 2 components → [time x 1] per trial
%   4) Grand average across trials for distractor (label==1) and ND (label==0)
%
% Note: The 7 channel pairs must match the xDAWN weight order:
%   {P1–P2, P3–P4, P5–P6, P7–P8, PO3–PO4, PO5–PO6, PO7–PO8}

% ---- Panel names default ----
if nargin < 6 || isempty(panelNames)
    panelNames = {'Offline','Online'};
end
if numel(panelNames) ~= 2
    error('panelNames must be a 1x2 cell array of strings, e.g., {''Calibration'',''Online''}.');
end

% ---- Validate decoder ----
W = decoder.spatialFilter.diff;
if size(W,1) ~= 7
    error('decoder.spatialFilter.diff must have 7 rows (one per L/R pair). Got %dx%d.', size(W,1), size(W,2));
end
if size(W,2) < 2
    error('decoder.spatialFilter.diff must have at least 2 columns (top-2 components).');
end
W = W(:,1:2); % ensure 7x2

% ---- Electrode ROIs (paired order matters) ----
LeftElec  = {'P1','P3','P5','P7','PO3','PO5','PO7'};
RightElec = {'P2','P4','P6','P8','PO4','PO6','PO8'};
[okL, lIdx] = ismember(LeftElec,  params.chanLabels);
[okR, rIdx] = ismember(RightElec, params.chanLabels);
if ~all(okL) || ~all(okR)
    missing = [LeftElec(~okL), RightElec(~okR)];
    error('Missing required channels: %s', strjoin(missing, ', '));
end

% ---- Orientation sign based on dSide ----
%   "left"  decoder: use R - L orientation
%   "right" decoder: use L - R orientation
switch lower(string(dSide))
    case "left"
        orient = +1; % +1 * (R - L)
    case "right"
        orient = -1; % -1 * (R - L) = (L - R)
    otherwise
        error('dSide must be "left" or "right".');
end

% ---- Figure ----
figure('Color','w', 'Units','inches', 'Position',[1 1 4 6]);
tiledlayout(2,1, 'TileSpacing','compact', 'Padding','compact');
panelLetters = {'A','B'};
datasets = {origData, bestData};
yL = [-5 8]; % µV (adjust as needed)

for p = 1:2
    ax = nexttile; hold(ax,'on');
    D = datasets{p};

    % labels as column
    D.labels = D.labels(:);
    dTrials  = (D.labels == 1);
    ndTrials = (D.labels == 0);

    % ---- Baseline correction ----
    bIdx = find(params.epochTime >= params.baseline_window(1) & ...
                params.epochTime <= params.baseline_window(2));
    base = mean(D.data(bIdx, :, :), 1);
    D.data = D.data - base;

    T = size(D.data,1);
    N = size(D.data,3);

    % ---- xDAWN-processed 1D trace per trial ----
    xproj = zeros(T, N);

    for n = 1:N
        % Extract ROI time series for this trial: [T x 7] each
        Lroi = squeeze(D.data(:, lIdx, n));  % T x 7
        Rroi = squeeze(D.data(:, rIdx, n));  % T x 7
        if isvector(Lroi), Lroi = Lroi(:)'; end
        if isvector(Rroi), Rroi = Rroi(:)'; end

        % Base 7-channel difference is (R - L); flip sign if dSide=="right"
        diff7 = (Rroi - Lroi) * orient;  % T x 7

        % Project with xDAWN (7x2), then average the 2 components
        comps = diff7 * W;               % (T x 7) * (7 x 2) = T x 2
        xproj(:,n) = mean(comps, 2);     % T x 1
    end

    % ---- Grand averages ----
    waveD  = mean(xproj(:, dTrials),  2, 'omitnan');
    waveND = mean(xproj(:, ndTrials), 2, 'omitnan');

    % ---- Shaded window (example) ----
    patch([0.2 0.5 0.5 0.2], [yL(1) yL(1) yL(2) yL(2)], ...
          [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.5, 'HandleVisibility','off');

    % ---- Plot ----
    h1 = plot(ax, params.epochTime, waveD,  'LineWidth',2, 'Color', params.plotColor{1});
    h2 = plot(ax, params.epochTime, waveND, 'LineWidth',2, 'Color', params.plotColor{5});

    % ---- Reference lines ----
    xline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');
    yline(ax, 0, '--', 'LineWidth',1.5, 'HandleVisibility','off');

    % ---- Axes/labels ----
    xlim(ax,[-0.1 0.65]);
    ylim(ax,yL);
    xticks(ax,0:0.1:max(params.epochTime));
    xlabel(ax,'Time (s)', 'FontName','Arial', 'FontSize',10);
    ylabel(ax,'Amplitude (\muV)', 'FontName','Arial', 'FontSize',10);
    title(ax, panelNames{p}, 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    legend(ax, [h1 h2], {'Distractor','No distractor'}, ...
        'Box','on', 'FontSize',10, 'Location','northeast');

    text(ax, -0.08, 1.02, panelLetters{p}, ...
        'Units','normalized', 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    set(ax, 'FontName','Arial', 'FontSize',10, 'LineWidth',1);
    box(ax,'off'); hold(ax,'off');
end
end
