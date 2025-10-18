function plotERPOffvsOnlineAllD_xDAWN(origData, bestData, params, decoderL, decoderR, panelNames,showRT)
% plotERPOffvsOnlineAllD_xDAWN
% xDAWN applied to per-trial 7-ch difference (paired L/R electrodes), then
% average the top-2 components and plot grand averages.
%
% D.labels: 0 = no distractor, 1 = distractor RIGHT, 2 = distractor LEFT
% Use weights:
%   labels==1 -> decoderL.spatialFilter.diff (apply to L-R)
%   labels==2 -> decoderR.spatialFilter.diff (apply to R-L)
%   labels==0 -> decoderL.spatialFilter.diff (apply to random flip of R-L)
%
% Inputs:
%   origData, bestData: .data [T x C x N], .labels [1xN] or [Nx1]
%   params: .chanLabels, .epochTime, .baseline_window, .plotColor
%   decoderL/decoderR: structs with field .spatialFilter.diff (7x2)
%   panelNames (optional): 1x2 cellstr

% ---- Defaults for panel names ----
if nargin < 6 || isempty(panelNames)
    panelNames = {'Offline','Online'};
end
if numel(panelNames) ~= 2
    error('panelNames must be a 1x2 cell array of strings.');
end
if nargin < 7 || isempty(showRT)
    showRT = 0;   % default: don’t show RT
end

% ---- Validate xDAWN matrices ----
WL = decoderL.spatialFilter.diff;   % expected 7x2
WR = decoderR.spatialFilter.diff;   % expected 7x2
if ~ismatrix(WL) || ~all(size(WL)==[7 2]) || ~ismatrix(WR) || ~all(size(WR)==[7 2])
    error('decoderL/decoderR .spatialFilter.diff must be 7x2 matrices (weights for top-2 comps).');
end

% ---- Electrode pairs (order matters & must be paired) ----
LeftElec  = {'P1','P3','P5','P7','PO3','PO5','PO7'};
RightElec = {'P2','P4','P6','P8','PO4','PO6','PO8'};

% map labels to channel indices (paired)
[isL, lIdx] = ismember(LeftElec,  params.chanLabels);
[isR, rIdx] = ismember(RightElec, params.chanLabels);
if ~all(isL) || ~all(isR)
    missing = [LeftElec(~isL), RightElec(~isR)];
    error('Missing required channels: %s', strjoin(missing, ', '));
end

% ---- Figure layout ----
figure('Color','w', 'Units','inches', 'Position',[1 1 4.2 6.2]);
tlo = tiledlayout(2,1, 'TileSpacing','compact', 'Padding','compact'); %#ok<NASGU>
annotations = {'A','B'};
datasets = {origData, bestData};
yL = [-4 4]; % µV

% Optional reproducibility for ND random flipping:
if isfield(params,'rng_seed') && ~isempty(params.rng_seed)
    rng(params.rng_seed);
end

for p = 1:2
    ax = nexttile;
    hold(ax,'on');
    D = datasets{p};

    % labels as row vector
    if size(D.labels,1) > 1, D.labels = D.labels(:)'; end

    dTrials  = (D.labels == 1) | (D.labels == 2);
    ndTrials = (D.labels == 0);

    % ---- Baseline correction (per trial, per channel) ----
    baseline_idx = find(params.epochTime >= params.baseline_window(1) & ...
                        params.epochTime <= params.baseline_window(2));
    baseline = mean(D.data(baseline_idx, :, :), 1);
    D.data = D.data - baseline;

    T = size(D.data,1);
    N = size(D.data,3);
    diffAll_xdawn = zeros(T, N); % final per-trial 1D trace after xDAWN&comp-avg

    % ---- Build per-trial 7ch differences & apply xDAWN ----
    for n = 1:N
        lab = D.labels(n);

        % Extract paired L/R time series for this trial: [T x 7] each
        Lroi = squeeze(D.data(:, lIdx, n)); % T x 7
        Rroi = squeeze(D.data(:, rIdx, n)); % T x 7
        if isvector(Lroi), Lroi = Lroi(:)'; end
        if isvector(Rroi), Rroi = Rroi(:)'; end

        switch lab
            case 1   % distractor RIGHT -> use L-R, weights WL
                diff7 = Lroi - Rroi;      % [T x 7]
                W = WL;                   % [7 x 2]
            case 2   % distractor LEFT -> use R-L, weights WR
                diff7 = Rroi - Lroi;
                W = WR;
            otherwise % 0: no distractor -> random sign of (R-L), weights WL
                base = Rroi - Lroi;
                if rand > 0.5, sgn = +1; else, sgn = -1; end
                diff7 = base * sgn;
                W = WL;
        end

        % xDAWN projection to top-2 components: [T x 2]
        comps = diff7 * W;   % (T x 7) * (7 x 2)

        % Average the 2 components -> [T x 1]
        diffAll_xdawn(:, n) = mean(comps, 2);
    end

    % ---- Grand averages (xDAWN-processed) ----
    if any(dTrials)
        waveD  = mean(diffAll_xdawn(:, dTrials), 2);
    else
        waveD  = zeros(T,1);
    end
    if any(ndTrials)
        waveND = mean(diffAll_xdawn(:, ndTrials), 2);
    else
        waveND = zeros(T,1);
    end

    % ---- Optional analysis window shading (example: 0.2–0.5 s) ----
    patch([0.2 0.5 0.5 0.2], [yL(1) yL(1) yL(2) yL(2)], ...
        [0.9 0.9 0.9], 'EdgeColor','none', 'FaceAlpha',0.5, 'HandleVisibility','off');

    % ---- Plot waveforms ----
    h1 = plot(ax, params.epochTime, waveD,  'LineWidth',2, 'Color', params.plotColor{1});
    h2 = plot(ax, params.epochTime, waveND, 'LineWidth',2, 'Color', params.plotColor{5});

    % ---- Zero lines ----
    xline(ax, 0, '--', 'LineWidth',1.2, 'Color',[0.4 0.4 0.4], 'HandleVisibility','off');
    yline(ax, 0, '--', 'LineWidth',1.2, 'Color',[0.4 0.4 0.4], 'HandleVisibility','off');

    % ---- Axes/labels ----
    xlim(ax,[-0.1 0.8]);
    ylim(ax,yL);
    xticks(ax,0:0.1:max(params.epochTime));
    xlabel(ax,'Time (s)', 'FontName','Arial', 'FontSize',10);
    ylabel(ax,'Amplitude (\muV)', 'FontName','Arial', 'FontSize',10);
    title(ax, sprintf('%s', panelNames{p}), 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    % ---- Legend ----
    legend(ax, [h1 h2], {'Distractor','No distractor'}, ...
        'Box','on', 'FontSize',10, 'Location','northeast');
    % ---- Optional: RT overlay (vertical mean lines + shaded ±1 SD), RT in ms ----
    if showRT && isfield(D,'RT') && ~isempty(D.RT)
        % Colors for RT overlays
        colD  = [0.80 0.25 0.10];   % red   - Distractor
        colND = [0.10 0.65 0.25];   % green - No Distractor
        aFill = 0.12;               % alpha for shaded SD region

        % Split RT by condition (already in ms)
        RT  = double(D.RT(:));
        RTd = RT(dTrials);
        RTn = RT(ndTrials);

        % Means / SDs in ms
        mD  = mean(RTd, 'omitnan'); sD  = std(RTd, 'omitnan');
        mN  = mean(RTn, 'omitnan'); sN  = std(RTn, 'omitnan');

        % Skip if no valid numbers
        if ~(isnan(mD) || isnan(mN))
            % Convert to seconds for x-axis (params.epochTime is in seconds)
            mD_s = mD/1000; sD_s = sD/1000;
            mN_s = mN/1000; sN_s = sN/1000;

            % Clamp SD bands to current x-limits
            xl = xlim(ax); yl = ylim(ax);
            d0 = max(xl(1), mD_s - sD_s); d1 = min(xl(2), mD_s + sD_s);
            n0 = max(xl(1), mN_s - sN_s); n1 = min(xl(2), mN_s + sN_s);

%             % Distractor shaded band (±1 SD)
%             if d1 > d0 && ~isnan(d0) && ~isnan(d1)
%                 patch(ax, [d0 d0 d1 d1], [yl(1) yl(2) yl(2) yl(1)], colD, ...
%                     'FaceAlpha', aFill, 'EdgeColor', 'none', 'HandleVisibility','off');
%             end
%             % No-distractor shaded band (±1 SD)
%             if n1 > n0 && ~isnan(n0) && ~isnan(n1)
%                 patch(ax, [n0 n0 n1 n1], [yl(1) yl(2) yl(2) yl(1)], colND, ...
%                     'FaceAlpha', aFill, 'EdgeColor', 'none', 'HandleVisibility','off');
%             end

            % Vertical mean lines
            xline(ax, mD_s,  '-', 'LineWidth', 1.6, 'Color', colD,  'HandleVisibility','off');
            xline(ax, mN_s,  '-', 'LineWidth', 1.6, 'Color', colND, 'HandleVisibility','off');

            % ND - D difference (ms), shown top-right
            delta_ms = mN - mD;  % (No Distractor) - (Distractor)
            txt = sprintf('ND - D = %.0f ms', delta_ms);
            text(ax, 0.98, 0.05, txt, ...
                'Units','normalized', 'HorizontalAlignment','right', ...
                'VerticalAlignment','bottom', 'FontName','Arial', ...
                'FontSize',9, 'Color',[0.15 0.15 0.15], 'Interpreter','none');
        else
            % Optional: note if a condition is missing
            % warning('Skipping RT overlay: missing valid RTs for one or both conditions in panel %d.', p);
        end
    end

    % ---- Panel label (A/B) ----
    text(ax, -0.08, 1.02, annotations{p}, ...
        'Units','normalized', 'FontName','Arial', 'FontSize',12, 'FontWeight','bold');

    set(ax, 'FontName','Arial', 'FontSize',10, 'LineWidth',1);
    box(ax,'off');
    hold(ax,'off');
end
end
