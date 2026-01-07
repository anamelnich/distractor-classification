function plotERPOffvsOnlineAllD_xDAWN(origData, bestData, params, decoderL, decoderR, decoderN, panelNames, showRT)
% plotERPOffvsOnlineAllD_xDAWN
% Single- or multi-subject version.
%
% If inputs are structs -> single subject.
% If inputs are 1xN cell arrays -> group mode:
%   - apply individual xDAWN (decoderL/R/N) to each subject
%   - compute per-subject D / ND waveforms
%   - average across subjects and plot group waveforms.
%
% D.labels: 0 = no distractor, 1 = distractor RIGHT, 2 = distractor LEFT
% Use weights:
%   labels==1 -> decoderL.spatialFilter.diff (L-R)
%   labels==2 -> decoderR.spatialFilter.diff (R-L)
%   labels==0 -> decoderN.spatialFilter.diff (random sign of R-L)

if nargin < 7 || isempty(panelNames)
    panelNames = {'Offline','Online'};
end
if numel(panelNames) ~= 2
    error('panelNames must be a 1x2 cell array of strings.');
end
if nargin < 8 || isempty(showRT)
    showRT = 0;
end

% Electrode pairs
LeftElec  = {'P1','P3','P5','P7','PO3','PO5','PO7'};
RightElec = {'P2','P4','P6','P8','PO4','PO6','PO8'};

[isL, lIdx] = ismember(LeftElec,  params.chanLabels);
[isR, rIdx] = ismember(RightElec, params.chanLabels);
if ~all(isL) || ~all(isR)
    missing = [LeftElec(~isL), RightElec(~isR)];
    error('Missing required channels: %s', strjoin(missing, ', '));
end

% Optional reproducibility
if isfield(params,'rng_seed') && ~isempty(params.rng_seed)
    rng(params.rng_seed);
end

isGroup = iscell(origData);
T       = numel(params.epochTime);
yL      = [-4 4]; % µV

% =====================================================================
% SINGLE-SUBJECT MODE
% =====================================================================
if ~isGroup
    WL = decoderL.spatialFilter.diff; % left distractor trials
    WR = decoderR.spatialFilter.diff; % right distractor trials
    WN = decoderN.spatialFilter.diff;
    if ~isequal(size(WL),[7 2]) || ~isequal(size(WR),[7 2]) || ~isequal(size(WN),[7 2])
        error('decoderL/R/N.spatialFilter.diff must all be 7x2.');
    end

    datasets    = {origData, bestData};
    annotations = {'A','B'};

    figure('Color','w','Units','inches','Position',[1 1 4.2 6.2]);
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    for p = 1:2
        ax = nexttile; hold(ax,'on');
        D  = datasets{p};

        [waveD, waveND, RTd, RTn] = local_xdawn_subject(D.epochs, params, WL, WR, WN, lIdx, rIdx);

        % Shaded analysis window (example)
        patch(ax,[0.15 0.5 0.5 0.15], [yL(1) yL(1) yL(2) yL(2)], ...
            [0.9 0.9 0.9], 'EdgeColor','none','FaceAlpha',0.5, ...
            'HandleVisibility','off');

        h1 = plot(ax, params.epochTime, waveD,  'LineWidth',2,'Color',params.plotColor{1});
        h2 = plot(ax, params.epochTime, waveND, 'LineWidth',2,'Color',params.plotColor{5});

        xline(ax,0,'--','LineWidth',1.2,'Color',[0.4 0.4 0.4],'HandleVisibility','off');
        yline(ax,0,'--','LineWidth',1.2,'Color',[0.4 0.4 0.4],'HandleVisibility','off');

        xlim(ax,[-0.1 0.8]); ylim(ax,yL);
        xticks(ax,0:0.1:max(params.epochTime));
        xlabel(ax,'Time (s)','FontName','Arial','FontSize',10);
        ylabel(ax,'Amplitude (\muV)','FontName','Arial','FontSize',10);
        title(ax,panelNames{p},'FontName','Arial','FontSize',12,'FontWeight','bold');

        legend(ax,[h1 h2],{'Distractor','No distractor'}, ...
            'Box','on','FontSize',10,'Location','northeast');

        if showRT && ~isempty(RTd) && ~isempty(RTn)
            local_plot_rt_overlay(ax, RTd, RTn);
        end

        text(ax,-0.08,1.02,annotations{p}, ...
            'Units','normalized','FontName','Arial', ...
            'FontSize',12,'FontWeight','bold');

        set(ax,'FontName','Arial','FontSize',10,'LineWidth',1);
        box(ax,'off'); hold(ax,'off');
    end

    return;
end

% =====================================================================
% GROUP MODE: cell arrays
% =====================================================================
nSubj = numel(origData);
if numel(bestData) ~= nSubj || numel(decoderL) ~= nSubj || ...
   numel(decoderR) ~= nSubj || numel(decoderN) ~= nSubj
    error('origData, bestData, decoderL/R/N must all be 1xN cell arrays with same length.');
end

waveD_group  = cell(1,2);
waveND_group = cell(1,2);
RTd_all      = cell(1,2);
RTn_all      = cell(1,2);

datasetsCell = {origData, bestData};

for p = 1:2
    waveD_mat  = nan(T, nSubj);
    waveND_mat = nan(T, nSubj);
    RTd_all{p} = [];
    RTn_all{p} = [];

    for sj = 1:nSubj
        D     = datasetsCell{p}{sj};
        decL  = decoderL{sj};
        decR  = decoderR{sj};
        decN  = decoderN{sj};

        if ~isfield(decL,'spatialFilter') || ~isfield(decR,'spatialFilter') || ~isfield(decN,'spatialFilter')
            warning('Subject %d: missing spatialFilter in one of decoders; skipping.', sj);
            continue;
        end
        WL = decL.spatialFilter.diff;
        WR = decR.spatialFilter.diff;
        WN = decN.spatialFilter.diff;
        if ~isequal(size(WL),[7 2]) || ~isequal(size(WR),[7 2]) || ~isequal(size(WN),[7 2])
            warning('Subject %d: decoderL/R/N.spatialFilter.diff must be 7x2; skipping.', sj);
            continue;
        end

        [waveD, waveND, RTd, RTn] = local_xdawn_subject(D.epochs, params, WL, WR, WN, lIdx, rIdx);

        if numel(waveD) == T,  waveD_mat(:,sj)  = waveD(:);  end
        if numel(waveND) == T, waveND_mat(:,sj) = waveND(:); end

        if ~isempty(RTd), RTd_all{p} = [RTd_all{p}; RTd(:)]; end
        if ~isempty(RTn), RTn_all{p} = [RTn_all{p}; RTn(:)]; end
    end

    waveD_group{p}  = nanmean(waveD_mat,  2);
    waveND_group{p} = nanmean(waveND_mat, 2);
end

% ======================= PLOT GROUP =======================
figure('Color','w','Units','inches','Position',[1 1 4.2 6.2]);
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
annotations = {'A','B'};

for p = 1:2
    ax = nexttile; hold(ax,'on');

    waveD  = waveD_group{p};
    waveND = waveND_group{p};

    patch(ax,[0.15 0.5 0.5 0.15], [yL(1) yL(1) yL(2) yL(2)], ...
        [0.9 0.9 0.9], 'EdgeColor','none','FaceAlpha',0.5, ...
        'HandleVisibility','off');

    h1 = plot(ax, params.epochTime, waveD,  'LineWidth',2,'Color',params.plotColor{1});
    h2 = plot(ax, params.epochTime, waveND, 'LineWidth',2,'Color',params.plotColor{5});

    xline(ax,0,'--','LineWidth',1.2,'Color',[0.4 0.4 0.4],'HandleVisibility','off');
    yline(ax,0,'--','LineWidth',1.2,'Color',[0.4 0.4 0.4],'HandleVisibility','off');

    xlim(ax,[-0.1 0.8]); ylim(ax,yL);
    xticks(ax,0:0.1:max(params.epochTime));
    xlabel(ax,'Time (s)','FontName','Arial','FontSize',10);
    ylabel(ax,'Amplitude (\muV)','FontName','Arial','FontSize',10);
    title(ax,panelNames{p},'FontName','Arial','FontSize',12,'FontWeight','bold');

    legend(ax,[h1 h2],{'Distractor','No distractor'}, ...
        'Box','on','FontSize',10,'Location','northeast');

    if showRT && ~isempty(RTd_all{p}) && ~isempty(RTn_all{p})
        local_plot_rt_overlay(ax, RTd_all{p}, RTn_all{p});
    end

    text(ax,-0.08,1.02,annotations{p}, ...
        'Units','normalized','FontName','Arial', ...
        'FontSize',12,'FontWeight','bold');

    set(ax,'FontName','Arial','FontSize',10,'LineWidth',1);
    box(ax,'off'); hold(ax,'off');
end

end

% =====================================================================
% Helpers
% =====================================================================

function [waveD, waveND, RTd, RTn] = local_xdawn_subject(D, params, WL, WR, WN, lIdx, rIdx)
% Apply xDAWN (WL/WR/WN) to one subject & one dataset.

if size(D.labels,1) > 1, D.labels = D.labels(:)'; end

dTrials  = (D.labels == 1) | (D.labels == 2);
ndTrials = (D.labels == 0);

baseline_idx = find(params.epochTime >= params.baseline_window(1) & ...
                    params.epochTime <= params.baseline_window(2));
baseline = mean(D.data(baseline_idx, :, :), 1);
D.data   = D.data - baseline;

T = size(D.data,1);
N = size(D.data,3);
diffAll_xdawn = zeros(T, N);

for n = 1:N
    lab = D.labels(n);

    Lroi = squeeze(D.data(:, lIdx, n)); % T x 7
    Rroi = squeeze(D.data(:, rIdx, n)); % T x 7
    if isvector(Lroi), Lroi = Lroi(:)'; end
    if isvector(Rroi), Rroi = Rroi(:)'; end

    switch lab
        case 1   % distractor RIGHT -> L-R, WL
            diff7 = Lroi - Rroi;
            W = WR;
        case 2   % distractor LEFT -> R-L, WR
            diff7 = Rroi - Lroi;
            W = WL;
        otherwise % 0: no distractor -> random sign of (R-L), WN
            base = Rroi - Lroi;
            if rand > 0.5, sgn = +1; else, sgn = -1; end
            diff7 = base * sgn;
            W = WN;
    end

    comps = diff7 * W;           % T x 2
    diffAll_xdawn(:,n) = mean(comps,2);
end

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

RTd = [];
RTn = [];
if isfield(D,'RT') && ~isempty(D.RT)
    RT  = double(D.RT(:));
    RTd = RT(dTrials);
    RTn = RT(ndTrials);
end
end

function local_plot_rt_overlay(ax, RTd_ms, RTn_ms)
colD  = [0.80 0.25 0.10];
colND = [0.10 0.65 0.25];

mD  = mean(RTd_ms, 'omitnan');
mN  = mean(RTn_ms, 'omitnan');
if isnan(mD) || isnan(mN), return; end

mD_s = mD/1000;
mN_s = mN/1000;

xline(ax, mD_s, '-', 'LineWidth',1.6,'Color',colD,  'HandleVisibility','off');
xline(ax, mN_s, '-', 'LineWidth',1.6,'Color',colND, 'HandleVisibility','off');

delta_ms = mN - mD;
txt = sprintf('ND - D = %.0f ms', delta_ms);
text(ax, 0.98, 0.05, txt, ...
    'Units','normalized','HorizontalAlignment','right', ...
    'VerticalAlignment','bottom','FontName','Arial', ...
    'FontSize',9,'Color',[0.15 0.15 0.15],'Interpreter','none');
end

