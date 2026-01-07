function plotERPOffvsOnlineAllD_PO7PO8(origData, bestData, params, decoderL, decoderR, decoderN, panelNames, showRT)
% plotERPOffvsOnlineAllD_PO7PO8
% PO7/PO8-based version of plotERPOffvsOnlineAllD_xDAWN.
%
% Call signature kept compatible with xDAWN version:
%   plotERPOffvsOnlineAllD_PO7PO8( ...
%       calibDataCell, ...
%       finalDataCell, ...
%       cfg, ...
%       decLCell, ...
%       decRCell, ...
%       decNCell, ...
%       {'Pre Intervention','Post Intervention'}, ...
%       1);
%
% Inputs:
%   origData : struct OR 1xN cell array of structs with field .epochs
%   bestData : struct OR 1xN cell array of structs with field .epochs
%   params   : struct with fields:
%                .chanLabels
%                .epochTime
%                .baseline_window
%                .plotColor (cell, at least {1} and {5})
%                .rng_seed (optional)
%   decoderL/R/N : kept for compatibility, ignored here
%
% D.labels: 0 = no distractor, 1 = distractor RIGHT, 2 = distractor LEFT
%
% For each trial:
%   label==1 (distractor RIGHT) -> contralateral (PO7) - ipsi (PO8)
%   label==2 (distractor LEFT)  -> contralateral (PO8) - ipsi (PO7)
%   label==0 (no distractor)    -> random sign of (PO8 - PO7) to balance

% ----------------- Handle optional args -----------------
if nargin < 7 || isempty(panelNames)
    panelNames = {'Offline','Online'};
end
if numel(panelNames) ~= 2
    error('panelNames must be a 1x2 cell array of strings.');
end
if nargin < 8 || isempty(showRT)
    showRT = 0;
end

% ----------------- Find PO7 / PO8 -----------------
[isPO7, idxPO7] = ismember('PO7', params.chanLabels);
[isPO8, idxPO8] = ismember('PO8', params.chanLabels);
if ~isPO7 || ~isPO8
    missing = {};
    if ~isPO7, missing{end+1} = 'PO7'; end %#ok<AGROW>
    if ~isPO8, missing{end+1} = 'PO8'; end %#ok<AGROW>
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
    datasets    = {origData, bestData};
    annotations = {'A','B'};

    figure('Color','w','Units','inches','Position',[1 1 4.2 6.2]);
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    for p = 1:2
        ax = nexttile; hold(ax,'on');
        S  = datasets{p};

        if ~isfield(S,'epochs')
            error('Single-subject struct must have field .epochs.');
        end
        E = S.epochs;

        [waveD, waveND, RTd, RTn] = local_PO7PO8_subject(E, params, idxPO7, idxPO8);

        % Shaded analysis window (example: 0.15–0.5 s)
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
if numel(bestData) ~= nSubj
    error('origData and bestData must be 1xN cell arrays with same length.');
end
% decoderL/R/N are ignored, but we can sanity-check size if you want:
if nargin >= 4 && ~isempty(decoderL)
    if numel(decoderL) ~= nSubj || numel(decoderR) ~= nSubj || numel(decoderN) ~= nSubj
        warning('decoderL/R/N lengths do not match origData; they are ignored anyway.');
    end
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
        S = datasetsCell{p}{sj};
        if ~isfield(S,'epochs')
            warning('Subject %d, panel %d missing .epochs; skipping.', sj, p);
            continue;
        end
        E = S.epochs;

        [waveD, waveND, RTd, RTn] = local_PO7PO8_subject(E, params, idxPO7, idxPO8);

        if numel(waveD) == T,  waveD_mat(:,sj)  = waveD(:);  end
        if numel(waveND) == T, waveND_mat(:,sj) = waveND(:); end

        if ~isempty(RTd), RTd_all{p} = [RTd_all{p}; RTd(:)]; end %#ok<AGROW>
        if ~isempty(RTn), RTn_all{p} = [RTn_all{p}; RTn(:)]; end %#ok<AGROW>
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

function [waveD, waveND, RTd, RTn] = local_PO7PO8_subject(E, params, idxPO7, idxPO8)
% Compute PO7/PO8-based Pd-like waveform for one subject & one dataset.
%
% E.data   : [T x C x N]
% E.labels : [1 x N] or [N x 1], with 0/1/2
% E.RT     : [N x 1] or [1 x N] in ms (optional)

if size(E.labels,1) > 1
    labels = E.labels(:)';
else
    labels = E.labels;
end

dTrials  = (labels == 1) | (labels == 2);
ndTrials = (labels == 0);

% Baseline correction
baseline_idx = find(params.epochTime >= params.baseline_window(1) & ...
                    params.epochTime <= params.baseline_window(2));
baseline = mean(E.data(baseline_idx, :, :), 1);
dataBC   = E.data - baseline;

T = size(dataBC,1);
N = size(dataBC,3);
diffAll = zeros(T, N);

for n = 1:N
    lab = labels(n);

    L = squeeze(dataBC(:, idxPO7, n)); % PO7
    R = squeeze(dataBC(:, idxPO8, n)); % PO8
    if isrow(L), L = L'; end
    if isrow(R), R = R'; end

    switch lab
        case 1   % distractor RIGHT -> contralateral (PO7) - ipsi (PO8)
            diff = L - R;
        case 2   % distractor LEFT  -> contralateral (PO8) - ipsi (PO7)
            diff = R - L;
        otherwise % 0: no distractor -> random sign of (R-L)
            base = R - L;
            if rand > 0.5, sgn = +1; else, sgn = -1; end
            diff = base * sgn;
    end

    diffAll(:,n) = diff;
end

if any(dTrials)
    waveD  = mean(diffAll(:, dTrials), 2);
else
    waveD  = zeros(T,1);
end
if any(ndTrials)
    waveND = mean(diffAll(:, ndTrials), 2);
else
    waveND = zeros(T,1);
end

RTd = [];
RTn = [];
if isfield(E,'RT') && ~isempty(E.RT)
    RT  = double(E.RT(:));
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

