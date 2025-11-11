% assumes epoched data and decoder R and L is loaded 

sessions = {'decoding1','decoding2','decoding3','decoding4','decoding5','training2'};

featuresR_orig = compute_features(decoderR,data.training1.epochs.data);
featuresL_orig = compute_features(decoderL,data.training1.epochs.data);
num_features = length(featuresL_orig(:,1));

%% Kolmogorov–Smirnov test
KSstats = [];
for si = 1:numel(sessions)
    sf = sessions{si};
    if ~isfield(data, sf)
        warning('Missing %s in data. Skipping.', sf);
        continue;
    end
    epochs = data.(sf).epochs.data;

    featuresR_new = compute_features(decoderR,epochs);
    featuresL_new = compute_features(decoderL,epochs);
    
    KSstats.(sf).hR      = nan(1, num_features);
    KSstats.(sf).pR      = nan(1, num_features);
    KSstats.(sf).ksstatR = nan(1, num_features);
    KSstats.(sf).hL      = nan(1, num_features);
    KSstats.(sf).pL      = nan(1, num_features);
    KSstats.(sf).ksstatL = nan(1, num_features);

    for feature = 1:num_features
        % Right decoder
        [hR, pR, ksR] = kstest2(featuresR_orig(feature, :), featuresR_new(feature, :));
        KSstats.(sf).hR(feature)      = hR;
        KSstats.(sf).pR(feature)      = pR;
        KSstats.(sf).ksstatR(feature) = ksR;

        % Left decoder
        [hL, pL, ksL] = kstest2(featuresL_orig(feature, :), featuresL_new(feature, :));
        KSstats.(sf).hL(feature)      = hL;
        KSstats.(sf).pL(feature)      = pL;
        KSstats.(sf).ksstatL(feature) = ksL;
    end

end

%% === Tiled histograms per decoder (R and L), one tile per session, with % significant ===
% Requires: KSstats.(session).ksstatR / ksstatL / pR / pL and 'sessions' cell array.
% Significance uses Benjamini–Hochberg FDR (q<0.05) computed per session & decoder.

% ----- Style -----
burnt  = [191, 87,  0] / 255;   % UT burnt orange (R)
grayL  = [160,165,170]/ 255;    % neutral gray (L)
slate  = [ 40, 44, 52] / 255;   % dark gray for axes/text
thr    = [0.10 0.20 0.30];      % reference lines

% --- Consistent binning across all sessions/decoders ---
allD = [];
for si = 1:numel(sessions)
    sf = sessions{si};
    if ~isfield(KSstats, sf), continue; end
    allD = [allD; KSstats.(sf).ksstatR(:); KSstats.(sf).ksstatL(:)]; 
end
allD = allD(~isnan(allD));
if isempty(allD)
    warning('No KS distances found. Aborting plot.');
    return;
end
xmax  = min(1, max(0.05, ceil(max(allD)*20)/20)); % round up to nearest .05
edges = 0:0.05:max(0.6, xmax);                    % consistent bins

% --- Helper to compute max y-limit (probability) for shared axes ---
getMaxProb = @(vals) max( histcounts(vals(~isnan(vals)), edges, 'Normalization','probability') );

% ========== RIGHT DECODER FIGURE ==========
% Pre-compute max prob across all R sessions
maxProbR = 0;
for si = 1:numel(sessions)
    sf = sessions{si}; if ~isfield(KSstats, sf), continue; end
    maxProbR = max(maxProbR, getMaxProb(KSstats.(sf).ksstatR));
end
maxProbR = maxProbR * 1.10; % pad bysf 10%

nS = numel(sessions);
nCols = ceil(sqrt(nS));
nRows = ceil(nS / nCols);

figR = figure('Color','w','Units','inches','Position',[1 1 8.5 6.5]);
tlR  = tiledlayout(figR, nRows, nCols, 'Padding','compact', 'TileSpacing','compact');
title(tlR, 'KS distance of features — Right decoder (vs calibration)', ...
    'FontName','Arial','FontSize',14,'FontWeight','bold','Color',slate);

for si = 1:nS
    sf = sessions{si};
    ax = nexttile; hold(ax, 'on');
    if ~isfield(KSstats, sf)
        title(ax, sprintf('%s (missing)', sf), 'FontName','Arial','Color',[0.5 0.5 0.5]);
        axis(ax, 'off'); continue;
    end

    % Data
    D  = KSstats.(sf).ksstatR(:);
    p  = KSstats.(sf).pR(:);
    D  = D(~isnan(D));
    p  = p(~isnan(p));

    % FDR (BH) per session for R
    q = bh_fdr(p);
    pctSig = 100 * mean(q < 0.05);

    % Histogram
    histogram(ax, D, edges, 'Normalization','probability', ...
        'FaceColor', burnt, 'EdgeColor', burnt*0.7, 'LineWidth', 0.75, 'FaceAlpha', 0.9);

    % Reference lines
    for k = 1:numel(thr)
        xline(ax, thr(k), '--', 'LineWidth', 1.1, 'Color', [0 0 0 0.35]);
    end

    % Labels
    title(ax, sprintf('%s vs calibration', sf), ...
        'FontName','Arial','FontSize',11.5,'Color',slate);
    if si > (nRows-1)*nCols, xlabel(ax, 'KS distance', 'FontName','Arial','FontSize',10.5,'Color',slate); end
    if mod(si-1,nCols)==0, ylabel(ax, 'Probability', 'FontName','Arial','FontSize',10.5,'Color',slate); end

    % Axes style
    ax.LineWidth = 1.1; ax.FontName = 'Arial'; ax.FontSize = 10.5;
    ax.XColor = slate; ax.YColor = slate; box(ax, 'on'); set(ax,'Layer','top');
    xlim(ax, [edges(1) edges(end)]); ylim(ax, [0 maxProbR]);
    grid(ax, 'on'); ax.XGrid = 'off'; ax.YGrid = 'on'; ax.GridColor = [0 0 0 0.08];

    % --- Annotation: % significant (FDR q<0.05) ---
    yl = ylim(ax);
    txt = sprintf('q<0.05: %.0f%%', pctSig);
    text(ax, edges(1) + 0.55*(edges(end)-edges(1)), yl(2)*0.92, txt, ...
        'FontName','Arial','FontSize',10.5,'Color',slate, 'FontWeight','bold');
end

% ========== LEFT DECODER FIGURE ==========
% Pre-compute max prob across all L sessions
maxProbL = 0;
for si = 1:numel(sessions)
    sf = sessions{si}; if ~isfield(KSstats, sf), continue; end
    maxProbL = max(maxProbL, getMaxProb(KSstats.(sf).ksstatL));
end
maxProbL = maxProbL * 1.10; % pad by 10%

figL = figure('Color','w','Units','inches','Position',[1 1 8.5 6.5]);
tlL  = tiledlayout(figL, nRows, nCols, 'Padding','compact', 'TileSpacing','compact');
title(tlL, 'KS distance of features — Left decoder (vs calibration)', ...
    'FontName','Arial','FontSize',14,'FontWeight','bold','Color',slate);

for si = 1:nS
    sf = sessions{si};
    ax = nexttile; hold(ax, 'on');
    if ~isfield(KSstats, sf)
        title(ax, sprintf('%s (missing)', sf), 'FontName','Arial','Color',[0.5 0.5 0.5]);
        axis(ax, 'off'); continue;
    end

    % Data
    D  = KSstats.(sf).ksstatL(:);
    p  = KSstats.(sf).pL(:);
    D  = D(~isnan(D));
    p  = p(~isnan(p));

    % FDR (BH) per session for L
    q = bh_fdr(p);
    pctSig = 100 * mean(q < 0.05);

    % Histogram
    histogram(ax, D, edges, 'Normalization','probability', ...
        'FaceColor', grayL, 'EdgeColor', grayL*0.6, 'LineWidth', 0.75, 'FaceAlpha', 0.9);

    % Reference lines
    for k = 1:numel(thr)
        xline(ax, thr(k), '--', 'LineWidth', 1.1, 'Color', [0 0 0 0.35]);
    end

    % Labels
    title(ax, sprintf('%s vs calibration', sf), ...
        'FontName','Arial','FontSize',11.5,'Color',slate);
    if si > (nRows-1)*nCols, xlabel(ax, 'KS distance', 'FontName','Arial','FontSize',10.5,'Color',slate); end
    if mod(si-1,nCols)==0, ylabel(ax, 'Probability', 'FontName','Arial','FontSize',10.5,'Color',slate); end

    % Axes style
    ax.LineWidth = 1.1; ax.FontName = 'Arial'; ax.FontSize = 10.5;
    ax.XColor = slate; ax.YColor = slate; box(ax, 'on'); set(ax,'Layer','top');
    xlim(ax, [edges(1) edges(end)]); ylim(ax, [0 maxProbL]);
    grid(ax, 'on'); ax.XGrid = 'off'; ax.YGrid = 'on'; ax.GridColor = [0 0 0 0.08];

    % --- Annotation: % significant (FDR q<0.05) ---
    yl = ylim(ax);
    txt = sprintf('q<0.05: %.0f%%', pctSig);
    text(ax, edges(1) + 0.55*(edges(end)-edges(1)), yl(2)*0.92, txt, ...
        'FontName','Arial','FontSize',10.5,'Color',slate, 'FontWeight','bold');
end

%% Sequential KS distances
%% ---------- Setup ----------
sessions_seq = {'decoding1','decoding2','decoding3','decoding4','decoding5'};
sessions_all = [sessions_seq, {'training2'}];   % for computing features once

% Baseline/orig features (from training1)
featuresR_orig = compute_features(decoderR, data.training1.epochs.data); % [nFeatures x nTrials]
featuresL_orig = compute_features(decoderL, data.training1.epochs.data);
num_features   = size(featuresR_orig,1);

% Normalize orientation to [features x trials]
nf = num_features;

feat.R = struct();  % feat.R.(session) = [nFeatures x nTrials]
feat.L = struct();

for si = 1:numel(sessions_all)
    sf = sessions_all{si};
    if ~isfield(data, sf)
        warning('Missing %s in data. Skipping feature computation.', sf);
        continue;
    end
    epochs = data.(sf).epochs.data;
    Fr = compute_features(decoderR, epochs);
    Fl = compute_features(decoderL, epochs);
    feat.R.(sf) = Fr;
    feat.L.(sf) = Fl;
end

% Add names for "orig" baseline
feat.R.orig = featuresR_orig;
feat.L.orig = featuresL_orig;

% Sequential pairs + final orig vs training2
pairs = [
    "orig"       , "decoding1";
    "decoding1"  , "decoding2";
    "decoding2"  , "decoding3";
    "decoding3"  , "decoding4";
    "decoding4"  , "decoding5";
    "orig"       , "training2"
];

KSstats = struct();

for k = 1:size(pairs,1)
    A = char(pairs(k,1));
    B = char(pairs(k,2));
    tag = sprintf('%s_vs_%s', A, B);

    % Guard: ensure both sets exist
    if ~isfield(feat.R, A) || ~isfield(feat.R, B) || isempty(feat.R.(A)) || isempty(feat.R.(B))
        warning('Missing features for pair %s; skipping.', tag);
        continue;
    end

    % Preallocate
    KSstats.(tag).hR      = false(1, nf);
    KSstats.(tag).pR      = nan(1, nf);
    KSstats.(tag).ksstatR = nan(1, nf);
    KSstats.(tag).hL      = false(1, nf);
    KSstats.(tag).pL      = nan(1, nf);
    KSstats.(tag).ksstatL = nan(1, nf);

    FR_A = feat.R.(A);  FR_B = feat.R.(B);  % [nf x nTrials]
    FL_A = feat.L.(A);  FL_B = feat.L.(B);

    for f = 1:nf
        % Right decoder
        xa = FR_A(f, :); xb = FR_B(f, :);
        xa = xa(~isnan(xa)); xb = xb(~isnan(xb));      % drop NaNs if any
        if ~isempty(xa) && ~isempty(xb)
            [hR, pR, ksR] = kstest2(xa, xb);
            KSstats.(tag).hR(f)      = logical(hR);
            KSstats.(tag).pR(f)      = pR;
            KSstats.(tag).ksstatR(f) = ksR;
        end

        % Left decoder
        ya = FL_A(f, :); yb = FL_B(f, :);
        ya = ya(~isnan(ya)); yb = yb(~isnan(yb));
        if ~isempty(ya) && ~isempty(yb)
            [hL, pL, ksL] = kstest2(ya, yb);
            KSstats.(tag).hL(f)      = logical(hL);
            KSstats.(tag).pL(f)      = pL;
            KSstats.(tag).ksstatL(f) = ksL;
        end
    end

    % Optional: multiple-comparison correction (per comparison)
    % [~, ~, ~, qR] = fdr_bh(KSstats.(tag).pR);  % if you have an FDR function
    % [~, ~, ~, qL] = fdr_bh(KSstats.(tag).pL);
    % KSstats.(tag).qR = qR; KSstats.(tag).qL = qL;
end
%% ===== Tiled histograms for sequential KS comparisons (R & L) =====
% Expects:
%   pairs = [
%       "orig","decoding1";
%       "decoding1","decoding2";
%       "decoding2","decoding3";
%       "decoding3","decoding4";
%       "decoding4","decoding5";
%       "orig","training2"
%   ];
%   KSstats.<A>_vs_<B>.ksstatR / pR / ksstatL / pL

% ---------- Style ----------
burnt  = [191, 87,  0]/255;   % UT burnt orange (R)
grayL  = [160,165,170]/255;   % neutral gray (L)
slate  = [ 40, 44, 52]/255;   % dark gray for axes/text
thr    = [0.10 0.20 0.30];    % reference KS thresholds

% ---------- Build tag list from pairs ----------
pairTags = strings(size(pairs,1),1);
for i = 1:size(pairs,1)
    pairTags(i) = sprintf('%s_vs_%s', pairs(i,1), pairs(i,2));
end

% ---------- Collect all distances to make consistent bins ----------
allD = [];
for i = 1:numel(pairTags)
    tag = char(pairTags(i));
    if ~isfield(KSstats, tag), continue; end
    allD = [allD; KSstats.(tag).ksstatR(:); KSstats.(tag).ksstatL(:)];
end
allD = allD(~isnan(allD));
if isempty(allD)
    warning('No KS distances found. Aborting plot.');
    return;
end

xmax  = min(1, max(0.05, ceil(max(allD)*20)/20));  % round up to nearest .05
edges = 0:0.05:max(0.6, xmax);

% Helper for max prob across panels
getMaxProb = @(vals) max( histcounts(vals(~isnan(vals)), edges, 'Normalization','probability') );

% Layout
nP = numel(pairTags);
nCols = ceil(sqrt(nP));
nRows = ceil(nP / nCols);

% ========== RIGHT DECODER ==========
maxProbR = 0;
for i = 1:nP
    tag = char(pairTags(i)); if ~isfield(KSstats, tag), continue; end
    maxProbR = max(maxProbR, getMaxProb(KSstats.(tag).ksstatR));
end
maxProbR = maxProbR * 1.10;  % 10% headroom
maxProbL = 0.9;

figR = figure('Color','w','Units','inches','Position',[1 1 8.5 6.5]);
tlR  = tiledlayout(figR, nRows, nCols, 'Padding','compact', 'TileSpacing','compact');
title(tlR, 'KS distance of features — Right decoder (sequential pairs)', ...
    'FontName','Arial','FontSize',14,'FontWeight','bold','Color',slate);

for i = 1:nP
    tag = char(pairTags(i));
    ax = nexttile; hold(ax,'on');

    if ~isfield(KSstats, tag)
        title(ax, sprintf('%s (missing)', strrep(tag,'_','\_')), 'FontName','Arial','Color',[0.5 0.5 0.5]);
        axis(ax,'off'); continue;
    end

    D = KSstats.(tag).ksstatR(:);
    p = KSstats.(tag).pR(:);
    D = D(~isnan(D)); p = p(~isnan(p));

    % FDR (BH) per panel
    q = bh_fdr(p);
    pctSig = 100 * mean(q < 0.05);

    % Histogram
    histogram(ax, D, edges, 'Normalization','probability', ...
        'FaceColor', burnt, 'EdgeColor', burnt*0.7, 'LineWidth', 0.75, 'FaceAlpha', 0.9);

    % Reference lines
    for k = 1:numel(thr)
        xline(ax, thr(k), '--', 'LineWidth', 1.1, 'Color', [0 0 0 0.35]);
    end

    % Labels
    % Turn "A_vs_B" into "A → B"
    title(ax, strrep(strrep(tag,'_vs_',' vs '),'_','\_'), ...
    'FontName','Arial','FontSize',11.5,'Color',slate);
    if i > (nRows-1)*nCols
        xlabel(ax, 'KS distance', 'FontName','Arial','FontSize',10.5,'Color',slate);
    end
    if mod(i-1,nCols)==0
        ylabel(ax, 'Probability', 'FontName','Arial','FontSize',10.5,'Color',slate);
    end

    % Axes
    ax.LineWidth = 1.1; ax.FontName = 'Arial'; ax.FontSize = 10.5;
    ax.XColor = slate; ax.YColor = slate; box(ax,'on'); set(ax,'Layer','top');
    xlim(ax, [edges(1) edges(end)]); ylim(ax, [0 maxProbR]);
    grid(ax,'on'); ax.XGrid='off'; ax.YGrid='on'; ax.GridColor=[0 0 0 0.08];

    % Annotation: % significant
    yl = ylim(ax);
    txt = sprintf('q<0.05: %.0f%%', pctSig);
    text(ax, edges(1) + 0.55*(edges(end)-edges(1)), yl(2)*0.92, txt, ...
        'FontName','Arial','FontSize',10.5,'Color',slate,'FontWeight','bold');
end

% ========== LEFT DECODER ==========
maxProbL = 0;
for i = 1:nP
    tag = char(pairTags(i)); if ~isfield(KSstats, tag), continue; end
    maxProbL = max(maxProbL, getMaxProb(KSstats.(tag).ksstatL));
end
maxProbL = maxProbL * 1.10;
maxProbL = 0.9;

figL = figure('Color','w','Units','inches','Position',[1 1 8.5 6.5]);
tlL  = tiledlayout(figL, nRows, nCols, 'Padding','compact', 'TileSpacing','compact');
title(tlL, 'KS distance of features — Left decoder (sequential pairs)', ...
    'FontName','Arial','FontSize',14,'FontWeight','bold','Color',slate);

for i = 1:nP
    tag = char(pairTags(i));
    ax = nexttile; hold(ax,'on');

    if ~isfield(KSstats, tag)
        title(ax, sprintf('%s (missing)', strrep(tag,'_','\_')), 'FontName','Arial','Color',[0.5 0.5 0.5]);
        axis(ax,'off'); continue;
    end

    D = KSstats.(tag).ksstatL(:);
    p = KSstats.(tag).pL(:);
    D = D(~isnan(D)); p = p(~isnan(p));

    q = bh_fdr(p);
    pctSig = 100 * mean(q < 0.05);

    histogram(ax, D, edges, 'Normalization','probability', ...
        'FaceColor', grayL, 'EdgeColor', grayL*0.6, 'LineWidth', 0.75, 'FaceAlpha', 0.9);

    for k = 1:numel(thr)
        xline(ax, thr(k), '--', 'LineWidth', 1.1, 'Color', [0 0 0 0.35]);
    end

    title(ax, strrep(strrep(tag,'_vs_',' vs '),'_','\_'), ...
    'FontName','Arial','FontSize',11.5,'Color',slate);
    if i > (nRows-1)*nCols
        xlabel(ax, 'KS distance', 'FontName','Arial','FontSize',10.5,'Color',slate);
    end
    if mod(i-1,nCols)==0
        ylabel(ax, 'Probability', 'FontName','Arial','FontSize',10.5,'Color',slate);
    end

    ax.LineWidth = 1.1; ax.FontName = 'Arial'; ax.FontSize = 10.5;
    ax.XColor = slate; ax.YColor = slate; box(ax,'on'); set(ax,'Layer','top');
    xlim(ax, [edges(1) edges(end)]); ylim(ax, [0 maxProbL]);
    grid(ax,'on'); ax.XGrid='off'; ax.YGrid='on'; ax.GridColor=[0 0 0 0.08];

    yl = ylim(ax);
    txt = sprintf('q<0.05: %.0f%%', pctSig);
    text(ax, edges(1) + 0.55*(edges(end)-edges(1)), yl(2)*0.92, txt, ...
        'FontName','Arial','FontSize',10.5,'Color',slate,'FontWeight','bold');
end



%% ---------- Helper: Benjamini–Hochberg FDR ----------
function q = bh_fdr(p)
    p = p(:);
    n = numel(p);
    [ps, idx] = sort(p);
    qtmp = ps .* (n ./ (1:n)');
    % enforce monotonicity
    for i = n-1:-1:1
        qtmp(i) = min(qtmp(i), qtmp(i+1));
    end
    q = nan(n,1);
    q(idx) = min(qtmp, 1);
end

