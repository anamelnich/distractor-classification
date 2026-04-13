function OUT = analyze_pd_r2_groups(expSubjects, ctrlSubjects, cacheDir, chanlocs, roiIdx)
% analyze_pd_r2_groups
%
% Inputs:
%   expSubjects : cell array of experimental subject IDs
%   ctrlSubjects: cell array of control subject IDs
%   cacheDir    : folder containing subject cache files
%   chanlocs    : EEGLAB chanlocs struct
%   roiIdx      : vector of electrode indices for ROI
%
% Example:
%   load chanlocs64.mat
%   roiIdx = [24,25,26,27,28,29,50,51,52,53,54,55,56,57,62,63];
%   OUT = analyze_pd_r2_groups(expSubjects, ctrlSubjects, './cache', chanlocs, roiIdx);

if nargin < 5 || isempty(roiIdx)
    roiIdx = [24,25,26,27,28,29,50,51,52,53,54,55,56,57,62,63];
end

% ----------------------------
% Load per-subject electrode maps
% ----------------------------
EXP  = load_group_subject_maps(expSubjects,  cacheDir, chanlocs);
CTRL = load_group_subject_maps(ctrlSubjects, cacheDir, chanlocs);

OUT = struct();
OUT.exp  = EXP;
OUT.ctrl = CTRL;
OUT.roiIdx = roiIdx;

% ----------------------------
% ROI subject-level summaries
% ----------------------------
exp_pre_roi   = mean(EXP.MAV_pre(roiIdx,:),  1, 'omitnan')';
exp_post_roi  = mean(EXP.MAV_post(roiIdx,:), 1, 'omitnan')';
ctrl_pre_roi  = mean(CTRL.MAV_pre(roiIdx,:),  1, 'omitnan')';
ctrl_post_roi = mean(CTRL.MAV_post(roiIdx,:), 1, 'omitnan')';

OUT.roi.pre_exp   = exp_pre_roi;
OUT.roi.post_exp  = exp_post_roi;
OUT.roi.pre_ctrl  = ctrl_pre_roi;
OUT.roi.post_ctrl = ctrl_post_roi;

% change-score test (equivalent to Group x Time interaction)
delta_exp  = exp_post_roi  - exp_pre_roi;
delta_ctrl = ctrl_post_roi - ctrl_pre_roi;

[~, p_roi, ~, stats_roi] = ttest2(delta_exp, delta_ctrl);
d_roi = cohend(delta_exp, delta_ctrl);

OUT.roi.delta_exp  = delta_exp;
OUT.roi.delta_ctrl = delta_ctrl;
OUT.roi.change_ttest.p = p_roi;
OUT.roi.change_ttest.t = stats_roi.tstat;
OUT.roi.change_ttest.df = stats_roi.df;
OUT.roi.change_ttest.d = d_roi;

% mixed ANOVA via fitrm
T = table( ...
    categorical([repmat("exp",numel(exp_pre_roi),1); repmat("ctrl",numel(ctrl_pre_roi),1)]), ...
    [exp_pre_roi;  ctrl_pre_roi], ...
    [exp_post_roi; ctrl_post_roi], ...
    'VariableNames', {'Group','Pre','Post'});

Meas = table(categorical(["Pre";"Post"]), 'VariableNames', {'Time'});
rm = fitrm(T, 'Pre-Post ~ Group', 'WithinDesign', Meas);
ranova_tbl = ranova(rm, 'WithinModel', 'Time');

OUT.roi.anova = ranova_tbl;
OUT.roi.eta_p2.Time = compute_partial_eta(ranova_tbl, 'Time');
OUT.roi.eta_p2.GroupTime = compute_partial_eta(ranova_tbl, 'Group:Time');

fprintf('\n=== ROI mixed ANOVA (Group x Time) ===\n');
disp(ranova_tbl)
fprintf('ROI change-score test: t(%d)=%.3f, p=%.4f, d=%.3f\n', ...
    stats_roi.df, stats_roi.tstat, p_roi, d_roi);
fprintf('Partial eta^2: Time = %.3f, Group×Time = %.3f\n', ...
    OUT.roi.eta_p2.Time, OUT.roi.eta_p2.GroupTime);

% ----------------------------
% Exploratory electrode-wise statistics
% ----------------------------
% Compare post-pre change between groups at each electrode
exp_delta  = EXP.MAV_post  - EXP.MAV_pre;   % chan x subj
ctrl_delta = CTRL.MAV_post - CTRL.MAV_pre;  % chan x subj

nChan = size(exp_delta,1);
p_elec = nan(nChan,1);
t_elec = nan(nChan,1);

for ch = 1:nChan
    x = exp_delta(ch,:)';
    y = ctrl_delta(ch,:)';

    % remove NaNs within each group separately
    x = x(~isnan(x));
    y = y(~isnan(y));

    if numel(x) >= 2 && numel(y) >= 2
        [~, ptmp, ~, stmp] = ttest2(x, y);
        p_elec(ch) = ptmp;
        t_elec(ch) = stmp.tstat;
    end
end

[h_fdr, p_fdr] = fdr_bh(p_elec, 0.05);

OUT.electrode.delta_exp = exp_delta;
OUT.electrode.delta_ctrl = ctrl_delta;
OUT.electrode.p = p_elec;
OUT.electrode.t = t_elec;
OUT.electrode.h_fdr = h_fdr;
OUT.electrode.p_fdr = p_fdr;

% ----------------------------
% Group means for plotting
% ----------------------------
OUT.mean.exp_pre   = mean(EXP.MAV_pre,  2, 'omitnan');
OUT.mean.exp_post  = mean(EXP.MAV_post, 2, 'omitnan');
OUT.mean.ctrl_pre  = mean(CTRL.MAV_pre,  2, 'omitnan');
OUT.mean.ctrl_post = mean(CTRL.MAV_post, 2, 'omitnan');

OUT.mean.exp_delta  = OUT.mean.exp_post  - OUT.mean.exp_pre;
OUT.mean.ctrl_delta = OUT.mean.ctrl_post - OUT.mean.ctrl_pre;

% ----------------------------
% Plot 1: ROI pre/post
% ----------------------------
plot_roi_prepost(exp_pre_roi, exp_post_roi, ctrl_pre_roi, ctrl_post_roi);

% ----------------------------
% Plot 2: ROI change
% ----------------------------
plot_roi_change(delta_exp, delta_ctrl);

% ----------------------------
% Plot 3: Topographies
% ----------------------------
plot_group_topos(OUT.mean, EXP.chanlocsOrd);

% ----------------------------
% Plot 4: Electrode-wise t-map with FDR-marked sig electrodes
% ----------------------------
plot_tmap_with_sig(t_elec, h_fdr, EXP.chanlocsOrd);

end

% =========================================================================
function G = load_group_subject_maps(subjectIDs, cacheDir, chanlocs)

MAP_MAV_PRE  = [];
MAP_MAV_POST = [];
usedIDs = {};
chanlocsOrdRef = [];
idxFoundRef = [];

for i = 1:numel(subjectIDs)
    sid = subjectIDs{i};

    % flexible matching
    f = dir(fullfile(cacheDir, sprintf('%s*cache*.mat', sid)));
    if isempty(f)
        warning('No cache file found for %s in %s', sid, cacheDir);
        continue
    end

    filePath = fullfile(f(1).folder, f(1).name);

    try
        C = load(filePath);

        if ~isfield(C,'training1') || ~isfield(C,'training2') || ~isfield(C,'cfg')
            warning('Skipping %s: missing training1/training2/cfg', sid);
            continue
        end
        if ~isfield(C.training1,'epochs') || isempty(C.training1.epochs) || ...
           ~isfield(C.training2,'epochs') || isempty(C.training2.epochs)
            warning('Skipping %s: missing epochs', sid);
            continue
        end

        outPre  = computePdR2_pairDiffTopos(C.training1, C.cfg, chanlocs);
        outPost = computePdR2_pairDiffTopos(C.training2, C.cfg, chanlocs);

        if ~isfield(outPre,'map_combined_MAV') || ~isfield(outPost,'map_combined_MAV')
            warning('Skipping %s: missing map_combined_MAV', sid);
            continue
        end

        vM_pre  = outPre.map_combined_MAV(:);
        vM_post = outPost.map_combined_MAV(:);

        if isempty(chanlocsOrdRef)
            [chanlocsOrdRef, idxFoundRef] = local_reorder_chanlocs_by_labels(chanlocs, C.cfg.chanLabels);
        end

        MAP_MAV_PRE(:,end+1)  = vM_pre;   %#ok<AGROW>
        MAP_MAV_POST(:,end+1) = vM_post;  %#ok<AGROW>
        usedIDs{end+1} = sid; %#ok<AGROW>

    catch ME
        warning('Skipping %s: %s', sid, ME.message);
        continue
    end
end

G = struct();
G.MAV_pre = MAP_MAV_PRE;    % chan x subj
G.MAV_post = MAP_MAV_POST;  % chan x subj
G.usedIDs = usedIDs;
G.chanlocsOrd = chanlocsOrdRef;
G.idxFound = idxFoundRef;
end

% =========================================================================
function plot_roi_prepost(pre_exp, post_exp, pre_ctrl, post_ctrl)

figure('Color','w','Units','pixels','Position',[100 100 560 460]); 
hold on
x = [1 2];

col_exp  = [0.20 0.55 0.85];
col_ctrl = [0.85 0.30 0.30];
light_exp  = [0.75 0.85 0.95];
light_ctrl = [0.95 0.80 0.80];

for i = 1:numel(pre_exp)
    if ~isnan(pre_exp(i)) && ~isnan(post_exp(i))
        plot(x, [pre_exp(i), post_exp(i)], '-', 'Color', light_exp, 'LineWidth', 1);
    end
end
for i = 1:numel(pre_ctrl)
    if ~isnan(pre_ctrl(i)) && ~isnan(post_ctrl(i))
        plot(x, [pre_ctrl(i), post_ctrl(i)], '-', 'Color', light_ctrl, 'LineWidth', 1);
    end
end

m_exp  = [mean(pre_exp,'omitnan'), mean(post_exp,'omitnan')];
m_ctrl = [mean(pre_ctrl,'omitnan'), mean(post_ctrl,'omitnan')];
s_exp  = [sem1(pre_exp), sem1(post_exp)];
s_ctrl = [sem1(pre_ctrl), sem1(post_ctrl)];

errorbar(x, m_exp, s_exp, 'Color', col_exp, 'LineWidth', 2, 'CapSize', 0);
errorbar(x, m_ctrl, s_ctrl, 'Color', col_ctrl, 'LineWidth', 2, 'CapSize', 0);

h1 = plot(x, m_exp, '-o', 'Color', col_exp, 'MarkerFaceColor', col_exp, ...
    'MarkerEdgeColor', col_exp, 'LineWidth', 2.5, 'MarkerSize', 6);
h2 = plot(x, m_ctrl, '-o', 'Color', col_ctrl, 'MarkerFaceColor', col_ctrl, ...
    'MarkerEdgeColor', col_ctrl, 'LineWidth', 2.5, 'MarkerSize', 6);

xlim([0.7 2.3])
xticks([1 2]); xticklabels({'Pre','Post'})
ylabel('R^2 (Posterior ROI)')
title('Posterior ROI Neural Discriminability')
legend([h1 h2], {'Experimental','Control'}, 'Location','best', 'Box','off')
set(gca, 'FontName','Arial', 'FontSize',13, 'LineWidth',1.2, 'TickDir','out', 'Box','off')
end

% =========================================================================
function plot_roi_change(delta_exp, delta_ctrl)

figure('Color','w','Units','pixels','Position',[120 120 420 420]); 
hold on

col_exp  = [0.20 0.55 0.85];
col_ctrl = [0.85 0.30 0.30];
light_exp  = [0.75 0.85 0.95];
light_ctrl = [0.95 0.80 0.80];

scatter(ones(sum(~isnan(delta_exp)),1), delta_exp(~isnan(delta_exp)), 36, ...
    'MarkerFaceColor', light_exp, 'MarkerEdgeColor', col_exp, ...
    'LineWidth', 0.9, 'jitter','on', 'jitterAmount',0.08);

scatter(2*ones(sum(~isnan(delta_ctrl)),1), delta_ctrl(~isnan(delta_ctrl)), 36, ...
    'MarkerFaceColor', light_ctrl, 'MarkerEdgeColor', col_ctrl, ...
    'LineWidth', 0.9, 'jitter','on', 'jitterAmount',0.08);

plot(1, mean(delta_exp,'omitnan'), 'o', 'Color', col_exp, ...
    'MarkerFaceColor', col_exp, 'MarkerSize', 9, 'LineWidth',1.5);
plot(2, mean(delta_ctrl,'omitnan'), 'o', 'Color', col_ctrl, ...
    'MarkerFaceColor', col_ctrl, 'MarkerSize', 9, 'LineWidth',1.5);

yline(0,'--','Color',[0.6 0.6 0.6],'LineWidth',1)
xlim([0.5 2.5])
xticks([1 2]); xticklabels({'Experimental','Control'})
ylabel('\Delta R^2 (Post - Pre)')
title('Posterior ROI Change in Discriminability')
set(gca, 'FontName','Arial', 'FontSize',13, 'LineWidth',1.2, 'TickDir','out', 'Box','off')
end

% =========================================================================
function plot_group_topos(M, chanlocsOrd)

plotArgs = {'electrodes','off','plotrad',0.7,'headrad',0.50};
clim = max(abs([M.exp_pre; M.exp_post; M.ctrl_pre; M.ctrl_post]),[],'omitnan');
dclim = max(abs([M.exp_delta; M.ctrl_delta]),[],'omitnan');

figure('Color','w','Units','pixels','Position',[100 100 960 620]);

subplot(2,3,1)
topoplot(M.exp_pre, chanlocsOrd, plotArgs{:}); colorbar; caxis([0 clim]); title('EXP Pre (R^2)')

subplot(2,3,2)
topoplot(M.exp_post, chanlocsOrd, plotArgs{:}); colorbar; caxis([0 clim]); title('EXP Post (R^2)')

subplot(2,3,3)
topoplot(M.exp_delta, chanlocsOrd, plotArgs{:}); colorbar; caxis([-dclim dclim]); title('EXP \DeltaR^2')

subplot(2,3,4)
topoplot(M.ctrl_pre, chanlocsOrd, plotArgs{:}); colorbar; caxis([0 clim]); title('CTRL Pre (R^2)')

subplot(2,3,5)
topoplot(M.ctrl_post, chanlocsOrd, plotArgs{:}); colorbar; caxis([0 clim]); title('CTRL Post (R^2)')

subplot(2,3,6)
topoplot(M.ctrl_delta, chanlocsOrd, plotArgs{:}); colorbar; caxis([-dclim dclim]); title('CTRL \DeltaR^2')
colormap(parula)
end

% =========================================================================
function plot_tmap_with_sig(tmap, h_fdr, chanlocsOrd)

figure('Color','w','Units','pixels','Position',[140 140 420 380]);
topoplot(tmap, chanlocsOrd, 'electrodes','off','plotrad',0.7,'headrad',0.50);
colorbar
title('Experimental vs Control \DeltaR^2 t-map')
set(gca, 'FontName','Arial', 'FontSize',12)

sigIdx = find(h_fdr == 1);
if ~isempty(sigIdx)
    hold on
    for k = 1:numel(sigIdx)
        ch = sigIdx(k);
        plot(chanlocsOrd(ch).X, chanlocsOrd(ch).Y, 'ko', ...
            'MarkerSize', 6, 'LineWidth', 1.2);
    end
end
end

% =========================================================================
function eta_p2 = compute_partial_eta(tbl, effectName)
rowNames = string(tbl.Properties.RowNames);

idx_effect = find(rowNames == effectName, 1);
if isempty(idx_effect)
    idx_effect = find(contains(rowNames, effectName) & ~contains(rowNames, 'Error'), 1);
end
if isempty(idx_effect)
    eta_p2 = NaN;
    return
end

withinPart = effectName;
withinPart = strrep(withinPart, '(Intercept):', '');
withinPart = strrep(withinPart, 'Group:', '');
withinPart = strrep(withinPart, ':Group', '');
withinPart = regexprep(withinPart, '(^:|:$)', '');
withinPart = regexprep(withinPart, '::', ':');

targetError = "Error(" + withinPart + ")";
idx_error = find(rowNames == targetError, 1);
if isempty(idx_error)
    idx_error = find(contains(rowNames, "Error(") & contains(rowNames, withinPart), 1);
end
if isempty(idx_error)
    eta_p2 = NaN;
    return
end

eta_p2 = tbl.SumSq(idx_effect) / (tbl.SumSq(idx_effect) + tbl.SumSq(idx_error));
end

% =========================================================================
function [chanlocs_ord, idxInChanlocs] = local_reorder_chanlocs_by_labels(chanlocs, chanLabels)
nChan = numel(chanLabels);
chanlocs_ord  = repmat(chanlocs(1), 1, nChan);
idxInChanlocs = nan(nChan,1);

clabs = upper(string(strtrim({chanlocs.labels})));
for i = 1:nChan
    targ = upper(string(strtrim(chanLabels{i})));
    j = find(clabs == targ, 1, 'first');
    if ~isempty(j)
        chanlocs_ord(i)  = chanlocs(j);
        idxInChanlocs(i) = j;
    end
end
end

% =========================================================================
function d = cohend(x,y)
x = x(~isnan(x));
y = y(~isnan(y));
d = (mean(x)-mean(y)) / sqrt((var(x)+var(y))/2);
end

% =========================================================================
function s = sem1(x)
x = x(~isnan(x));
if isempty(x), s = NaN; else, s = std(x)/sqrt(numel(x)); end
end

% =========================================================================
function [h, p_adj] = fdr_bh(p, q)
% Benjamini-Hochberg FDR
if nargin < 2, q = 0.05; end
p = p(:);
nanMask = isnan(p);
p2 = p(~nanMask);

[ps, idx] = sort(p2);
m = numel(ps);
th = (1:m)'/m * q;
w = find(ps <= th, 1, 'last');

h2 = false(m,1);
if ~isempty(w)
    h2(ps <= th(w)) = true;
end

% adjusted p-values
p_adj2 = nan(m,1);
for i = 1:m
    p_adj2(i) = min(1, min(m * ps(i:end) ./ (i:m)'));
end

h = false(size(p));
p_adj = nan(size(p));

tmpH = false(m,1); tmpH(idx) = h2;
tmpP = nan(m,1);   tmpP(idx) = p_adj2;

h(~nanMask) = tmpH;
p_adj(~nanMask) = tmpP;
end