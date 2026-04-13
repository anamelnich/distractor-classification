function OUT = analyze_stroop_cache_groups(expSubjects, ctrlSubjects, cacheDir)
% analyze_stroop_cache_groups
%
% Usage:
%   expSubjects  = {'e21','e22','e25','e26','e29','e30','e31','e32','e38','e39'};
%   ctrlSubjects = {'e23','e24','e27','e33','e36','e37'};
%   OUT = analyze_stroop_cache_groups(expSubjects, ctrlSubjects, './cache');
%
% Inputs:
%   expSubjects  : cell array of experimental subject IDs
%   ctrlSubjects : cell array of control subject IDs
%   cacheDir     : folder containing *_cache_TRAIN_STROOP.mat
%
% Output:
%   OUT.exp, OUT.ctrl   : group-specific outputs
%   OUT.combinedTable   : combined summary table

if nargin < 3 || isempty(cacheDir)
    cacheDir = './cache';
end

OUT = struct();
OUT.cacheDir = cacheDir;

% --- analyze each group separately (no plotting here) ---
OUT.exp  = analyze_one_group(expSubjects,  cacheDir, 'Experimental');
OUT.ctrl = analyze_one_group(ctrlSubjects, cacheDir, 'Control');

% --- combined table ---
grpExp  = repmat("Experimental", numel(OUT.exp.subjects), 1);
grpCtrl = repmat("Control",      numel(OUT.ctrl.subjects), 1);

Texp = OUT.exp.table;
Tctrl = OUT.ctrl.table;

Texp.Group = grpExp;
Tctrl.Group = grpCtrl;

OUT.combinedTable = [movevars(Texp, 'Group', 'Before', 'Subject'); ...
                     movevars(Tctrl, 'Group', 'Before', 'Subject')];

disp(OUT.combinedTable)

% --- print group-level % removed ---
fprintf('\nAverage %% trials removed after cleaning:\n');
fprintf('Experimental: Pre = %.2f%%, Post = %.2f%%\n', ...
    OUT.exp.values.mean_pct_pre_removed, OUT.exp.values.mean_pct_post_removed);
fprintf('Control:      Pre = %.2f%%, Post = %.2f%%\n', ...
    OUT.ctrl.values.mean_pct_pre_removed, OUT.ctrl.values.mean_pct_post_removed);

% --- combined plots ---
plot_accuracy_groups(OUT.exp.values.pre_acc, OUT.exp.values.post_acc, ...
                     OUT.ctrl.values.pre_acc, OUT.ctrl.values.post_acc);

plot_rt_groups(OUT.exp.values.pre_rt_cong, OUT.exp.values.pre_rt_inc, ...
               OUT.exp.values.post_rt_cong, OUT.exp.values.post_rt_inc, ...
               OUT.ctrl.values.pre_rt_cong, OUT.ctrl.values.pre_rt_inc, ...
               OUT.ctrl.values.post_rt_cong, OUT.ctrl.values.post_rt_inc);

plot_stroop_groups(OUT.exp.values.pre_stroop, OUT.exp.values.post_stroop, ...
                   OUT.ctrl.values.pre_stroop, OUT.ctrl.values.post_stroop);

end

% =========================================================================
function G = analyze_one_group(subjects, cacheDir, groupName)

nS = numel(subjects);

pre_acc   = nan(nS,1);
post_acc  = nan(nS,1);

pre_rt_cong  = nan(nS,1);
pre_rt_inc   = nan(nS,1);
post_rt_cong = nan(nS,1);
post_rt_inc  = nan(nS,1);

pre_stroop  = nan(nS,1);
post_stroop = nan(nS,1);

n_pre_raw   = nan(nS,1);
n_pre_keep  = nan(nS,1);
n_post_raw  = nan(nS,1);
n_post_keep = nan(nS,1);

pct_pre_removed  = nan(nS,1);
pct_post_removed = nan(nS,1);

subjData = struct([]);

for i = 1:nS
    sid = subjects{i};
    f = fullfile(cacheDir, sprintf('%s_cache_TRAIN_STROOP.mat', sid));

    if ~isfile(f)
        warning('Missing cache file for %s: %s', sid, f);
        continue
    end

    C = load(f);

    if ~isfield(C, 'stroop1') || ~isfield(C, 'stroop2')
        warning('File for %s missing stroop1 or stroop2: %s', sid, f);
        continue
    end

    pre  = process_stroop_session(C.stroop1, sid, 'pre');
    post = process_stroop_session(C.stroop2, sid, 'post');

    pre_acc(i)   = pre.accuracy;
    post_acc(i)  = post.accuracy;

    pre_rt_cong(i)  = pre.rt_congruent;
    pre_rt_inc(i)   = pre.rt_incongruent;
    post_rt_cong(i) = post.rt_congruent;
    post_rt_inc(i)  = post.rt_incongruent;

    pre_stroop(i)  = pre.stroop_effect;
    post_stroop(i) = post.stroop_effect;

    n_pre_raw(i)   = pre.n_raw;
    n_pre_keep(i)  = pre.n_kept;
    n_post_raw(i)  = post.n_raw;
    n_post_keep(i) = post.n_kept;

    pct_pre_removed(i)  = pre.pct_removed;
    pct_post_removed(i) = post.pct_removed;

    subjData(i).subject = sid;
    subjData(i).file = f;
    subjData(i).pre = pre;
    subjData(i).post = post;
end

G = struct();
G.groupName = groupName;
G.subjects = subjects(:);
G.cacheDir = cacheDir;
G.subject = subjData;

G.values = struct();
G.values.pre_acc   = pre_acc;
G.values.post_acc  = post_acc;

G.values.pre_rt_cong   = pre_rt_cong;
G.values.pre_rt_inc    = pre_rt_inc;
G.values.post_rt_cong  = post_rt_cong;
G.values.post_rt_inc   = post_rt_inc;

G.values.pre_stroop   = pre_stroop;
G.values.post_stroop  = post_stroop;

G.values.n_pre_raw    = n_pre_raw;
G.values.n_pre_keep   = n_pre_keep;
G.values.n_post_raw   = n_post_raw;
G.values.n_post_keep  = n_post_keep;

G.values.pct_pre_removed  = pct_pre_removed;
G.values.pct_post_removed = pct_post_removed;
G.values.mean_pct_pre_removed  = mean(pct_pre_removed, 'omitnan');
G.values.mean_pct_post_removed = mean(pct_post_removed, 'omitnan');

G.table = table( ...
    subjects(:), ...
    pre_acc, post_acc, ...
    pre_rt_cong, pre_rt_inc, post_rt_cong, post_rt_inc, ...
    pre_stroop, post_stroop, ...
    n_pre_raw, n_pre_keep, pct_pre_removed, ...
    n_post_raw, n_post_keep, pct_post_removed, ...
    'VariableNames', { ...
    'Subject', ...
    'Acc_Pre', 'Acc_Post', ...
    'RTcong_Pre', 'RTinc_Pre', 'RTcong_Post', 'RTinc_Post', ...
    'Stroop_Pre', 'Stroop_Post', ...
    'Nraw_Pre', 'Nkeep_Pre', 'PctRemoved_Pre', ...
    'Nraw_Post', 'Nkeep_Post', 'PctRemoved_Post'});

fprintf('\n%s: %% Trials removed after cleaning\n', groupName);
for i = 1:nS
    if ~isnan(pct_pre_removed(i)) || ~isnan(pct_post_removed(i))
        fprintf('%s: Pre = %.2f%%, Post = %.2f%%\n', ...
            subjects{i}, pct_pre_removed(i), pct_post_removed(i));
    end
end

end

% =========================================================================
function S = process_stroop_session(stroopStruct, sid, label)

if ~isfield(stroopStruct, 'beh')
    error('Subject %s %s: missing .beh', sid, label);
end

beh = stroopStruct.beh;

trial_type = to_col(get_field_any(beh, {'trial_type','Trial_Type'}));
resp       = to_col(get_field_any(beh, {'Response','response'}));
rt         = to_col(get_field_any(beh, {'Reaction_Time','reaction_time','RT'}));

trial_type = double(trial_type);
resp = double(resp);
rt = double(rt);

if numel(trial_type) ~= numel(resp) || numel(resp) ~= numel(rt)
    error('Subject %s %s: mismatched vector lengths', sid, label);
end

S = struct();
S.n_raw = numel(resp);

keep = (resp ~= 3) & ~isnan(rt) & ~isnan(resp) & ~isnan(trial_type);
trial_type = trial_type(keep);
resp = resp(keep);
rt = rt(keep);

mu = mean(rt, 'omitnan');
sd = std(rt, 'omitnan');

if isnan(sd) || sd == 0
    keep2 = true(size(rt));
else
    keep2 = (rt >= mu - 3*sd) & (rt <= mu + 3*sd);
end

trial_type = trial_type(keep2);
resp = resp(keep2);
rt = rt(keep2);

S.n_kept = numel(resp);
S.n_removed = S.n_raw - S.n_kept;

if S.n_raw > 0
    S.pct_removed = 100 * S.n_removed / S.n_raw;
else
    S.pct_removed = NaN;
end

S.cleaned_trial_type = trial_type;
S.cleaned_response = resp;
S.cleaned_rt = rt;

S.accuracy = mean(resp == 1, 'omitnan');

isCorrect = (resp == 1);
isCong = (trial_type == 1);
isInc  = (trial_type == 2);

S.rt_congruent   = mean(rt(isCorrect & isCong), 'omitnan');
S.rt_incongruent = mean(rt(isCorrect & isInc),  'omitnan');
S.stroop_effect  = S.rt_incongruent - S.rt_congruent;

S.n_correct_cong = sum(isCorrect & isCong);
S.n_correct_inc  = sum(isCorrect & isInc);

end

% =========================================================================
function x = get_field_any(s, names)
for k = 1:numel(names)
    if isfield(s, names{k})
        x = s.(names{k});
        return
    end
end
error('Could not find any of these fields: %s', strjoin(names, ', '));
end

% =========================================================================
function x = to_col(x)
if iscell(x)
    x = cell2mat(x);
end
if isstring(x) || ischar(x)
    x = double(x);
end
x = x(:);
end

% =========================================================================
function plot_accuracy_groups(pre_exp, post_exp, pre_ctrl, post_ctrl)

pre_exp  = pre_exp * 100;
post_exp = post_exp * 100;
pre_ctrl  = pre_ctrl * 100;
post_ctrl = post_ctrl * 100;

figure('Color','w', 'Name','Stroop_Accuracy_Groups', ...
    'Units','pixels', 'Position',[100 100 620 500]);
hold on

x = [1 2];

col_exp   = [0.20 0.55 0.85];
col_ctrl  = [0.85 0.30 0.30];
light_exp = [0.75 0.85 0.95];
light_ctrl = [0.95 0.80 0.80];

for i = 1:numel(pre_exp)
    if ~isnan(pre_exp(i)) && ~isnan(post_exp(i))
        plot(x, [pre_exp(i) post_exp(i)], '-', 'Color', light_exp, 'LineWidth', 1);
    end
end

for i = 1:numel(pre_ctrl)
    if ~isnan(pre_ctrl(i)) && ~isnan(post_ctrl(i))
        plot(x, [pre_ctrl(i) post_ctrl(i)], '-', 'Color', light_ctrl, 'LineWidth', 1);
    end
end

m_exp = [mean(pre_exp,'omitnan'), mean(post_exp,'omitnan')];
m_ctrl = [mean(pre_ctrl,'omitnan'), mean(post_ctrl,'omitnan')];

s_exp = [sem1(pre_exp), sem1(post_exp)];
s_ctrl = [sem1(pre_ctrl), sem1(post_ctrl)];

errorbar(x, m_exp, s_exp, 'Color', col_exp, 'LineWidth', 2, 'CapSize', 0);
errorbar(x, m_ctrl, s_ctrl, 'Color', col_ctrl, 'LineWidth', 2, 'CapSize', 0);

h1 = plot(x, m_exp, '-o', 'Color', col_exp, 'MarkerFaceColor', col_exp, ...
    'MarkerEdgeColor', col_exp, 'LineWidth', 2.5, 'MarkerSize', 6);
h2 = plot(x, m_ctrl, '-o', 'Color', col_ctrl, 'MarkerFaceColor', col_ctrl, ...
    'MarkerEdgeColor', col_ctrl, 'LineWidth', 2.5, 'MarkerSize', 6);

xlim([0.7 2.3])
ylim([90 100])
xticks([1 2])
xticklabels({'Pre','Post'})
ylabel('Accuracy (%)')
title('Stroop Accuracy')
legend([h1 h2], {'Experimental','Control'}, 'Location','southwest', 'Box','off')

set(gca, 'FontName','Arial', 'FontSize',13, 'LineWidth',1.2, ...
    'TickDir','out', 'Box','off')
end

% =========================================================================
function plot_rt_groups(pre_cong_exp, pre_inc_exp, post_cong_exp, post_inc_exp, ...
                        pre_cong_ctrl, pre_inc_ctrl, post_cong_ctrl, post_inc_ctrl)

figure('Color','w', 'Name','Stroop_RT_Groups', ...
    'Units','pixels', 'Position',[120 120 760 520]);
hold on

x = [1 2 4 5];

col_exp_cong  = [0.20 0.55 0.85];
col_exp_inc   = [0.45 0.70 0.95];
col_ctrl_cong = [0.85 0.30 0.30];
col_ctrl_inc  = [0.95 0.55 0.55];

m_exp = [mean(pre_cong_exp,'omitnan'), mean(pre_inc_exp,'omitnan'), ...
         mean(post_cong_exp,'omitnan'), mean(post_inc_exp,'omitnan')];
m_ctrl = [mean(pre_cong_ctrl,'omitnan'), mean(pre_inc_ctrl,'omitnan'), ...
          mean(post_cong_ctrl,'omitnan'), mean(post_inc_ctrl,'omitnan')];

s_exp = [sem1(pre_cong_exp), sem1(pre_inc_exp), sem1(post_cong_exp), sem1(post_inc_exp)];
s_ctrl = [sem1(pre_cong_ctrl), sem1(pre_inc_ctrl), sem1(post_cong_ctrl), sem1(post_inc_ctrl)];

offset = 0.10;
x_exp = x - offset;
x_ctrl = x + offset;

errorbar(x_exp, m_exp, s_exp, 'k', 'LineStyle','none', 'LineWidth',1.2, 'CapSize',0);
errorbar(x_ctrl, m_ctrl, s_ctrl, 'k', 'LineStyle','none', 'LineWidth',1.2, 'CapSize',0);

h1 = plot(x_exp([1 3]), m_exp([1 3]), '-o', 'Color', col_exp_cong, ...
    'MarkerFaceColor', col_exp_cong, 'MarkerEdgeColor', col_exp_cong, ...
    'LineWidth', 2.2, 'MarkerSize', 6);
h2 = plot(x_exp([2 4]), m_exp([2 4]), '-o', 'Color', col_exp_inc, ...
    'MarkerFaceColor', col_exp_inc, 'MarkerEdgeColor', col_exp_inc, ...
    'LineWidth', 2.2, 'MarkerSize', 6);

h3 = plot(x_ctrl([1 3]), m_ctrl([1 3]), '-s', 'Color', col_ctrl_cong, ...
    'MarkerFaceColor', col_ctrl_cong, 'MarkerEdgeColor', col_ctrl_cong, ...
    'LineWidth', 2.2, 'MarkerSize', 6);
h4 = plot(x_ctrl([2 4]), m_ctrl([2 4]), '-s', 'Color', col_ctrl_inc, ...
    'MarkerFaceColor', col_ctrl_inc, 'MarkerEdgeColor', col_ctrl_inc, ...
    'LineWidth', 2.2, 'MarkerSize', 6);

xlim([0.5 5.5])
xticks(x)
xticklabels({'Pre Cong','Pre Incong','Post Cong','Post Incong'})
ylabel('Reaction Time (ms)')
title('Stroop Reaction Time')
legend([h1 h2 h3 h4], ...
    {'Exp Cong','Exp Incong','Ctrl Cong','Ctrl Incong'}, ...
    'Location','northwest', 'Box','off')

set(gca, 'FontName','Arial', 'FontSize',13, 'LineWidth',1.2, ...
    'TickDir','out', 'Box','off')
end

% =========================================================================
function plot_stroop_groups(pre_exp, post_exp, pre_ctrl, post_ctrl)

figure('Color','w', 'Name','Stroop_Effect_Groups', ...
    'Units','pixels', 'Position',[140 140 620 500]);
hold on

x = [1 2];

col_exp   = [0.20 0.55 0.85];
col_ctrl  = [0.85 0.30 0.30];
light_exp = [0.75 0.85 0.95];
light_ctrl = [0.95 0.80 0.80];

yline(0, '--', 'Color', [0.65 0.65 0.65], 'LineWidth', 1.0);

for i = 1:numel(pre_exp)
    if ~isnan(pre_exp(i)) && ~isnan(post_exp(i))
        plot(x, [pre_exp(i) post_exp(i)], '-', 'Color', light_exp, 'LineWidth', 1);
    end
end

for i = 1:numel(pre_ctrl)
    if ~isnan(pre_ctrl(i)) && ~isnan(post_ctrl(i))
        plot(x, [pre_ctrl(i) post_ctrl(i)], '-', 'Color', light_ctrl, 'LineWidth', 1);
    end
end

m_exp = [mean(pre_exp,'omitnan'), mean(post_exp,'omitnan')];
m_ctrl = [mean(pre_ctrl,'omitnan'), mean(post_ctrl,'omitnan')];

s_exp = [sem1(pre_exp), sem1(post_exp)];
s_ctrl = [sem1(pre_ctrl), sem1(post_ctrl)];

errorbar(x, m_exp, s_exp, 'Color', col_exp, 'LineWidth', 2, 'CapSize', 0);
errorbar(x, m_ctrl, s_ctrl, 'Color', col_ctrl, 'LineWidth', 2, 'CapSize', 0);

h1 = plot(x, m_exp, '-o', 'Color', col_exp, 'MarkerFaceColor', col_exp, ...
    'MarkerEdgeColor', col_exp, 'LineWidth', 2.5, 'MarkerSize', 6);
h2 = plot(x, m_ctrl, '-o', 'Color', col_ctrl, 'MarkerFaceColor', col_ctrl, ...
    'MarkerEdgeColor', col_ctrl, 'LineWidth', 2.5, 'MarkerSize', 6);

xlim([0.7 2.3])
xticks([1 2])
xticklabels({'Pre','Post'})
ylabel('Incongruent - Congruent RT (ms)')
title('Stroop Effect')
legend([h1 h2], {'Experimental','Control'}, 'Location','northwest', 'Box','off')

set(gca, 'FontName','Arial', 'FontSize',13, 'LineWidth',1.2, ...
    'TickDir','out', 'Box','off')
end

% =========================================================================
function s = sem1(x)
x = x(~isnan(x));
if isempty(x)
    s = NaN;
else
    s = std(x) / sqrt(numel(x));
end
end