function OUT = analyze_stroop_cache(subjects, cacheDir)
% analyze_stroop_cache  Clean and analyze Stroop data from cache files.
%
% Usage:
%   subjects = {'e23','e24','e27','e33','e36','e37'};
%   OUT = analyze_stroop_cache(subjects);
%   OUT = analyze_stroop_cache(subjects, './cache');
%
% Inputs:
%   subjects : cell array of subject IDs, e.g. {'e23','e24'}
%   cacheDir : folder containing files like e36_cache_TRAIN_STROOP.mat
%              default = './cache'
%
% Assumptions in each cache file:
%   stroop1 = pre
%   stroop2 = post
%   stroopX.beh.trial_type     : 1 = congruent, 2 = incongruent
%   stroopX.beh.Response       : 1 = correct, 2 = incorrect, 3 = timeout
%   stroopX.beh.Reaction_Time  : numeric RT
%
% Cleaning:
%   1) remove timeout trials (Response == 3)
%   2) remove RT outliers outside mean +/- 2*SD within that session
%
% Metrics after cleaning:
%   - accuracy = proportion correct among retained trials
%   - mean RT (correct retained trials only), separately for congruent/incongruent
%   - Stroop effect = mean RT incongruent - mean RT congruent
%
% Output:
%   OUT struct containing subject-level and group-level values.

if nargin < 2 || isempty(cacheDir)
    cacheDir = './cache';
end

nS = numel(subjects);

% -------------------- storage --------------------
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

% -------------------- loop subjects --------------------
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

% -------------------- package values --------------------
OUT = struct();
OUT.subjects = subjects(:);
OUT.cacheDir = cacheDir;
OUT.subject = subjData;

OUT.values = struct();
OUT.values.pre_acc   = pre_acc;
OUT.values.post_acc  = post_acc;

OUT.values.pre_rt_cong   = pre_rt_cong;
OUT.values.pre_rt_inc    = pre_rt_inc;
OUT.values.post_rt_cong  = post_rt_cong;
OUT.values.post_rt_inc   = post_rt_inc;

OUT.values.pre_stroop   = pre_stroop;
OUT.values.post_stroop  = post_stroop;

OUT.values.n_pre_raw    = n_pre_raw;
OUT.values.n_pre_keep   = n_pre_keep;
OUT.values.n_post_raw   = n_post_raw;
OUT.values.n_post_keep  = n_post_keep;

OUT.values.pct_pre_removed  = pct_pre_removed;
OUT.values.pct_post_removed = pct_post_removed;
OUT.values.mean_pct_pre_removed  = mean(pct_pre_removed, 'omitnan');
OUT.values.mean_pct_post_removed = mean(pct_post_removed, 'omitnan');

% -------------------- summary table --------------------
OUT.table = table( ...
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

disp(OUT.table)
fprintf('\n%% Trials removed after cleaning:\n');
for i = 1:nS
    if ~isnan(pct_pre_removed(i)) || ~isnan(pct_post_removed(i))
        fprintf('%s: Pre = %.2f%%, Post = %.2f%%\n', ...
            subjects{i}, pct_pre_removed(i), pct_post_removed(i));
    end
end

fprintf('\nAverage %% trials removed:\n');
fprintf('Pre  = %.2f%%\n', mean(pct_pre_removed, 'omitnan'));
fprintf('Post = %.2f%%\n', mean(pct_post_removed, 'omitnan'));

% -------------------- plots --------------------
plot_accuracy(pre_acc, post_acc, subjects);
plot_rt(pre_rt_cong, pre_rt_inc, post_rt_cong, post_rt_inc, subjects);
plot_stroop(pre_stroop, post_stroop, subjects);

end

% ========================================================================
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

% remove timeout
keep = (resp ~= 3) & ~isnan(rt) & ~isnan(resp) & ~isnan(trial_type);
trial_type = trial_type(keep);
resp = resp(keep);
rt = rt(keep);

% remove RT outliers ±2 SD within session
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

% accuracy on retained trials
S.accuracy = mean(resp == 1, 'omitnan');

% RT on correct retained trials only
isCorrect = (resp == 1);
isCong = (trial_type == 1);
isInc  = (trial_type == 2);

S.rt_congruent   = mean(rt(isCorrect & isCong), 'omitnan');
S.rt_incongruent = mean(rt(isCorrect & isInc),  'omitnan');
S.stroop_effect  = S.rt_incongruent - S.rt_congruent;

S.n_correct_cong = sum(isCorrect & isCong);
S.n_correct_inc  = sum(isCorrect & isInc);

end

% ========================================================================
function x = get_field_any(s, names)
for k = 1:numel(names)
    if isfield(s, names{k})
        x = s.(names{k});
        return
    end
end
error('Could not find any of these fields: %s', strjoin(names, ', '));
end

% ========================================================================
function x = to_col(x)
if iscell(x)
    x = cell2mat(x);
end
if isstring(x) || ischar(x)
    x = double(x);
end
x = x(:);
end

% ========================================================================
function plot_accuracy(pre_acc, post_acc, subjects)

% --- convert to percent ---
pre_acc  = pre_acc * 100;
post_acc = post_acc * 100;

fig = figure('Color','w', 'Name','Stroop_Accuracy_PrePost', ...
    'Units','pixels', 'Position',[100 100 560 500]);
hold on

x = [1 2];
lineColor = [0.75 0.75 0.75];
meanColor = [0.10 0.10 0.10];

for i = 1:numel(pre_acc)
    if ~isnan(pre_acc(i)) && ~isnan(post_acc(i))
        plot(x, [pre_acc(i) post_acc(i)], '-', 'Color', lineColor, 'LineWidth', 1.0);
        scatter(x, [pre_acc(i) post_acc(i)], 35, 'MarkerFaceColor','w', ...
            'MarkerEdgeColor', lineColor, 'LineWidth', 1.0);
    end
end

m = [mean(pre_acc,'omitnan'), mean(post_acc,'omitnan')];
sem = [sem1(pre_acc), sem1(post_acc)];

errorbar(x, m, sem, 'k', 'LineWidth', 1.6, 'CapSize', 0);
plot(x, m, '-o', 'Color', meanColor, 'LineWidth', 2.5, ...
    'MarkerFaceColor', meanColor, 'MarkerEdgeColor', meanColor, 'MarkerSize', 7);

xlim([0.7 2.3])
ylim([90 100]) % percent scale
xticks([1 2])
xticklabels({'Pre','Post'})
ylabel('Accuracy (%)')
title('Stroop Accuracy')

set(gca, 'FontName','Arial', 'FontSize',13, 'LineWidth',1.2, ...
    'TickDir','out', 'Box','off')
end

% ========================================================================
function plot_rt(pre_cong, pre_inc, post_cong, post_inc, subjects)

fig = figure('Color','w', 'Name','Stroop_RT_PrePost_CongIncong', ...
    'Units','pixels', 'Position',[120 120 760 520]);
hold on

% x positions
x = [1 2 4 5]; % pre cong, pre inc, post cong, post inc

% colors
colCong = [0.25 0.55 0.85];
colInc  = [0.85 0.35 0.30];
lineColor = [0.82 0.82 0.82];

for i = 1:numel(pre_cong)
    yi = [pre_cong(i), pre_inc(i), post_cong(i), post_inc(i)];
    if sum(~isnan(yi)) >= 2
        %plot(x, yi, '-', 'Color', lineColor, 'LineWidth', 0.9);
        scatter(x([1 3]), yi([1 3]), 28, 'MarkerFaceColor','w', ...
            'MarkerEdgeColor', colCong, 'LineWidth', 1.0);
        scatter(x([2 4]), yi([2 4]), 28, 'MarkerFaceColor','w', ...
            'MarkerEdgeColor', colInc, 'LineWidth', 1.0);
    end
end

m = [mean(pre_cong,'omitnan'), mean(pre_inc,'omitnan'), ...
     mean(post_cong,'omitnan'), mean(post_inc,'omitnan')];
s = [sem1(pre_cong), sem1(pre_inc), sem1(post_cong), sem1(post_inc)];

errorbar(x, m, s, 'k', 'LineStyle','none', 'LineWidth',1.4, 'CapSize',0);
plot(x([1 3]), m([1 3]), '-o', 'Color', colCong, 'LineWidth',2.5, ...
    'MarkerFaceColor', colCong, 'MarkerEdgeColor', colCong, 'MarkerSize',7);
plot(x([2 4]), m([2 4]), '-o', 'Color', colInc, 'LineWidth',2.5, ...
    'MarkerFaceColor', colInc, 'MarkerEdgeColor', colInc, 'MarkerSize',7);

xlim([0.5 5.5])
ylim([450 850])
xticks(x)
xticklabels({'Pre Cong','Pre Incong','Post Cong','Post Incong'})
ylabel('Reaction Time')
title('Stroop Reaction Time')
set(gca, 'FontName','Arial', 'FontSize',13, 'LineWidth',1.2, ...
    'TickDir','out', 'Box','off')
end

% ========================================================================
function plot_stroop(pre_eff, post_eff, subjects)

fig = figure('Color','w', 'Name','Stroop_Effect_PrePost', ...
    'Units','pixels', 'Position',[140 140 560 500]);
hold on

x = [1 2];
lineColor = [0.75 0.75 0.75];
meanColor = [0.20 0.20 0.20];
zeroColor = [0.6 0.6 0.6];

yline(0, '--', 'Color', zeroColor, 'LineWidth', 1.0);

for i = 1:numel(pre_eff)
    if ~isnan(pre_eff(i)) && ~isnan(post_eff(i))
        plot(x, [pre_eff(i) post_eff(i)], '-', 'Color', lineColor, 'LineWidth', 1.0);
        scatter(x, [pre_eff(i) post_eff(i)], 35, 'MarkerFaceColor','w', ...
            'MarkerEdgeColor', lineColor, 'LineWidth', 1.0);
    end
end

m = [mean(pre_eff,'omitnan'), mean(post_eff,'omitnan')];
sem = [sem1(pre_eff), sem1(post_eff)];

errorbar(x, m, sem, 'k', 'LineWidth',1.6, 'CapSize',0);
plot(x, m, '-o', 'Color', meanColor, 'LineWidth',2.5, ...
    'MarkerFaceColor', meanColor, 'MarkerEdgeColor', meanColor, 'MarkerSize',7);

xlim([0.7 2.3])
ylim([0 200])
xticks([1 2])
xticklabels({'Pre','Post'})
ylabel('Incongruent - Congruent RT')
title('Stroop Effect')
set(gca, 'FontName','Arial', 'FontSize',13, 'LineWidth',1.2, ...
    'TickDir','out', 'Box','off')
end

% ========================================================================
function s = sem1(x)
x = x(~isnan(x));
if isempty(x)
    s = NaN;
else
    s = std(x) / sqrt(numel(x));
end
end