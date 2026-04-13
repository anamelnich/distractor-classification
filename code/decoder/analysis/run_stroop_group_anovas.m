function STATS = run_stroop_group_anovas(stroop_exp, stroop_ctrl)

STATS = struct();

%% =========================
%% 1) ACCURACY: Group x Time
%% =========================
acc_pre_exp   = stroop_exp.values.pre_acc(:);
acc_post_exp  = stroop_exp.values.post_acc(:);

acc_pre_ctrl  = stroop_ctrl.values.pre_acc(:);
acc_post_ctrl = stroop_ctrl.values.post_acc(:);

T_acc = table( ...
    categorical([repmat("exp",numel(acc_pre_exp),1); repmat("ctrl",numel(acc_pre_ctrl),1)]), ...
    [acc_pre_exp;  acc_pre_ctrl], ...
    [acc_post_exp; acc_post_ctrl], ...
    'VariableNames', {'Group','Pre','Post'});

Meas_acc = table(categorical(["Pre"; "Post"]), 'VariableNames', {'Time'});

rm_acc = fitrm(T_acc, 'Pre-Post ~ Group', 'WithinDesign', Meas_acc);
ranova_acc = ranova(rm_acc, 'WithinModel', 'Time');

STATS.accuracy.ranova = ranova_acc;
STATS.accuracy.eta_p2.Time = compute_partial_eta(ranova_acc, 'Time');
STATS.accuracy.eta_p2.GroupTime = compute_partial_eta(ranova_acc, 'Group:Time');

fprintf('\n================ ACCURACY: Group x Time ================\n');
disp(ranova_acc)
fprintf('Partial eta^2:\n');
fprintf('  Time       = %.3f\n', STATS.accuracy.eta_p2.Time);
fprintf('  Group×Time = %.3f\n', STATS.accuracy.eta_p2.GroupTime);

%% ====================================
%% 2) RT: Group x Time x Congruency
%% ====================================
rt_pre_cong_exp   = stroop_exp.values.pre_rt_cong(:);
rt_pre_inc_exp    = stroop_exp.values.pre_rt_inc(:);
rt_post_cong_exp  = stroop_exp.values.post_rt_cong(:);
rt_post_inc_exp   = stroop_exp.values.post_rt_inc(:);

rt_pre_cong_ctrl  = stroop_ctrl.values.pre_rt_cong(:);
rt_pre_inc_ctrl   = stroop_ctrl.values.pre_rt_inc(:);
rt_post_cong_ctrl = stroop_ctrl.values.post_rt_cong(:);
rt_post_inc_ctrl  = stroop_ctrl.values.post_rt_inc(:);

T_rt = table( ...
    categorical([repmat("exp",numel(rt_pre_cong_exp),1); repmat("ctrl",numel(rt_pre_cong_ctrl),1)]), ...
    [rt_pre_cong_exp;  rt_pre_cong_ctrl], ...
    [rt_pre_inc_exp;   rt_pre_inc_ctrl], ...
    [rt_post_cong_exp; rt_post_cong_ctrl], ...
    [rt_post_inc_exp;  rt_post_inc_ctrl], ...
    'VariableNames', {'Group','PreCong','PreInc','PostCong','PostInc'});

% IMPORTANT: 4x2 table, one row per repeated-measure column
Time = categorical(["Pre"; "Pre"; "Post"; "Post"]);
Congruency = categorical(["Cong"; "Inc"; "Cong"; "Inc"]);
Meas_rt = table(Time, Congruency);

rm_rt = fitrm(T_rt, 'PreCong-PostInc ~ Group', 'WithinDesign', Meas_rt);
ranova_rt = ranova(rm_rt, 'WithinModel', 'Time*Congruency');

STATS.rt.ranova = ranova_rt;
STATS.rt.eta_p2.Time = compute_partial_eta(ranova_rt, 'Time');
STATS.rt.eta_p2.Congruency = compute_partial_eta(ranova_rt, 'Congruency');
STATS.rt.eta_p2.TimeCongruency = compute_partial_eta(ranova_rt, 'Time:Congruency');
STATS.rt.eta_p2.GroupTime = compute_partial_eta(ranova_rt, 'Group:Time');
STATS.rt.eta_p2.GroupCongruency = compute_partial_eta(ranova_rt, 'Group:Congruency');
STATS.rt.eta_p2.GroupTimeCongruency = compute_partial_eta(ranova_rt, 'Group:Time:Congruency');

fprintf('\n================ RT: Group x Time x Congruency ================\n');
disp(ranova_rt)
fprintf('Partial eta^2:\n');
fprintf('  Time                  = %.3f\n', STATS.rt.eta_p2.Time);
fprintf('  Congruency            = %.3f\n', STATS.rt.eta_p2.Congruency);
fprintf('  Time×Congruency       = %.3f\n', STATS.rt.eta_p2.TimeCongruency);
fprintf('  Group×Time            = %.3f\n', STATS.rt.eta_p2.GroupTime);
fprintf('  Group×Congruency      = %.3f\n', STATS.rt.eta_p2.GroupCongruency);
fprintf('  Group×Time×Congruency = %.3f\n', STATS.rt.eta_p2.GroupTimeCongruency);

%% =========================
%% 3) RT COST: Group x Time
%% =========================
cost_pre_exp   = stroop_exp.values.pre_stroop(:);
cost_post_exp  = stroop_exp.values.post_stroop(:);

cost_pre_ctrl  = stroop_ctrl.values.pre_stroop(:);
cost_post_ctrl = stroop_ctrl.values.post_stroop(:);

T_cost = table( ...
    categorical([repmat("exp",numel(cost_pre_exp),1); repmat("ctrl",numel(cost_pre_ctrl),1)]), ...
    [cost_pre_exp;  cost_pre_ctrl], ...
    [cost_post_exp; cost_post_ctrl], ...
    'VariableNames', {'Group','Pre','Post'});

Meas_cost = table(categorical(["Pre"; "Post"]), 'VariableNames', {'Time'});

rm_cost = fitrm(T_cost, 'Pre-Post ~ Group', 'WithinDesign', Meas_cost);
ranova_cost = ranova(rm_cost, 'WithinModel', 'Time');

STATS.cost.ranova = ranova_cost;
STATS.cost.eta_p2.Time = compute_partial_eta(ranova_cost, 'Time');
STATS.cost.eta_p2.GroupTime = compute_partial_eta(ranova_cost, 'Group:Time');

fprintf('\n================ RT COST: Group x Time ================\n');
disp(ranova_cost)
fprintf('Partial eta^2:\n');
fprintf('  Time       = %.3f\n', STATS.cost.eta_p2.Time);
fprintf('  Group×Time = %.3f\n', STATS.cost.eta_p2.GroupTime);

end

function eta_p2 = compute_partial_eta(tbl, effectName)
% partial eta^2 = SS_effect / (SS_effect + SS_error)

rowNames = string(tbl.Properties.RowNames);

% --- find effect row ---
idx_effect = find(rowNames == effectName, 1);

if isempty(idx_effect)
    idx_effect = find(contains(rowNames, effectName) & ~contains(rowNames, 'Error'), 1);
end

if isempty(idx_effect)
    eta_p2 = NaN;
    warning('Effect "%s" not found in ranova table.', effectName);
    return
end

% --- determine correct error term ---
% remove Group and Intercept, since the error row is based only on within-subject factors
withinPart = effectName;
withinPart = strrep(withinPart, '(Intercept):', '');
withinPart = strrep(withinPart, 'Group:', '');
withinPart = strrep(withinPart, ':Group', '');

% clean possible double colons or leftovers
withinPart = regexprep(withinPart, '(^:|:$)', '');
withinPart = regexprep(withinPart, '::', ':');

% expected error row
targetError = "Error(" + withinPart + ")";

idx_error = find(rowNames == targetError, 1);

if isempty(idx_error)
    % fallback if MATLAB formats row names slightly differently
    idx_error = find(contains(rowNames, "Error(") & contains(rowNames, withinPart), 1);
end

if isempty(idx_error)
    eta_p2 = NaN;
    warning('Could not find matching error row for effect "%s".', effectName);
    return
end

ss_effect = tbl.SumSq(idx_effect);
ss_error  = tbl.SumSq(idx_error);

eta_p2 = ss_effect / (ss_effect + ss_error);
end