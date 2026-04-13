
%% Load thresholds
% --- Load ---
t = load(sprintf('./../../cnbiLoop/online_info/%s_thrlog.mat', subjectID));
thrLog = t.thrLog;

% Flip thrN (your existing step)
vals = num2cell(1 - [thrLog.thrN]);
[thrLog.thrN] = vals{:};

% --- Parse timestamps ---
d = {thrLog.timestamp}';
d = datetime(d, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

% Put into a table for easier cleaning + stable indexing
T = table((1:numel(thrLog))', d, 'VariableNames', {'origIdx','dt'});
T.dayKey = dateshift(T.dt, 'start', 'day');

% Sort globally so "session order" is chronological
T = sortrows(T, 'dt');

% Session grouping = day
[G, dayList] = findgroups(T.dayKey);

% Expected runs per session (session 1..5 in chronological order)
expectedRuns = [6 8 8 8 6];

% Track removals (optional but useful for debugging)
T.removed_bug     = false(height(T),1);
T.removed_practice = false(height(T),1);

% --- Pass 1: remove bug rows (within-session dt < 4 minutes) ---
minGap = minutes(4);

for g = 1:max(G)
    idx = find(G == g);
    if numel(idx) < 2, continue; end

    dtDiff  = diff(T.dt(idx));
    badNext = dtDiff < minGap;      % small gap between consecutive rows
    badPrev = [badNext; false];     % drop the EARLIER row in each bad pair
    badRows = idx(badPrev);

    T.removed_bug(badRows) = true;
end

% Drop bug rows
T = T(~T.removed_bug, :);

% Recompute groups after dropping rows
[G, dayList] = findgroups(T.dayKey);

% --- Pass 2: remove practice if session has more runs than expected ---
nSess = max(G);
for g = 1:nSess
    idx = find(G == g);
    nRuns = numel(idx);

    if g <= numel(expectedRuns)
        expN = expectedRuns(g);
    else
        % If you ever have >5 sessions, default to "no expectation"
        expN = nRuns;
    end

    if nRuns > expN
        % Assume practice is the first row in that session
        firstRow = idx(1);
        T.removed_practice(firstRow) = true;
    end
end

% Drop practice rows
T = T(~T.removed_practice, :);

% Recompute groups after dropping rows
[G, dayList] = findgroups(T.dayKey);

% --- Assign Session/Run on cleaned set ---
T.Session = G;

% run index within each session
runCells = splitapply(@(x){(1:numel(x))'}, T.dt, G);
T.Run = vertcat(runCells{:});

% --- Write back into thrLog (cleaned) ---
thrLog_clean = thrLog(T.origIdx); % keep only surviving rows, original structs
for i = 1:numel(thrLog_clean)
    thrLog_clean(i).Session = T.Session(i);
    thrLog_clean(i).Run     = T.Run(i);
end
thrLog = thrLog_clean;
clear thrLog_clean
% (Optional) sanity printout
counts = splitapply(@numel, T.Run, T.Session);
disp(table((1:numel(counts))', counts, 'VariableNames', {'Session','NumRuns'}));



%% Load posteriors

opDir = './../../cnbiLoop/online_info';

F = dir(fullfile(opDir, sprintf('%s_OnlinePosteriors_*.mat', subjectID)));
if isempty(F)
    error('No OnlinePosteriors files found for %s in %s', subjectID, opDir);
end

fn = {F.name};
pat = ['^' regexptranslate('escape',subjectID) '_OnlinePosteriors_(\d{8})\.mat$'];
tok = regexp(fn, pat, 'tokens');

% keep only matching files
isMatch = ~cellfun(@isempty, tok);
F = F(isMatch);
tok = tok(isMatch);

if numel(F) < 5
    error('Found only %d matching OnlinePosteriors files for %s (need 5). Check filenames in %s.', ...
        numel(F), subjectID, opDir);
end

dates = cellfun(@(c) str2double(c{1}{1}), tok);

% sort by date
[~, ord] = sort(dates);
F = F(ord);

% load first five
p1 = load(fullfile(F(1).folder, F(1).name));
p2 = load(fullfile(F(2).folder, F(2).name));
p3 = load(fullfile(F(3).folder, F(3).name));
p4 = load(fullfile(F(4).folder, F(4).name));
p5 = load(fullfile(F(5).folder, F(5).name));

% Put them in a cell so we can loop
P = {p1, p2, p3, p4, p5};

% expected #rows per session
expectedN = [360 480 480 480 360];
trialPerRun = 60;

for s = 1:numel(P)

    % --- Get the numeric matrix (nTrials x 3) from the loaded struct ---
    fn = fieldnames(P{s});
    A  = P{s}.(fn{1});   % assumes the .mat contains exactly one main variable

    if size(A,2) ~= 3
        warning('Session %d: expected N x 3, but got %d x %d.', s, size(A,1), size(A,2));
    end

    n = size(A,1);
    expN = expectedN(s);

    if n < expN
        warning('Session %d: OnlinePosteriors has %d rows, expected %d. Leaving as-is.', s, n, expN);

    elseif n > expN
        if n >= trialPerRun
            warning('Session %d: OnlinePosteriors has %d rows (> %d). Removing first %d rows (practice).', ...
                s, n, expN, trialPerRun);
            A = A(trialPerRun+1:end, :);
        else
            warning('Session %d: has %d rows (> expected) but < %d, cannot drop first run cleanly. Leaving as-is.', ...
                s, n, trialPerRun);
        end
    end

    % Optional sanity check: should be multiple of 60 after cleaning
    if mod(size(A,1), trialPerRun) ~= 0
        warning('Session %d: after cleaning, %d rows is not a multiple of %d.', ...
            s, size(A,1), trialPerRun);
    end

    % Write back into the same struct (preserve variable name)
    P{s}.(fn{1}) = A;
end

% Unpack back out if you still want p1..p5 variables
p1 = P{1}; p2 = P{2}; p3 = P{3}; p4 = P{4}; p5 = P{5};

%%
% === Config ===
runsize   = 60;                 % trials per run (fixed)
sessions  = 1:5;                % decoding1..decoding5
sessFields = arrayfun(@(s)sprintf('decoding%d', s), sessions, 'UniformOutput', false);

gapRuns   = 1;                  % visual gap (in "run slots") between sessions
showLabels= false;               % label sessions above the axis

%% === Storage per session ===
acc_runs  = cell(numel(sessions),1);
tpr_runs  = cell(numel(sessions),1);
tnr_runs  = cell(numel(sessions),1);
amb_runs  = cell(numel(sessions),1);

for si = 1:numel(sessions)
    sf = sessFields{si};
    if ~isfield(data, sf)
        warning('Missing %s in data. Skipping.', sf);
        continue;
    end
    beh = data.(sf).beh;
    if ~isfield(beh, 'BCI_output') || ~isfield(beh, 'trial_type')
        warning('%s.beh missing BCI_output or trial_type. Skipping.', sf);
        continue;
    end

    y_true = beh.trial_type(:);     % 1 = distractor, 0 = no distractor
    y_pred = beh.BCI_output(:);     % 1 = distractor, 0 = no distractor, 3 = ambivalent
    ntr    = numel(y_true);

    if numel(y_pred) ~= ntr
        error('%s: BCI_output and trial_type must be the same length.', sf);
    end
    if mod(ntr, runsize) ~= 0
        warning('%s: trials (%d) not divisible by %d; last partial run will be ignored.', sf, ntr, runsize);
    end

    nruns  = floor(ntr / runsize);
    A  = nan(nruns,1);
    TPR = nan(nruns,1);
    TNR = nan(nruns,1);
    AMB = zeros(nruns,1);

    for r = 1:nruns
        idx = (r-1)*runsize + (1:runsize);

        yt   = y_true(idx);
        yp   = y_pred(idx);

        ambMask = (yp == 3);              % ambivalent -> ignore in metrics
        AMB(r)  = sum(ambMask);

        keep = ~ambMask;                   % non-ambivalent trials for metric calc
        if ~any(keep)
            A(r)   = NaN; TPR(r) = NaN; TNR(r) = NaN;
            continue;
        end

        yt_k = yt(keep);
        yp_k = yp(keep);  % only 0/1 remain

        % Accuracy
        A(r) = mean(yp_k == yt_k);

        % TPR (sensitivity for distractor trials)
        posMask = (yt_k == 1);
        if any(posMask)
            TPR(r) = mean(yp_k(posMask) == 1);
        else
            TPR(r) = NaN;
        end

        % TNR (specificity for no-distractor trials)
        negMask = (yt_k == 0);
        if any(negMask)
            TNR(r) = mean(yp_k(negMask) == 0);
        else
            TNR(r) = NaN;
        end
    end

    acc_runs{si} = A(:);
    tpr_runs{si} = TPR(:);
    tnr_runs{si} = TNR(:);
    amb_runs{si} = AMB(:);
end

% === Concatenate with gaps ===
concat_x   = [];          % continuous x (run index with gaps)
concat_acc = [];
concat_tpr = [];
concat_tnr = [];
concat_amb = [];

sess_start_idx = [];      % for drawing separators/labels
sess_end_idx   = [];
x_cursor = 0;

for si = 1:numel(sessions)
    A   = acc_runs{si};
    TPR = tpr_runs{si};
    TNR = tnr_runs{si};
    AMB = amb_runs{si};

    if isempty(A), continue; end
    nruns = numel(A);

    % segment for this session
    x_seg = x_cursor + (1:nruns);

    concat_x   = [concat_x,   x_seg];
    concat_acc = [concat_acc; A(:)];
    concat_tpr = [concat_tpr; TPR(:)];
    concat_tnr = [concat_tnr; TNR(:)];
    concat_amb = [concat_amb; AMB(:)];

    sess_start_idx(end+1) = x_seg(1);
    sess_end_idx(end+1)   = x_seg(end);

    % gap after session (as NaNs so lines break)
    if si < numel(sessions)
        x_gap = x_seg(end) + (1:gapRuns);
        concat_x   = [concat_x,   x_gap];
        concat_acc = [concat_acc; nan(gapRuns,1)];
        concat_tpr = [concat_tpr; nan(gapRuns,1)];
        concat_tnr = [concat_tnr; nan(gapRuns,1)];
        concat_amb = [concat_amb; nan(gapRuns,1)];

        x_cursor = x_gap(end);
    else
        x_cursor = x_seg(end);
    end
end

%% Thresholds for plotting
S = thrLog; 
Session = [S.Session]';
Run     = [S.Run]';
Margin = [S.margin]';
ThrR   = [S.thrR]';
ThrL   = [S.thrL]';
ThrN   = [S.thrN]';
Timestamp = datetime({S.timestamp}', ...
    'InputFormat','yyyy-MM-dd HH:mm:ss');

T = table(Session, Run, Margin, ThrR, ThrL, ThrN, Timestamp, ...
    'VariableNames', ...
    {'Session','Run','Margin','ThresholdR','ThresholdL','ThresholdN','Timestamp'});

%% === Concatenate thresholds/margin to mirror concat_x timeline ===
thrR_all = []; thrL_all = []; thrN_all = []; marg_all = [];
x_thr    = [];
x_cursor = 0;
sess_end_idx_thr = [];

for si = 1:numel(sessions)
    s = sessions(si);

    % Take this session’s threshold rows, ordered by Run
    rows = T(T.Session == s, :);
    if isempty(rows), continue; end
    rows = sortrows(rows, 'Run');

    % How many runs are in accuracy for this session?
    A = acc_runs{si};
    if isempty(A), continue; end
    nruns_acc = numel(A);

    % If Run indices in T aren’t contiguous / start at 1, just take the first nruns in order
    if height(rows) ~= nruns_acc
        warning('Session %d: thresholds (%d) != accuracy (%d). Using min length.', ...
            s, height(rows), nruns_acc);
    end
    nruns = min(height(rows), nruns_acc);
    rows  = rows(1:nruns, :);

    % Build x segment and append
    x_seg = x_cursor + (1:nruns);

    thrR_all = [thrR_all; rows.ThresholdR];
    thrL_all = [thrL_all; rows.ThresholdL];
    thrN_all = [thrN_all; rows.ThresholdN];
    marg_all = [marg_all; rows.Margin];
    x_thr    = [x_thr, x_seg];

    sess_end_idx_thr(end+1) = x_seg(end);

    % Insert visual gap (NaNs) between sessions, same as accuracy timeline
    if si < numel(sessions)
        x_gap    = x_seg(end) + (1:gapRuns);
        x_thr    = [x_thr, x_gap];
        thrR_all = [thrR_all; nan(gapRuns,1)];
        thrL_all = [thrL_all; nan(gapRuns,1)];
        thrN_all = [thrN_all; nan(gapRuns,1)];
        marg_all = [marg_all; nan(gapRuns,1)];
        x_cursor = x_gap(end);
    else
        x_cursor = x_seg(end);
    end
end
% Ensure per-session rows are strictly increasing in Run
% and that we never exceed accuracy length
assert(numel(x_thr) == numel(concat_x), 'x_thr and concat_x should align in length once both are built.');
assert(isequal(size(thrR_all), size(concat_acc)), 'Lengths should match for plotting/overlay.');

%% Plot accuracy + thresholds

% ---------- Global styling ----------
set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultAxesFontSize',14);        % larger for readability (Nature ~8 pt after scaling)
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultLineLineWidth',1.6);
set(groot,'defaultAxesLineWidth',1);
set(groot,'defaultAxesTickDir','out');
set(groot,'defaultAxesBox','off');

% ---------- Color palette ----------
colAcc = [0.15 0.4 0.8];   % blue
colR   = [0.85 0.2 0.1];   % red
colL   = [0.0 0.6 0.45];   % teal
colN   = [0.6 0.4 0.85];   % purple
colM   = [0.35 0.35 0.35]; % gray (margin)

% ---------- Marker subsampling ----------
if ~isempty(concat_x)
    stepIdx  = max(1, floor(numel(concat_x)/60));
    mkIdxAcc = 1:stepIdx:numel(concat_x);
else
    mkIdxAcc = [];
end
if ~isempty(x_thr)
    stepIdxT = max(1, floor(numel(x_thr)/60));
    mkIdxThr = 1:stepIdxT:numel(x_thr);
else
    mkIdxThr = [];
end

% ---------- Figure ----------
fig = figure('Color','w','Units','inches','Position',[1 1 8.5 5.0]); % full-width Nature figure
tiledlayout(1,1,'Padding','tight','TileSpacing','compact');
nexttile;

% ---------- Left axis: Accuracy ----------
yyaxis left
hAcc = plot(concat_x, concat_acc, '-o', ...
    'Color', colAcc, 'MarkerSize', 4.5, 'MarkerIndices', mkIdxAcc, ...
    'LineWidth', 2.0, 'DisplayName','Accuracy');
hold on;

yl_left = [0 1];
ylim(yl_left);
xlabel('Run index','FontSize',15,'FontWeight','bold');
ylabel('Accuracy','FontSize',15,'FontWeight','bold');
title('Run-wise Accuracy and Adaptive Thresholds','FontSize',17,'FontWeight','bold');
grid on;

yChance = yline(0.5,'--','FontSize',12,'Color',[0.5 0.5 0.5],'Alpha',0.7);
yChance.LabelVerticalAlignment = 'bottom';

draw_session_dividers(sess_end_idx, sessions, yl_left(2)+0.01, false);

% ---------- Right axis: Thresholds ----------
yyaxis right
hold on;
hR = plot(x_thr, thrR_all, '-',  'Color', colR, 'LineWidth', 2.0, ...
    'Marker','o', 'MarkerIndices', mkIdxThr, 'MarkerSize', 4, 'DisplayName','ThresholdR');
hL = plot(x_thr, thrL_all, '--', 'Color', colL, 'LineWidth', 2.0, ...
    'Marker','s', 'MarkerIndices', mkIdxThr, 'MarkerSize', 4, 'DisplayName','ThresholdL');
hN = plot(x_thr, thrN_all, ':',  'Color', colN, 'LineWidth', 2.2, ...
    'Marker','^', 'MarkerIndices', mkIdxThr, 'MarkerSize', 4, 'DisplayName','ThresholdN');

plotMargin = true; % include margin in Nature plots
if plotMargin
    hM = plot(x_thr, marg_all, '-.', 'Color', colM, 'LineWidth', 1.8, ...
        'Marker','d', 'MarkerIndices', mkIdxThr, 'MarkerSize', 4, 'DisplayName','Margin');
end

ylim([0 1]);
ylabel('Threshold / Margin','FontSize',15,'FontWeight','bold');

% ---------- Legend ----------
lgdHandles = [hAcc, hR, hL, hN];
lgdLabels  = {'Accuracy','ThresholdR','ThresholdL','ThresholdN'};
if plotMargin
    lgdHandles = [lgdHandles, hM];
    lgdLabels  = [lgdLabels, {'Margin'}];
end
lgd = legend(lgdHandles, lgdLabels, 'Location','northeastoutside');
set(lgd,'Box','off','FontSize',13,'AutoUpdate','off');

% ---------- Final adjustments ----------
xlim([min(concat_x) max(concat_x)]);
ax = gca;
ax.YAxis(1).Color = colAcc; % left axis color
ax.YAxis(2).Color = [0.25 0.25 0.25]; % right axis color

% Optional annotation or panel letter
annotation('textbox',[0.01 0.95 0.05 0.05],'String','a',...
    'FontWeight','bold','FontSize',16,'EdgeColor','none');

%% %%%%%%%%%%%%%%%%%%%%%%% Compute AUC and AUPRC (per-session) %%%%%%%%%%%%%%%%%%%%%%

% Inputs you already have:
% sessions, sessFields, data.(sf).beh.trial_type, data.(sf).beh.BCI_output (optional), runsize

% ---- Load posteriors per session ----
Pcell = {
    p1.OnlinePosteriors
    p2.OnlinePosteriors
    p3.OnlinePosteriors
    p4.OnlinePosteriors
    p5.OnlinePosteriors
};
assert(numel(Pcell) == numel(sessions), 'Number of posterior files must match number of sessions.');

auc_session   = nan(numel(sessions),1);
auprc_session = nan(numel(sessions),1);
pr_chance     = nan(numel(sessions),1);
n_eff_trials  = zeros(numel(sessions),1);

for si = 1:numel(sessions)
    sf = sessFields{si};
    if ~isfield(data, sf) || ~isfield(data.(sf),'beh')
        warning('Missing %s.beh; skipping session %d.', sf, si);
        continue;
    end
    beh = data.(sf).beh;
    if ~isfield(beh,'trial_type')
        warning('%s.beh missing trial_type; skipping.', sf);
        continue;
    end

    y_true = beh.trial_type(:);   % 1=distractor, 0=no-distractor
    ntr_beh = numel(y_true);

    % --- From OnlinePosteriors: [score, threshold_used, output_code] ---
    P = Pcell{si};
    if size(P,2) < 3
        error('OnlinePosteriors for session %d must be n×3 (score, threshold, output_code).', si);
    end
    score_sess = double(P(:,1));   % posterior for class 1
    y_out_raw  = double(P(:,3));   % 1=distr, 2=no-distr, 3=ambiv
    % Map 2->0 to match your labeling
    y_out = y_out_raw;
    y_out(y_out == 2) = 0;

    % --- Length alignment (warn & truncate to common length) ---
    ntr_post = numel(score_sess);
    nmatch   = min(ntr_beh, ntr_post);
    if nmatch ~= ntr_beh || nmatch ~= ntr_post
        warning('Session %d: behavior trials (%d) vs posteriors (%d). Using first %d.', si, ntr_beh, ntr_post, nmatch);
    end
    y_true = y_true(1:nmatch);
    score_sess = score_sess(1:nmatch);
    y_out = y_out(1:nmatch);

    % --- Remove ambivalent (3) for metrics ---
    keep = (y_out ~= 3);
    yk   = y_true(keep);
    sk   = score_sess(keep);
    n_eff_trials(si) = numel(yk);

    if numel(yk) == 0 || numel(unique(yk)) < 2
        auc_session(si)   = NaN;
        auprc_session(si) = NaN;
        pr_chance(si)     = NaN;
        continue;
    end

    % PR baseline = prevalence of positives
    pr_chance(si) = mean(yk == 1);

    % AUROC & AUPRC (uses Statistics and Machine Learning Toolbox)
    auc_session(si)   = local_safe_auc(yk, sk);
    auprc_session(si) = local_safe_auprc(yk, sk);
end

%% ========= Plot: AUROC (left) & AUPRC (right) over sessions =========

% Publication styling
set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultAxesFontSize',14);
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultLineLineWidth',1.6);
set(groot,'defaultAxesLineWidth',1);
set(groot,'defaultAxesTickDir','out');
set(groot,'defaultAxesBox','off');

colAUC   = [0.15 0.40 0.80];  % blue
colAUPRC = [0.85 0.20 0.10];  % red
colPRC0  = [0.40 0.40 0.40];  % gray

xS = 1:numel(sessions);

figS = figure('Color','w','Units','inches','Position',[1 1 7.5 4.5]);
tiledlayout(1,1,'Padding','tight','TileSpacing','compact');
nexttile; hold on;

% AUROC
p1 = plot(xS, auc_session, '-o', ...
    'Color', colAUC, 'MarkerSize', 5.5, 'LineWidth', 2.0, ...
    'DisplayName','AUROC');

% AUPRC
p2 = plot(xS, auprc_session, '-s', ...
    'Color', colAUPRC, 'MarkerSize', 5.5, 'LineWidth', 2.0, ...
    'DisplayName','AUPRC');

% PR chance (prevalence)
p3 = plot(xS, pr_chance, ':^', ...
    'Color', colPRC0, 'MarkerSize', 5, 'LineWidth', 1.8, ...
    'DisplayName','PR chance');

% Chance line (shared)
yline(0.5,'--','Chance','Alpha',0.7, ...
    'Color',[0.5 0.5 0.5],'FontSize',12);

ylim([0.45 0.80]);
ylabel('AUC','FontSize',15,'FontWeight','bold');

grid on; box off;
xticks(xS);

% Session labels
if iscell(sessions)
    xlbl = cellfun(@char, sessions, 'UniformOutput', false);
elseif isstring(sessions)
    xlbl = cellstr(sessions(:)).';
elseif isnumeric(sessions)
    xlbl = arrayfun(@(v) sprintf('S%d', v), sessions(:).', 'UniformOutput', false);
else
    xlbl = arrayfun(@(k) sprintf('S%d', k), 1:numel(sessions), 'UniformOutput', false);
end
xticklabels(xlbl);

xlabel('Session','FontSize',15,'FontWeight','bold');
title('Per-Session performance (Ambivalent ignored)', ...
      'FontSize',15,'FontWeight','bold');

lgd = legend([p1 p2 p3], ...
    {'AUROC','AUPRC','PR Chance'}, ...
    'Location','northeastoutside');
set(lgd,'Box','off','AutoUpdate','off','FontSize',13);


%% RT comparison for distractor task %%%%%%%%

calibData = data.training1;
finalData = data.training2;

% Calibration
rtC_nd = calibData.beh.RT(calibData.beh.trial_type == 0);
rtC_d  = calibData.beh.RT(calibData.beh.trial_type == 1);

mC_nd = mean(rtC_nd, 'omitnan');
mC_d  = mean(rtC_d,  'omitnan');
diffC = mC_nd - mC_d;

% Final
rtF_nd = finalData.beh.RT(finalData.beh.trial_type == 0);
rtF_d  = finalData.beh.RT(finalData.beh.trial_type == 1);

mF_nd = mean(rtF_nd, 'omitnan');
mF_d  = mean(rtF_d,  'omitnan');
diffF = mF_nd - mF_d;
mC = [mC_nd mC_d];
mF = [mF_nd mF_d];
barData = [mC; mF];
%%
figure('Color','w','Units','inches','Position',[1 1 4.2 4]); hold on;

barData = [mC; mF];   % rows = calib/final, cols = ND/D
b = bar(barData,'grouped');
xticks(1:2);
xticklabels({'Pre','Post'});
b(1).FaceColor = cfg.plotColor{5};   % No distractor
b(2).FaceColor = cfg.plotColor{1};  % Distractor


ylabel('Reaction Time');
legend({'No distractor','Distractor'},'Location','northwest');
title('Mean RT by Trial Type');
ylim([400 800]);
grid on; box off;

figure('Color','w','Units','inches','Position',[1 1 4.2 4]);

bar([diffC, diffF], 'FaceColor',[0.4 0.4 0.4]);
set(gca,'XTickLabel',{'Pre','Post'});
ylabel('\Delta RT (No distractor − Distractor)', 'Interpreter','tex');
title('Distractor Cost');
yline(0,'--','Color',[0.5 0.5 0.5]);
ylim([-80, 80]);
grid on; box off;

%% Stroop Results

stroopCalib = data.stroop1;
stroopFinal = data.stroop2;

respC = stroopCalib.beh.Response;
keepC = (respC == 1);

ttC = stroopCalib.beh.Trial_Type(keepC);
rtC = stroopCalib.beh.Reaction_Time(keepC);

% make sure trial types are strings for strcmpi
ttC = string(ttC);

rtC_cong = rtC(strcmpi(ttC,'congruent'));
rtC_inc  = rtC(strcmpi(ttC,'incongruent'));

mC = [mean(rtC_cong,'omitnan'), mean(rtC_inc,'omitnan')];
diffC = mC(2) - mC(1);   % Incongruent − Congruent


respF = stroopFinal.beh.Response;
keepF = (respF == 1);

ttF = stroopFinal.beh.Trial_Type(keepF);
rtF = stroopFinal.beh.Reaction_Time(keepF);

ttF = string(ttF);

rtF_cong = rtF(strcmpi(ttF,'congruent'));
rtF_inc  = rtF(strcmpi(ttF,'incongruent'));

mF = [mean(rtF_cong,'omitnan'), mean(rtF_inc,'omitnan')];
diffF = mF(2) - mF(1);

figure('Color','w','Units','inches','Position',[1 1 4 4]); hold on;

barData = [mC; mF];   % rows = pre/post, cols = cong/inc
b = bar(barData,'grouped');

b(1).FaceColor = [0.3 0.7 0.4];   % Congruent
b(2).FaceColor = [0.8 0.3 0.3];   % Incongruent

xticks(1:2);
xticklabels({'Pre','Post'});

ylabel('Reaction Time');
legend({'Congruent','Incongruent'},'Location','northwest');
title('Stroop RT by Trial Type');
ylim([400 900]);
grid on; box off;

figure('Color','w','Units','inches','Position',[1 1 4 4]); hold on;

bar([diffC diffF],'FaceColor',[0.4 0.4 0.4]);

xticks(1:2);
xticklabels({'Pre','Post'});

ylabel('Δ RT (Incongruent − Congruent)');
title('Stroop Effect');
yline(0,'--','Color',[0.5 0.5 0.5]);
ylim([0 200]);

grid on; box off;

%% %%%%%% Pd plots %%%%%%%%%

calibData = safeCombine(data, 'training1');
finalData   = safeCombine(data, 'training2');
on1Data   = safeCombine(data, 'decoding1');
on2Data   = safeCombine(data, 'decoding2');
on3Data   = safeCombine(data, 'decoding3');
on4Data   = safeCombine(data, 'decoding4');
on5Data   = safeCombine(data, 'decoding5');
load(sprintf('./../decoders/%s_decoderL.mat',subjectID));
load(sprintf('./../decoders/%s_decoderR.mat',subjectID));
load(sprintf('./../decoders/%s_decoderN.mat',subjectID));
panelNames = {"Pre BCI","Post BCI"};

comparePd_rawPO78(data.training1,data.training2,cfg)
plotERPOffvsOnlineAllD_xDAWN(calibData, finalData, cfg, decoderL, decoderR, decoderN, panelNames,1)

load('chanlocs64.mat')
out = computePdR2_pairDiffTopos(data.training1, cfg, chanlocs);



%% %% Helper functions 
function out = safeCombine(data, topField)
% out = [] unless data.(topField).epochs exists and is nonempty
    out = [];
    if isfield(data, topField) && isfield(data.(topField), 'epochs') ...
            && ~isempty(data.(topField).epochs)
        out = combineEpochs({data.(topField).epochs});
    end
end
function auc = local_safe_auc(y, s)
    % y in {0,1}, s real-valued scores
    try
        [~,~,~,auc] = perfcurve(y, s, 1);
    catch
        y  = double(y); s = double(s);
        [~,~,~,auc] = perfcurve(y, s, 1);
    end
end

function auprc = local_safe_auprc(y, s)
    try
        [~,~,~,auprc] = perfcurve(y, s, 1, 'xCrit','reca','yCrit','prec');
    catch
        y  = double(y); s = double(s);
        [~,~,~,auprc] = perfcurve(y, s, 1, 'xCrit','reca','yCrit','prec');
    end
end

function draw_session_dividers(sess_end_idx, sessions, yTop, showLabels)
% Draw vertical dashed lines after each session and optional labels.
    if isempty(sess_end_idx), return; end
    hold on;
    for k = 1:numel(sess_end_idx)-1
        x = sess_end_idx(k) + 0.5; % between runs
        xline(x,':','Color',[0.6 0.6 0.6],'LineWidth',0.75,'Alpha',0.7);
        if showLabels
            text(x, yTop, sprintf('S%d', sessions(k+1)), ...
                'HorizontalAlignment','center','VerticalAlignment','bottom', ...
                'Color',[0.3 0.3 0.3],'FontSize',8);
        end
    end
end

