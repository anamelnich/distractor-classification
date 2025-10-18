% T = readtable('./../../data/e17_decoders/e17_thresholds_table.csv');
T = readtable('./../../data/e18_decoders/e18_thresholds_table.csv');
%% load posteriors
saved_decoder  = load('./../cnbiLoop/online_decoders/decoderR_e17_onlinePosteriors.mat'); %subject e17
% saved_decoder  = load('./../cnbiLoop/online_decoders/decoderR_e18_onlinePosteriors.mat'); %subject e18
posteriors_all = saved_decoder.decoderR.onlinePosteriors;

%% remove the first run from subject 17
posteriors_all = posteriors_all(541:end); %e17
% posteriors_all(2041:2100)=[]; %e18
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

%% === Build concatenated thresholds/margin to match your accuracy timeline ===
% T columns required: Session, Run, Margin, ThresholdR, ThresholdL, ThresholdN

thrR_all = []; thrL_all = []; thrN_all = []; marg_all = [];
x_thr    = [];                        % x that mirrors concat_x
x_cursor = 0;
sess_end_idx_thr = [];

for si = 1:numel(sessions)
    s = sessions(si);
    % Grab this session’s rows and sort by Run
    rows = T(T.Session == s, :);
    rows = sortrows(rows, 'Run');
    if isempty(rows), continue; end

    % If you computed accuracy runs earlier:
    nruns_acc = numel(acc_runs{si});
    if height(rows) ~= nruns_acc
        warning('Session %d: thresholds (%d runs) != accuracy (%d runs). Using min length.', ...
                 s, height(rows), nruns_acc);
    end
    nruns = min(height(rows), nruns_acc);

    % x segment for this session
    x_seg = x_cursor + (1:nruns);

    % Append thresholds/margin
    thrR_all = [thrR_all; rows.ThresholdR(1:nruns)];
    thrL_all = [thrL_all; rows.ThresholdL(1:nruns)];
    thrN_all = [thrN_all; rows.ThresholdN(1:nruns)];
    marg_all = [marg_all; rows.Margin(1:nruns)];
    x_thr    = [x_thr, x_seg];

    sess_end_idx_thr(end+1) = x_seg(end);

    % Insert a NaN gap (same as for accuracy) so lines break between sessions
    if si < numel(sessions)
        x_gap = x_seg(end) + (1:gapRuns);
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

%% === Plot accuracy (left axis) + thresholds & margin (right axis) ===
% ---------- Publication styling (do this once per session if you like) ----------
set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultAxesFontSize',9);           % 8–9 pt typical for Nature
set(groot,'defaultTextInterpreter','none');   % keep plain text labels
set(groot,'defaultLegendBox','off');          % no legend box
set(groot,'defaultLineLineWidth',1.2);
set(groot,'defaultAxesLineWidth',0.75);
set(groot,'defaultAxesTickDir','out');
set(groot,'defaultAxesBox','off');            % outer box off for cleaner look

% ---------- Color palette (colorblind-friendly-ish) ----------
colAcc = [0.15 0.4 0.8];   % blue
colR   = [0.8 0.25 0.1];   % red
colL   = [0.0 0.6 0.5];    % teal/green
colN   = [0.6 0.4 0.8];    % purple
colM   = [0.55 0.55 0.55]; % gray (margin)

% Reduce marker clutter: mark every ~5th point (adjust as needed)
if ~isempty(concat_x)
    stepIdx = max(1, floor(numel(concat_x)/50));   % ≤ ~50 markers total
    mkIdxAcc = 1:stepIdx:numel(concat_x);
else
    mkIdxAcc = [];
end
if ~isempty(x_thr)
    stepIdxT = max(1, floor(numel(x_thr)/50));
    mkIdxThr = 1:stepIdxT:numel(x_thr);
else
    mkIdxThr = [];
end

% ---------- Figure ----------
fig = figure('Color','w','Units','inches','Position',[1 1 5.2 3.1]); % Nature-ish aspect

% Left axis: Accuracy
yyaxis left
hAcc = plot(concat_x, concat_acc, '-o', ...
    'Color', colAcc, 'MarkerSize', 3.5, 'MarkerIndices', mkIdxAcc, ...
    'LineWidth', 1.6, 'DisplayName','Accuracy');
hold on;

% draw chance AFTER legend is set to AutoUpdate off (so it won't get included)
yl = [0.1 0.9];
ylim(yl);
yChance = yline(0.5,'--','Theoretical Chance','Alpha',0.5,'FontSize',9);
yChance.LabelVerticalAlignment = 'bottom';
yChance.Color = [0.4 0.4 0.4];

grid on; box off;
xlabel('Run index');
ylabel('Accuracy');
title('Accuracy & Thresholds per Run (Ambivalent ignored)');
draw_session_dividers(sess_end_idx, sessions, yl(2)+0.005, false);

% Right axis: thresholds + margin
yyaxis right
hold on;

hR = plot(x_thr, thrR_all, '-',  ...
    'Color', colR, 'LineWidth', 1.4, 'DisplayName','ThresholdR', ...
    'Marker','o', 'MarkerIndices', mkIdxThr, 'MarkerSize', 3);
hL = plot(x_thr, thrL_all, '--', ...
    'Color', colL, 'LineWidth', 1.4, 'DisplayName','ThresholdL', ...
    'Marker','s', 'MarkerIndices', mkIdxThr, 'MarkerSize', 3);
hN = plot(x_thr, thrN_all, ':',  ...
    'Color', colN, 'LineWidth', 1.6, 'DisplayName','ThresholdN', ...
    'Marker','^', 'MarkerIndices', mkIdxThr, 'MarkerSize', 3);

% Optional margin line (uncomment if you want it in this plot)
plotMargin = false;   % <--- set true to include Margin
if plotMargin
    hM = plot(x_thr, marg_all, '-.', ...
        'Color', colM, 'LineWidth', 1.2, 'DisplayName','Margin', ...
        'Marker','d', 'MarkerIndices', mkIdxThr, 'MarkerSize', 3);
end

% Set thresholds y-limits (0–1 makes sense; or tighten to data with padding)
ylim([0.1 0.9]);
ylabel('Thresholds');

% ---------- Legend (robust) ----------
% Turn off auto-update so later lines (e.g., xline/yline) won't hijack the legend
lgdHandles = [hAcc, hR, hL, hN];
lgdLabels  = {'Accuracy','ThresholdR','ThresholdL','ThresholdN'};
if plotMargin
    lgdHandles = [lgdHandles, hM];
    lgdLabels  = [lgdLabels,  {'Margin'}];
end

lgd = legend(lgdHandles, lgdLabels, 'Location','eastoutside');
set(lgd,'AutoUpdate','off');   % freeze contents

% Tighten layout: slightly shrink right margin so legend fits
outerpos = fig.OuterPosition;

%% === 2) TPR & TNR (continuous) ===
figure('Color','w','Units','inches','Position',[1 1 10 4]);
h1 = plot(concat_x, concat_tpr, '-o', 'LineWidth', 1.6, 'MarkerSize', 4, ...
          'DisplayName','TPR (distractor)');
hold on;
h2 = plot(concat_x, concat_tnr, '-o', 'LineWidth', 1.6, 'MarkerSize', 4, ...
          'DisplayName','TNR (no-distractor)');
yline(0.5,'--','Chance','Alpha',0.5);
grid on; box on;
xlabel('Run index (continuous across sessions)');
ylabel('Rate');
title('TPR & TNR per Run (Ambivalent trials ignored)');
legend([h1 h2],'Location','best','AutoUpdate','off');  
ylim([0.35 1]);
draw_session_dividers(sess_end_idx, sessions, 1.02, showLabels);

%% === 3) Ambivalent counts (continuous) ===
% --- Precompute & sanity checks ---
yAmb = (concat_amb ./ 60) * 100;     % ambivalent trials (%)
marg_all = marg_all * 100;           % convert from decimal to percent

mkIdxThr = mkIdxThr(mkIdxThr>=1 & mkIdxThr<=numel(x_thr));

%% --- Figure & plotting ---
figure('Color','w','Units','inches','Position',[1 1 7 4]);
hold on; grid on; box on;

p1 = plot(concat_x, yAmb, '-^', ...
    'LineWidth', 1.8, 'MarkerSize', 4, ...
    'DisplayName', 'Ambivalent (%)');

p2 = plot(x_thr, marg_all, '-o', ...
    'Color', colR, 'LineWidth', 1.4, ...
    'MarkerIndices', mkIdxThr, 'MarkerSize', 3, ...
    'DisplayName', 'ThresholdR (%)');

xlabel('Run index (continuous across sessions)');
ylabel('% Ambivalent trials');
title('Ambivalent Trials per Run');

ylim([0, 50]);
xlim([min([concat_x(:); x_thr(:)]) max([concat_x(:); x_thr(:)])]);
% 
set(gca, 'Layer','top', 'LineWidth',1, ...
    'FontName','Arial', 'FontSize',15);

legend('Location','best','AutoUpdate','off'); legend boxoff;

% Draw session dividers above highest point
draw_session_dividers(sess_end_idx, sessions, 50, showLabels);

hold off;



%% === 4) Combined: Acc/TPR/TNR (left) + Amb (right) ===
figure('Color','w','Units','inches','Position',[1 1 11 4.8]);
yyaxis left;
plot(concat_x, concat_acc, '-', 'LineWidth', 1.8, 'DisplayName','Accuracy'); hold on;
plot(concat_x, concat_tpr, '--', 'LineWidth', 1.5, 'DisplayName','TPR');
plot(concat_x, concat_tnr, ':',  'LineWidth', 1.5, 'DisplayName','TNR');
yline(0.5,'--','Chance','Alpha',0.4);
ylabel('Accuracy / TPR / TNR'); ylim([0 1]); grid on; box on;

yyaxis right;
plot(concat_x, concat_amb, '-.', 'LineWidth', 1.5, 'DisplayName','Ambivalent count');
ylabel('# Ambivalent');  ylim([0 40]);

xlabel('Run index (continuous across sessions)');
title('All Metrics per Run (Ambivalent ignored in rates)');
legend('Location','bestoutside', 'AutoUpdate','off');

% Optional: nicer xticks (every 5 runs, for example)
% xt = get(gca,'XLim');
% set(gca,'XTick',unique([1, sess_end_idx, round(linspace(xt(1), xt(2), 12))]));
%% === Build per-session summaries (means & std across runs) ===

% ---- Compute per-session stats ----
[MTLR, SDTLR] = sess_stats_TPR_LR(sessions, tpr_runs, T);   % (K x 3): TPR, ThrL, ThrR
[MTN,  SDTN]  = sess_stats_TNR_N(sessions, tnr_runs, T);    % (K x 2): TNR, ThrN

% ---- Styling defaults (Nature-ish) ----
set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultAxesFontSize',9);
set(groot,'defaultLegendBox','off');
set(groot,'defaultAxesTickDir','out');
set(groot,'defaultAxesLineWidth',0.75);
set(groot,'defaultAxesBox','off');

% ---- FIGURE A: TPR + ThrL + ThrR ----
colsA = [ 0.80 0.25 0.10;   % TPR (red)
          0.00 0.60 0.50;   % ThrL (teal)
          0.60 0.40 0.80 ]; % ThrR (purple)

figA = figure('Color','w','Units','inches','Position',[1 1 5.6 3.3]);
bA = bar(MTLR, 'grouped'); hold on;
for k = 1:numel(bA)
    bA(k).FaceColor = colsA(k,:); bA(k).EdgeColor = 'none';
end

% error bars
ng = size(MTLR,1); nb = size(MTLR,2);
xEndsA = nan(ng, nb);
for k = 1:nb, xEndsA(:,k) = bA(k).XEndPoints; end
for k = 1:nb
    errorbar(xEndsA(:,k), MTLR(:,k), SDTLR(:,k), 'k', 'linestyle','none', ...
        'LineWidth',0.8, 'CapSize',6);
end

xticks(1:numel(sessions));
xticklabels(arrayfun(@(s)sprintf('S%d',s), sessions,'UniformOutput',false));
ylabel('Percent (%)');
title('Per-session averages: TPR & Thresholds L/R');
ylim([15, 90]); grid on; box off;
legend({'TPR','ThresholdL','ThresholdR'}, 'Location','eastoutside');

% export (optional)
% exportgraphics(gca,'session_bars_TPR_LR.pdf','ContentType','vector');

% ---- FIGURE B: TNR + ThrN ----
colsB = [ 0.00 0.60 0.50;   % TNR (teal)
          0.90 0.60 0.00 ]; % ThrN (orange)

figB = figure('Color','w','Units','inches','Position',[1 1 4.8 3.3]);
bB = bar(MTN, 'grouped'); hold on;
for k = 1:numel(bB)
    bB(k).FaceColor = colsB(k,:); bB(k).EdgeColor = 'none';
end

ng = size(MTN,1); nb = size(MTN,2);
xEndsB = nan(ng, nb);
for k = 1:nb, xEndsB(:,k) = bB(k).XEndPoints; end
for k = 1:nb
    errorbar(xEndsB(:,k), MTN(:,k), SDTN(:,k), 'k', 'linestyle','none', ...
        'LineWidth',0.8, 'CapSize',6);
end

xticks(1:numel(sessions));
xticklabels(arrayfun(@(s)sprintf('S%d',s), sessions,'UniformOutput',false));
ylabel('Percent (%)');
title('Per-session averages: TNR & Threshold N');
ylim([50, 90]); grid on; box off;
legend({'TNR','ThresholdN'}, 'Location','eastoutside');


%% %%%%%%%%%%%%%%%%%%%%%%% Compute AUC and AUPRC %%%%%%%%%%%%%%%%%%%%%%
% ========= Inputs assumed =========
% sessions      : cellstr of session names (e.g., {'S1','S2','S3','S4','S5'})
% sessFields    : cellstr mirroring 'sessions' used to index 'data'
% data.(sf).beh : struct with fields: trial_type (0/1), BCI_output (0/1/3)
% runsize       : integer (60), only used for alignment checks
% posteriors_all: 1xN or Nx1 posterior scores for class 1, concatenated across sessions
%                 in the same trial order as data (session by session)

if size(posteriors_all,1)==1, posteriors_all = posteriors_all(:); end

% ========= Compute per-session AUROC & AUPRC =========
auc_session   = nan(numel(sessions),1);
auprc_session = nan(numel(sessions),1);
pr_chance     = nan(numel(sessions),1);   % baseline = positive prevalence after filtering
n_eff_trials  = zeros(numel(sessions),1); % usable (non-ambivalent) trials per session

cursor = 0;
for si = 1:numel(sessions)
    sf = sessFields{si};
    if ~isfield(data, sf) || ~isfield(data.(sf),'beh')
        warning('Missing %s.beh; skipping session.', sf);
        continue;
    end
    beh = data.(sf).beh;
    if ~isfield(beh,'trial_type') || ~isfield(beh,'BCI_output')
        warning('%s.beh missing trial_type or BCI_output; skipping.', sf);
        continue;
    end

    y_true = beh.trial_type(:);       % 1=distractor, 0=no-distractor
    y_out  = beh.BCI_output(:);       % 0/1, 3=ambivalent
    ntr    = numel(y_true);

    % ---- OPTION A (single concatenated posterior vector) ----
    idx_sess = cursor + (1:ntr);
    if idx_sess(end) > numel(posteriors_all)
        error('posteriors_all too short for session %d.', si);
    end
    score_sess = double(posteriors_all(idx_sess));
    cursor = idx_sess(end);

    % ---- OPTION B (per-session posterior field) ----
    % If you store posteriors per session instead, comment OPTION A above
    % and use this (adjust the fieldname as needed):
    % if isfield(beh,'posterior')
    %     score_sess = double(beh.posterior(:));
    %     if numel(score_sess) ~= ntr
    %         error('%s: posterior length (%d) != trials (%d).', sf, numel(score_sess), ntr);
    %     end
    % else
    %     warning('%s.beh missing posterior; skipping.', sf);
    %     continue;
    % end

    % Remove ambivalent
    keep = (y_out ~= 3);
    yk   = y_true(keep);
    sk   = score_sess(keep);
    n_eff_trials(si) = numel(yk);

    if numel(yk) == 0 || numel(unique(yk)) < 2
        % no usable trials or only one class → undefined curves
        auc_session(si)   = NaN;
        auprc_session(si) = NaN;
        pr_chance(si)     = NaN;
        continue;
    end

    % Baseline for PR curve = prevalence of positives
    pr_chance(si) = mean(yk == 1);

    % AUROC
    auc_session(si) = local_safe_auc(yk, sk);

    % AUPRC (Recall-Precision)
    auprc_session(si) = local_safe_auprc(yk, sk);
end

% Optional: sanity check full consumption of the posterior vector
if cursor ~= numel(posteriors_all)
    warning('posteriors_all has %d entries; consumed %d across sessions.', numel(posteriors_all), cursor);
end

%% ========= Plot: AUROC (left) & AUPRC (right) over sessions =========
% ---------- Publication styling ----------
set(groot,'defaultAxesFontName','Arial');
set(groot,'defaultTextFontName','Arial');
set(groot,'defaultAxesFontSize',9);
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultLegendBox','off');
set(groot,'defaultLineLineWidth',1.2);
set(groot,'defaultAxesLineWidth',0.75);
set(groot,'defaultAxesTickDir','out');
set(groot,'defaultAxesBox','off');

% Colors
colAUC   = [0.15 0.40 0.80];  % blue
colAUPRC = [0.80 0.25 0.10];  % red
colPRC0  = [0.40 0.40 0.40];  % gray (chance PR per session markers/line)

xS = 1:numel(sessions);

figS = figure('Color','w','Units','inches','Position',[1 1 5.2 3.1]);
yyaxis left
p1 = plot(xS, auc_session, '-o', 'Color', colAUC, 'MarkerSize', 5, 'LineWidth', 1.6, 'DisplayName','AUROC');
hold on;
yline(0.5,'--','ROC Chance','Alpha',0.5,'Color',[0.4 0.4 0.4]);
ylim([0.45 0.65]); % tweak if needed
ylabel('AUROC');

yyaxis right
p2 = plot(xS, auprc_session, '-s', 'Color', colAUPRC, 'MarkerSize', 5, 'LineWidth', 1.6, 'DisplayName','AUPRC');
hold on;
% AUPRC chance varies by prevalence; show as markers (or a line if you prefer)
p3 = plot(xS, pr_chance, ':^', 'Color', colPRC0, 'MarkerSize', 4, 'LineWidth', 1.2, 'DisplayName','PR Chance (prevalence)');
ylim([0.45 0.65]);
ylabel('AUPRC');

grid on; box off;
xticks(xS);
xticks(1:numel(sessions));

if iscell(sessions)
    % cell array: ensure each item is char
    if all(cellfun(@(x) ischar(x) || (isstring(x) && isscalar(x)), sessions))
        xlbl = cellfun(@char, sessions, 'UniformOutput', false);
    else
        xlbl = arrayfun(@(k) sprintf('S%d', k), 1:numel(sessions), 'UniformOutput', false);
    end
elseif isstring(sessions) && isvector(sessions)
    xlbl = cellstr(sessions(:)).';          % convert string array -> cellstr row
elseif isnumeric(sessions) && isvector(sessions)
    xlbl = arrayfun(@(v) sprintf('S%d', v), sessions(:).', 'UniformOutput', false);
else
    % fallback: S1..Sn
    xlbl = arrayfun(@(k) sprintf('S%d', k), 1:numel(sessions), 'UniformOutput', false);
end

xticklabels(xlbl);
xlabel('Session');
title('Per-Session AUROC & AUPRC (Ambivalent ignored)');

lgd = legend([p1 p2 p3], {'AUROC','AUPRC','PR Chance'}, 'Location','eastoutside');
set(lgd,'AutoUpdate','off');

%% ========= Helper functions =========
function A = local_safe_auc(y, s)
    try
        [~,~,~,A] = perfcurve(y, s, 1);
    catch
        A = NaN;
    end
end

function AP = local_safe_auprc(y, s)
    try
        [~,~,~,AP] = perfcurve(y, s, 1, 'xCrit','reca','yCrit','prec');
    catch
        AP = NaN;
    end
end


%% Helper: draw session dividers + labels
function draw_session_dividers(sess_end_idx, sessions, ytop, showLabels)
    for i = 1:numel(sess_end_idx)-1
        xline(sess_end_idx(i) + 0.5, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
    end
    if showLabels
        mids = round((sess_end_idx - [0, sess_end_idx(1:end-1)]) / 2) + [1, sess_end_idx(1:end-1)];
        for i = 1:numel(mids)
            text(mids(i), ytop, sprintf('S%d', i), ...
                 'HorizontalAlignment','center','VerticalAlignment','bottom', ...
                 'FontWeight','bold','Color',[0.2 0.2 0.2]);
        end
    end
end

%% ---- Helper to compute per-session mean/std with safe alignment ----
function [M, SD] = sess_stats_TPR_LR(sessions, tpr_runs, T)
    K = numel(sessions);
    M  = nan(K,3);  % [TPR, ThrL, ThrR]
    SD = nan(K,3);
    for si = 1:K
        s = sessions(si);
        tpr = tpr_runs{si}(:);
        rows = T(T.Session==s, :); rows = sortrows(rows,'Run');
        ThrL = rows.ThresholdL(:);
        ThrR = rows.ThresholdR(:);

        nmin = min([numel(tpr), numel(ThrL), numel(ThrR)]);
        tpr  = tpr(1:nmin);
        ThrL = ThrL(1:nmin);
        ThrR = ThrR(1:nmin);

        % convert to %
        tprP  = 100*tpr;  ThrLP = 100*ThrL;  ThrRP = 100*ThrR;

        M(si,:)  = [mean(tprP,'omitnan'),  mean(ThrLP,'omitnan'),  mean(ThrRP,'omitnan')];
        SD(si,:) = [std(tprP,'omitnan'),   std(ThrLP,'omitnan'),   std(ThrRP,'omitnan')];
    end
end

function [M, SD] = sess_stats_TNR_N(sessions, tnr_runs, T)
    K = numel(sessions);
    M  = nan(K,2);  % [TNR, ThrN]
    SD = nan(K,2);
    for si = 1:K
        s = sessions(si);
        tnr = tnr_runs{si}(:);
        rows = T(T.Session==s, :); rows = sortrows(rows,'Run');
        ThrN = rows.ThresholdN(:);

        nmin = min([numel(tnr), numel(ThrN)]);
        tnr  = tnr(1:nmin);
        ThrN = ThrN(1:nmin);

        % convert to %
        tnrP  = 100*tnr;  ThrNP = 100*ThrN;

        M(si,:)  = [mean(tnrP,'omitnan'),  mean(ThrNP,'omitnan')];
        SD(si,:) = [std(tnrP,'omitnan'),   std(ThrNP,'omitnan')];
    end
end