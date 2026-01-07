subjects   = {'e21','e22'};     % etc
nSubj      = numel(subjects);

runsize    = 60;
sessions   = 1:5;
nSessions  = numel(sessions);
sessFields = arrayfun(@(s)sprintf('decoding%d', s), sessions, 'UniformOutput', false);

gapRuns    = 1;

% per-subject/session storage for run-wise metrics
acc_runs_sub  = cell(nSessions, nSubj);
tpr_runs_sub  = cell(nSessions, nSubj);
tnr_runs_sub  = cell(nSessions, nSubj);
amb_runs_sub  = cell(nSessions, nSubj);

% per-subject/session storage for thresholds & margins
thrR_runs_sub = cell(nSessions, nSubj);
thrL_runs_sub = cell(nSessions, nSubj);
thrN_runs_sub = cell(nSessions, nSubj);
marg_runs_sub = cell(nSessions, nSubj);

% NEW: ERP/xDAWN inputs and decoders
calibDataCell = cell(1, nSubj);
finalDataCell = cell(1, nSubj);
decodingCell = cell(nSessions, nSubj);  

decLCell      = cell(1, nSubj);
decRCell      = cell(1, nSubj);
decNCell      = cell(1, nSubj);

cfg = [];   % will grab once from any decoderX.params

%% ==== Load Data ====
for sj = 1:nSubj
    subjectID = subjects{sj};
    fprintf('Processing subject %s...\n', subjectID);

    S = load(sprintf('data/%s_data.mat', subjectID));
    if ~isfield(S, 'data')
        error('%s_data.mat missing variable "data".', subjectID);
    end
    data = S.data;

    % ---------- grab training1 / training2 for ERP xDAWN plots ----------
    if isfield(data, 'training1')
        calibDataCell{sj} = data.training1;
    else
        warning('%s: missing data.training1 (calibration).', subjectID);
        calibDataCell{sj} = [];
    end

    if isfield(data, 'training2')
        finalDataCell{sj} = data.training2;
    else
        warning('%s: missing data.training2 (final/online).', subjectID);
        finalDataCell{sj} = [];
    end
    for si = 1:nSessions
        sf = sessFields{si};   % 'decoding1'...'decoding5'

        if isfield(data, sf)
            decodingCell{si, sj} = data.(sf); 
        else
            decodingCell{si, sj} = [];
            warning('%s: missing %s', subjectID, sf);
        end
    end

    % ---------- grab decoders L, R, N from the MAT (top-level or inside data) ----------
    % decoderL
    if isfield(S, 'decoderL')
        decL = S.decoderL;
    elseif isfield(data, 'decoderL')
        decL = data.decoderL;
    else
        error('%s: no decoderL found in file or in data.', subjectID);
    end
    decLCell{sj} = decL;

    % decoderR
    if isfield(S, 'decoderR')
        decR = S.decoderR;
    elseif isfield(data, 'decoderR')
        decR = data.decoderR;
    else
        error('%s: no decoderR found in file or in data.', subjectID);
    end
    decRCell{sj} = decR;

    % decoderN
    if isfield(S, 'decoderN')
        decN = S.decoderN;
    elseif isfield(data, 'decoderN')
        decN = data.decoderN;
    else
        error('%s: no decoderN found in file or in data.', subjectID);
    end
    decNCell{sj} = decN;

    % ---------- grab cfg (params) once from any decoder ----------
    if isempty(cfg)
        if isfield(decR, 'params')
            cfg = decR.params;
        elseif isfield(decL, 'params')
            cfg = decL.params;
        elseif isfield(decN, 'params')
            cfg = decN.params;
        else
            error('%s: no params field found in any decoder.', subjectID);
        end
    end

    % ---- thrLog with Session/Run ----
    if ~isfield(data, 'thrLog')
        error('No thrLog found for %s.', subjectID);
    end
    thrLog = data.thrLog;

    d = datetime({thrLog.timestamp}', 'InputFormat','yyyy-MM-dd HH:mm:ss');
    G = findgroups(dateshift(d, 'start', 'day'));
    idxCells  = splitapply(@(x) {(1:numel(x))'}, d, G);
    runInSess = vertcat(idxCells{:});

    for k = 1:numel(thrLog)
        thrLog(k).Session = G(k);
        thrLog(k).Run     = runInSess(k);
    end

    Session = [thrLog.Session]';
    Run     = [thrLog.Run]';
    Margin  = [thrLog.margin]';
    ThrR    = [thrLog.thrR]';
    ThrL    = [thrLog.thrL]';
    ThrN    = [thrLog.thrN]';

    Tsub = table(Session, Run, Margin, ThrR, ThrL, ThrN, d, ...
        'VariableNames', {'Session','Run','Margin','ThresholdR','ThresholdL','ThresholdN','Timestamp'});

    % ---- Per-session behavior + thresholds ----
    for si = 1:nSessions
        sf = sessFields{si};
        if ~isfield(data, sf)
            warning('%s: missing %s. Skipping.', subjectID, sf);
            continue;
        end
        if ~isfield(data.(sf),'beh')
            warning('%s: %s.beh missing. Skipping.', subjectID, sf);
            continue;
        end

        beh = data.(sf).beh;
        if ~all(isfield(beh, {'BCI_output','trial_type'}))
            warning('%s: %s.beh missing BCI_output or trial_type. Skipping.', subjectID, sf);
            continue;
        end

        y_true = beh.trial_type(:);
        y_pred = beh.BCI_output(:);
        ntr    = numel(y_true);

        if numel(y_pred) ~= ntr
            error('%s, %s: BCI_output/trial_type length mismatch.', subjectID, sf);
        end
        if mod(ntr, runsize) ~= 0
            warning('%s, %s: %d trials not divisible by %d; last run ignored.', ...
                subjectID, sf, ntr, runsize);
        end

        nruns = floor(ntr / runsize);
        A   = nan(nruns,1);
        TPR = nan(nruns,1);
        TNR = nan(nruns,1);
        AMB = zeros(nruns,1);

        for r = 1:nruns
            idx = (r-1)*runsize + (1:runsize);
            yt  = y_true(idx);
            yp  = y_pred(idx);

            ambMask = (yp == 3);
            AMB(r)  = sum(ambMask);

            keep = ~ambMask;
            if ~any(keep), continue; end

            yt_k = yt(keep);
            yp_k = yp(keep);  % 0/1 only

            A(r) = mean(yp_k == yt_k);

            pos = (yt_k == 1);
            if any(pos), TPR(r) = mean(yp_k(pos) == 1); end

            neg = (yt_k == 0);
            if any(neg), TNR(r) = mean(yp_k(neg) == 0); end
        end

        acc_runs_sub{si,sj} = A;
        tpr_runs_sub{si,sj} = TPR;
        tnr_runs_sub{si,sj} = TNR;
        amb_runs_sub{si,sj} = AMB;

        % thresholds for this session
        rows = sortrows(Tsub(Tsub.Session == sessions(si), :), 'Run');
        if isempty(rows)
            warning('%s: no thresholds for session %d.', subjectID, sessions(si));
            continue;
        end
        if height(rows) ~= nruns
            warning('%s, S%d: thresholds (%d) vs runs (%d). Using min.', ...
                subjectID, sessions(si), height(rows), nruns);
        end
        nr = min(height(rows), nruns);
        rows = rows(1:nr, :);

        thrR_runs_sub{si,sj} = rows.ThresholdR;
        thrL_runs_sub{si,sj} = rows.ThresholdL;
        thrN_runs_sub{si,sj} = rows.ThresholdN;
        marg_runs_sub{si,sj} = rows.Margin;
    end
end



%% ==== SESSION-WISE AVERAGES ACROSS SUBJECTS ====

acc_runs = cell(nSessions,1);
tpr_runs = cell(nSessions,1);
tnr_runs = cell(nSessions,1);
amb_runs = cell(nSessions,1);

thrR_allSess = cell(nSessions,1);
thrL_allSess = cell(nSessions,1);
thrN_allSess = cell(nSessions,1);
marg_allSess = cell(nSessions,1);

for si = 1:nSessions
    % number of runs available for this session across subjects
    nruns_subj = cellfun(@numel, acc_runs_sub(si,:));
    maxRuns    = max(nruns_subj);

    A_mat   = nan(maxRuns, nSubj);
    TPR_mat = nan(maxRuns, nSubj);
    TNR_mat = nan(maxRuns, nSubj);
    AMB_mat = nan(maxRuns, nSubj);

    ThrR_mat = nan(maxRuns, nSubj);
    ThrL_mat = nan(maxRuns, nSubj);
    ThrN_mat = nan(maxRuns, nSubj);
    Marg_mat = nan(maxRuns, nSubj);

    for sj = 1:nSubj
        A   = acc_runs_sub{si,sj};  if ~isempty(A),   A_mat(1:numel(A),sj)   = A(:);   end
        TPR = tpr_runs_sub{si,sj};  if ~isempty(TPR), TPR_mat(1:numel(TPR),sj)=TPR(:); end
        TNR = tnr_runs_sub{si,sj};  if ~isempty(TNR), TNR_mat(1:numel(TNR),sj)=TNR(:); end
        AMB = amb_runs_sub{si,sj};  if ~isempty(AMB), AMB_mat(1:numel(AMB),sj)=AMB(:); end

        rR = thrR_runs_sub{si,sj};  if ~isempty(rR),  ThrR_mat(1:numel(rR),sj)=rR(:);  end
        rL = thrL_runs_sub{si,sj};  if ~isempty(rL),  ThrL_mat(1:numel(rL),sj)=rL(:);  end
        rN = thrN_runs_sub{si,sj};  if ~isempty(rN),  ThrN_mat(1:numel(rN),sj)=rN(:);  end
        mM = marg_runs_sub{si,sj};  if ~isempty(mM),  Marg_mat(1:numel(mM),sj)=mM(:);  end
    end

    acc_runs{si} = nanmean(A_mat,   2);
    tpr_runs{si} = nanmean(TPR_mat, 2);
    tnr_runs{si} = nanmean(TNR_mat, 2);
    amb_runs{si} = nanmean(AMB_mat, 2);

    thrR_allSess{si} = nanmean(ThrR_mat, 2);
    thrL_allSess{si} = nanmean(ThrL_mat, 2);
    thrN_allSess{si} = nanmean(ThrN_mat, 2);
    marg_allSess{si} = nanmean(Marg_mat, 2);
end

%% ==== CONCATENATED TIMELINE FOR PLOTTING ====

concat_x   = [];
concat_acc = [];
concat_tpr = [];
concat_tnr = [];
concat_amb = [];

thrR_all = [];
thrL_all = [];
thrN_all = [];
marg_all = [];
x_thr    = [];

sess_start_idx = [];
sess_end_idx   = [];
x_cursor       = 0;

for si = 1:nSessions
    A   = acc_runs{si};
    TPR = tpr_runs{si};
    TNR = tnr_runs{si};
    AMB = amb_runs{si};

    if isempty(A), continue; end
    nruns = numel(A);

    x_seg = x_cursor + (1:nruns);

    concat_x   = [concat_x,   x_seg];
    concat_acc = [concat_acc; A(:)];
    concat_tpr = [concat_tpr; TPR(:)];
    concat_tnr = [concat_tnr; TNR(:)];
    concat_amb = [concat_amb; AMB(:)];

    sess_start_idx(end+1) = x_seg(1);
    sess_end_idx(end+1)   = x_seg(end);

    rR = thrR_allSess{si}(:);
    rL = thrL_allSess{si}(:);
    rN = thrN_allSess{si}(:);
    mM = marg_allSess{si}(:);

    nr_thr = min([numel(rR), numel(rL), numel(rN), numel(mM), nruns]);
    rR = rR(1:nr_thr); rL = rL(1:nr_thr); rN = rN(1:nr_thr); mM = mM(1:nr_thr);

    thrR_all = [thrR_all; rR];
    thrL_all = [thrL_all; rL];
    thrN_all = [thrN_all; rN];
    marg_all = [marg_all; mM];
    x_thr    = [x_thr, x_seg(1:nr_thr)];

    if si < nSessions
        x_gap = x_seg(end) + (1:gapRuns);
        concat_x   = [concat_x,   x_gap];
        concat_acc = [concat_acc; nan(gapRuns,1)];
        concat_tpr = [concat_tpr; nan(gapRuns,1)];
        concat_tnr = [concat_tnr; nan(gapRuns,1)];
        concat_amb = [concat_amb; nan(gapRuns,1)];

        x_thr    = [x_thr,    x_gap];
        thrR_all = [thrR_all; nan(gapRuns,1)];
        thrL_all = [thrL_all; nan(gapRuns,1)];
        thrN_all = [thrN_all; nan(gapRuns,1)];
        marg_all = [marg_all; nan(gapRuns,1)];

        x_cursor = x_gap(end);
    else
        x_cursor = x_seg(end);
    end
end

%% ==== PLOT: ACCURACY + THRESHOLDS ====

set_pub_defaults;

colAcc = [0.15 0.40 0.80];
colR   = [0.85 0.20 0.10];
colL   = [0.00 0.60 0.45];
colN   = [0.60 0.40 0.85];
colM   = [0.35 0.35 0.35];

if isempty(concat_x), mkIdxAcc = []; else
    stepIdx  = max(1, floor(numel(concat_x)/60));
    mkIdxAcc = 1:stepIdx:numel(concat_x);
end
if isempty(x_thr), mkIdxThr = []; else
    stepIdxT = max(1, floor(numel(x_thr)/60));
    mkIdxThr = 1:stepIdxT:numel(x_thr);
end

figure('Color','w','Units','inches','Position',[1 1 8.5 5.0]);
tiledlayout(1,1,'Padding','tight','TileSpacing','compact');
nexttile;

hAcc = plot(concat_x, concat_acc, '-o', ...
    'Color', colAcc, 'MarkerSize', 4.5, 'MarkerIndices', mkIdxAcc, ...
    'LineWidth', 2.0, 'DisplayName','Accuracy');
hold on;

yl = [0 1];
ylim(yl);
xlabel('Run index','FontSize',15,'FontWeight','bold');
ylabel('Accuracy / Threshold / Margin','FontSize',15,'FontWeight','bold');
title('Run-wise Accuracy and Adaptive Thresholds','FontSize',17,'FontWeight','bold');
grid on;

yChance = yline(0.5,'--','FontSize',12,'Color',[0.5 0.5 0.5],'Alpha',0.7);
yChance.LabelVerticalAlignment = 'bottom';

draw_session_dividers(sess_end_idx, sessions, yl(2)+0.01, false);

hR = plot(x_thr, thrR_all, '-',  'Color', colR, 'LineWidth', 2.0, ...
    'Marker','o', 'MarkerIndices', mkIdxThr, 'MarkerSize', 4, 'DisplayName','ThresholdR');
hL = plot(x_thr, thrL_all, '--', 'Color', colL, 'LineWidth', 2.0, ...
    'Marker','s', 'MarkerIndices', mkIdxThr, 'MarkerSize', 4, 'DisplayName','ThresholdL');
hN = plot(x_thr, 1-thrN_all, ':',  'Color', colN, 'LineWidth', 2.2, ...
    'Marker','^', 'MarkerIndices', mkIdxThr, 'MarkerSize', 4, 'DisplayName','ThresholdN');

plotMargin = true;
if plotMargin
    hM = plot(x_thr, marg_all, '-.', 'Color', colM, 'LineWidth', 1.8, ...
        'Marker','d', 'MarkerIndices', mkIdxThr, 'MarkerSize', 4, 'DisplayName','Margin');
end

lgdHandles = [hAcc, hR, hL, hN];
lgdLabels  = {'Accuracy','ThresholdR','ThresholdL','ThresholdN'};
if plotMargin
    lgdHandles = [lgdHandles, hM];
    lgdLabels  = [lgdLabels, {'Margin'}];
end
lgd = legend(lgdHandles, lgdLabels, 'Location','northeastoutside');
set(lgd,'Box','off','FontSize',13,'AutoUpdate','off');

xlim([min(concat_x) max(concat_x)]);
ax = gca; ax.YAxis(1).Color = 'k';

annotation('textbox',[0.01 0.95 0.05 0.05],'String','a', ...
    'FontWeight','bold','FontSize',16,'EdgeColor','none');

%% ==== AUC + AUPRC (group-averaged, per-session) ====

auc_session_sub   = nan(nSessions, nSubj);
auprc_session_sub = nan(nSessions, nSubj);
pr_chance_sub     = nan(nSessions, nSubj);
n_eff_trials_sub  = zeros(nSessions, nSubj);

for sj = 1:nSubj
    subjectID = subjects{sj};
    fprintf('AUC/AUPRC: %s...\n', subjectID);

    S = load(sprintf('data/%s_data.mat', subjectID));
    if ~isfield(S, 'data'), error('%s_data.mat missing "data".', subjectID); end
    data = S.data;

    for si = 1:nSessions
        sf = sessFields{si};
        if ~isfield(data, sf) || ~isfield(data.(sf),'beh')
            warning('%s: missing %s.beh; skipping S%d.', subjectID, sf, sessions(si));
            continue;
        end

        dec = data.(sf);
        if ~isfield(dec,'posteriors')
            warning('%s: %s missing posteriors; skipping S%d.', subjectID, sf, sessions(si));
            continue;
        end

        beh = dec.beh;
        if ~all(isfield(beh, {'trial_type','BCI_output'}))
            warning('%s: %s.beh missing trial_type/BCI_output; skipping.', subjectID, sf);
            continue;
        end

        y_true = beh.trial_type(:);
        y_out  = beh.BCI_output(:);

        P = dec.posteriors;
        if size(P,2) < 1
            warning('%s: %s.posteriors has <1 column; skipping.', subjectID, sf);
            continue;
        end
        score = double(P(:,1));

        nBeh  = numel(y_true);
        nOut  = numel(y_out);
        nPost = numel(score);
        n     = min([nBeh, nOut, nPost]);
        if n < nBeh || n < nOut || n < nPost
            warning('%s, S%d: beh=%d, out=%d, post=%d. Using first %d.', ...
                subjectID, sessions(si), nBeh, nOut, nPost, n);
        end

        y_true = y_true(1:n);
        y_out  = y_out(1:n);
        score  = score(1:n);

        keep = (y_out ~= 3);
        yk   = y_true(keep);
        sk   = score(keep);

        n_eff_trials_sub(si,sj) = numel(yk);
        if numel(yk) == 0 || numel(unique(yk)) < 2, continue; end

        pr_chance_sub(si,sj)     = mean(yk == 1);
        auc_session_sub(si,sj)   = local_safe_auc(yk, sk);
        auprc_session_sub(si,sj) = local_safe_auprc(yk, sk);
    end
end

auc_session   = nanmean(auc_session_sub,   2);
auprc_session = nanmean(auprc_session_sub, 2);
pr_chance     = nanmean(pr_chance_sub,     2);
n_eff_trials  = nanmean(n_eff_trials_sub,  2);


%% ==== PLOT: AUROC & AUPRC OVER SESSIONS ====

set_pub_defaults;

colAUC   = [0.15 0.40 0.80];
colAUPRC = [0.85 0.20 0.10];
colPRC0  = [0.40 0.40 0.40];

xS = 1:nSessions;

figure('Color','w','Units','inches','Position',[1 1 7.5 4.5]);
tiledlayout(1,1,'Padding','tight','TileSpacing','compact');
nexttile;

yyaxis left
p1 = plot(xS, auc_session, '-o', 'Color', colAUC, 'MarkerSize', 5.5, 'LineWidth', 2.0);
hold on;
yline(0.5,'--','ROC Chance','Alpha',0.7,'Color',[0.5 0.5 0.5],'FontSize',12);

yl1 = [max(0.45, min([auc_session; 0.5]) - 0.03), ...
       min(1,     max([auc_session; 0.5]) + 0.03)];
ylim(yl1);
ylabel('AUROC','FontSize',15,'FontWeight','bold');

yyaxis right
p2 = plot(xS, auprc_session, '-s', 'Color', colAUPRC, 'MarkerSize', 5.5, 'LineWidth', 2.0);
hold on;
p3 = plot(xS, pr_chance, ':^', 'Color', colPRC0, 'MarkerSize', 5, 'LineWidth', 1.8);

yl2 = [max(0.45, min([auprc_session; pr_chance]) - 0.03), ...
       min(1,     max([auprc_session; pr_chance]) + 0.03)];
ylim(yl2);
ylabel('AUPRC','FontSize',15,'FontWeight','bold');

grid on; box off;
xticks(xS);

if isnumeric(sessions)
    xlbl = arrayfun(@(v) sprintf('S%d', v), sessions(:).', 'uni', 0);
elseif isstring(sessions)
    xlbl = cellstr(sessions(:)).';
elseif iscell(sessions)
    xlbl = sessions;
else
    xlbl = arrayfun(@(k) sprintf('S%d', k), 1:nSessions, 'uni', 0);
end
xticklabels(xlbl);
xlabel('Session','FontSize',15,'FontWeight','bold');
title('Per-Session AUROC & AUPRC', ...
    'FontSize',17,'FontWeight','bold');

lgd = legend([p1 p2 p3], {'AUROC','AUPRC','PR Chance'}, 'Location','northeastoutside');
set(lgd,'Box','off','AutoUpdate','off','FontSize',13);

ax = gca;
ax.YAxis(1).Color = 'k';
ax.YAxis(2).Color = 'k';

%% ==== Plot Pd before and after intervention - all subjects, xDAWN ===
plotERPOffvsOnlineAllD_xDAWN( ...
    calibDataCell, ...
    finalDataCell, ...
    cfg, ...
    decLCell, ...
    decRCell, ...
    decNCell, ...
    {'Pre Intervention','Post Intervention'}, ...
    1);
%% ==== Plot Pd before and after intervention - individual subjects ===
for sj = 1:nSubj
    subjectID = subjects{sj};

    Dcal = calibDataCell{sj};
    Dfin = finalDataCell{sj};

    if isempty(Dcal) || isempty(Dfin)
        warning('Subject %s: missing training1 or training2; skipping.', subjectID);
        continue;
    end

    decL = decLCell{sj};
    decR = decRCell{sj};
    decN = decNCell{sj};

    % Nice subject-specific panel names
    panelNames = { ...
        sprintf('%s – Pre Intervention', subjectID), ...
        sprintf('%s – Post Intervention', subjectID)};

    plotERPOffvsOnlineAllD_xDAWN( ...
        Dcal, ...       % struct (single-subject mode)
        Dfin, ...       % struct
        cfg, ...        % same params for everyone
        decL, ...
        decR, ...
        decN, ...
        panelNames, ...
        1);             % showRT = 1 (or 0 if you don’t want RT overlays)
end
%% ==== Plot Pd before and after intervention - all subjects, PO7/PO8 ===
plotERPOffvsOnlineAllD_PO7PO8( ...
    calibDataCell, ...
    finalDataCell, ...
    cfg, ...
    decLCell, ...
    decRCell, ...
    decNCell, ...
    {'Pre Intervention','Post Intervention'}, ...
    1);
%% ==== Plot Pd (PO7/PO8) before and after intervention – individual subjects ===
for sj = 1:nSubj
    subjectID = subjects{sj};

    Dcal = calibDataCell{sj};
    Dfin = finalDataCell{sj};

    if isempty(Dcal) || isempty(Dfin)
        warning('Subject %s: missing training1 or training2; skipping.', subjectID);
        continue;
    end

    % You can still grab decoders for compatibility, even though this
    % PO7/PO8 function ignores them internally
    decL = decLCell{sj};
    decR = decRCell{sj};
    decN = decNCell{sj};

    % Nice subject-specific panel names
    panelNames = { ...
        sprintf('%s – Pre Intervention', subjectID), ...
        sprintf('%s – Post Intervention', subjectID)};

    plotERPOffvsOnlineAllD_PO7PO8( ...
        Dcal, ...       % struct (single-subject mode)
        Dfin, ...       % struct
        cfg,  ...       % same params for everyone
        decL, ...       % ignored but kept for signature compatibility
        decR, ...
        decN, ...
        panelNames, ...
        1);             % showRT = 1 (set to 0 if you don’t want RT overlays)
end


%% ==== Compute Pd AUC over online sessions - PO7/PO8 ====

[auc_session, auc_subj] = computePdAUC_PO7PO8(decodingCell, cfg, [0.15 0.50], subjects);
%%

auc_eachSubject = cell(1, nSubj);    % store per-subject outputs
auc_sessions_each = cell(1, nSubj);  % store per-subject session means

for sj = 1:nSubj

    fprintf('\n========== Running Pd AUC for subject %s ==========\n', subjects{sj});

    % Extract decodingCell for this subject only
    decoding_oneSubj = decodingCell(:, sj);

    % Run the AUC function for only this subject
    [auc_session_sj, auc_subj_sj] = computePdAUC_PO7PO8( ...
        decoding_oneSubj, cfg, [0.15 0.50], subjects(sj) );

    % Store results
    auc_sessions_each{sj} = auc_session_sj;  % nSessions x 1
    auc_eachSubject{sj}   = auc_subj_sj;     % nSessions x 1 (since 1 subj)

end

%% ==== Compute Pd AUC over online sessions - xDAWN ====
[auc_sess, auc_subj] = computePdAUC_xDAWN(decodingCell, decLCell, decRCell, cfg, [0.15 0.50], subjects);
%%
timeWin = [0.15 0.50];    % 150–500 ms

for sj = 1:nSubj
    fprintf('\n=== Plotting subject %s (%d/%d) ===\n', subjects{sj}, sj, nSubj);

    % Run function for single subject: pass only that column
    decoding_sj = decodingCell(:, sj);   % nSessions x 1
    decL_sj     = decLCell(sj);
    decR_sj     = decRCell(sj);

    % Compute + plot
    computePdAUC_xDAWN(decoding_sj, decL_sj, decR_sj, cfg, timeWin, subjects(sj));

end


%% === Helper Functions ===

function set_pub_defaults
set(groot, 'defaultAxesFontName','Arial', ...
           'defaultTextFontName','Arial', ...
           'defaultAxesFontSize',14, ...
           'defaultTextInterpreter','none', ...
           'defaultLineLineWidth',1.6, ...
           'defaultAxesLineWidth',1, ...
           'defaultAxesTickDir','out', ...
           'defaultAxesBox','off');
end

function draw_session_dividers(sess_end_idx, sessions, yTop, showLabels)
if isempty(sess_end_idx), return; end
hold on;
for k = 1:numel(sess_end_idx)-1
    x = sess_end_idx(k) + 0.5;
    xline(x,':','Color',[0.6 0.6 0.6],'LineWidth',0.75,'Alpha',0.7);
    if showLabels
        text(x, yTop, sprintf('S%d', sessions(k+1)), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'Color',[0.3 0.3 0.3],'FontSize',8);
    end
end
end

function auc = local_safe_auc(y, s)
try
    [~,~,~,auc] = perfcurve(y, s, 1);
catch
    y = double(y); s = double(s);
    [~,~,~,auc] = perfcurve(y, s, 1);
end
end

function auprc = local_safe_auprc(y, s)
try
    [~,~,~,auprc] = perfcurve(y, s, 1, 'xCrit','reca','yCrit','prec');
catch
    y = double(y); s = double(s);
    [~,~,~,auprc] = perfcurve(y, s, 1, 'xCrit','reca','yCrit','prec');
end
end
