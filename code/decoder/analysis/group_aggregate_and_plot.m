function GROUP = group_aggregate_and_plot(SUBJ, subjectIDs)

nS = numel(SUBJ);

% ---------- collect run timelines ----------
maxLen = 0;
for i = 1:nS
    if ~isfield(SUBJ{i},'runTimeline') || isempty(SUBJ{i}.runTimeline), continue; end
    maxLen = max(maxLen, numel(SUBJ{i}.runTimeline.concat_x));
end

ACC = nan(nS, maxLen);
THR_R = nan(nS, maxLen);
THR_L = nan(nS, maxLen);
THR_N = nan(nS, maxLen);
MARG  = nan(nS, maxLen);

for i = 1:nS
    if isempty(SUBJ{i}) || ~isfield(SUBJ{i},'runTimeline'), continue; end

    rt = SUBJ{i}.runTimeline;
    tt = SUBJ{i}.thrTimeline;

    L = numel(rt.concat_x);
    ACC(i,1:L) = rt.concat_acc(:);

    if ~isempty(tt)
        THR_R(i,1:L) = tt.thrR_all(:);
        THR_L(i,1:L) = tt.thrL_all(:);
        THR_N(i,1:L) = tt.thrN_all(:);
        MARG(i, 1:L) = tt.marg_all(:);
    end
end

x = 1:maxLen;

% ---------- mean ± sem ----------
mACC = nanmean(ACC,1);
sACC = nanstd(ACC,0,1) ./ sqrt(sum(~isnan(ACC),1));

mR = nanmean(THR_R,1); sR = nanstd(THR_R,0,1) ./ sqrt(sum(~isnan(THR_R),1));
mL = nanmean(THR_L,1); sL = nanstd(THR_L,0,1) ./ sqrt(sum(~isnan(THR_L),1));
mN = nanmean(THR_N,1); sN = nanstd(THR_N,0,1) ./ sqrt(sum(~isnan(THR_N),1));
mM = nanmean(MARG, 1); sM = nanstd(MARG, 0,1) ./ sqrt(sum(~isnan(MARG),1));
% --- Fit lines for thresholds (group mean across runs) ---
[slR, bR, stR] = fit_line_nan(x, mR);
[slL, bL, stL] = fit_line_nan(x, mL);
[slN, bN, stN] = fit_line_nan(x, mN);


% ---------- Plot group run-wise accuracy + thresholds ----------
figure('Color','w','Units','inches','Position',[1 1 9 5]); hold on;

yyaxis left
shaded_sem(x, mACC, sACC); hold on;
plot(x, mACC, 'LineWidth', 2.2);
yline(0.5,'--','Alpha',0.7);
ylim([0 1]);
xlabel('Run index');
ylabel('Accuracy (mean ± SEM)');
grid on; box off;

yyaxis right
shaded_sem(x, mR, sR); hold on;
plot(x, mR, 'LineWidth', 1.8);
shaded_sem(x, mL, sL); plot(x, mL, '--','LineWidth',1.8);
shaded_sem(x, mN, sN); plot(x, mN, ':','LineWidth',2.0);
shaded_sem(x, mM, sM); plot(x, mM, '-.','LineWidth',1.6);
ylim([0 1]);
ylabel('Threshold / Margin (mean ± SEM)');


title(sprintf('Group: Run-wise Accuracy + Thresholds (n=%d)', nS));
legend({'Accuracy','Chance',...
        'Threshold Right Distractor','Threshold Left Distractor','Threshold No Distractor','Ambivalence Margin'}, ...
        'Location','northeastoutside');




% ---------- AUROC/AUPRC per session ----------
aucMat   = nan(nS, 5);
auprcMat = nan(nS, 5);
prcMat   = nan(nS, 5);

for i = 1:nS
    if ~isfield(SUBJ{i},'aucOut') || isempty(SUBJ{i}.aucOut), continue; end
    aucMat(i,:)   = SUBJ{i}.aucOut.auc_session(:)';
    auprcMat(i,:) = SUBJ{i}.aucOut.auprc_session(:)';
    prcMat(i,:)   = SUBJ{i}.aucOut.pr_chance(:)';
end

mAUC   = nanmean(aucMat,1);   sAUC   = nanstd(aucMat,0,1) ./ sqrt(sum(~isnan(aucMat),1));
mAUPRC = nanmean(auprcMat,1); sAUPRC = nanstd(auprcMat,0,1) ./ sqrt(sum(~isnan(auprcMat),1));
mPRC0  = nanmean(prcMat,1);   sPRC0  = nanstd(prcMat,0,1) ./ sqrt(sum(~isnan(prcMat),1));
% --- Fit lines for AUROC and AUPRC across sessions ---
xS = 1:5;
[slAUC, bAUC, stAUC] = fit_line_nan(xS, mAUC);
[slPR,  bPR,  stPR ] = fit_line_nan(xS, mAUPRC);

figure('Color','w','Units','inches','Position',[1 1 7.5 4.5]); hold on;

errorbar(xS, mAUC,   sAUC,   '-o','LineWidth',2);
errorbar(xS, mAUPRC, sAUPRC, '-s','LineWidth',2);
errorbar(xS, mPRC0,  sPRC0,  ':^','LineWidth',1.6);

yline(0.5,'--','Alpha',0.7);
ylim([0.45 0.85]);
xticks(xS); xticklabels(arrayfun(@(k)sprintf('S%d',k), xS,'UniformOutput',false));
xlabel('Session');
ylabel('AUC (mean ± SEM)');
title(sprintf('Group: AUROC/AUPRC (Ambivalent ignored), n=%d', nS));
grid on; box off;
legend({'AUROC','AUPRC','PR chance','Chance'},'Location','northeastoutside');

% ---------- Package output ----------
GROUP = struct();
GROUP.subjectIDs = subjectIDs;
GROUP.SUBJ = SUBJ;
GROUP.ACC = ACC;
GROUP.THR_R = THR_R; GROUP.THR_L = THR_L; GROUP.THR_N = THR_N; GROUP.MARG = MARG;
GROUP.aucMat = aucMat; GROUP.auprcMat = auprcMat; GROUP.prcMat = prcMat;

end
function [slope, intercept, stats] = fit_line_nan(x, y)
% Fits y ~ intercept + slope*x ignoring NaNs. Returns slope, intercept and stats.
% stats: struct with fields p, r2, n
    x = x(:); 
    y = y(:);

    ok = ~isnan(x) & ~isnan(y);
    x2 = x(ok); y2 = y(ok);
    if numel(x2) < 2
        slope = NaN; intercept = NaN;
        stats = struct('p',NaN,'r2',NaN,'n',numel(x2));
        return;
    end

    % slope/intercept
    p = polyfit(x2, y2, 1);
    slope = p(1);
    intercept = p(2);

    % optional inferential stats
    lm = fitlm(x2, y2);
    stats = struct();
    stats.p  = lm.Coefficients.pValue(2);   % p-value for slope
    stats.r2 = lm.Rsquared.Ordinary;
    stats.n  = numel(x2);
end


function h = plot_fit_line(ax, x, slope, intercept, lineStyle)
% Plots the fitted line on axis ax using x range.
    yhat = slope*x + intercept;
    axes(ax);
    h = plot(x, yhat, lineStyle, 'LineWidth', 2.2, 'HandleVisibility','off');
end
