function G = group_plot_rt_and_stroop_prepost(cacheFiles, cfgRef)
% Group-level RT plots for:
%  1) Distractor task (training1 vs training2): ND vs D, and ND-D cost
%  2) Stroop (stroop1 vs stroop2): Congruent vs Incongruent (correct only), and Inc-Cong effect
%
% cacheFiles : cellstr of per-subject cache .mat paths
% cfgRef     : optional cfg for colors (uses cfgRef.plotColor{5} and {1} if provided)

% ---------- storage (rows = subjects) ----------
mC_nd = []; mC_d = []; diffC = [];
mF_nd = []; mF_d = []; diffF = [];

mSC_cong = []; mSC_inc = []; diffSC = [];
mSF_cong = []; mSF_inc = []; diffSF = [];

usedIDs = {};

for i = 1:numel(cacheFiles)
    try
        C = load(cacheFiles{i});
        sid = '';
        if isfield(C,'subjectID'), sid = C.subjectID; end

        % ---------- Distractor task (training1/training2) ----------
        if isfield(C,'training1') && isfield(C,'training2') && ...
           isfield(C.training1,'beh') && isfield(C.training2,'beh') && ...
           isfield(C.training1.beh,'RT') && isfield(C.training1.beh,'trial_type') && ...
           isfield(C.training2.beh,'RT') && isfield(C.training2.beh,'trial_type')

            rtC = C.training1.beh.RT;
            ttC = C.training1.beh.trial_type;

            rtF = C.training2.beh.RT;
            ttF = C.training2.beh.trial_type;

            % Pre
            rtC_nd_i = rtC(ttC==0);
            rtC_d_i  = rtC(ttC==1);

            mC_nd_i = mean(rtC_nd_i,'omitnan');
            mC_d_i  = mean(rtC_d_i,'omitnan');
            diffC_i = mC_nd_i - mC_d_i;   % ND − D

            % Post
            rtF_nd_i = rtF(ttF==0);
            rtF_d_i  = rtF(ttF==1);

            mF_nd_i = mean(rtF_nd_i,'omitnan');
            mF_d_i  = mean(rtF_d_i,'omitnan');
            diffF_i = mF_nd_i - mF_d_i;

        else
            fprintf('Skipping subject %d (%s): missing training RT fields\n', i, sid);
            continue; % must have training RTs to include subject
        end

        % ---------- Stroop (stroop1/stroop2): correct only ----------
        okStroop = isfield(C,'stroop1') && isfield(C,'stroop2') && ...
                   isfield(C.stroop1,'beh') && isfield(C.stroop2,'beh') && ...
                   isfield(C.stroop1.beh,'Response') && isfield(C.stroop1.beh,'Trial_Type') && isfield(C.stroop1.beh,'Reaction_Time') && ...
                   isfield(C.stroop2.beh,'Response') && isfield(C.stroop2.beh,'Trial_Type') && isfield(C.stroop2.beh,'Reaction_Time');

        if okStroop
            % Pre
            keepC = (C.stroop1.beh.Response == 1);
            ttC_s = string(C.stroop1.beh.Trial_Type(keepC));
            rtC_s = C.stroop1.beh.Reaction_Time(keepC);

            rtC_cong = rtC_s(strcmpi(ttC_s,'congruent'));
            rtC_inc  = rtC_s(strcmpi(ttC_s,'incongruent'));

            mSC_cong_i = mean(rtC_cong,'omitnan');
            mSC_inc_i  = mean(rtC_inc,'omitnan');
            diffSC_i   = mSC_inc_i - mSC_cong_i;  % Inc − Cong

            % Post
            keepF = (C.stroop2.beh.Response == 1);
            ttF_s = string(C.stroop2.beh.Trial_Type(keepF));
            rtF_s = C.stroop2.beh.Reaction_Time(keepF);

            rtF_cong = rtF_s(strcmpi(ttF_s,'congruent'));
            rtF_inc  = rtF_s(strcmpi(ttF_s,'incongruent'));

            mSF_cong_i = mean(rtF_cong,'omitnan');
            mSF_inc_i  = mean(rtF_inc,'omitnan');
            diffSF_i   = mSF_inc_i - mSF_cong_i;

        else
            % allow missing stroop; mark as NaN for this subject
            mSC_cong_i = NaN; mSC_inc_i = NaN; diffSC_i = NaN;
            mSF_cong_i = NaN; mSF_inc_i = NaN; diffSF_i = NaN;
        end

        % ---------- append ----------
        mC_nd(end+1,1) = mC_nd_i; %#ok<AGROW>
        mC_d(end+1,1)  = mC_d_i;  %#ok<AGROW>
        diffC(end+1,1) = diffC_i; %#ok<AGROW>

        mF_nd(end+1,1) = mF_nd_i; %#ok<AGROW>
        mF_d(end+1,1)  = mF_d_i;  %#ok<AGROW>
        diffF(end+1,1) = diffF_i; %#ok<AGROW>

        mSC_cong(end+1,1) = mSC_cong_i; %#ok<AGROW>
        mSC_inc(end+1,1)  = mSC_inc_i;  %#ok<AGROW>
        diffSC(end+1,1)   = diffSC_i;   %#ok<AGROW>

        mSF_cong(end+1,1) = mSF_cong_i; %#ok<AGROW>
        mSF_inc(end+1,1)  = mSF_inc_i;  %#ok<AGROW>
        diffSF(end+1,1)   = diffSF_i;   %#ok<AGROW>

        usedIDs{end+1} = sid; %#ok<AGROW>

    catch
        continue;
    end
end

N = numel(mC_nd);

% ---------- colors ----------
useCfg = (nargin >= 2) && ~isempty(cfgRef) && isfield(cfgRef,'plotColor') && numel(cfgRef.plotColor) >= 5;
if useCfg
    colND = cfgRef.plotColor{5};  % ND
    colD  = cfgRef.plotColor{1};  % D
else
    colND = [0.2 0.6 0.8];
    colD  = [0.85 0.2 0.1];
end

% ---------- helper for mean±SEM ----------
mean_sem = @(x) deal(mean(x,'omitnan'), std(x,'omitnan') ./ sqrt(sum(~isnan(x))));

%% ===================== Plot 1: Distractor RT (ND vs D) =====================
[m1_nd, se1_nd] = mean_sem(mC_nd);
[m1_d,  se1_d ] = mean_sem(mC_d);
[m2_nd, se2_nd] = mean_sem(mF_nd);
[m2_d,  se2_d ] = mean_sem(mF_d);

barData = [m1_nd m1_d; m2_nd m2_d];
errData = [se1_nd se1_d; se2_nd se2_d];

figure('Color','w','Units','inches','Position',[1 1 4.8 4]); hold on;
b = bar(barData,'grouped');
b(1).FaceColor = colND;
b(2).FaceColor = colD;

% errorbars (grouped bar geometry)
ng = size(barData,1); nb = size(barData,2);
x = nan(nb,ng);
for k = 1:nb, x(k,:) = b(k).XEndPoints; end
errorbar(x', barData, errData, 'k', 'linestyle','none','LineWidth',1.2);

xticks(1:2); xticklabels({'Pre','Post'});
ylabel('Reaction Time');
ylim([300 700]);
legend({'No distractor','Distractor'},'Location','northwest');
title(sprintf('Mean RT by Trial Type (n=%d)', N));
grid on; box off;

%% ===================== Plot 2: Distractor Cost (ND − D) =====================
[mDiffPre, seDiffPre]   = mean_sem(diffC);
[mDiffPost, seDiffPost] = mean_sem(diffF);

figure('Color','w','Units','inches','Position',[1 1 4.6 4]); hold on;
bar([mDiffPre, mDiffPost], 'FaceColor',[0.4 0.4 0.4]);
errorbar([1 2], [mDiffPre mDiffPost], [seDiffPre seDiffPost], 'k', 'linestyle','none','LineWidth',1.2);
xticks(1:2); xticklabels({'Pre','Post'});
ylabel('\Delta RT (No distractor − Distractor)', 'Interpreter','tex');
title('Distractor Cost');
yline(0,'--','Color',[0.5 0.5 0.5]);
grid on; box off;

%% ===================== Plot 3: Stroop RT (Cong vs Inc) =====================
% (some subjects may have NaNs if stroop missing)
[mC_cong, seC_cong] = mean_sem(mSC_cong);
[mC_inc,  seC_inc ] = mean_sem(mSC_inc);
[mF_cong, seF_cong] = mean_sem(mSF_cong);
[mF_inc,  seF_inc ] = mean_sem(mSF_inc);

barDataS = [mC_cong mC_inc; mF_cong mF_inc];
errDataS = [seC_cong seC_inc; seF_cong seF_inc];

figure('Color','w','Units','inches','Position',[1 1 4.8 4]); hold on;
b = bar(barDataS,'grouped');
b(1).FaceColor = [0.3 0.7 0.4]; % Cong
b(2).FaceColor = [0.8 0.3 0.3]; % Inc

ng = size(barDataS,1); nb = size(barDataS,2);
x = nan(nb,ng);
for k = 1:nb, x(k,:) = b(k).XEndPoints; end
errorbar(x', barDataS, errDataS, 'k', 'linestyle','none','LineWidth',1.2);

xticks(1:2); xticklabels({'Pre','Post'});
ylabel('Reaction Time');
ylim([300 900]);
legend({'Congruent','Incongruent'},'Location','northwest');
title(sprintf('Stroop RT by Trial Type (n=%d)', sum(~isnan(mSC_cong))));
grid on; box off;

%% ===================== Plot 4: Stroop Effect (Inc − Cong) =====================
[mSEpre, seSEpre]   = mean_sem(diffSC);
[mSEpost, seSEpost] = mean_sem(diffSF);

figure('Color','w','Units','inches','Position',[1 1 4.6 4]); hold on;
bar([mSEpre mSEpost], 'FaceColor',[0.4 0.4 0.4]);
errorbar([1 2], [mSEpre mSEpost], [seSEpre seSEpost], 'k', 'linestyle','none','LineWidth',1.2);
xticks(1:2); xticklabels({'Pre','Post'});
ylabel('Δ RT (Incongruent − Congruent)');
title('Stroop Effect');
yline(0,'--','Color',[0.5 0.5 0.5]);
grid on; box off;

% ---------- package outputs ----------
G = struct();
G.N = N;
G.usedIDs = usedIDs;

G.distractor = struct('mC_nd',mC_nd,'mC_d',mC_d,'diffC',diffC, ...
                      'mF_nd',mF_nd,'mF_d',mF_d,'diffF',diffF);

G.stroop = struct('mC_cong',mSC_cong,'mC_inc',mSC_inc,'diffC',diffSC, ...
                  'mF_cong',mSF_cong,'mF_inc',mSF_inc,'diffF',diffSF);

end
