function out = computePdR2_pairDiffTopos(data, params, chanlocFile)
% computePdR2_pairDiffTopos
%
% Computes pairwise hemisphere-difference features (L-R and R-L), then r^2 for:
%   (1) Right distractor (label==1) vs Rest (labels==0 or 2) using  (L-R)
%   (2) Left  distractor (label==2) vs Rest (labels==0 or 1) using  (R-L)
%
% Steps:
%  1) Baseline correction (per trial, per channel)
%  2) Build L/R homologous pairs from params.chanLabels (excluding midline, EOG, M1, M2)
%  3) For each pair, compute mean(abs(diff)) in 0.2–0.5 s window (0 sec = sample 256)
%  4) Compute signed and unsigned r^2 per pair
%  5) Plot topoplots (pair value duplicated onto both electrodes for visualization)
%
% Inputs:
%  data.epochs.data   : [nTime x nChan x nTrials] (e.g., 768 x 64 x 480)
%  data.epochs.labels : [nTrials x 1] with {0=ND, 1=RD, 2=LD}
%  params.chanLabels  : [nChan x 1] cellstr channel names (64x1)
%  params.epochTime   : (optional) [nTime x 1] time in seconds
%  params.baseline_window : (optional) [t1 t2] seconds, default [-0.2 0]
%  params.fsamp       : (optional) sampling rate (Hz), used if epochTime missing
%  chanlocFile        : (optional) path to chan64.mat (EEGLAB-style chanlocs struct)
%
% Output struct out contains:
%  out.pairs            : table of (L,R) labels/indices
%  out.r2_RD_signed     : [nPairs x 1]
%  out.r2_RD_unsigned   : [nPairs x 1]
%  out.r2_LD_signed     : [nPairs x 1]
%  out.r2_LD_unsigned   : [nPairs x 1]
%  out.map64_RD_signed  : [64 x 1] values for topoplot (duplicated per pair)
%  out.map64_LD_signed  : [64 x 1]
%
% Notes:
%  - If you want ND-only as “rest”, change masks below.
%  - Requires EEGLAB topoplot() if you plot.

%% ---- pull arrays ----
X = data.epochs.data;    % nTime x nChan x nTrials
y = data.epochs.labels(:);
assert(ndims(X)==3, 'data.epochs.data must be time x chan x trials');
[nTime,nChan,nTr] = size(X);
assert(numel(y)==nTr, 'labels length must match #trials');

chanLabels = params.chanLabels(:);
assert(numel(chanLabels)==nChan, 'params.chanLabels must match #channels');

%% ---- time vector + indices ----
t = local_get_time_vector(params, nTime);   % seconds, length nTime

% baseline window
if isfield(params,'baseline_window') && numel(params.baseline_window)==2
    bw = params.baseline_window;
else
    bw = [-0.2 0];
end
bIdx = (t >= bw(1)) & (t <= bw(2));
assert(any(bIdx), 'Baseline window has no samples. Check epochTime/fsamp.');

% Pd window (0.2–0.5s)
wIdx = (t >= 0.2) & (t <= 0.5);
assert(any(wIdx), '0.2–0.5s window has no samples. Check epochTime/fsamp.');

%% ---- 1) baseline correction (per trial, per channel) ----
% baseline: 1 x nChan x nTrials
base = mean(X(bIdx,:,:), 1);
Xbc  = X - base;

%% ---- 2) build L/R pairs (exclude midline + EOG + M1/M2) ----
excludeNames = {'EOG','M1','M2','TP7','TP8','FT7','FT8','T7','T8','FP2','FP1','AF7','AF3','AF4','AF8'};
isExcl = false(nChan,1);

for i = 1:nChan
    lab = upper(strtrim(chanLabels{i}));
    if any(strcmpi(lab, excludeNames))
        isExcl(i) = true;
        continue;
    end
    % midline heuristic: ends with Z (FPZ, FZ, FCZ, CZ, PZ, POZ, OZ, AFZ, etc.)
    if endsWith(lab,'Z')
        isExcl(i) = true;
        continue;
    end
end

% map label -> index (case-insensitive)
labMap = containers.Map();
for i = 1:nChan
    labMap(upper(strtrim(chanLabels{i}))) = i;
end

% candidate "left" electrodes: those with odd/left-sided suffixes that have a right counterpart
pairs = []; % rows: [iL iR]
pairL = {};
pairR = {};

for i = 1:nChan
    if isExcl(i), continue; end
    Llab = upper(strtrim(chanLabels{i}));

    % treat as left if it has a valid right counterpart per naming rules
    Rlab = local_right_counterpart(Llab);
    if isempty(Rlab), continue; end
    if ~isKey(labMap, Rlab), continue; end

    iR = labMap(Rlab);
    if isExcl(iR), continue; end

    % ensure we only add each pair once
    if i < iR
        pairs(end+1,:) = [i iR]; %#ok<AGROW>
        pairL{end+1,1} = Llab;   %#ok<AGROW>
        pairR{end+1,1} = Rlab;   %#ok<AGROW>
    end
end

assert(~isempty(pairs), 'No L/R pairs found. Check chanLabels naming.');

nPairs = size(pairs,1);

%% ---- 3) compute pair-difference features (MAV + positive AUC) ----
% Preallocate feature vectors per pair (nPairs x nTrials)
feat_LminusR_MAV    = zeros(nPairs, nTr);
feat_RminusL_MAV    = zeros(nPairs, nTr);
feat_LminusR_AUCpos = zeros(nPairs, nTr);
feat_RminusL_AUCpos = zeros(nPairs, nTr);

dt = mean(diff(t(wIdx)));              % seconds per sample in window
if ~isfinite(dt) || dt<=0
    % fallback (shouldn't happen if t is sane)
    dt = 1;
end

for p = 1:nPairs
    iL = pairs(p,1);
    iR = pairs(p,2);

    % diff time x trials
    dLR = squeeze(Xbc(:,iL,:) - Xbc(:,iR,:));   % [nTime x nTrials]
    dRL = -dLR;

    % --- Mean absolute value (MAV) ---
    feat_LminusR_MAV(p,:) = mean(abs(dLR(wIdx,:)), 1);
    feat_RminusL_MAV(p,:) = mean(abs(dRL(wIdx,:)), 1);

    % --- Positive area under curve (AUCpos) ---
    % area = sum(positive part) * dt  (units: uV*s)
    feat_LminusR_AUCpos(p,:) = sum(max(dLR(wIdx,:), 0), 1) * dt;
    feat_RminusL_AUCpos(p,:) = sum(max(dRL(wIdx,:), 0), 1) * dt;
end


%% ---- 4) class masks + r^2 per pair (MAV + AUCpos) ----
mRD1 = (y==1);
mRD0 = (y~=1);  % rest = 0 or 2
mLD1 = (y==2);
mLD0 = (y~=2);  % rest = 0 or 1

% MAV r2
r2_RD_MAV_u = zeros(nPairs,1);
r2_LD_MAV_u = zeros(nPairs,1);

% AUCpos r2
r2_RD_AUC_u = zeros(nPairs,1);
r2_LD_AUC_u = zeros(nPairs,1);

for p = 1:nPairs
    xRD = feat_LminusR_MAV(p,:);
    [r2_RD_MAV_u(p), ~] = local_r2_signed(xRD(mRD1), xRD(mRD0));

    xLD = feat_RminusL_MAV(p,:);
    [r2_LD_MAV_u(p), ~] = local_r2_signed(xLD(mLD1), xLD(mLD0));

    xRD = feat_LminusR_AUCpos(p,:);
    [r2_RD_AUC_u(p), ~] = local_r2_signed(xRD(mRD1), xRD(mRD0));

    xLD = feat_RminusL_AUCpos(p,:);
    [r2_LD_AUC_u(p), ~] = local_r2_signed(xLD(mLD1), xLD(mLD0));
end

%% ---- 5) build 64-channel vectors for topoplot (CONTRA ONLY) ----
map_RD_MAV = nan(nChan,1);   % RD contra only (LEFT hemi)
map_LD_MAV = nan(nChan,1);   % LD contra only (RIGHT hemi)
map_RD_AUC = nan(nChan,1);
map_LD_AUC = nan(nChan,1);

for p = 1:nPairs
    iL = pairs(p,1);
    iR = pairs(p,2);

    map_RD_MAV(iL) = r2_RD_MAV_u(p);
    map_LD_MAV(iR) = r2_LD_MAV_u(p);

    map_RD_AUC(iL) = r2_RD_AUC_u(p);
    map_LD_AUC(iR) = r2_LD_AUC_u(p);
end

map_RD_MAV(isExcl) = nan;  map_LD_MAV(isExcl) = nan;
map_RD_AUC(isExcl) = nan;  map_LD_AUC(isExcl) = nan;

out.map_RD_MAV_contra = map_RD_MAV;
out.map_LD_MAV_contra = map_LD_MAV;
out.map_RD_AUC_contra = map_RD_AUC;
out.map_LD_AUC_contra = map_LD_AUC;

%% ---- plot (optional): 2x2 (MAV + AUCpos) ----
if nargin >= 3 && ~isempty(chanlocFile)

    if ischar(chanlocFile) || (isstring(chanlocFile) && isscalar(chanlocFile))
        chanlocs_in = local_load_chanlocs(chanlocFile);
    elseif isstruct(chanlocFile) && isfield(chanlocFile, 'labels')
        chanlocs_in = chanlocFile;
    else
        error('Third input must be a chanlocs struct array or a .mat filename.');
    end

    [chanlocs_ord, idxFound] = local_reorder_chanlocs_by_labels(chanlocs_in, params.chanLabels);

    % settings
    plotArgs = {'electrodes','labels','plotrad',0.7,'headrad',0.50,'electcolor','w'};

    % --- MAV ---
    v1 = out.map_RD_MAV_contra(:);
    v2 = out.map_LD_MAV_contra(:);
    keep1 = ~isnan(idxFound) & ~isnan(v1);
    keep2 = ~isnan(idxFound) & ~isnan(v2);
    vv = [v1(keep1); v2(keep2)];
    clim_MAV = [0 max(vv)];
    if isempty(clim_MAV(2)) || clim_MAV(2)==0, clim_MAV = [0 1e-3]; end

    % --- AUCpos ---
    a1 = out.map_RD_AUC_contra(:);
    a2 = out.map_LD_AUC_contra(:);
    keep3 = ~isnan(idxFound) & ~isnan(a1);
    keep4 = ~isnan(idxFound) & ~isnan(a2);
    aa = [a1(keep3); a2(keep4)];
    clim_AUC = [0 max(aa)];
    if isempty(clim_AUC(2)) || clim_AUC(2)==0, clim_AUC = [0 1e-3]; end

    % figure('units','normalized','position',[0.05 0.1 0.9 0.7]);
    % 
    % % 1) RD MAV
    % subplot(2,2,1);
    % topoplot(v1(keep1), chanlocs_ord(keep1), plotArgs{:});
    % caxis(clim_MAV); colorbar;
    % title('MAV: RD vs Rest (CONTRA LEFT)');
    % 
    % % 2) LD MAV
    % subplot(2,2,2);
    % topoplot(v2(keep2), chanlocs_ord(keep2), plotArgs{:});
    % caxis(clim_MAV); colorbar;
    % title('MAV: LD vs Rest (CONTRA RIGHT)');
    % 
    % % 3) RD AUCpos
    % subplot(2,2,3);
    % topoplot(a1(keep3), chanlocs_ord(keep3), plotArgs{:});
    % caxis(clim_AUC); colorbar;
    % title('Pos AUC: RD vs Rest (CONTRA LEFT)');
    % 
    % % 4) LD AUCpos
    % subplot(2,2,4);
    % topoplot(a2(keep4), chanlocs_ord(keep4), plotArgs{:});
    % caxis(clim_AUC); colorbar;
    % title('Pos AUC: LD vs Rest (CONTRA RIGHT)');
end


%% ---- 6) Combined contralateral maps (MAV + AUCpos) ----
map_combined_MAV = nan(nChan,1);
map_combined_AUC = nan(nChan,1);

for p = 1:nPairs
    iL = pairs(p,1);
    iR = pairs(p,2);

    % LEFT hemi comes from RD
    map_combined_MAV(iL) = r2_RD_MAV_u(p);
    map_combined_AUC(iL) = r2_RD_AUC_u(p);

    % RIGHT hemi comes from LD
    map_combined_MAV(iR) = r2_LD_MAV_u(p);
    map_combined_AUC(iR) = r2_LD_AUC_u(p);
end

map_combined_MAV(isExcl) = nan;
map_combined_AUC(isExcl) = nan;

out.map_combined_MAV = map_combined_MAV;
out.map_combined_AUC = map_combined_AUC;
%% ---- Combined Contralateral: MAV + AUC in ONE figure ----

v_comb_MAV = out.map_combined_MAV(:);
v_comb_AUC = out.map_combined_AUC(:);

keep_MAV = ~isnan(idxFound) & ~isnan(v_comb_MAV);
keep_AUC = ~isnan(idxFound) & ~isnan(v_comb_AUC);

% Color limits
vv = v_comb_MAV(keep_MAV);
clim_MAV = [0 max(vv)];
if isempty(clim_MAV(2)) || clim_MAV(2)==0
    clim_MAV = [0 1e-3];
end

aa = v_comb_AUC(keep_AUC);
clim_AUC = [0 max(aa)];
if isempty(clim_AUC(2)) || clim_AUC(2)==0
    clim_AUC = [0 1e-3];
end

% figure('units','normalized','position',[0.3 0.3 0.6 0.45]);

% %% ---- MAV ----
% subplot(1,2,1)
% topoplot(v_comb_MAV(keep_MAV), chanlocs_ord(keep_MAV), ...
%     plotArgs{:});
% caxis(clim_MAV);
% colorbar;
% title('Combined Contralateral – MAV');
% 
% %% ---- AUC ----
% subplot(1,2,2)
% topoplot(v_comb_AUC(keep_AUC), chanlocs_ord(keep_AUC), ...
%     plotArgs{:});
% caxis(clim_AUC);
% colorbar;
% title('Combined Contralateral – Positive AUC');




% --- package outputs ---
out = struct();
out.pairs = table(pairs(:,1), pairs(:,2), pairL, pairR, ...
    'VariableNames', {'idxL','idxR','labelL','labelR'});
out.isExcluded = isExcl;
out.time = t;
out.winIdx = wIdx;

% IMPORTANT: store the actual 64×1 maps (not v_comb_*)
out.map_combined_MAV = map_combined_MAV;
out.map_combined_AUC = map_combined_AUC;




end

%% ===================== helpers =====================

function t = local_get_time_vector(params, nTime)
% Prefer params.epochTime. Otherwise use fsamp and "0 sec = sample 256".
    if isfield(params,'epochTime') && numel(params.epochTime)==nTime
        t = params.epochTime(:);
        return;
    end

    assert(isfield(params,'fsamp') && ~isempty(params.fsamp), ...
        'Need params.epochTime or params.fsamp to define time.');
    fs = params.fsamp;

    sampleZero = 256; % given by you: time 0 is sample 256
    % samples are 1..nTime
    t = ((1:nTime) - sampleZero) ./ fs;
    t = t(:);
end

function Rlab = local_right_counterpart(Llab)
% Given a label like 'P7' or 'AF3' or 'P9' returns 'P8'/'AF4'/'P10'.
% Returns '' if it doesn't look like a left-lateral label.
    Llab = upper(strtrim(Llab));

    % must end with a digit
    d = regexp(Llab,'(\d+)$','tokens','once');
    if isempty(d)
        Rlab = '';
        return;
    end
    num = str2double(d{1});
    prefix = regexprep(Llab,'\d+$','');

    % only map odd->even and 9->10
    if num==9
        Rnum = 10;
    elseif mod(num,2)==1
        Rnum = num + 1;
    else
        % already even => treat as right, don't create a pair from it
        Rlab = '';
        return;
    end

    Rlab = sprintf('%s%d', prefix, Rnum);
end

function [r2_unsigned, r2_signed] = local_r2_signed(x1, x0)
% r^2 = (mu1-mu0)^2 / (var1+var0); signed by sign(mu1-mu0)
    x1 = x1(:); x0 = x0(:);
    mu1 = mean(x1); mu0 = mean(x0);
    v1  = var(x1, 1);  % population var (divide by N) for stability
    v0  = var(x0, 1);

    denom = (v1 + v0);
    if denom <= 0 || isnan(denom)
        r2_unsigned = 0;
        r2_signed   = 0;
        return;
    end

    r2_unsigned = ((mu1 - mu0)^2) / denom;
    r2_signed   = sign(mu1 - mu0) * r2_unsigned;
end
function [chanlocs_ord, idxInChanlocs] = local_reorder_chanlocs_by_labels(chanlocs, chanLabels)
% Reorder chanlocs struct array to match chanLabels (case-insensitive).
% Returns:
%   chanlocs_ord     : 1 x nChan struct array in chanLabels order (missing -> empty struct)
%   idxInChanlocs    : nChan x 1 indices into original chanlocs (NaN if not found)

    nChan = numel(chanLabels);
    chanlocs_ord  = repmat(struct('labels',[],'theta',[],'radius',[],'X',[],'Y',[],'Z',[], ...
                                 'sph_theta',[],'sph_phi',[],'sph_radius',[],'type',[],'urchan',[],'ref',[]), ...
                           1, nChan);
    idxInChanlocs = nan(nChan,1);

    % build lookup: LABEL -> index
    clabs = cellfun(@(s) upper(strtrim(s)), {chanlocs.labels}, 'UniformOutput', false);
    for i = 1:nChan
        targ = upper(strtrim(chanLabels{i}));
        j = find(strcmp(clabs, targ), 1, 'first');
        if ~isempty(j)
            chanlocs_ord(i)  = chanlocs(j);
            idxInChanlocs(i) = j;
        else
            % leave empty struct; caller should skip this index
            idxInChanlocs(i) = nan;
        end
    end
end
