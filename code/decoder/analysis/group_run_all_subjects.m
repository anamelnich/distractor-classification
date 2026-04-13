function GROUP = group_run_all_subjects(subjectIDs, doParallel)
% GROUP = group_run_all_subjects({'e21','e22','e23'}, true)
%
% - Adds ./analysis and ../functions to path
% - Per subject:
%     * load thresholds (thrlog)
%     * load online posteriors (decoding1..5)
%     * load behavior (loadData)
%     * (optional heavy) preprocess EEG + epoch and cache
%     * compute metrics
% - Group:
%     * mean±SEM plots across subjects

%% -------------------- PATHS --------------------
thisDir = fileparts(mfilename('fullpath'));   % .../analysis
addpath(genpath(thisDir));                    % REQUIRED: add ./analysis
addpath(genpath(fullfile(thisDir, '..', '..', 'functions')));

% ---- adjust these if needed ----
% paths.dataPath   = fullfile(pwd, '..','..','data');                   % your loadData base
paths.dataPath = '/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction/project_healthy';
paths.opDir      = '/Users/hililbby/Library/CloudStorage/Box-Box/CNBI/Attention_distraction/project_healthy/online_info'; % OnlinePosteriors + thrlog
paths.cacheDir   = fullfile(thisDir, 'cache');                        % where we store preprocessed EEG/epochs
if ~exist(paths.cacheDir, 'dir'); mkdir(paths.cacheDir); end

%% -------------------- PARPOOL --------------------
if nargin < 2, doParallel = true; end
if doParallel
    p = gcp('nocreate');
    if isempty(p)
        parpool('IdleTimeout', 120); %#ok<*NOPRT>
    end
end

%% -------------------- RUN SUBJECTS --------------------
nS = numel(subjectIDs);
SUBJ = cell(nS,1);

if doParallel
    parfor si = 1:nS
        SUBJ{si} = process_one_subject(subjectIDs{si}, paths);
    end
else
    for si = 1:nS
        SUBJ{si} = process_one_subject(subjectIDs{si}, paths);
    end
end

%% -------------------- GROUP AGGREGATION --------------------
GROUP = group_aggregate_and_plot(SUBJ, subjectIDs);
% --- Collect cache files that were successfully created/found ---
cacheFiles = cell(nS,1);
for si = 1:nS
    if ~isempty(SUBJ{si}) && isfield(SUBJ{si},'cacheFile') && ~isempty(SUBJ{si}.cacheFile) ...
            && isfile(SUBJ{si}.cacheFile)
        cacheFiles{si} = SUBJ{si}.cacheFile;
    end
end
cacheFiles = cacheFiles(~cellfun(@isempty, cacheFiles));

% --- Group EEG plots ---
tmp = load('chanlocs64.mat');         % adjust var name if needed
if isfield(tmp,'chanlocs')
    chanlocs = tmp.chanlocs;
else
    % if your file loads a different variable name, set it here:
    chanlocs = tmp.chanlocs64; 
end

GROUP.GPD = group_plot_pd_po78(cacheFiles);                 % group Pd waveform (PO7/PO8)
GROUP.GT  = group_plot_pd_topos(cacheFiles, chanlocs);      % group topo (combined contra MAV/AUC)
GROUP.GRT = group_plot_rt_and_stroop_prepost(cacheFiles);

stamp = datestr(now, 'yyyymmdd_HHMMSS');
groupFile = fullfile(paths.cacheDir, ['GROUP_' stamp '.mat']);
save(groupFile, 'GROUP', '-v7.3');
end
