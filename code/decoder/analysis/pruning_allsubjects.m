
% Script to run computeModel w/ pruning for multiple subjects, save results, and plot

% ----- User settings -----
subjectList = {'e1','e2','e3','e4','e5','e6'};  % <-- fill in your subject IDs
resultsDir  = 'results';           % directory to save per-subject .mat files

% Create results directory if needed
if ~exist(resultsDir,'dir')
    mkdir(resultsDir);
end

% ----- Loop: compute and save per-subject data -----
nSubs = numel(subjectList);
trainingDataAll = cell(nSubs,1);
bestItrDataAll = cell(nSubs,1);
for iSub = 1:nSubs
    subjID = subjectList{iSub};
    fprintf('Running computeModel for %s...\n', subjID);
    [triData, biData] = computeModel(subjID);
    trainingDataAll{iSub} = triData;
    bestItrDataAll{iSub}  = biData;
    % save(fullfile(resultsDir, [subjID '_results.mat']), 'triData', 'biData');
end
% save(fullfile(resultsDir,'all_results.mat'), 'trainingDataAll', 'bestItrDataAll');

%% ----- Aggregate across subjects -----
% Start with first subject's data
bigOrig = trainingDataAll{1};
bigBest = bestItrDataAll{1};
for i = 2:nSubs
    td = trainingDataAll{i};
    bd = bestItrDataAll{i};
    % concatenate trials
    bigOrig.data       = cat(3, bigOrig.data,       td.data);
    bigOrig.labels     = [bigOrig.labels;     td.labels];
    bigOrig.file_id    = [bigOrig.file_id;    td.file_id];
    bigOrig.RT         = [bigOrig.RT;         td.RT];
    bigOrig.tpos       = [bigOrig.tpos;       td.tpos];
    bigOrig.posteriors = [bigOrig.posteriors; td.posteriors];

    bigBest.data       = cat(3, bigBest.data,       bd.data);
    bigBest.labels     = [bigBest.labels;     bd.labels];
    bigBest.file_id    = [bigBest.file_id;    bd.file_id];
    bigBest.RT         = [bigBest.RT;         bd.RT];
    bigBest.tpos       = [bigBest.tpos;       bd.tpos];
    bigBest.posteriors = [bigBest.posteriors; bd.posteriors];
end

%% ----- Plot averaged results -----
% Assume cfg is in workspace or loaded externally
% 1) ERP difference waves

plotERPpruned(bigOrig, bigBest, cfg);

%% 2) Target position distributions

plotTargetPosDistribution(bigOrig, bigBest, cfg);

% 3) Run pruning summary

plotRunPruning(bigOrig, bigBest);
