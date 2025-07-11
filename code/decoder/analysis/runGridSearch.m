
% runGridSearch_parallel.m

% 1) Build the full list of tasks ---------------------------------------

subjects = {'e1','e2','e3','e4','e5','e6','e8','e10','e11','e12','e13','e14','e15'};
% subjects = {'e1','e2','e3','e4','e5','e6'};

addpath(genpath('../functions'));
tasks = struct('subject',{},'cfg',{});  % preallocate empty
tidx = 0;

for si = 1:numel(subjects)
    subjID = subjects{si};
    fprintf('Queuing %s…\n', subjID);


    for baseline = true
      for balance = [true,false]
        for roiOpt = {'None'}
          baseCfg = defaultCfg();
          baseCfg.baseline_iscompute = baseline;
          baseCfg.balance_iscompute  = balance;
          baseCfg.roi                = roiOpt{1};

          for sfType = {'xDAWN'}
            baseCfg.spatialFilter.type = sfType{1};
            if strcmp(sfType{1},'None')
              compsList = NaN;
            else
              compsList = 2;
            end
            for nComp = compsList
              baseCfg.spatialFilter.nComp = nComp;

              % your 3 PSD options
              psdOptions = {...
                struct('is_compute',false,'roi',[],'freq_range',[]),...
                struct('is_compute',true,'roi','all','freq_range',8:1:14)};
              for p = psdOptions
                opt = p{1};
                baseCfg.psd.is_compute = opt.is_compute;
                baseCfg.psd.roi        = opt.roi;
                baseCfg.psd.freq_range = opt.freq_range;

                for erpFlag = [true,false]
                  baseCfg.features.erp_iscompute = erpFlag;
                  for diffFlag = [true,false]
                    baseCfg.features.diffwave_iscompute = diffFlag;
                    for normFlag = true
                      baseCfg.classify.is_normalize = normFlag;
                      if normFlag
                        normTypes = {'zscore'};
                      else
                        normTypes = {''};
                      end
                      for nt = normTypes
                        baseCfg.classify.normtype = nt{1};

                        for redType = {'r2'}
                          baseCfg.classify.reduction.type = redType{1};
                          for nFeat = 30
                            baseCfg.reduction.numfeats = nFeat;
                            for mdl = {'linear'}
                              cfg = baseCfg;            % copy the “base” + these few overrides
                              cfg.classify.type = mdl{1};

                              % queue up one task
                              tidx = tidx + 1;
                              tasks(tidx).subject = subjID;
                              tasks(tidx).cfg     = cfg;

                            end
                          end
                        end

                      end
                    end
                  end
                end

              end
            end
          end
        end
      end
    end
end

numTasks = numel(tasks);
results5(numTasks) = struct('subject',[],'cfg',[],'perf',[],'error',[]);

% 2) Run them in parallel -----------------------------------------------
parpool('local');  % start a pool if you haven't already

parfor ti = 1:numTasks
    subjID = tasks(ti).subject;
    cfg     = tasks(ti).cfg;

    results5(ti).subject = subjID;
    results5(ti).cfg     = cfg;
    try
        perf = computeModel(subjID, cfg);
        results5(ti).perf  = perf;
        results5(ti).error = [];
    catch ME
        warning('Task %d (%s) failed: %s', ti, subjID, ME.message);
        results5(ti).perf  = struct('aucRight',NaN,'kappaRight',NaN,'aucLeft',NaN,'kappaLeft',NaN);
        results5(ti).error = ME;
    end
end

% 3) Save ----------------------------------------------------------------
T5 = struct2table(results5);
save('gridSearchResults5_parfor.mat','results5','T5');
writetable(T5,'gridSearchResults5_parfor.csv');
fprintf('Done. %d tasks completed.\n', numTasks);


%% Plot results
subjects = {'e1','e2','e3','e4','e5','e6'};
plotGridSearch(T2, 'auprc', subjects);

%% Find top configurations
topConfigsB = findTopConfigs(T4, 'auprc', 'bilateral', 5);
topConfigsR = findTopConfigs(T4, 'auprc', 'right', 5);
topConfigsL = findTopConfigs(T4, 'auprc', 'left', 5);
%%
idx = [3];
selected = T2.cfg(idx);

[cf, cv] = findCommonConfigFields(T2, idx); 



%% Find bottom configurations
bottomB = findBottomConfigs(T, 'auprc', 'bilateral', 10, subjects);
bottomR = findBottomConfigs(T, 'auprc', 'right', 10, subjects);
bottomL = findBottomConfigs(T, 'auprc', 'left', 10, subjects);

%% Find top params
subjects = {'e1','e2','e3','e4','e5','e6'};
summary = analyzeParamEffect(T, 'spatialFilter.type', 'auprc', 'right',subjects);