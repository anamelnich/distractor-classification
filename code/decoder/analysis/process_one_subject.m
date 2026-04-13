function OUT = process_one_subject(subjectID, paths)
% Returns a struct OUT with per-subject metrics + (optional) EEG epochs in cache.

OUT = struct();
OUT.subjectID = subjectID;

%% -------------------- LOAD BEHAVIOR (FAST-ish) --------------------
% Your loadData already brings behavior + headers etc.
data = loadData_box(paths.dataPath, subjectID);  % <-- your existing function
if exist('sopen.mat','file'); delete('sopen.mat'); end

OUT.hasData = ~isempty(fieldnames(data));

%% -------------------- THRESHOLDS --------------------
thrLog = load_thresholds_clean(subjectID, paths.opDir); % helper below
OUT.thrLog = thrLog;

%% -------------------- ONLINE POSTERIORS --------------------
Pcell = load_online_posteriors_clean(subjectID, paths.opDir); % helper below
OUT.Pcell = Pcell; % cell {S1..S5}, each n×3

%% -------------------- RUNWISE METRICS (decoding1..5) --------------------
[runTimeline, sessionInfo] = compute_runwise_metrics(data);
OUT.runTimeline = runTimeline;   % concat_x, acc, tpr, tnr, amb, sess_end_idx
OUT.sessionInfo = sessionInfo;   % per-session #runs etc.

%% -------------------- THRESHOLDS ON SAME TIMELINE --------------------
thrTimeline = align_thresholds_to_runs(thrLog, runTimeline);
OUT.thrTimeline = thrTimeline;   % x_thr, thrR/L/N, margin aligned to concat_x

%% -------------------- AUC/AUPRC PER SESSION --------------------
aucOut = compute_auc_auprc_per_session(data, Pcell);
OUT.aucOut = aucOut;             % auc_session, auprc_session, pr_chance, n_eff

%% -------------------- RT / STROOP EFFECTS (if present) --------------------
OUT.rtOut     = compute_rt_effects(data);      % training1 vs training2 (if exists)
OUT.stroopOut = compute_stroop_effects(data);  % stroop1 vs stroop2 (if exists)

%% -------------------- OPTIONAL: HEAVY EEG PREPROCESS + CACHE (ONLY training1/2) --------------------
try
    cacheFile = fullfile(paths.cacheDir, sprintf('%s_cache_TRAIN_STROOP.mat', subjectID));

    if exist(cacheFile, 'file')
        OUT.cacheFile = cacheFile;
        OUT.cached = true;
        return;
    end

    % --- Build cfg from training1 header ---
    cfg = setParams(data.training1.header);
    cfg.fsamp = data.training1.header.SampleRate;
    cfg.eegChannels     = 1:64;
    cfg.eogChannels     = 65:66;
    cfg.triggerChannel  = 67;

    cfg.chanLabels = data.training1.header.Label;
    cfg.chanLabels(65:67) = [];

    % --- Only keep the fields we care about ---
    keepFields = {'training1','training2','stroop1','stroop2'};
    allF = fieldnames(data);
    for i = 1:numel(allF)
        if ~ismember(allF{i}, keepFields)
            data = rmfield(data, allF{i});
        end
    end

    % ---------- EEG preprocess ONLY for training1/2 ----------
    eegFields = {'training1','training2'};
    for i = 1:numel(eegFields)
        f = eegFields{i};
        if ~isfield(data,f) || isempty(data.(f)), continue; end
        trigtype = 2;
        data.(f) = preprocessDataset(data.(f), cfg, f, trigtype);
    end

    % bandpass
    [b, a] = butter(cfg.spectralFilter.order, cfg.spectralFilter.freqs./(cfg.fsamp/2), 'bandpass');
    cfg.spectralFilter.b = b;
    cfg.spectralFilter.a = a;

    for i = 1:numel(eegFields)
        f = eegFields{i};
        if ~isfield(data,f) || ~isfield(data.(f),'data'), continue; end
        data.(f).data = filter(b, a, data.(f).data);
    end

    % epoch
    for i = 1:numel(eegFields)
        f = eegFields{i};
        if ~isfield(data,f) || ~isfield(data.(f),'index'), continue; end
        d = data.(f);

        epochs.data    = nan(length(cfg.epochSamples), length(cfg.chanLabels), length(d.index.pos));
        epochs.labels  = d.index.typ;
        epochs.file_id = nan(length(d.index.typ), 1);

        for t = 1:length(d.index.pos)
            epochs.data(:,:,t) = d.data(d.index.pos(t) + cfg.epochSamples, :);
            epochs.file_id(t)  = find(d.index.pos(t) <= d.eof, 1, 'first');
        end
        epochs.eof = d.eof;

        data.(f).epochs = epochs;
    end

    % ---------- STOOP: keep behavior only (no EEG needed) ----------
    stroopFields = {'stroop1','stroop2'};
    for i = 1:numel(stroopFields)
        f = stroopFields{i};
        if ~isfield(data,f) || isempty(data.(f)), continue; end
        % strip big raw fields if present
        if isfield(data.(f),'data');   data.(f) = rmfield(data.(f),'data');   end
        if isfield(data.(f),'index');  data.(f) = rmfield(data.(f),'index');  end
        if isfield(data.(f),'epochs'); data.(f) = rmfield(data.(f),'epochs'); end
        if isfield(data.(f),'eof');    data.(f) = rmfield(data.(f),'eof');    end
    end

    % ---------- Save cache ----------
    C = struct();
    C.subjectID = subjectID;
    C.cfg = cfg;
    if isfield(data,'training1'), C.training1 = data.training1; end
    if isfield(data,'training2'), C.training2 = data.training2; end
    if isfield(data,'stroop1'),   C.stroop1   = data.stroop1;   end
    if isfield(data,'stroop2'),   C.stroop2   = data.stroop2;   end

    save(cacheFile, '-struct', 'C', '-v7.3');

    OUT.cacheFile = cacheFile;
    OUT.cached = true;

catch ME
    OUT.cached = false;
    OUT.cacheFile = '';
    OUT.cacheError = ME.message;
end


end
