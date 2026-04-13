function data = loadData(dataPath, subjectID)
% loadData loads EEG and behavioral data for a subject across multiple days
%
%   data = loadData(dataPath, subjectID)
%
% Output:
%   data.<taskType><n> – struct with fields:
%       .data, .header, .eof   (EEG)
%       .beh                   (behavior struct)

% 1) Find all day folders
dayFolders = dir(fullfile(dataPath, [subjectID '_20*']));
if isempty(dayFolders)
    error('No day folders found for subject %s in %s', subjectID, dataPath);
end

% 2) Collect sessions by task and day
sessions = struct();
for d = 1:numel(dayFolders)
    dayName = dayFolders(d).name;
    dayPath = fullfile(dayFolders(d).folder, dayName);
    % extract YYYYMMDD
    tk = regexp(dayName, ['^' subjectID '_(\d{8})$'], 'tokens');
    if isempty(tk), continue; end
    dayField = ['d' tk{1}{1}];

    % list subfolders
    subDirs = dir(dayPath);
    subNames = setdiff({subDirs([subDirs.isdir]).name}, {'.','..'});
    for iSub = 1:numel(subNames)
        sub = subNames{iSub};
        tk2 = regexp(sub, ['^' subjectID '_\d{14}_(\w+)$'], 'tokens');
        if isempty(tk2)
            fprintf('SKIP (regex no match): %s\n', sub);
            continue;
        end


        subP = fullfile(dayPath, sub);
        tk2 = regexp(sub, ['^' subjectID '_\d{14}_(\w+)$'], 'tokens');
        if isempty(tk2), continue; end
        taskType = lower(tk2{1}{1});

        % load EEG
        eeg = [];
        gdfF = dir(fullfile(subP,'*.gdf'));
        if ~isempty(gdfF)
            fp = fullfile(gdfF(1).folder, gdfF(1).name);
            [sig, hdr] = sload(fp);
            eeg.data   = sig;
            eeg.header = hdr;
            eeg.eof    = size(sig,1);
        else
            fprintf('MISSING GDF: %s\n', subP);
        end

        % load behavior
        beh = [];
    
        switch taskType
            case 'stroop'
                f = fullfile(subP, [sub '.behoutput.txt']);
                if ~isfile(f)
                    fprintf('MISSING stroop beh: %s\n', f);
                else
                    beh = loadStroop(f);
                end
            case {'training','decoding','validation'}
                af = fullfile(subP, [sub '.analysis.txt']);
                tf = fullfile(subP, [sub '.triggers.txt']);
                if ~isfile(af), fprintf('MISSING analysis: %s\n', af); end
                if ~isfile(tf), fprintf('MISSING triggers: %s\n', tf); end
                if isfile(af) && isfile(tf)
                    beh = loadAnalysis(af, tf, taskType,subjectID);
                end
        end

        % group sessions
        if ~isfield(sessions, taskType)
            sessions.(taskType) = struct();
        end
        if ~isfield(sessions.(taskType), dayField)
            sessions.(taskType).(dayField) = { struct('eeg',eeg,'beh',beh) };
        else
            sessions.(taskType).(dayField){end+1} = struct('eeg',eeg,'beh',beh);
        end
    end
end

% 3) Concatenate per-day and build output fields
data = struct();
types = fieldnames(sessions);
for t = 1:numel(types)
    typ = types{t};
    days = sort(fieldnames(sessions.(typ)));
    for iDay = 1:numel(days)
        list = sessions.(typ).(days{iDay});  % cell of structs
        % concatenate EEG
        combinedEeg = [];
        for j = 1:numel(list)
            if ~isempty(list{j}.eeg)
                combinedEeg = concatenateSession(combinedEeg, list{j}.eeg);
            end
        end
        % concatenate behavior
        combinedBeh = [];
        for j = 1:numel(list)
            if ~isempty(list{j}.beh)
                combinedBeh = concatenateBehSession(combinedBeh, list{j}.beh);
            end
        end
        % attach behavior to EEG struct
        combinedEeg.beh = combinedBeh;
        % assign to data.<taskType><n>
        fieldName = [typ num2str(iDay)];
        data.(fieldName) = combinedEeg;
    end
end
end

%% ─── Subfunction: loadStroop ──────────────────────────────────────────────
function beh = loadStroop(file)
    opts = detectImportOptions(file,'FileType','text','Delimiter','\t');
    opts.DataLines = [2 Inf];
    T = readtable(file, opts);
    beh = table2struct(T, 'ToScalar', true);
    n = height(T);
    beh.trial_type = zeros(n,1);
    for k = 1:n
        s = lower(T.Trial_Type{k});
        switch s
            case 'neutral'
                v = 0;
            case 'congruent'
                v = 1;
            case 'incongruent'
                v = 2;
            otherwise
                v = NaN;
        end
        beh.trial_type(k) = v;
    end
end

%% ─── Subfunction: loadAnalysis ───────────────────────────────────────────
function beh = loadAnalysis(analysisFile, triggersFile, taskType, subjectID)
    % Read behavioral and trigger data
    A = readmatrix(analysisFile);
    Traw = readmatrix(triggersFile);
    % Clean triggers
    Tclean = Traw;
    Tclean(Tclean(:,2)==6 | Tclean(:,2)==60, :) = [];
    dup = find(diff(Tclean(:,3))==1) + 1;
    Tclean(dup, :) = [];

    % Always store cleaned triggers
    triggers = Tclean;

    % Determine variable names based on column count
    baseVars = {'trial','trial_type','response','tpos','dpos','dot','ITI','BCI_output'};
    ncol = size(A,2);
    if strcmp(taskType,'decoding')
        if ncol == numel(baseVars)+1
            vars = [baseVars, {'class'}];
        elseif ncol == numel(baseVars)
            vars = baseVars;
        else
            error('Unexpected number of columns (%d) for decoding in %s', ncol, analysisFile);
        end
    else
        if ncol ~= numel(baseVars)
            error('Unexpected number of columns (%d) for %s in %s', ncol, taskType, analysisFile);
        end
        vars = baseVars;
    end

    % Build behavior struct
    beh = cell2struct(mat2cell(A, size(A,1), ones(1,ncol)), vars, 2);

    % Attach triggers
    beh.triggers = triggers;

    % Compute RT with error handling
    try
        
        starts = triggers(ismember(triggers(:,2), [8 32 44]), 3);
        resp   = triggers(triggers(:,2)==64,3);
        fprintf('[%s] %s: #starts=%d, #resp=%d\n', subjectID, taskType, numel(starts), numel(resp));

        beh.RT = resp - starts;
    catch ME
        warning('Error computing RT for %s %s: %s', subjectID,taskType, ME.message);
    end
end

%% ─── Subfunction: concatenateSession ────────────────────────────────────
function combined = concatenateSession(combined, newS)
    % If this is the first session, just copy it over (and keep its eof)
    if isempty(combined)
        combined        = newS;
        combined.eof    = newS.eof(:);   % make sure it’s a column
        return;
    end

    % How many samples we already had?
    prevLen = size(combined.data,1);

    % Merge EVENT info if present
    if isfield(combined.header,'EVENT') && isfield(newS.header,'EVENT')
        pos = newS.header.EVENT.POS + prevLen;
        combined.header.EVENT.TYP = [combined.header.EVENT.TYP; newS.header.EVENT.TYP];
        combined.header.EVENT.POS = [combined.header.EVENT.POS; pos];
    end

    % Concatenate the raw data
    combined.data = [combined.data; newS.data];

    % Append the new eof *offset* by how many we already had
    combined.eof  = [combined.eof; newS.eof(:) + prevLen];
end


%% ─── Subfunction: concatenateBehSession ─────────────────────────────────
function combined = concatenateBehSession(combined, newB)
    if isempty(combined), combined = newB; return; end
    flds = fieldnames(newB);
    for i = 1:numel(flds)
        fld = flds{i};
        if isfield(combined, fld)
            combined.(fld) = [combined.(fld); newB.(fld)];
        else
            warning('concatenateBehSession:MissingField', 'Field "%s" not in combined—skipping.', fld);
        end
    end
end