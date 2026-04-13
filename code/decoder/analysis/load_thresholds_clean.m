function thrLog = load_thresholds_clean(subjectID, opDir)

t = load(fullfile(opDir, sprintf('%s_thrlog.mat', subjectID)));
thrLog = t.thrLog;

% flip thrN
vals = num2cell(1 - [thrLog.thrN]);
[thrLog.thrN] = vals{:};

% parse timestamps
d = {thrLog.timestamp}';
d = datetime(d, 'InputFormat', 'yyyy-MM-dd HH:mm:ss');

T = table((1:numel(thrLog))', d, 'VariableNames', {'origIdx','dt'});
T.dayKey = dateshift(T.dt, 'start', 'day');
T = sortrows(T, 'dt');

[G, ~] = findgroups(T.dayKey);

expectedRuns = [6 8 8 8 6];
T.removed_bug      = false(height(T),1);
T.removed_practice = false(height(T),1);

% Pass 1: bug rows (within-session dt < 4 min)
minGap = minutes(4);
for g = 1:max(G)
    idx = find(G == g);
    if numel(idx) < 2, continue; end
    dtDiff  = diff(T.dt(idx));
    badNext = dtDiff < minGap;
    badPrev = [badNext; false];       % drop earlier row
    badRows = idx(badPrev);
    T.removed_bug(badRows) = true;
end
T = T(~T.removed_bug, :);

% regroup
[G, ~] = findgroups(T.dayKey);

% Pass 2: remove practice if more than expected
nSess = max(G);
for g = 1:nSess
    idx = find(G == g);
    nRuns = numel(idx);
    expN = (g <= numel(expectedRuns)) * expectedRuns(min(g,numel(expectedRuns))) + (g > numel(expectedRuns))*nRuns;
    if nRuns > expN
        T.removed_practice(idx(1)) = true; % first row is practice
    end
end
T = T(~T.removed_practice, :);

% regroup + assign session/run
[G, ~] = findgroups(T.dayKey);
T.Session = G;
runCells = splitapply(@(x){(1:numel(x))'}, T.dt, G);
T.Run = vertcat(runCells{:});

% write back to struct
thrLog_clean = thrLog(T.origIdx);
for i = 1:numel(thrLog_clean)
    thrLog_clean(i).Session = T.Session(i);
    thrLog_clean(i).Run     = T.Run(i);
end
thrLog = thrLog_clean;

end
