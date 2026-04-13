function thrTimeline = align_thresholds_to_runs(thrLog, runTimeline)

if isempty(thrLog) || isempty(runTimeline) || isempty(runTimeline.concat_x)
    thrTimeline = [];
    return;
end

S = thrLog;
Session = [S.Session]';
Run     = [S.Run]';
Margin  = [S.margin]';
ThrR    = [S.thrR]';
ThrL    = [S.thrL]';
ThrN    = [S.thrN]';

T = table(Session, Run, Margin, ThrR, ThrL, ThrN, ...
    'VariableNames', {'Session','Run','Margin','ThresholdR','ThresholdL','ThresholdN'});

% Mirror your concat timeline: 5 sessions + 1-run gap
gapRuns = 1;
sessions = 1:5;

thrR_all = []; thrL_all = []; thrN_all = []; marg_all = [];
x_cursor = 0;

% We need nruns per session from runTimeline:
% (we infer by splitting concat_acc on NaN gaps)
acc = runTimeline.concat_acc;
isGap = isnan(acc);
blocks = find_blocks(~isGap); % helper below: consecutive true segments

expectedRuns = [6 8 8 8 6];

for si = 1:5
    expN = expectedRuns(si);

    % initialize slots with NaN (prevents shifting)
    thrR_s = nan(expN,1);
    thrL_s = nan(expN,1);
    thrN_s = nan(expN,1);
    marg_s = nan(expN,1);

    % fill by matching Session+Run
    for r = 1:expN
        idx = find(T.Session==si & T.Run==r, 1, 'first');
        if ~isempty(idx)
            thrR_s(r) = T.ThresholdR(idx);
            thrL_s(r) = T.ThresholdL(idx);
            thrN_s(r) = T.ThresholdN(idx);
            marg_s(r) = T.Margin(idx);
        end
    end

    % append session
    thrR_all = [thrR_all; thrR_s];
    thrL_all = [thrL_all; thrL_s];
    thrN_all = [thrN_all; thrN_s];
    marg_all = [marg_all; marg_s];

    % add gap after sessions 1..4
    if si < 5
        thrR_all = [thrR_all; nan(gapRuns,1)];
        thrL_all = [thrL_all; nan(gapRuns,1)];
        thrN_all = [thrN_all; nan(gapRuns,1)];
        marg_all = [marg_all; nan(gapRuns,1)];
    end
end

% pad/truncate to match concat length
L = numel(runTimeline.concat_x);
thrR_all = pad_to_length(thrR_all, L);
thrL_all = pad_to_length(thrL_all, L);
thrN_all = pad_to_length(thrN_all, L);
marg_all = pad_to_length(marg_all, L);

thrTimeline = struct();
thrTimeline.thrR_all = thrR_all;
thrTimeline.thrL_all = thrL_all;
thrTimeline.thrN_all = thrN_all;
thrTimeline.marg_all = marg_all;

end

function v = pad_to_length(v, L)
v = v(:);
if numel(v) < L
    v(end+1:L,1) = nan;
elseif numel(v) > L
    v = v(1:L);
end
end

function blocks = find_blocks(mask)
% mask logical vector; returns struct array with start/end/len of true segments
mask = mask(:);
d = diff([false; mask; false]);
starts = find(d==1);
ends   = find(d==-1)-1;
blocks = struct('start',num2cell(starts),'end',num2cell(ends),'len',num2cell(ends-starts+1));
end
