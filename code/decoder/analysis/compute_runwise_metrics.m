function [runTimeline, sessionInfo] = compute_runwise_metrics(data)

runsize = 60;
sessions = 1:5;
sessFields = arrayfun(@(s)sprintf('decoding%d', s), sessions, 'UniformOutput', false);
gapRuns = 1;

expectedRuns = [6 8 8 8 6];   % <-- enforce this structure

acc_runs = cell(5,1);
tpr_runs = cell(5,1);
tnr_runs = cell(5,1);
amb_runs = cell(5,1);

for si = 1:5
    sf = sessFields{si};
    expN = expectedRuns(si);

    % Default: all NaNs (so missing session or missing runs become NaN slots)
    A   = nan(expN,1);
    TPR = nan(expN,1);
    TNR = nan(expN,1);
    AMB = nan(expN,1);

    if isfield(data, sf) && ~isempty(data.(sf)) && isfield(data.(sf),'beh')
        beh = data.(sf).beh;

        if isfield(beh,'BCI_output') && isfield(beh,'trial_type')
            y_true = beh.trial_type(:);
            y_pred = beh.BCI_output(:);

            ntr = numel(y_true);
            nruns_found = floor(ntr / runsize);      % how many complete runs exist
            nruns_use   = min(nruns_found, expN);    % only fill up to expected

            for r = 1:nruns_use
                idx = (r-1)*runsize + (1:runsize);
                yt = y_true(idx);
                yp = y_pred(idx);

                ambMask = (yp == 3);
                AMB(r) = sum(ambMask);

                keep = ~ambMask;
                if ~any(keep), continue; end

                yt_k = yt(keep);
                yp_k = yp(keep);

                A(r) = mean(yp_k == yt_k);

                posMask = (yt_k == 1);
                if any(posMask), TPR(r) = mean(yp_k(posMask) == 1); end

                negMask = (yt_k == 0);
                if any(negMask), TNR(r) = mean(yp_k(negMask) == 0); end
            end
        end
    end

    acc_runs{si} = A;
    tpr_runs{si} = TPR;
    tnr_runs{si} = TNR;
    amb_runs{si} = AMB;
end

% concatenate with gaps (now always the same length across subjects)
concat_x = [];
concat_acc = [];
concat_tpr = [];
concat_tnr = [];
concat_amb = [];
sess_end_idx = [];
x_cursor = 0;

for si = 1:5
    A = acc_runs{si};                 % always length expectedRuns(si)
    nruns = numel(A);
    x_seg = x_cursor + (1:nruns);

    concat_x   = [concat_x, x_seg];
    concat_acc = [concat_acc; A(:)];
    concat_tpr = [concat_tpr; tpr_runs{si}(:)];
    concat_tnr = [concat_tnr; tnr_runs{si}(:)];
    concat_amb = [concat_amb; amb_runs{si}(:)];

    sess_end_idx(end+1) = x_seg(end);

    if si < 5
        x_gap = x_seg(end) + (1:gapRuns);
        concat_x   = [concat_x, x_gap];
        concat_acc = [concat_acc; nan(gapRuns,1)];
        concat_tpr = [concat_tpr; nan(gapRuns,1)];
        concat_tnr = [concat_tnr; nan(gapRuns,1)];
        concat_amb = [concat_amb; nan(gapRuns,1)];
        x_cursor = x_gap(end);
    else
        x_cursor = x_seg(end);
    end
end

runTimeline = struct();
runTimeline.concat_x   = concat_x(:);
runTimeline.concat_acc = concat_acc(:);
runTimeline.concat_tpr = concat_tpr(:);
runTimeline.concat_tnr = concat_tnr(:);
runTimeline.concat_amb = concat_amb(:);
runTimeline.sess_end_idx = sess_end_idx;

sessionInfo = struct();
sessionInfo.acc_runs = acc_runs;
sessionInfo.expectedRuns = expectedRuns;

end

