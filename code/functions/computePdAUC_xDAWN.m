function [auc_session, auc_subj] = computePdAUC_xDAWN(decodingCell, decLCell, decRCell, params, timeWin, subjLabels)
% computePdAUC_PO7PO8 (xDAWN version)
%
% Compute Pd (contra–ipsi) positive AUC across decoding sessions using xDAWN
% filters from decoderL / decoderR, works for single subject or group.
%
% Inputs:
%   decodingCell : nSessions x nSubj cell array
%                  decodingCell{si,sj} or decodingCell{si,sj}.epochs must have:
%                      .data   [T x C x N]
%                      .labels [1 x N] or [N x 1]
%                      labels: 0 = ND, 1 = distractor RIGHT, 2 = distractor LEFT
%
%   decLCell     : 1 x nSubj cell array, each with .spatialFilter.diff (7x2)
%   decRCell     : 1 x nSubj cell array, each with .spatialFilter.diff (7x2)
%
%   params       : struct with fields:
%                      .chanLabels       (cellstr)
%                      .epochTime        (vector, seconds)
%                      .baseline_window  ([t1 t2] seconds)
%
%   timeWin      : [tStart tEnd] seconds for AUC (e.g. [0.15 0.50])
%                  default: [0.15 0.50]
%
%   subjLabels   : (optional) 1 x nSubj cell array of subject IDs for plot title
%
% Outputs:
%   auc_session  : [nSessions x 1] mean positive AUC across subjects
%   auc_subj     : [nSessions x nSubj] per-session AUC per subject

    if nargin < 5 || isempty(timeWin)
        timeWin = [0.15 0.50];  % default 150–500 ms
    end

    [nSessions, nSubj] = size(decodingCell);
    if nargin < 6 || isempty(subjLabels)
        subjLabels = arrayfun(@(j)sprintf('Subj%d', j), 1:nSubj, 'UniformOutput', false);
    end

    auc_subj = nan(nSessions, nSubj);

    % ---- xDAWN electrode pairs (must match your decoder filters) ----
    LeftElec  = {'P1','P3','P5','P7','PO3','PO5','PO7'};
    RightElec = {'P2','P4','P6','P8','PO4','PO6','PO8'};

    chanLabels = params.chanLabels;
    [isL, lIdx] = ismember(LeftElec,  chanLabels);
    [isR, rIdx] = ismember(RightElec, chanLabels);
    if ~all(isL) || ~all(isR)
        missing = [LeftElec(~isL), RightElec(~isR)];
        error('Missing required channels for xDAWN pairs: %s', strjoin(missing, ', '));
    end

    % Time vector and windows
    t = params.epochTime(:);
    if numel(t) < 2
        error('params.epochTime must be a non-empty time vector.');
    end

    idxWin = (t >= timeWin(1)) & (t <= timeWin(2));
    if ~any(idxWin)
        error('No samples in AUC time window [%.3f %.3f] s.', timeWin(1), timeWin(2));
    end

    baseIdx = (t >= params.baseline_window(1)) & (t <= params.baseline_window(2));
    if ~any(baseIdx)
        error('No samples in baseline window [%.3f %.3f] s.', ...
              params.baseline_window(1), params.baseline_window(2));
    end

    % ================== LOOP OVER SUBJECTS × SESSIONS ==================
    for sj = 1:nSubj

        % Grab subject-specific xDAWN weights
        decL = decLCell{sj};
        decR = decRCell{sj};
        if isempty(decL) || isempty(decR) || ...
           ~isfield(decL,'spatialFilter') || ~isfield(decR,'spatialFilter') || ...
           ~isfield(decL.spatialFilter,'diff') || ~isfield(decR.spatialFilter,'diff')
            warning('Subj %d (%s): invalid decoderL/decoderR; skipping.', sj, subjLabels{sj});
            continue;
        end

        WL = decL.spatialFilter.diff;   % 7 x 2
        WR = decR.spatialFilter.diff;   % 7 x 2
        if ~isequal(size(WL),[7 2]) || ~isequal(size(WR),[7 2])
            warning('Subj %d (%s): decoderL/R.spatialFilter.diff must be 7x2; skipping.', ...
                    sj, subjLabels{sj});
            continue;
        end

        for si = 1:nSessions

            Dwrap = decodingCell{si, sj};
            if isempty(Dwrap)
                continue;
            end

            % Allow either decodingCell{si,sj}.epochs or direct struct
            if isfield(Dwrap, 'epochs')
                D = Dwrap.epochs;
            else
                D = Dwrap;
            end

            if ~isfield(D, 'data') || ~isfield(D, 'labels')
                warning('Subj %d (%s), Sess %d: missing data or labels; skipping.', ...
                        sj, subjLabels{sj}, si);
                continue;
            end

            X = D.data;        % [T x C x N]
            labels = D.labels; % [1 x N] or [N x 1]

            if size(labels,1) > 1
                labels = labels(:)';
            end

            [T, ~, N] = size(X);
            if numel(t) ~= T
                warning('Subj %d (%s), Sess %d: time length mismatch; skipping.', ...
                        sj, subjLabels{sj}, si);
                continue;
            end

            % Distractor trials only (labels 1 or 2)
            dTrials = (labels == 1) | (labels == 2);
            if ~any(dTrials)
                warning('Subj %d (%s), Sess %d: no distractor trials.', ...
                        sj, subjLabels{sj}, si);
                continue;
            end

            % ---- Baseline correction ----
            baseline = mean(X(baseIdx, :, :), 1);  % [1 x C x N]
            X = X - baseline;

            % ---- xDAWN projection (7ch L/R differences) ----
            diffAll_xdawn = nan(T, N);  % one xDAWN trace per trial

            for n = 1:N
                lab = labels(n);
                if ~(lab == 1 || lab == 2)
                    continue;  % skip ND/other labels
                end

                % Extract paired L/R: [T x 7] each
                Lroi = squeeze(X(:, lIdx, n));  % T x 7
                Rroi = squeeze(X(:, rIdx, n));  % T x 7
                if isvector(Lroi), Lroi = Lroi(:)'; end
                if isvector(Rroi), Rroi = Rroi(:)'; end

                % Following your original convention:
                % labels==1 (distractor RIGHT) -> use decoderL, L-R
                % labels==2 (distractor LEFT)  -> use decoderR, R-L
                switch lab
                    case 1   % distractor RIGHT
                        diff7 = Lroi - Rroi;   % [T x 7]
                        W = WR;               % [7 x 2]
                    case 2   % distractor LEFT
                        diff7 = Rroi - Lroi;   % [T x 7]
                        W = WL;               % [7 x 2]
                    otherwise
                        continue;
                end

                comps = diff7 * W;                    % [T x 2]
                diffAll_xdawn(:, n) = mean(comps, 2); % average top-2 comps -> [T x 1]
            end

            validIdx = dTrials & ~all(isnan(diffAll_xdawn), 1);
            if ~any(validIdx)
                warning('Subj %d (%s), Sess %d: no valid distractor trials after xDAWN.', ...
                        sj, subjLabels{sj}, si);
                continue;
            end

            % Grand-average Pd wave for this subject/session
            waveD = mean(diffAll_xdawn(:, validIdx), 2);   % [T x 1]

            % ---- Positive-only AUC in time window ----
            seg = waveD(idxWin);
            seg(seg < 0) = 0;                % keep only positive portion
            auc_val = trapz(t(idxWin), seg); % µV * s

            auc_subj(si, sj) = auc_val;
        end
    end

    % ================== GROUP / SUBJECT SUMMARY ==================
    auc_session = nanmean(auc_subj, 2);  % [nSessions x 1]

    % ================== PLOTTING ==================
    figure('Color','w','Units','inches','Position',[1 1 5 3.5]); hold on;
    burntOrange = [191 87 0] / 255;

    if nSubj > 1
        % ---- Group bar + subject dots ----
        bar(1:nSessions, auc_session, 'FaceColor', burntOrange);

        % Jittered subject dots
        for si = 1:nSessions
            y = auc_subj(si, :);
            x = si + 0.05*(rand(size(y)) - 0.5); % small jitter
            plot(x(~isnan(y)), y(~isnan(y)), 'k.', 'MarkerSize', 10);
        end

        title('Pd positive AUC (xDAWN)');
    else
        % ---- Single subject ----
        bar(1:nSessions, auc_subj(:,1), 'FaceColor', burntOrange);
        title(sprintf('Pd positive AUC (xDAWN) - %s', subjLabels{1}));
    end

    ylabel(sprintf('Positive AUC (%.0f–%.0f ms)', ...
        timeWin(1)*1000, timeWin(2)*1000));

    xticks(1:nSessions);
    xticklabels(arrayfun(@(s) sprintf('Session %d', s), 1:nSessions, 'UniformOutput', false));
    grid on; box off;

end
