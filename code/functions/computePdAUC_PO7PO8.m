function [auc_session, auc_subj] = computePdAUC_PO7PO8(decodingCell, params, timeWin, subjLabels)
% computePdAUC_PO7PO8
%
% Compute Pd (contra–ipsi) positive AUC at PO7/PO8 across decoding sessions,
% works for single subject (nSubj = 1) or group (nSubj > 1).
%
% Inputs:
%   decodingCell : nSessions x nSubj cell array
%                  decodingCell{si,sj} or decodingCell{si,sj}.epochs must have:
%                      .data   [T x C x N]
%                      .labels [1 x N] or [N x 1]
%                      labels: 0 = ND, 1 = distractor RIGHT, 2 = distractor LEFT
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
%
% Usage examples:
%   % Group:
%   [auc_sess, auc_subj] = computePdAUC_PO7PO8(decodingCell, cfg, [0.15 0.50], subjects);
%
%   % Single subject (e.g. sj = 2):
%   [auc_sess, auc_subj] = computePdAUC_PO7PO8(decodingCell(:,2), cfg, [0.15 0.50], {'e22'});

    if nargin < 3 || isempty(timeWin)
        timeWin = [0.15 0.50];  % default 150–500 ms
    end

    [nSessions, nSubj] = size(decodingCell);
    if nargin < 4 || isempty(subjLabels)
        subjLabels = arrayfun(@(j)sprintf('Subj%d', j), 1:nSubj, 'UniformOutput', false);
    end

    auc_subj = nan(nSessions, nSubj);

    % ---- Find PO7 / PO8 channel indices ----
    chanLabels = params.chanLabels;
    idxPO7 = find(strcmpi(chanLabels, 'PO7'), 1);
    idxPO8 = find(strcmpi(chanLabels, 'PO8'), 1);
    if isempty(idxPO7) || isempty(idxPO8)
        error('PO7 and/or PO8 not found in params.chanLabels.');
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
        for si = 1:nSessions

            Dwrap = decodingCell{si, sj};
            if isempty(Dwrap)
                continue;
            end

            % Allow either decodingCell{si,sj}.epochs or direct struct Dwrap
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

            [T, C, N] = size(X); %#ok<NASGU>
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

            % ---- Compute PO7/PO8 contra–ipsi difference per trial ----
            diffAll = nan(T, N);  % one trace per trial

            for n = 1:N
                lab = labels(n);
                if ~(lab == 1 || lab == 2)
                    continue;  % skip ND/other labels
                end

                po7 = squeeze(X(:, idxPO7, n));  % [T x 1]
                po8 = squeeze(X(:, idxPO8, n));  % [T x 1]

                if lab == 1
                    % distractor RIGHT -> contralateral = PO7
                    diffAll(:, n) = po7 - po8;   % contra - ipsi
                else
                    % distractor LEFT -> contralateral = PO8
                    diffAll(:, n) = po8 - po7;   % contra - ipsi
                end
            end

            validIdx = dTrials & ~all(isnan(diffAll), 1);
            if ~any(validIdx)
                warning('Subj %d (%s), Sess %d: no valid distractor trials after diff.', ...
                        sj, subjLabels{sj}, si);
                continue;
            end

            % Grand-average Pd wave for this subject/session
            waveD = mean(diffAll(:, validIdx), 2);   % [T x 1]

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
        bar(1:nSessions, auc_session, 'FaceColor',burntOrange);

        % Jittered subject dots
        for si = 1:nSessions
            y = auc_subj(si, :);
            x = si + 0.05*(rand(size(y)) - 0.5); % small jitter
            plot(x(~isnan(y)), y(~isnan(y)), 'k.', 'MarkerSize', 10);
        end

        title('Pd positive AUC at PO7/PO8');
    else
        % ---- Single subject: simple bar or line plot ----
        bar(1:nSessions, auc_subj(:,1), 'FaceColor',burntOrange);
        title(sprintf('Pd positive AUC (PO7/PO8) - %s', subjLabels{1}));
    end

    ylabel(sprintf('Positive AUC (%.0f–%.0f ms)', ...
        timeWin(1)*1000, timeWin(2)*1000));

    xticks(1:nSessions);
    xticklabels(arrayfun(@(s) sprintf('Session %d', s), 1:nSessions, 'UniformOutput', false));
    grid on; box off;

end

