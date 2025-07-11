function plotGridSearch(T, measureName, subjectsToPlot)
% plotGridSearch  Plot average + individual subject metrics across configs
%   plotGridSearch(T, measureName) uses all subjects.
%   plotGridSearch(T, measureName, subjectsToPlot) uses only those you list.
%
%   measureName is e.g. 'auprc','kappa','accuracy','tpr','tnr'.

% Default to all subjects if none specified
if nargin<3 || isempty(subjectsToPlot)
    subjectsToPlot = unique(T.subject, 'stable');
end
subjectsToPlot = subjectsToPlot(:);

% Pre-allocate
numSubjects = numel(subjectsToPlot);

% Determine number of configs per subject by inspecting the first subject
rows1 = find(strcmp(T.subject, subjectsToPlot{1}));
nConfigs = numel(rows1);
% Verify consistency
for s = 2:numSubjects
    if numel(find(strcmp(T.subject, subjectsToPlot{s}))) ~= nConfigs
        error('Subject %s has a different number of configs.', subjectsToPlot{s});
    end
end

% Initialize data matrices: rows=subjects, cols=configs
data_bilat = nan(numSubjects, nConfigs);
data_right = nan(numSubjects, nConfigs);
data_left  = nan(numSubjects, nConfigs);

% Fill in the data
for s = 1:numSubjects
    rows = find(strcmp(T.subject, subjectsToPlot{s}));
    for k = 1:nConfigs
        perf = T.perf{rows(k)};
        % Bilateral
        if isfield(perf, 'bilateral') && isfield(perf.bilateral, measureName)
            data_bilat(s,k) = perf.bilateral.(measureName);
        end
        % Right
        if isfield(perf, 'right') && isfield(perf.right, measureName)
            data_right(s,k) = perf.right.(measureName);
        end
        % Left
        if isfield(perf, 'left') && isfield(perf.left, measureName)
            data_left(s,k) = perf.left.(measureName);
        end
    end
end

% Compute subject‐wise means
avg_bilat = nanmean(data_bilat,1);
avg_right = nanmean(data_right,1);
avg_left  = nanmean(data_left,1);

%% Plot 1: Bilateral decoder
figure;
subplot(2,1,1);
bar(avg_bilat, 'FaceColor',[0.2 0.6 0.8]);
hold on;
for s = 1:numSubjects
    scatter(1:nConfigs, data_bilat(s,:), 50, 'k', 'filled');
end
hold off;
xlabel('Parameter Combination');
ylabel(measureName);
title(sprintf('Bilateral Decoder (%s) – subjects: %s', measureName, ...
    strjoin(subjectsToPlot',',')));
ylim([0 1]);
grid on;

%% Plot 2: Right vs Left decoders
subplot(2,1,2);
bar([avg_right; avg_left]','grouped');
hold on;
offset = 0.15;
for k = 1:nConfigs
    xR = k - offset;
    xL = k + offset;
    scatter(repmat(xR, numSubjects,1), data_right(:,k), 50, 'r', 'filled');
    scatter(repmat(xL, numSubjects,1), data_left(:,k), 50, 'b', 'filled');
end
hold off;
xlabel('Parameter Combination');
ylabel(measureName);
legend('Right','Left','Location','best');
title(sprintf('Right vs Left Decoders (%s) – subjects: %s', measureName, ...
    strjoin(subjectsToPlot',',')));
ylim([0 1]);
grid on;

end
