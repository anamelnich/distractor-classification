% ————————————————
% 1) Extract AUPRCs for combo #2
% ————————————————
% assume:
%   performance : 468×4 table
%   correlations: 13×1 numeric
correlations = load('allSubjectResults6.mat');
correlations = correlations.corrs;
correlations = correlations.aucs;
correlations = cellfun(@(v) mean(v,'omitnan'), correlations);
performance = load('gridSearchResults_parfor.mat');
performance = performance.T;
% How many combos per subject?
%%
nParams = 36;
nSubs   = numel(correlations);

% Which variable holds the structs?
perfVar = performance.Properties.VariableNames{3};
perfCol = performance.(perfVar);  % 468×1 cell or struct array

% preallocate
leftAUPRC  = nan(nSubs,1);
rightAUPRC = nan(nSubs,1);
avgAUPRC   = nan(nSubs,1);

for s = 1:nSubs
    rowIdx = nParams*(s-1) + 2;  % e.g. 2, 38, 74, …

    % pull out the struct for that row
    if iscell(perfCol)
        ps = perfCol{rowIdx};
    else
        ps = perfCol(rowIdx);
    end

    % store left & right
    leftAUPRC(s)  = ps.left.auprc;
    rightAUPRC(s) = ps.right.auprc;
    avgAUPRC(s)   = mean([leftAUPRC(s), rightAUPRC(s)]);
end

% ————————————————————
%  Exclude subject #n
% ————————————————————
% excludeIdx = [8,10,12];
% keep       = true(nSubs,1);
% keep(excludeIdx) = false;
% 
% corr_use   = correlations(keep);
% X_use      = [avgAUPRC, leftAUPRC, rightAUPRC];
% X_use      = X_use(keep, :);
% metrics    = {'Avg','Left','Right'};

corr_use = correlations;
X_use      = [avgAUPRC, leftAUPRC, rightAUPRC];
metrics    = {'Avg','Left','Right'};

% ————————————————————
%%  2) Plotting
% ————————————————————
for k = 1:3
    x   = X_use(:,k);
    lbl = metrics{k};
    [r,p] = corr(corr_use, x, 'Type','Pearson');

    % Figure
    figure('Units','inches','Position',[1 1 4 4]);
    scatter(x, corr_use, 80, 'k', 'filled'); hold on;
    hL = lsline; set(hL,'LineWidth',2,'Color',[.3 .3 .3]);
    hold off;

    % Labels & title
    xlabel(sprintf('%s AUPRC', lbl), 'FontSize',14,'FontName','Arial');
    ylabel('Peak Amplitude', 'FontSize',14,'FontName','Arial');
    title(sprintf('r=%.2f, p=%.3f',r,p), ...
          'FontSize',16,'FontName','Arial');

    % Axis styling
    ax = gca;
    set(ax, 'FontName','Arial','FontSize',12,'LineWidth',1.5, ...
           'Box','on','TickDir','out');
    ti = ax.TightInset;
    ax.Position = [ti(1:2), 1-ti(1)-ti(3), 1-ti(2)-ti(4)];
end

