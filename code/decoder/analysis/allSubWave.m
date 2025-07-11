%% runAllSubjects.m
% subjects = {'e1','e2','e3','e4','e5','e6','e8','e11','e12','e13','e14','e15'};
subjects = {'e1','e2','e3','e4','e5','e13'};
nSub     = numel(subjects);

nT = 768;

allD   = nan(nSub, nT);
allND  = nan(nSub, nT);
rtD    = nan(nSub,1);
rtND   = nan(nSub,1);

%— loop through subjects
for i = 1:nSub
    subj = subjects{i};
    try
        fprintf('Processing %s…\n', subj);
        [meanD, meanND, rtD_mean, rtND_mean] = computeModel(subj);
        
        allD(i,:)    = meanD(:)';     % subject i, distractor waveform
        allND(i,:)   = meanND(:)';    % subject i, no-distractor waveform
        rtD(i)       = rtD_mean;      % subject i mean RT, distractor
        rtND(i)      = rtND_mean;     % subject i mean RT, no-distractor
        
    catch ME
        warning('  → failed on %s: %s', subj, ME.message);
        % leave this row as NaNs
    end
end
%%
outFile = 'allSubjectWaveformsD2e1to6n13.mat';
save(outFile, 'subjects', 'allD', 'allND', 'rtD', 'rtND');

%% — compute grand-average ± SEM
gmeanD   = mean(allD,   1, 'omitnan');
gsemD    = std( allD,0, 1, 'omitnan')/sqrt(nSub);
gmeanND  = mean(allND,  1, 'omitnan');
gsemND   = std( allND,0,1, 'omitnan')/sqrt(nSub);

%— compute group RT means (and SD if you like)
% rtD_mean   = mean(rtD,  'omitnan')/1000;  
% rtD_sd     = std(rtD,   'omitnan')/1000;
% rtND_mean  = mean(rtND, 'omitnan')/1000;
% rtND_sd    = std(rtND,  'omitnan')/1000;

%%
% Perform a cluster‐based permutation test on the within‐subject difference 
% waveforms (allD − allND) to identify time regions where distractor and 
% no‐distractor responses differ, while controlling family‐wise error.  
% Within the specified window (0.1–0.5 s), the function:
%   1. Computes a paired t‐statistic at each time point.
%   2. Thresholds these t‐values at p < 0.05 to form preliminary “suprathreshold” samples.
%   3. Groups adjacent suprathreshold samples into clusters.
%   4. Calculates a cluster‐statistic as the sum of |t| within each cluster.
%   5. Builds a null distribution of maximum cluster‐statistics by randomly 
%      sign‐flipping each subject’s difference waveform across 1000 permutations.
%   6. Assigns each observed cluster a family‐wise error–corrected p‐value 
%      based on its percentile in the null distribution.
%   7. Returns a logical mask of time points belonging to significant clusters.
t = cfg.epochTime;
stat = clusterPermTest(allD, allND, t, [0.1 0.5], 0.05, 1000);


%% — plot
%— assume in workspace:
%   t            : [1×nT] time in seconds (e.g. -0.5:1/512:1)
%   gmeanD,gsemD : [1×nT] grand mean & SEM, distractor
%   gmeanND,gsemND : [1×nT] grand mean & SEM, no-distractor
%   rtD, rtND    : [nSub×1] per-subject mean RT in ms

% convert RT to seconds
% rtD_mean  = mean(rtD,  'omitnan')/1000;

% custom colors
colD   = [0.698, 0.133, 0.133];  % red
colND  = [0,     0.502, 0];      % green
colRT  = [0.3,   0.3,   0.3];    % gray
colCW  = [0.8,   0.8,   0.8];    % light gray

yrange = [-0.85 0.85];

figure('Units','inches','Position',[1 1 6 4], ...
       'PaperPositionMode','auto','Color','w','Renderer','painters');
hold on;

% 1) Classification window (no legend)
hCW = patch([0.1 0.5 0.5 0.1], [yrange([1 1 2 2])], colCW, ...
            'FaceAlpha',0.4, 'EdgeColor','none', 'HandleVisibility','off');

% 2) SEM shading (no legend)
fill([t, fliplr(t)], [gmeanD-gsemD, fliplr(gmeanD+gsemD)], ...
     colD,  'FaceAlpha',0.2, 'EdgeColor','none', 'HandleVisibility','off');
fill([t, fliplr(t)], [gmeanND-gsemND, fliplr(gmeanND+gsemND)], ...
     colND, 'FaceAlpha',0.2, 'EdgeColor','none', 'HandleVisibility','off');


% 3) Mean traces (capture handles for legend)
hLineD  = plot(t, gmeanD,  'Color',colD,  'LineWidth',2);
hLineND = plot(t, gmeanND,'Color',colND, 'LineWidth',2);

% 4) Solid zero baselines
yline(0, 'k-', 'LineWidth',1.5, 'HandleVisibility','off');
xline(0, 'k-', 'LineWidth',1.5, 'HandleVisibility','off');

% 5) RT mean line (no legend)
% xline(rtD_mean, '--', 'Color',colRT, 'LineWidth',1.5, 'HandleVisibility','off');

% 6) Significance ticks and single star
sigTimes = t(stat.mask);
if ~isempty(sigTimes)
    % y‐positions for dots and the single star
    sigYdots  = 0.7;
    sigYstar  = 0.75;
    % plot a dot at every significant time point
    plot(sigTimes, repmat(sigYdots,1,numel(sigTimes)), ...
         'k.', 'MarkerSize',10, 'HandleVisibility','off');
    % place one star at the center of the significant cluster
    sigX = mean(sigTimes);
    text(sigX, sigYstar, '*', ...
         'HorizontalAlignment','center', ...
         'FontSize',16, 'Color','k', 'HandleVisibility','off');
end

% 7) Axes limits & styling
xlim([-0.1 0.7]);
ylim(yrange);
xlabel('Time (s)',       'FontSize',14,'FontName','Times New Roman');
ylabel('Δ Voltage (µV)', 'FontSize',14,'FontName','Times New Roman');
set(gca, ...
    'FontName','Times New Roman', ...
    'FontSize',12, ...
    'LineWidth',1, ...
    'TickDir','out', ...
    'Box','off');

% 8) Legend
legend([hLineD, hLineND], ...
       {'Distractor (±SEM)','No Distractor (±SEM)'}, ...
       'Location','NorthEast','FontSize',12);

title('Grand-average Difference Waveforms','FontSize',16,'FontName','Times New Roman');
hold off;


%% 8) Export high-res PDF
print(gcf, 'Fig1_diffWaveform.pdf', '-dpdf', '-r300');





