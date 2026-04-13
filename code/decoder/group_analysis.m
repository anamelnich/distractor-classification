subjects = {'e23','e24','e27','e33','e36','e37'}; % control
subjects = {'e21','e22','e25','e26','e29','e30','e31','e32','e38','e39'}; %experimental
subjects = {'e21','e22','e25','e29','e32','e38'}; %experimental clean
GROUP = group_run_all_subjects(subjects, false);

%%

cacheDir = './cache';
files = dir(fullfile(cacheDir, '*.mat'));
cacheFiles = fullfile({files.folder}, {files.name});

%%
group_plot_rt_and_stroop_prepost(cacheFiles);

%%
% --- Define output folder ---
figDir = fullfile('.', '..','..','..','figures_postproc');
if ~exist(figDir, 'dir'); mkdir(figDir); end

% --- Get all open figures ---
figs = findall(0, 'Type', 'figure');
figs = flipud(figs); % optional: preserve creation order

% --- Save each figure ---
for i = 1:numel(figs)
    fig = figs(i);
    
    name = get(fig, 'Name');
    if isempty(name)
        name = sprintf('figure_%02d', i);
    end
    
    % sanitize filename
    name = regexprep(name, '[^\w\d\-]', '_');
    
    % default paths
    pdfPath = fullfile(figDir, [name 'r2compare_PO2.pdf']);
    figPath = fullfile(figDir, [name 'r2compare_PO2.fig']);
    
    % avoid overwrite
    if isfile(pdfPath) || isfile(figPath)
        pdfPath = fullfile(figDir, sprintf('%s_%02d.pdf', name, i));
        figPath = fullfile(figDir, sprintf('%s_%02d.fig', name, i));
    end
    
    % save
    exportgraphics(fig, pdfPath, 'ContentType', 'vector');
    savefig(fig, figPath);
end

fprintf('Saved %d figures (.pdf + .fig) to %s\n', numel(figs), figDir);

%% Stroop in depth analysis

subjects = {'e23','e24','e27','e33','e36','e37'};
stroop_ctrl = analyze_stroop_cache(subjects, './cache');
stamp = datestr(now, 'yyyymmdd_HHMMSS');
savePath = fullfile('./cache', ['group_stroop_control_' stamp '.mat']);
save(savePath, 'stroop_ctrl', '-v7.3');

subjects = {'e21','e22','e25','e26','e29','e30','e31','e32','e38','e39'}; %experimental
stroop_exp = analyze_stroop_cache(subjects, './cache');
stamp = datestr(now, 'yyyymmdd_HHMMSS');
savePath = fullfile('./cache', ['group_stroop_exp_' stamp '.mat']);
save(savePath, 'stroop_exp', '-v7.3');

%cntrl and exptl togehter
expSubjects  = {'e21','e22','e25','e26','e29','e30','e31','e32','e38','e39'};
ctrlSubjects = {'e23','e24','e27','e33','e36','e37'};
OUT = analyze_stroop_cache_groups(expSubjects, ctrlSubjects, './cache');

%% r2 topoplots

exp = load("/Users/hililbby/Library/Mobile Documents/com~apple~CloudDocs/UT Austin/JM/distractor-classification/code/decoder/analysis/cache/GROUP_20260323_123840.mat");
ctrl = load("/Users/hililbby/Library/Mobile Documents/com~apple~CloudDocs/UT Austin/JM/distractor-classification/code/decoder/analysis/cache/GROUP_20260323_125328.mat");
%%
load chanlocs64.mat
if exist('chanlocs64','var')
    chanlocs = chanlocs64;
end

expSubjects  = {'e21','e22','e25','e26','e29','e30','e31','e32','e38','e39'};
ctrlSubjects = {'e23','e24','e27','e33','e36','e37'};

OUT = analyze_pd_r2_groups(expSubjects, ctrlSubjects, './cache', chanlocs);

%% %% STATISTICAL TESTS
%% distractor cost
% --- Extract ---
pre_exp  = exp.GRT.distractor.diffC(:);
post_exp = exp.GRT.distractor.diffF(:);

pre_ctrl  = ctrl.GRT.distractor.diffC(:);
post_ctrl = ctrl.GRT.distractor.diffF(:);

% --- Build wide table ---
T = table;

T.Group = [repmat("exp", numel(pre_exp),1); ...
           repmat("ctrl", numel(pre_ctrl),1)];

T.Pre  = [pre_exp;  pre_ctrl];
T.Post = [post_exp; post_ctrl];

% --- Within design ---
Meas = table(["Pre"; "Post"], 'VariableNames', {'Time'});

% --- Fit RM model ---
rm = fitrm(T, 'Pre-Post ~ Group', 'WithinDesign', Meas);

ranovatbl = ranova(rm, 'WithinModel', 'Time');
disp(ranovatbl)

[~,~,ci,~] = ttest2(delta_exp, delta_ctrl);
disp('Confidence Interval')
disp(ci)

% Effect size 
% from your table
SS_effect = 1453.1;
SS_error  = 6085.9;

eta_p2 = SS_effect / (SS_effect + SS_error);
fprintf('Partial eta^2 (Group×Time) = %.3f\n', eta_p2)

%% stroop
% Load files from cache

stroop_stats = run_stroop_group_anovas(stroop_exp, stroop_ctrl);
