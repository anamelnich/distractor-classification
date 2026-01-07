function formatData(subjectID)

%% ====================== Load EEG ====================== %%
clearvars -except subjectID cfg;
close all; rng('default');
addpath(genpath('../../functions'));

%% Load Data 

dataPath = [pwd '/../../../data/'];
data = loadData(dataPath, subjectID);
delete sopen.mat

%% Set Parameters and Preprocess
cfg = setParams(data.training1.header);
cfg.fsamp = data.training1.header.SampleRate;  

cfg.eegChannels = 1:64; 
cfg.eogChannels = 65:66;
cfg.triggerChannel = 67;

cfg.chanLabels = data.training1.header.Label;
cfg.chanLabels(65:67)=[];
 

fields = fieldnames(data);
for i = 1:numel(fields)
    fname = fields{i};
    if isempty(data.(fname))
        data = rmfield(data, fname);
        continue;
    end

    trigtype = 2;
    data.(fname) = preprocessDataset(data.(fname), cfg, fname, trigtype);
    
end

fields = fieldnames(data);

%% ==================== Bandpass Filter ======================= %%
[b, a] = butter(cfg.spectralFilter.order, cfg.spectralFilter.freqs./(cfg.fsamp/2), 'bandpass');
cfg.spectralFilter.b = b;
cfg.spectralFilter.a = a;

for i = 1:numel(fields)
    fname = fields{i};
    data.(fname).data = filter(b, a, data.(fname).data);
end

%% ======================== Epoching ========================== %%
for i = 1:numel(fields)
    
    fname = fields{i};
    d = data.(fname);
    
    epochs.data = nan(length(cfg.epochSamples), length(cfg.chanLabels), length(d.index.pos));
    epochs.labels = d.index.typ;
    epochs.file_id = nan(length(d.index.typ), 1);

    for t = 1:length(d.index.pos)
        epochs.data(:, :, t) = d.data(d.index.pos(t) + cfg.epochSamples, :);
        epochs.file_id(t) = find(d.index.pos(t) <= d.eof, 1, 'first');
    end
    
    data.(fname).epochs = epochs;
    data.(fname).epochs.eof = d.eof;
    if isfield(d, 'beh') && isfield(d.beh, 'RT')
        data.(fname).epochs.RT = d.beh.RT;
    end
    if isfield(d, 'beh') && isfield(d.beh, 'tpos')
        data.(fname).epochs.RT = d.beh.RT;
    end
end
%% ====================== Load Posteriors ====================== %%

pattern = sprintf('%s_OnlinePosteriors_*.mat', subjectID);
files = dir(fullfile('./../../cnbiLoop/online_info/', pattern));
keepIdx = ~contains({files.name}, '_wPracticethr');
files = files(keepIdx);

fileDates = zeros(length(files),1);
for i = 1:length(files)
    name = files(i).name;
    dateStr = regexp(name, '\d{8}', 'match', 'once');
    fileDates(i) = str2double(dateStr);
end

[~, idx] = sort(fileDates);
files = files(idx);

for i = 1:length(files)
    filepath = fullfile(files(i).folder, files(i).name);
    S = load(filepath, 'OnlinePosteriors');
    
    fieldName = sprintf('decoding%d', i);
    data.(fieldName).posteriors = S.OnlinePosteriors;
end

%% ====================== Load Models ====================== %%
load(sprintf('../decoders/%s_decoderR.mat',subjectID));
load(sprintf('../decoders/%s_decoderL.mat',subjectID));
load(sprintf('../decoders/%s_decoderN.mat',subjectID));
data.decoderR  = decoderR;
data.decoderL  = decoderL;
data.decoderN  = decoderN;

%% ====================== Load Ambivalence Margins ====================== %%

folderPath = './../../cnbiLoop/online_info/';
thrfile = fullfile(folderPath, sprintf('%s_thrlog.mat', subjectID));
Sthr    = load(thrfile);
thrlog  = Sthr.thrLog;
data.thrLog = thrlog;

%% ====================== Load Pruned Data ====================== %%
load(sprintf('./../data/%s_prunedL.mat',subjectID));
load(sprintf('./../data/%s_prunedR.mat',subjectID));
data.training1.bestItrDataL = bestItrDataL;
data.training1.bestItrDataR = bestItrDataR;
%% ====================== Subject ID ====================== %%
data.subjectID = subjectID;
%% ====================== Save ====================== %%
save(sprintf('../data/%s_data',subjectID),'data','-v7.3')
end





