% Configures various predefined parameters and saves it into a structure 
% variable 'cfg'. 
function params = setParams(header)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %% General Information %%
    %%%%%%%%%%%%%%%%%%%%%%%%%
    params.fsamp = header.SampleRate;  

    params.eegChannels = 1:64; 
    params.eogChannels = 65:66;
    params.triggerChannel = 67;

    params.chanLabels = header.Label;
    params.chanLabels(65:67)=[];

    params.plotOption = {'LineWidth', 2};
    params.plotColor = {
    [0.698, 0.133, 0.133],      % Red
    [0, 0, 1],                  % Blue
    [0.941, 0.502, 0.502],      % Light Red
    [0.6, 0.6, 1],              % Light Blue
    [0, 0.502, 0],              % Green
    [0, 0, 0],                  % Black
    [0.196, 0.804, 0.196],      % Light green
    [0.804, 0.361, 0.361],      % Red3 (Indian red)
    [0.196, 0.804, 0.196],      % Green3 (Lime green)
            };

    %%%%%%%%%%%%%%
    %% Epoching %%
    %%%%%%%%%%%%%%
    params.epochSamples = -0.5*params.fsamp+1:1.0*params.fsamp;
    params.epochTime = params.epochSamples./params.fsamp;
    params.epochOnset = find(params.epochTime == 0);
    
    %%%%%%%%%%%%%%%%%%%%%
    %% Spectral Filter %%
    %%%%%%%%%%%%%%%%%%%%%
    params.spectralFilter.freqs = [1 30]; 
    params.spectralFilter.order = 2;  
    
    params.EOG.spectralFilter.freqs = [1 10];  
    params.EOG.spectralFilter.order = 2;  

    %%%%%%%%%%%%%%%%%%%%
    %% ROI Selection %%%
    %%%%%%%%%%%%%%%%%%%%
    params.roi = 'P/PO'; % {'None', 'P/PO'}

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Baseline Correction %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    params.baseline_iscompute = true;
    params.baseline_window = [-0.2, 0];

    %%%%%%%%%%%%%%%%%%%%
    %% Spatial Filter %%
    %%%%%%%%%%%%%%%%%%%%
    params.spatialFilter.type = 'CCA';  
    params.spatialFilter.time = round(0.15*params.fsamp)+1:round(0.5*params.fsamp);
    params.spatialFilter.time = params.spatialFilter.time + params.epochOnset;
    params.spatialFilter.nComp = 2;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Power Spectral Density %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    params.psd.is_compute = true;
    params.psd.type = 'stockwell';  % {'stockwell'}
    params.psd.time = round(0.15*params.fsamp)+1:round(0.5*params.fsamp);
    params.psd.time = params.psd.time + params.epochOnset;
    params.psd.window = hanning(length(params.psd.time));
    params.psd.nfft  = 4*params.fsamp;
    params.psd.overlap = [];
    params.psd.freq_range = [4:1:7];

    %%%%%%%%%%%%%%
    %% Features %%
    %%%%%%%%%%%%%%
    params.features.erp_iscompute = false;
    params.features.diffwave_iscompute = true;

    %%%%%%%%%%%%%%%%%%%%%%
    %% Resampling Ratio %%
    %%%%%%%%%%%%%%%%%%%%%%
    params.resample.is_compute = true;
    params.resample.ratio = round(params.fsamp / 64);
    params.resample.time = round(0.15*params.fsamp)+1:round(0.5*params.fsamp);
    params.resample.time = params.resample.time + params.epochOnset;

    %%%%%%%%%%%%%%%%
    %% Classifier %%
    %%%%%%%%%%%%%%%%
    params.classify.is_normalize = false;
    params.classify.reduction.type = 'r2'; % {'pca', 'lasso', 'r2', 'None'}
    params.classify.type = 'diaglinear'; % {'linear', 'diaglinear'}
      
end