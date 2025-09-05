function ndf_main(thrR, thrL, thrN, margin)

global stream ndf ID ids idm

% warning('off', 'all');
% Include any required toolboxes
ndf_include(); %adds paths to CNBI toolkit and eegc3
addpath(genpath('../decoder'));
addpath(genpath('../functions'));

skip_iterations = true;
% Prepare and enter main loop
try
    load('./decoderR.mat');
    load('./decoderL.mat');
    load('./decoderN.mat');

    disp('Decoder Updated at');
    disp(decoderR.datetime);

    if nargin>=1 && ~isempty(thrR),  decoderR.threshold  = thrR;  end
    if nargin>=2 && ~isempty(thrL),  decoderL.threshold  = thrL;  end
    if nargin>=3 && ~isempty(thrN),  decoderN.threshold  = thrN;  end
    if nargin>=4 && ~isempty(margin), decoderR.thresholdMargin = margin; end

    ndf_initialization(); %sets up ndf configuration, should automatically setup ndf with 64 ch based on incoming data
    decoderR = initializeParams(decoderR);
    cleanupObj = onCleanup(@() ndf_down(decoderR));

    tid_attach(ID);
    disp('[ndf] Receiving NDF frames...');

    %%
    update_flag = false;
    %% Main Loop %%
    while(true)
        tic
        [ndf.frame, ndf.size] = ndf_read(ndf.sink, ndf.conf, ndf.frame); %read data, outputs frame = data and count = data size
        time_frame = 1000*toc;

        % Acquisition is down, exit
        if(ndf.size == 0)
            disp('[ndf] Broken pipe');
            break;
        end

        if ((time_frame < 20) && (skip_iterations)) %skip iteration if reading data takes less than 20 msec 
            disp(['Skipping iteration. Time was: ' num2str(time_frame) ' ms']);
        else
            skip_iterations = false;

            eeg_input = ndf.frame.eeg; %(samples x eeg_channels)
            eog_input = ndf.frame.exg; %(samples x exg_channels)
            trigger_input = ndf.frame.tri; %(samples x tri_channels)

            %store EEG and trigger data in stream, includes bandpass filter ...
            % based on spatial filter in decoder, also has EOG filter (commented out)
            ndf_store_signals([eeg_input, eog_input], trigger_input, decoderR); 

            if (~any(isnan(stream.eeg(:))))
                %returns sample (out of 768) where one of these triggers is found
                % first_index = find(ismember(stream.trigger, [102 104 100 110]), 1, 'first'); 
                first_index = find(ismember(stream.trigger, [8 32 44]), 1, 'first'); % ND, Dright, Dleft
                %disp(first_index)
                if (first_index >= 256) & (first_index <= 308) % 0.5 sec baseline, need 256 for decoder.baseline_idx to work correctly
                    label_value = stream.trigger(first_index);
                    fprintf('Label value at first_index (%d): %d\n', first_index, label_value);
                    if label_value == 32
                        [ex_posterior, ~] = singleClassificationRight(decoderR,...
                            stream.eeg((first_index - 256):end, decoderR.eegChannels));
                        threshold = decoderR.threshold;
                    elseif label_value == 44
                        [ex_posterior, ~] = singleClassificationRight(decoderL,...
                            stream.eeg((first_index - 256):end, decoderL.eegChannels));
                        threshold = decoderL.threshold;
                    elseif label_value == 8
                        [ex_posterior, ~] = singleClassificationRight(decoderN,...
                            stream.eeg((first_index - 256):end, decoderN.eegChannels));
                        threshold = decoderN.threshold;
                    end
                    disp(['Time Frame: ' num2str(time_frame, '%.2f') ' Posteriors: ' num2str(ex_posterior, ' %.2f')]);
                    decoderR.onlinePosteriors = [decoderR.onlinePosteriors, ex_posterior];
                    stream.trigger(first_index) = 0;

                    diff = ex_posterior - threshold;
                    if abs(diff) <= decoderR.thresholdMargin
                        code = 3;
                    else
                        code = (diff > 0) + 1; % diff>0 → code=2 or Pd, else (diff<0) → code=1 or no Pd
                    end
                    sendTiD(code);
                    % sendTiD(1 + (ex_posterior > decoder.decision_threshold)); % sends 1 if below threshold, 2 if above
                end
            end

            if (receiveTiD() == 20)
                break;
            end
        end
    end
folderPath = './online_decoders';
timestamp = datestr(now, 'yyyymmdd_HHMMSS');
filenameR = fullfile(folderPath,['decoderR_' timestamp '.mat']);
save(filenameR, 'decoderR');
save('./decoderR.mat', 'decoderR');

filenameL = fullfile(folderPath,['decoderL_' timestamp '.mat']);
save(filenameL, 'decoderL');
save('./decoderL.mat', 'decoderL');

filenameN = fullfile(folderPath,['decoderN_' timestamp '.mat']);
save(filenameN, 'decoderN');
save('./decoderN.mat', 'decoderN');

fprintf('Decoder ambivalence margin: %.4f\n', decoderR.thresholdMargin);
fprintf('DecoderR threshold: %.4f\n', decoderR.threshold);
fprintf('DecoderL threshold: %.4f\n', decoderL.threshold);
fprintf('DecoderN threshold: %.4f\n\n', decoderN.threshold);

% Save thresholds
logFile = fullfile(folderPath, 'thresholds_log.txt');
fid = fopen(logFile, 'a');  % append mode (creates file if it doesn't exist)
if fid ~= -1
    fprintf(fid, 'Run timestamp: %s\n', datestr(now,'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, 'Decoder ambivalence margin: %.4f\n', decoderR.thresholdMargin);
    fprintf(fid, 'DecoderR threshold: %.4f\n', decoderR.threshold);
    fprintf(fid, 'DecoderL threshold: %.4f\n', decoderL.threshold);
    fprintf(fid, 'DecoderN threshold: %.4f\n\n', decoderN.threshold);
    fclose(fid);
else
    warning('Could not open thresholds log file for writing.');
end

catch exception
    ndf_printexception(exception);
end
end
