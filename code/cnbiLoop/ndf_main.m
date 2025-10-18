function ndf_main(subjectID,thrR, thrL, thrN, margin)

global stream ndf ID ids idm

% warning('off', 'all');
% Include any required toolboxes
ndf_include(); %adds paths to CNBI toolkit and eegc3
addpath(genpath('../decoder'));
addpath(genpath('../functions'));
folderPath = './online_info';

skip_iterations = true;
% Prepare and enter main loop
try
    load(sprintf('../decoder/decoders/e%d_decoderR.mat', subjectID));
    load(sprintf('../decoder/decoders/e%d_decoderL.mat', subjectID));
    load(sprintf('../decoder/decoders/e%d_decoderN.mat', subjectID));

    disp('Decoder Updated at');
    disp(decoderR.datetime);
    
    load(sprintf('./online_info/e%d_thrlog.mat',subjectID));
    lastthr = thrLog(end);
    lastthr.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    if nargin>=2 && ~isempty(thrR),  lastthr.thrR  = thrR;  end
    if nargin>=3 && ~isempty(thrL),  lastthr.thrL  = thrL;  end
    if nargin>=4 && ~isempty(thrN),  lastthr.thrN  = thrN;  end
    if nargin>=5 && ~isempty(margin), lastthr.margin = margin; end
    
    timestamp = datestr(now, 'yyyymmdd');
    op_path = fullfile(folderPath, sprintf('e%d_OnlinePosteriors_%s.mat', subjectID, timestamp));
    if isfile(op_path)
        load(op_path, 'OnlinePosteriors');
    else
        OnlinePosteriors = [];
    end

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
                    win = first_index + decoderR.params.epochSamples;
                    if label_value == 32
                        [ex_posterior, ~] = singleClassificationRight(decoderR,...
                            stream.eeg(win, decoderR.eegChannels));
                        threshold = lastthr.thrR;
                    elseif label_value == 44
                        [ex_posterior, ~] = singleClassificationRight(decoderL,...
                            stream.eeg(win, decoderL.eegChannels));
                        threshold = lastthr.thrL;
                    elseif label_value == 8
                        [ex_posterior, ~] = singleClassificationRight(decoderN,...
                            stream.eeg(win, decoderN.eegChannels));
                        threshold = lastthr.thrN;
                    end
                    disp(['Time Frame: ' num2str(time_frame, '%.2f') ' Posteriors: ' num2str(ex_posterior, ' %.2f')]);
                    stream.trigger(first_index) = 0;

                    diff = ex_posterior - threshold;
                    if abs(diff) <= lastthr.margin
                        code = 3;
                    else
                        code = (diff > 0) + 1; % diff>0 → code=2 or Pd, else (diff<0) → code=1 or no Pd
                    end
                    OnlinePosteriors(end+1, :) = [ex_posterior; threshold;code];
                    sendTiD(code);
                end
            end

            if (receiveTiD() == 20)
                break;
            end
        end
    end


thrLog(end+1) = lastthr;
save(sprintf('./online_info/e%d_thrlog.mat',subjectID),'thrLog');

save(op_path, 'OnlinePosteriors');

fprintf('Decoder ambivalence margin: %.4f\n', lastthr.margin);
fprintf('DecoderR threshold: %.4f\n', lastthr.thrR);
fprintf('DecoderL threshold: %.4f\n', lastthr.thrL);
fprintf('DecoderN threshold: %.4f\n\n', lastthr.thrN);

% Save thresholds
logname = sprintf('e%d_thresholds_log.txt',subjectID);
logFile = fullfile(folderPath, logname);
fid = fopen(logFile, 'a');  % append mode (creates file if it doesn't exist)
if fid ~= -1
    fprintf(fid, 'Run timestamp: %s\n', datestr(now,'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, 'Decoder ambivalence margin: %.4f\n', lastthr.margin);
    fprintf(fid, 'DecoderR threshold: %.4f\n', lastthr.thrR);
    fprintf(fid, 'DecoderL threshold: %.4f\n', lastthr.thrL);
    fprintf(fid, 'DecoderN threshold: %.4f\n\n', lastthr.thrN);
    fclose(fid);
else
    warning('Could not open thresholds log file for writing.');
end


catch exception
    ndf_printexception(exception);
end
end
