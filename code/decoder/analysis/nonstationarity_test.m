% assumes epoched data and decoder R and L is loaded 

sessions = {'decoding1','decoding2','decoding3','decoding4','decoding5'};

featuresR_orig = compute_features(decoderR,data.training1.epochs.data);
featuresL_orig = compute_features(decoderL,data.training1.epochs.data);

for si = 1:numel(sessions)
    sf = sessFields{si};
    if ~isfield(data, sf)
        warning('Missing %s in data. Skipping.', sf);
        continue;
    end
    epochs = data.(sf).epochs.data;

    featuresR_new = compute_features(decoderR,epochs);
    featuresL_new = compute_features(decoderL,epochs);


end
