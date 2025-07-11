% pull out labels and 2D coords
labels = chanlocs.labels;  
Xall   = chanlocs.X;  
Yall   = chanlocs.Y;

for d = 1:numel(decoders)
    for f = 1:numel(features)
        data = allIdx{d,f};
        if isempty(data), continue; end
        
        % 1) count & find top 10 exactly as before
        [uIdx, ~, ic] = unique(data);
        counts        = accumarray(ic,1);
        [~, order]    = sort(counts,'descend');
        nTop          = min(10,numel(order));
        topBins       = order(1:nTop);
        topChans      = uIdx(topBins);    % these are channel‐indices into chanlocs
        
        % 2) get their coords & names
        Xtop = Xall(topChans);
        Ytop = Yall(topChans);
        Ltop = labels(topChans);
        
        % 3) plot
        figure('Name',sprintf('%s – %s Topology',decoders{d},features{f}));
        hold on;
        % plot all electrodes lightly
        scatter(Xall, Yall, 20, [.8 .8 .8], 'filled');
        % highlight top-10
        scatter(Xtop, Ytop, 80, 'r', 'filled');
        % label them
        for k = 1:nTop
            text(Xtop(k)+1, Ytop(k)+1, Ltop{k}, ...
                 'FontSize',12, 'FontWeight','bold');
        end
        axis equal off;
        title(sprintf('%s – %s Top 10 channels', decoders{d}, features{f}));
        hold off;
    end
end
