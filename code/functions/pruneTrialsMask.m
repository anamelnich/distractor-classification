function pruneMask = pruneTrialsMask(labels, post, thr, pctRemove, master)
  
  N = numel(labels);
  pruneMask = true(N,1);

  low_conf  = 0.2;
  high_conf = 0.8;
  classes   = unique(labels);

  for c = classes(:)'
    % only consider still-active trials of this class
    idx_c = find(master & labels==c);
    n_c   = numel(idx_c);
    if n_c==0, continue; end

    p_c = post(idx_c);

    % numbers to drop (5% of current pool)
    nMis = ceil(pctRemove * n_c);
    nThr = ceil(pctRemove * n_c);

    % 1) high-confidence misclassifications
    if c==1
      misIdx = idx_c(p_c < low_conf);
    else
      misIdx = idx_c(p_c > high_conf);
    end
    if ~isempty(misIdx)
      confMis = abs(post(misIdx) - thr);
      [~, ord] = sort(confMis, 'descend');
      drop1 = misIdx(ord(1:min(nMis, numel(ord))));
    else
      drop1 = [];
    end

    % 2) near-threshold trials
    distThr = abs(p_c - thr);
    [~, ord2] = sort(distThr, 'ascend');
    drop2 = idx_c(ord2(1:min(nThr, numel(ord2))));

    % mark for pruning
    pruneMask(unique([drop1; drop2])) = false;
  end
end

% function keepMask = pruneTrialsMask(labels, posteriors, threshold, pctRemove)
% % PRUNETRIALSMASK  Mask out high-confidence misclassifications & near-threshold
% %
% % Inputs:
% %   labels       - (N×1) true labels, e.g. 0 or 1
% %   posteriors   - (N×1) model P(class==1) for each trial
% %   threshold    - scalar decision boundary (e.g. 0.5 or your opt. threshold)
% %   pctRemove    - fraction to remove per class for each type (e.g. 0.05)
% %
% % Output:
% %   keepMask     - (N×1) logical, true = keep, false = prune
% 
% N = numel(labels);
% keepMask = true(N,1);
% 
% % static “high confidence” cut-offs
% low_conf  = 0.2;
% high_conf = 0.8;
% 
% classes = unique(labels);
% for c = classes(:)'
%     % indices of this class
%     idx_c = find(labels==c);
%     p_c   = posteriors(idx_c);
%     n_c   = numel(idx_c);
% 
%     % 1) high-confidence misclassifications
%     if c==1
%       misIdx = idx_c(p_c < low_conf);
%     else
%       misIdx = idx_c(p_c > high_conf);
%     end
%     nMisRemove = ceil(pctRemove * n_c);
%     if ~isempty(misIdx)
%       % confidence = distance from threshold
%       confMis = abs(posteriors(misIdx) - threshold);
%       [~, ord] = sort(confMis, 'descend');
%       toRmMis = misIdx(ord(1:min(nMisRemove,numel(ord))));
%     else
%       toRmMis = [];
%     end
% 
%     % 2) near-threshold trials (lowest |p–threshold|)
%     distThr = abs(p_c - threshold);
%     [~, ord2] = sort(distThr, 'ascend');
%     nThrRemove = ceil(pctRemove * n_c);
%     toRmThr = idx_c(ord2(1:min(nThrRemove,numel(ord2))));
% 
%     % combine and drop
%     toRm = unique([toRmMis; toRmThr]);
%     keepMask(toRm) = false;
% end
% end
