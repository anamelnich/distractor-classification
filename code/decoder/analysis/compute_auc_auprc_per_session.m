function out = compute_auc_auprc_per_session(data, Pcell)

sessions = 1:5;
sessFields = arrayfun(@(s)sprintf('decoding%d', s), sessions, 'UniformOutput', false);

auc_session   = nan(5,1);
auprc_session = nan(5,1);
pr_chance     = nan(5,1);
n_eff_trials  = zeros(5,1);

for si = 1:5
    sf = sessFields{si};
    if ~isfield(data, sf) || ~isfield(data.(sf),'beh'), continue; end
    beh = data.(sf).beh;
    if ~isfield(beh,'trial_type'), continue; end

    y_true = beh.trial_type(:);
    P = Pcell{si};
    if size(P,2) < 3, continue; end

    score = double(P(:,1));
    y_out_raw = double(P(:,3));  % 1=distr,2=no,3=amb
    y_out = y_out_raw; y_out(y_out==2) = 0;

    nmatch = min(numel(y_true), numel(score));
    y_true = y_true(1:nmatch);
    score  = score(1:nmatch);
    y_out  = y_out(1:nmatch);

    keep = (y_out ~= 3);
    yk = y_true(keep);
    sk = score(keep);
    n_eff_trials(si) = numel(yk);

    if numel(yk) == 0 || numel(unique(yk)) < 2, continue; end

    pr_chance(si) = mean(yk==1);

    auc_session(si)   = local_safe_auc(yk, sk);
    auprc_session(si) = local_safe_auprc(yk, sk);
end

out = struct();
out.auc_session = auc_session;
out.auprc_session = auprc_session;
out.pr_chance = pr_chance;
out.n_eff_trials = n_eff_trials;

end

function auc = local_safe_auc(y, s)
y = double(y); s = double(s);
[~,~,~,auc] = perfcurve(y, s, 1);
end

function auprc = local_safe_auprc(y, s)
y = double(y); s = double(s);
[~,~,~,auprc] = perfcurve(y, s, 1, 'xCrit','reca','yCrit','prec');
end
