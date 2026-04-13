function Pcell = load_online_posteriors_clean(subjectID, opDir)

F = dir(fullfile(opDir, sprintf('%s_OnlinePosteriors_*.mat', subjectID)));
if isempty(F), error('No OnlinePosteriors files found for %s in %s', subjectID, opDir); end

fn = {F.name};
pat = ['^' regexptranslate('escape',subjectID) '_OnlinePosteriors_(\d{8})\.mat$'];
tok = regexp(fn, pat, 'tokens');
isMatch = ~cellfun(@isempty, tok);
F = F(isMatch);
tok = tok(isMatch);

if numel(F) < 5
    error('Found only %d matching OnlinePosteriors files for %s in %s', numel(F), subjectID, opDir);
end

dates = cellfun(@(c) str2double(c{1}{1}), tok);
[~, ord] = sort(dates);
F = F(ord);

Pcell = cell(5,1);
expectedN = [360 480 480 480 360];
trialPerRun = 60;

for s = 1:5
    tmp = load(fullfile(F(s).folder, F(s).name));
    fns = fieldnames(tmp);
    A = tmp.(fns{1});  % assume one variable
    n = size(A,1);
    expN = expectedN(s);

    if n > expN && n >= trialPerRun
        A = A(trialPerRun+1:end,:); % drop practice run
    end

    Pcell{s} = A;
end

end
