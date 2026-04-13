function out = compute_stroop_effects(data)
out = struct('has',false);

if ~isfield(data,'stroop1') || ~isfield(data,'stroop2'), return; end
if ~isfield(data.stroop1,'beh') || ~isfield(data.stroop2,'beh'), return; end

b1 = data.stroop1.beh;
b2 = data.stroop2.beh;

need = {'Response','Trial_Type','Reaction_Time'};
if ~all(isfield(b1,need)) || ~all(isfield(b2,need)), return; end

keep1 = (b1.Response == 1);
tt1 = string(b1.Trial_Type(keep1));
rt1 = b1.Reaction_Time(keep1);

keep2 = (b2.Response == 1);
tt2 = string(b2.Trial_Type(keep2));
rt2 = b2.Reaction_Time(keep2);

m1_cong = mean(rt1(strcmpi(tt1,'congruent')),'omitnan');
m1_inc  = mean(rt1(strcmpi(tt1,'incongruent')),'omitnan');
m2_cong = mean(rt2(strcmpi(tt2,'congruent')),'omitnan');
m2_inc  = mean(rt2(strcmpi(tt2,'incongruent')),'omitnan');

out.has = true;
out.pre.meanCong = m1_cong;
out.pre.meanInc  = m1_inc;
out.pre.effect   = m1_inc - m1_cong;

out.post.meanCong = m2_cong;
out.post.meanInc  = m2_inc;
out.post.effect   = m2_inc - m2_cong;

end
