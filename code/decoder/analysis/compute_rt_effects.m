function out = compute_rt_effects(data)
out = struct('has',false);

if ~isfield(data,'training1') || ~isfield(data,'training2'), return; end
if ~isfield(data.training1,'beh') || ~isfield(data.training2,'beh'), return; end
if ~isfield(data.training1.beh,'RT') || ~isfield(data.training1.beh,'trial_type'), return; end
if ~isfield(data.training2.beh,'RT') || ~isfield(data.training2.beh,'trial_type'), return; end

calib = data.training1.beh;
final = data.training2.beh;

rtC_nd = calib.RT(calib.trial_type==0);
rtC_d  = calib.RT(calib.trial_type==1);
rtF_nd = final.RT(final.trial_type==0);
rtF_d  = final.RT(final.trial_type==1);

mC_nd = mean(rtC_nd,'omitnan'); mC_d = mean(rtC_d,'omitnan');
mF_nd = mean(rtF_nd,'omitnan'); mF_d = mean(rtF_d,'omitnan');

out.has = true;
out.pre.meanND = mC_nd; out.pre.meanD = mC_d; out.pre.diff = mC_nd - mC_d;
out.post.meanND = mF_nd; out.post.meanD = mF_d; out.post.diff = mF_nd - mF_d;

end
