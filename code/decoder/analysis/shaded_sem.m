function shaded_sem(x, m, s)
% draws a transparent band m±s ignoring NaNs
x = x(:); m = m(:); s = s(:);
keep = ~isnan(m) & ~isnan(s);
x = x(keep); m = m(keep); s = s(keep);
if isempty(x), return; end
X = [x; flipud(x)];
Y = [m-s; flipud(m+s)];
patch(X, Y, 1, 'FaceAlpha',0.12, 'EdgeColor','none', 'HandleVisibility','off');
end
