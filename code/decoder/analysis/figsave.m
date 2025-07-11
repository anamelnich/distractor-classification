% find all figures
figs = findall(groot, 'Type','figure');

for k = 1:numel(figs)
    fh = figs(k);
    
    % choose a name: if the figure has a .Name, use that, otherwise use its number
    if ~isempty(fh.Name)
        name = fh.Name;
    else
        name = sprintf('Figure_%d', fh.Number);
    end
    
    % sanitize the filename (no spaces, illegal chars, etc.)
    name = matlab.lang.makeValidName(name);
    

    
    % save a PNG at 300 dpi
    saveas(fh, [name '.png']);
    % OR, for a vector PDF:
    % saveas(fh, [name '.pdf']);
end
%%
exportgraphics(gcf,'poster_decoder_performance.pdf','ContentType','vector');