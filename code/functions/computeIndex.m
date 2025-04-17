
function index = computeIndex(trigger,trigType)

    if trigType
        [pos, typ] = ismember(trigger, [102 104 100 110]); 
    else
        [pos, typ] = ismember(trigger, [202 204 100 110]);
    end 
    index.pos = find(pos); 
    typ_matched = typ(pos);
    index.typ = zeros(size(typ_matched));
    index.typ(typ_matched == 1) = 1; %dright
    index.typ(typ_matched == 2) = 2; %dleft
    index.typ(typ_matched > 2) = 0; %dnone

end
