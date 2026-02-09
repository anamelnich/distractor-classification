
function index = computeIndexEOG(trigger)
        [pos, typ] = ismember(trigger, [8 32 44 64]); % top right bottom left
        index.pos = find(pos); % Get positions of all distractor or no-distractor triggers
        typ_matched = typ(pos);
        index.typ = zeros(size(typ_matched));
        index.typ(typ_matched == 1) = 1; %top
        index.typ(typ_matched == 2) = 2; %right
        index.typ(typ_matched == 3) = 3; %bottom
        index.typ(typ_matched == 4) = 4; %left
end
