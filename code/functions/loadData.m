function data = loadData(dataPath, subjectID)
% loadData loads EEG recordings from multiple day folders for a subject.
%
% Assumptions:
%  - Day folders are named as: subjectID_YYYYMMDD
%  - Each day folder contains subfolders named: subjectID_YYYYMMDDHHMMSS_task
%
% Recordings for the same task on the same day are concatenated.
% Recordings from different days are stored separately (e.g., decoding1, decoding2).
%
% Inputs:
%   dataPath  - Path to the directory containing subject folders
%   subjectID - Subject identifier (e.g., 'e5')
%
% Output:
%   data      - Struct with fields like decoding1, training1, etc.


dayFolders = dir(fullfile(dataPath, [subjectID '_20*']));
if isempty(dayFolders)
    error('No day folders found for subject %s in %s', subjectID, dataPath);
end

tasks = struct(); 

for d = 1:length(dayFolders)
    dayFolderName = dayFolders(d).name;
    dayFolderPath = fullfile(dayFolders(d).folder, dayFolderName);

    dayTokens = regexp(dayFolderName, ['^' subjectID '_(\d{8})$'], 'tokens');
    if isempty(dayTokens)
        warning('Invalid day folder format: %s', dayFolderName);
        continue;
    end
    dayField = ['d' dayTokens{1}{1}];

    subFolderInfo = dir(dayFolderPath);
    subFolderNames = setdiff({subFolderInfo([subFolderInfo.isdir]).name}, {'.', '..'});

    for i = 1:length(subFolderNames)
        subFolderName = subFolderNames{i};
        subFolderPath = fullfile(dayFolderPath, subFolderName);

        % Extract task type
        tokens = regexp(subFolderName, ['^' subjectID '_\d{14}_(\w+)$'], 'tokens');
        if isempty(tokens)
            warning('Invalid subfolder format: %s', subFolderName);
            continue;
        end
        taskType = lower(tokens{1}{1});

        taskSession = loadTaskData(subFolderPath);

        if ~isfield(tasks, taskType)
            tasks.(taskType) = struct();
        end
        if ~isfield(tasks.(taskType), dayField)
            tasks.(taskType).(dayField) = {taskSession};
        else
            tasks.(taskType).(dayField){end+1} = taskSession;
        end
    end
end

data = struct();
taskTypes = fieldnames(tasks);
for t = 1:length(taskTypes)
    taskType = taskTypes{t};
    dayFields = sort(fieldnames(tasks.(taskType)));  % Sorted by date
    for i = 1:length(dayFields)
        combined = [];
        sessions = tasks.(taskType).(dayFields{i});
        for j = 1:length(sessions)
            combined = concatenateSession(combined, sessions{j});
        end
        fieldName = [taskType num2str(i)];
        data.(fieldName) = combined;
    end
end

end

%=====================================================================
% Helper Functions
%=====================================================================
function taskData = loadTaskData(folder)
% Loads EEG data from the first GDF file in the folder

files = dir(fullfile(folder, '*.gdf'));
if isempty(files)
    error('No GDF files found in folder %s', folder);
end

filePath = fullfile(files(1).folder, files(1).name);
[signal, header] = sload(filePath);

taskData.data = signal;
taskData.header = header;
taskData.eof = size(signal, 1);
end


function combined = concatenateSession(combined, newSession)
% Concatenates two EEG session structs along the time axis

if isempty(combined)
    combined = newSession;
    return;
end

offset = size(combined.data, 1);

if isfield(combined.header, 'EVENT') && isfield(newSession.header, 'EVENT')
    newEventPos = newSession.header.EVENT.POS + offset;
    combined.header.EVENT.TYP = cat(1, combined.header.EVENT.TYP, newSession.header.EVENT.TYP);
    combined.header.EVENT.POS = cat(1, combined.header.EVENT.POS, newEventPos);
end

combined.data = cat(1, combined.data, newSession.data);
combined.eof = cat(1, combined.eof, size(combined.data, 1));
end
