function file_full_names=get_files(file_dir,type)
    % GET_FILES - Get list of files from directory with specified type
    % 
    % Syntax: file_full_names = get_files(file_dir, type)
    %
    % Inputs:
    %   file_dir - String, directory path to search for files
    %   type     - String, file pattern/extension (e.g., '*.txt', '*.m')
    %
    % Outputs:
    %   file_full_names - Cell array of strings containing filenames
    %
    % Description:
    %   This function searches for files matching the specified pattern in the
    %   given directory. It automatically removes any hidden system files that
    %   start with '._' (common on macOS) before returning the clean file list.
    %
    % Example:
    %   files = get_files('/path/to/data/', '*.txt');
    %   files = get_files('./images/', '*.png');
    
    % Get initial list of files matching the pattern
    list = dir([file_dir, type]);
    
    % Clean up: remove any system files that start with '._'
    for i = 1:length(list)
        fileName = list(i).name;
    
        % Check if filename contains '._' (hidden system files)
        if contains(fileName, '._')
            % Construct full file path
            filePath = fullfile(file_dir, fileName);
            
            % Delete the system file
            delete(filePath);
        end
    end
    
    % Get updated file list after cleanup
    list = dir([file_dir, type]);

    % Extract filenames from directory structure
    file_names = {list.name}'; 
    file_full_names={};
    
    % Convert cell array to the desired output format
    for i =1:size(file_names,1)
        % Option 1: Return full paths (commented out)
        % file_full_names=[file_full_names;[file_dir char(file_names(i))]];
        
        % Option 2: Return just filenames (current implementation)
        file_full_names=[file_full_names;char(file_names(i))];
    end
end
