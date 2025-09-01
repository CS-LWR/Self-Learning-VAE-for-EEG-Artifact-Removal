function npy2mat(sourceDir, targetDir)
% CONVERTNPYTOMAT Converts all .npy files in a folder to .mat files.
%
%   convertNPYtoMAT(sourceDir, targetDir)
%
%   Inputs:
%     sourceDir - Directory containing .npy files
%     targetDir - Destination directory for .mat files
%
%   Each .npy file is loaded using readNPY and saved as a .mat file with the same name.
%   The variable inside each .mat file is saved as 'data'.

    % Validate input arguments
    if nargin < 2
        error('Both sourceDir and targetDir must be specified.');
    end

    % Create target directory if it does not exist
    if ~exist(targetDir, 'dir')
        mkdir(targetDir);
    end

    % List all .npy files in source directory
    npyFiles = dir(fullfile(sourceDir, '*.npy'));

    if isempty(npyFiles)
        fprintf('No .npy files found in: %s\n', sourceDir);
        return;
    end

    % Process each .npy file
    for k = 1:length(npyFiles)
        npyFilePath = fullfile(sourceDir, npyFiles(k).name);
        [~, fileName, ~] = fileparts(npyFiles(k).name);

        try
            % Load the .npy file
            data = readNPY(npyFilePath);
        catch ME
            warning('Failed to read file: %s\nError: %s\n', npyFilePath, ME.message);
            continue;
        end

        % Save as .mat file
        matFilePath = fullfile(targetDir, [fileName '.mat']);
        save(matFilePath, 'data');

        fprintf('Converted: %s --> %s\n', npyFilePath, matFilePath);
    end

    disp('All .npy files have been converted.');
end
