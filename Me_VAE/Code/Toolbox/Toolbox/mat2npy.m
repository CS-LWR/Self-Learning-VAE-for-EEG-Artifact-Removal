function mat2npy(sourceDir, targetDir)
% MAT2NPY Converts all variables in each .mat file to individual .npy files.
%
%   mat2npy(sourceDir, targetDir)
%
%   Inputs:
%     sourceDir - Directory containing .mat files
%     targetDir - Destination directory for .npy files
%
%   Each variable in a .mat file is saved as a separate .npy file, named as:
%     [matFileName]_[variableName].npy

    if nargin < 2
        error('Both sourceDir and targetDir must be specified.');
    end

    if ~exist(targetDir, 'dir')
        mkdir(targetDir);
    end

    matFiles = dir(fullfile(sourceDir, '*.mat'));

    if isempty(matFiles)
        fprintf('No .mat files found in: %s\n', sourceDir);
        return;
    end

    for k = 1:length(matFiles)
        matFilePath = fullfile(sourceDir, matFiles(k).name);
        [~, fileName, ~] = fileparts(matFiles(k).name);

        try
            vars = load(matFilePath);
            varNames = fieldnames(vars);

            for i = 1:numel(varNames)
                varName = varNames{i};
                varData = vars.(varName);

                npyFileName = sprintf('%s_%s.npy', fileName, varName);
                npyFilePath = fullfile(targetDir, npyFileName);

                try
                    writeNPY(varData, npyFilePath);
                    fprintf('Saved: %s → %s\n', varName, npyFilePath);
                catch ME_inner
                    warning('Failed to save variable "%s" from file "%s": %s\n', ...
                            varName, matFiles(k).name, ME_inner.message);
                end
            end

        catch ME
            warning('Failed to load .mat file: %s\nError: %s\n', matFilePath, ME.message);
        end
    end

    disp('All .mat variables have been converted to .npy.');
end
