close all;
clear all;

addpath('./KSVD_Matlab_ToolBox');

patchSize = 32;

templateDirectory = '/net/voxel03/misc/me/preetig/Documents/Code/mPICCS/2D/templates/colon/subject03';
dirInfo = dir(templateDirectory);
numSlices = length(dirInfo)-2;

   fileName = dirInfo(13).name;
   filePath = sprintf('%s/%s',templateDirectory,fileName);
   image1 = dicomread(filePath);

numTemplates = 7;
[H,W] = size(image1);
numPatches = H*W/(patchSize^2)*numTemplates;
dataSet = zeros(patchSize*patchSize, numPatches);
counter = 1;
tic
for i = 1:numTemplates
   fileName = dirInfo(13+10*(i-1)).name;
   filePath = sprintf('%s/%s',templateDirectory,fileName);
   image = dicomread(filePath);
    
    if(i==3)
        minimum = min(image(:));
        maximum = max(image(:));
    end
    for j=1:(H/patchSize)
        for k=1:(W/patchSize)
            dimH = (j-1)*patchSize + 1;
            dimW = (k-1)*patchSize + 1;
            dataSet(:,counter) = reshape(image(dimH:dimH+patchSize-1,dimW:dimW+patchSize-1),[patchSize*patchSize 1]);
            counter = counter + 1;
        end
    end
end
toc
meanPatch = mean(dataSet,2);

% Mean Centre the dataset
% size(repmat(meanPatch, [1,size(dataSet,2)]))
% size(dataSet)
dataSet = dataSet - repmat(meanPatch, [1, size(dataSet,2)]);

% Dataset of Patches Created
% Call to K-SVD

params = struct('K',140,'numIteration',1,'errorFlag',1,'preserveDCAtom',1, 'InitializationMethod', 'DataElements','errorGoal',1.0, 'displayProgress', 1);
%params = struct('K',100,'numIteration',1,'errorFlag',1,'preserveDCAtom',1, 'InitializationMethod', 'DataElements','displayProgress', 1);

tic
[dict, output] = KSVD(dataSet,params);
toc

save('dictionary1','dict','meanPatch','minimum','maximum','patchSize');
