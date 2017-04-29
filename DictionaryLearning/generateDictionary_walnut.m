close all;
clear all;

addpath('./KSVD_Matlab_ToolBox');

patchSize = 13;

 file = load('../../walnut.mat');
 img  = file.V;
 numTemplates = 7;

in = img(:,:,70);  
image1 = double(in(71:330,31:290));


[H,W] = size(image1);
numPatches = H*W/(patchSize^2)*numTemplates;
dataSet = zeros(patchSize*patchSize, numPatches);
counter = 1;
tic
for i = 1:numTemplates
    in = img(:,:,70+10*(i-1));  
    image = double(in(71:330,31:290));
    
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

params = struct('K',200,'numIteration',1,'errorFlag',1,'preserveDCAtom',1, 'InitializationMethod', 'DataElements','errorGoal',1.0, 'displayProgress', 1);
tic
[dict, output] = KSVD(dataSet,params);
toc

save('dictionary','dict','meanPatch','minimum','maximum','patchSize');
