clear all;

templateDirectory = './templatesBrain/';
dirInfo = dir(templateDirectory);
% numTemplates = length(dirInfo) - 2;
numTemplates = 10;
patchSize = 8;
numDims = 64;

% Get the image size
name = sprintf('%s%s',templateDirectory, dirInfo(3).name);
image1 = (imread(name));
[H,W] = size(image1);

numPatches = (W - (patchSize-1)) * (H-(patchSize-1))
dataSet = zeros(patchSize*patchSize, numPatches*numTemplates);

counter = 1;

for i = 3:numTemplates+2
    
    name = sprintf('%s%s', templateDirectory, dirInfo(i).name);
    image = double(imread(name));

    if(counter == 1)
        minimum = min(image(:));
        maximum = max(image(:)-minimum);
    end
      
    minY = 1;
    maxY = H - (patchSize-1);
    minX = 1;
    maxX = W - (patchSize-1);
    for j=minY:maxY
        for k=minX:maxX
            dataSet(:,counter) = reshape(image(j:j+patchSize-1,k:k+patchSize-1),[patchSize*patchSize 1]);
            counter = counter + 1;
        end
    end
end

% Dataset of Patches Created
% Make the PCA Space

meanPatch = mean(dataSet,2);
dataMeanCentred = dataSet - repmat(meanPatch,1,numPatches*numTemplates);
size(dataMeanCentred)
[U,S,V] = svd(dataMeanCentred','econ');
size(V)
V = V(:,1:numDims);

save('brainPatchBasedEigenSpace','V', 'meanPatch','patchSize','minimum','maximum');

% Random Checks
% ------------------------
% numTemplates
% numPatches
% counter
% size(dataSet)
% size(meanPatch)