templateDirectory = './TrainImages/';
dirInfo = dir(templateDirectory);
numTemplates = length(dirInfo) - 2;
patchSize = 9;


% Get the image size
name = sprintf('%s%s',templateDirectory, dirInfo(3).name);
image1 = double(imread(name));
[H,W] = size(image1);

numPatches = H*W/(patchSize^2)*numTemplates;
dataSet = zeros(patchSize*patchSize, numPatches);

counter = 1;

for i = 3:numTemplates
    name = sprintf('%s%s', templateDirectory, dirInfo(i).name);
    image = double(imread(name));
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

% Dataset of Patches Created
% Call to K-SVD

params = struct('K',200,'numIteration',1,'errorFlag',1,'preserveDCAtom',1, 'InitializationMethod', 'DataElements','errorGoal',1.0, 'displayProgress', 1);

[dict, output] = KSVD(dataSet,params);

save('dictionary','dict','minimum','maximum','patchSize');


