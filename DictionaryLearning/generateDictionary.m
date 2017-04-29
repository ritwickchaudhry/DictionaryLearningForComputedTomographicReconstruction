close all;
clear all;

addpath('./KSVD_Matlab_ToolBox');
templateDirectory = './templatesBrain/';
dirInfo = dir(templateDirectory);
numTemplates = length(dirInfo) -2;
patchSize = 30;
sliceNumber = 80;


% Get the image size
name = sprintf('%s%s',templateDirectory, dirInfo(3).name);
%image1 = double(imread(name));
dim = [181 217 181];
fid = fopen(name);
data = fread(fid, prod(dim), 'uint8');
img = reshape(data, dim);
% image1 = img(1:180,21:200,sliceNumber);
% 
image1  = zeros(300,300);
image1(61:60+181,41:40+217) = img(1:181,1:217,sliceNumber); 

[H,W] = size(image1);

numPatches = H*W/(patchSize^2)*numTemplates;
dataSet = zeros(patchSize*patchSize, numPatches);

counter = 1;
figure;
for i = 1:numTemplates
    name = sprintf('%s%s', templateDirectory, dirInfo(i+2).name);
    %image = double(imread(name));
     dim = [181 217 181];
    fid = fopen(name);
    data = fread(fid, prod(dim), 'uint8');
    img = reshape(data, dim);   
   
    image  = zeros(300,300);
    image(61:60+181,41:40+217) = img(1:181,1:217,sliceNumber); 
    imshow(image,[]);title(i);pause(0.3);
    if(i==1)
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

meanPatch = mean(dataSet,2);

% Mean Centre the dataset
% size(repmat(meanPatch, [1,size(dataSet,2)]))
% size(dataSet)
dataSet = dataSet - repmat(meanPatch, [1, size(dataSet,2)]);

% Dataset of Patches Created
% Call to K-SVD

params = struct('K',200,'numIteration',1,'errorFlag',1,'preserveDCAtom',1, 'InitializationMethod', 'DataElements','errorGoal',1.0, 'displayProgress', 1);

[dict, output] = KSVD(dataSet,params);

save('dictionary_brainweb_patch_30','dict','meanPatch','minimum','maximum','patchSize');
