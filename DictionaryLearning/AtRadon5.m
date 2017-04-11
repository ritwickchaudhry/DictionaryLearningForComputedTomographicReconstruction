function b = AtRadon5(b_data,idx,dim,patchSize, numAngles,lambda2)
%% Split the Column into the components
H = dim(1);
W = dim(2);
numPatches = ((H*W)/(patchSize*patchSize));
s1 = size(b_data,1)-(numPatches*(patchSize^2));
s2 = size(b_data,1);

startPt = 1;
endPt =  s1;
Y1 = b_data(startPt:endPt);
Y1 = reshape(Y1,[(size(Y1,1))/numAngles numAngles]);

startPt = endPt + 1;
endPt = s2;
Y2 = b_data(startPt:endPt);

%% Get the DCT of the image from the projections

angles = idx(1:numAngles);
backProjImg =  iradon(Y1,angles,'linear','Cosine');
backProjImg = backProjImg(2:2+dim(1)-1,2:2+dim(2)-1);
image = backProjImg;

% Final Y Achieved

%% Find the Final Patches in the given vector and rearrange the image

counter = 0;

H = dim(1);
W = dim(2);

numPatches = ((H*W)/(patchSize*patchSize));

for j=1:(H/patchSize)
    for k=1:(W/patchSize)
	dimH = (j-1)*patchSize + 1;
	dimW = (k-1)*patchSize + 1;
    patchStart = (counter*patchSize*patchSize) + 1;
    patchEnd = patchStart + (patchSize*patchSize) - 1;
    patch = Y2(patchStart:patchEnd);
    patch = ((lambda2/numPatches)) * patch;
    patch = reshape(patch,[patchSize patchSize]);
    image(dimH:dimH+patchSize-1,dimW:dimW+patchSize-1) = image(dimH:dimH+patchSize-1,dimW:dimW+patchSize-1)+patch;
    counter = counter + 1;
    end
end

% -------------- Final Image Obtained

% Collectively Take DCT of the sum to get theta
b = reshape(dct2(image),[dim(1)*dim(2) 1]);
