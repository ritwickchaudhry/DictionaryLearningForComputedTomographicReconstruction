function b = AtRadon5(b_data,idx,dim,patchSize, numAngles,lambda2)
%% Split the Column into the components
H = dim(1);
W = dim(2);
numPatches = (H - (patchSize-1))*(W - (patchSize-1));
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

minY = 1;
maxY = H - (patchSize-1);
minX = 1;
maxX = W - (patchSize-1);

for j=minY:maxY
    for k=minX:maxX
        patchStart = (counter*patchSize*patchSize) + 1;
        patchEnd = patchStart + (patchSize*patchSize) - 1;
        patch = Y2(patchStart:patchEnd);
        patch = ((lambda2/numPatches)^2) * patch;
        patch = reshape(patch,[patchSize patchSize]);
        image(j:j+patchSize-1,k:k+patchSize-1) = image(j:j+patchSize-1,k:k+patchSize-1)+patch;
        counter = counter + 1;
    end
end

% -------------- Final Image Obtained

% Collectively Take DCT of the sum to get theta
b = reshape(dct2(image),[dim(1)*dim(2) 1]);