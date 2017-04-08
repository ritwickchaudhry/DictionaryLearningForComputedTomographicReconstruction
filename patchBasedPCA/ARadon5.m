function b = ARadon5(X_data,idx,dim,patchSize,numAngles,lambda2)
%% Get the radon of the image
startPt = 1;
endPt = dim(1)*dim(2);
X = X_data(startPt:endPt);
X = reshape(X,[dim(1) dim(2)]);
X = idct2(X);

angles = idx(1:numAngles);

radProj = radon(X,angles);
radProjVec = reshape(radProj,[size(radProj,1)*size(radProj,2) 1]);

b = radProjVec;
% got Y (Radon of IDCT of Theta)

%% Get the Individual Patches
H = dim(1);
W = dim(2);
numPatches = (W - (patchSize-1)) * (H-(patchSize-1));
minY = 1;
maxY = H - (patchSize-1);
minX = 1;
maxX = W - (patchSize-1);

for j=minY:maxY
    for k=minX:maxX
        b = cat(1,b,(lambda2/numPatches)*reshape(X(j:j+patchSize-1,k:k+patchSize-1),[patchSize*patchSize 1]));
    end
end

