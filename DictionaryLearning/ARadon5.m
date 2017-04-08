function b = ARadon5(X_data,idx,dim,patchSize, numAngles,lambda2)
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

numPatches = dim(1)*dim(2)/patchSize*patchSize;
H = dim(1);
W = dim(2);
for j=1:(H/patchSize)
    for k=1:(W/patchSize)
        dimH = (j-1)*patchSize + 1;
        dimW = (k-1)*patchSize + 1;
        b = cat(1,b,(lambda2/numPatches)*reshape(X(dimH:dimH+patchSize-1,dimW:dimW+patchSize-1),[patchSize*patchSize 1]));
    end
end


