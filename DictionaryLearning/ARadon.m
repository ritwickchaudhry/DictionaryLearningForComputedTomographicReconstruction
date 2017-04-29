
function b = ARadon(X_data,idx,dim,numAngles)

startPt = 1;
endPt = dim(1)*dim(2);
X = X_data(startPt:endPt);
X = reshape(X,[dim(1) dim(2)]);
X = idct2(X);

startPt = 1;
endPt = numAngles;
angles = idx(startPt:endPt);

radProj = radon(X,angles);  

b1 = reshape(radProj,[size(radProj,1)*size(radProj,2) 1]);

b = b1;

