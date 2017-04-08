function b = ARadon1(X_data,idx,dim,numAngles,lambda1)


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

vectX =  lambda1.*reshape(X,[dim(1)*dim(2) 1]); 
b = cat(1,b1,vectX);  


