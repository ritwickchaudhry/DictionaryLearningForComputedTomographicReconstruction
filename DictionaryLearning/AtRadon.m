function X_final = AtRadon(b_data,idx,dim,numAngles)
    
s = size(b_data,1);

startPt = 1;
endPt = s;
b = b_data(startPt:endPt);
b = reshape(b,[(size(b,1))/numAngles numAngles]);

startPt = 1;
endPt = numAngles;
angles = idx(startPt:endPt);    

backProjImg  =  iradon(b,angles,'linear','Cosine');

backProjImg = backProjImg(2:2+dim(1)-1,2:2+dim(2)-1);
X = dct2(backProjImg);
X = reshape(X,[size(backProjImg,1)*size(backProjImg,2) 1]);       


X_final = X;






