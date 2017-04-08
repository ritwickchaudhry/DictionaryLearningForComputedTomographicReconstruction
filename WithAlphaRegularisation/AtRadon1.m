function X_final = AtRadon1(b_data,idx,dim,numAngles,lambda1)
    
s1 = size(b_data,1)-(dim(1)*dim(2)); %m
s2 = size(b_data,1);  %(m+n)

startPt = 1;
endPt =  s1;
b = b_data(startPt:endPt);
b = reshape(b,[(size(b,1))/numAngles numAngles]);

startPt = endPt + 1;
endPt = s2;
Y2 = b_data(startPt:endPt);

startPt = 1;
endPt = numAngles;
angles = idx(startPt:endPt);

backProjImg =  iradon(b,angles,'linear','Cosine');    

backProjImg = backProjImg(2:2+dim(1)-1,2:2+dim(2)-1);
X = dct2(backProjImg);
X = reshape(X,[size(backProjImg,1)*size(backProjImg,2) 1]);  
       
%-------------- Modification for the PICCS part -starts------------------
  
temp = dct2(reshape(Y2,[dim(1) dim(2)]));
temp = reshape(temp,[dim(1)*dim(2) 1]);
X = X + (lambda1^2).*temp;

%-------------- Modification for the PICCS part -ends------------------

X_final = X;


   
    
   




