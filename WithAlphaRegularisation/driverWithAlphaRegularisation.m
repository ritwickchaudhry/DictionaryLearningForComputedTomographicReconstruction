clear all;
close all;

% Load the Eigen Space information learned from the templates.

es = load('EigenSpaceBrain.mat');
eigenVecs = es.eigenVecs;
% eigenVals = es.eigenVals;
numDim = size(eigenVecs,2);
meanTemplate = es.meanTemplate;
minimum = es.minimum;
maximum = es.maximum;
folderName = 'testWithAlphasBrain';
mkdir(folderName);

% Now read a new test image and find its weights

% testIm = double(imread('test_1.pgm'));
testIm = double(imread('testBrain.png'));
% testIm = testIm(13:180,1:168);
temp = testIm;
testIm = testIm(:);
alphaTest = eigenVecs'*(testIm-meanTemplate);
 

% Reconstruct the test image from its weights

recon = zeros(size(meanTemplate));
for j = 1:numDim
    recon = recon + (alphaTest(j)*eigenVecs(:,j));
end
recon = recon + meanTemplate;
imshow([reshape(testIm,size(temp)) reshape(recon,size(temp))],[]);

%-----The actual experiment starts...-----------------------------------------------------------

testIm = reshape(testIm,size(temp));
input = testIm;
load('dir_vectors_3668.mat');
id = 1:40;
dim = size(input);

lambda = [0.1]; 
rel_tol = 0.01; % relative target duality gap
angleSet = [5 10 40];
lambda1 = [0.1];
numCycles = 4;  

mseVal_FBP = zeros(length(angleSet),1);
mseVal = zeros(length(angleSet),length(lambda1));
mseVal_PICCS = zeros(length(angleSet),length(lambda1));
mseVal_CS = zeros(length(angleSet),length(lambda));

% mseValCropped = zeros(length(angleSet),length(lambda1));
% mseValCropped_PICCS = zeros(length(angleSet),length(lambda1));
% mseValCropped_CS = zeros(length(angleSet),length(lambda));


for la = 1:length(lambda)
    tic;
    for ang = 1:length(angleSet)

        numAngles = angleSet(ang);
        startAngle = 1;
        endAngle = numAngles;
        idx = id(startAngle:endAngle);
        CompleteAngleSet  = mtt(idx,:);
        idx1 = (atan(CompleteAngleSet(:,2)./CompleteAngleSet(:,1)))*180/pi;

        x = dct2(input);
        x = reshape(x, [dim(1)*dim(2) 1]);
        xArray = x;

        %----------------Without any prior  ------------------------------------------------------------------

        Afun = @(z) ARadon2(z,idx1,dim,numAngles);
        Atfun = @(z) AtRadon2(z,idx1,dim,numAngles);

        y =Afun(xArray);     % Measurements are generated here

%         %----------compute FBP result-------------------------------------------------
% 
        s = size(y,1);

        startPt = 1;
        endPt = s;
        b = y(startPt:endPt);
        b = reshape(b,[(size(b,1))/numAngles numAngles]);

        startPt = 1;
        endPt = numAngles;
        angles = idx1(startPt:endPt);    

        FBPresult  =  iradon(b,angles,'linear','Cosine');
        FBPresult = FBPresult(2:2+dim(1)-1,2:2+dim(2)-1);

        size(FBPresult);
        size(input);

        in = input;
        Nmr = (FBPresult-in).^2;
        mseVal_FBP(ang) = sqrt(sum(Nmr(:)/length(Nmr(:)))); % computing relative MSE value.
        
        FBPresult = FBPresult - minimum;
        FBPresult = FBPresult./maximum;
        
        outfileName = sprintf('%s/%d_angles_FBPresult',folderName,numAngles); 
        imwrite(FBPresult,outfileName,'png');   
        figure;imshow(FBPresult,[]);
        
%         in = in - minimum;
%         in = in./maximum;
%         Dnr = (in-mean2(in)).^2;
% 
%         
% 
% 
%         %--------------Compute the CS based result (using only l1_ls)-----------------------
        m = size(y,1);
        n = size(xArray,1);         

        [x_hat,status,history]=l1_ls_modified(Afun,Atfun,m,n,y,lambda(la),rel_tol);

        startPt = 1;
        endPt = size(x_hat,1);
        xp = x_hat(startPt:endPt);
        output = reshape(xp, [dim(1) dim(2)]);
        output = idct2(output);
        
        out = output;
        in = input;        
        Nmr = (out-in).^2;
%         Dnr = (in-mean2(in)).^2;

        mseVal_CS(ang,la) = sqrt(sum(Nmr(:))/length(Nmr(:))); % computing relative MSE value.        
        
        out = out - minimum;
        out = out./maximum;
        in = in - minimum;
        in = in./maximum;

        outfileName = sprintf('%s/%d_angles_without_prior_lambda_%d',folderName,numAngles,la); 
        imwrite(out,outfileName,'png');  
        outfileName = sprintf('%s.png',folderName); 
        imwrite(in,outfileName);
        figure;imshow(out,[]);    

%         mseVal_CS(ang) = sum(Nmr(:))/sum(Dnr(:)); % computing relative MSE value.

        
%         % Computing MSE over a selected region
%         inCropped = in(1:60,:);
%         outCropped = out(1:60,:);
%         Nmr = (outCropped-inCropped).^2;
% %         Dnr = inCropped-mean2(inCropped).^2;
%         mseValCropped_CS(ang,la) = sqrt(sum(Nmr(:))/length(Nmr(:))); % computing relative MSE value.
%         mseValCropped_CS(ang) = sum(Nmr(:))/sum(Dnr(:)); % computing relative MSE value.

%         outfileName = sprintf('%s_cropped.png',folderName); 
%         imwrite(inCropped,outfileName);

% 
%         
            
        %---------------------- With prior + l1_ls + regularisation with alpha ----------------------------------------------------------
        
        for lamb = 1:length(lambda1)
            % assume initial set of alphas to be all zeros

            Afun = @(z) ARadon1(z,idx1,dim,numAngles,lambda1(lamb));
            Atfun = @(z) AtRadon1(z,idx1,dim,numAngles,lambda1(lamb));

            Afun2 = @(z) ARadon3(z,eigenVecs);
            Atfun2 = @(z) AtRadon3(z,eigenVecs);
            
            alphas = cell(1,numCycles);

            for numIter = 1:numCycles         
                if numIter==1

                    startPt = 1;
                    endPt = dim(1)*dim(2);
                    X = xArray(startPt:endPt);
                    X = reshape(X,[dim(1) dim(2)]);
                    X = idct2(X);
                    startPt = 1;
                    endPt = numAngles;
                    angles = idx1(startPt:endPt);
                    radProj = radon(X,angles);    
                    y = reshape(radProj,[size(radProj,1)*size(radProj,2) 1]); 
%                     Actual Measurements - y
%                     Zero Alphas (That's why just mean template)
                    priorIm = meanTemplate(:);
                    temp = lambda1(lamb)*priorIm;  % including the prior term
                    y = cat(1,y,temp);

                else

                    startPt = 1;
                    endPt = dim(1)*dim(2);
                    X = xArray(startPt:endPt);
                    X = reshape(X,[dim(1) dim(2)]);
                    X = idct2(X);
                    startPt = 1;
                    endPt = numAngles;
                    angles = idx1(startPt:endPt);
                    radProj = radon(X,angles);    
                    y = reshape(radProj,[size(radProj,1)*size(radProj,2) 1]); 

                    result = output;
                    result  = result(:);

%                      Code to find the next set of optimum alphas       
                    alphas_y = (result - meanTemplate);
                    
%                     Hard Coded as of now
                    lambda3 = 0.1
                    
                    m = size(alphas_y,1);
                    n = size(eigenVecs,2);
                    [x_hat,status,history]=l1_ls_modified(Afun2,Atfun2,m,n,alphas_y,lambda3,rel_tol);                    
                    alphas{numIter} =  x_hat(:);
                    
                    priorIm = meanTemplate + eigenVecs*alphas{numIter};
                    temp = lambda1(lamb)*priorIm;
                    y = cat(1,y,temp);

                end                         

                m = size(y,1);
                n = size(xArray,1);         

                [x_hat,status,history]=l1_ls_modified(Afun,Atfun,m,n,y,lambda(la),rel_tol);


                startPt = 1;
                endPt = size(x_hat,1);
                xp = x_hat(startPt:endPt);

                output = reshape(xp, [dim(1) dim(2)]);
                output = idct2(output);          

            end
                in = input;
                out = output;

                Nmr = (out-in).^2;
%                 Dnr = in-mean2(in).^2;
                mseVal(ang,lamb) = sqrt(sum(Nmr(:))/length(Nmr(:))); % computing relative MSE value.
            
                in = in - minimum;
                in = in./maximum;
            
                out = out - minimum;
                out = out./maximum;

                outfileName = sprintf('%s/%d_angles_mPICCS_lambda1_%d_lamb_%d',folderName,numAngles,lamb,la); 
                imwrite(out,outfileName,'png');

%                 mseVal(ang,lamb) = sum(Nmr(:))/sum(Dnr(:)); % computing relative MSE value.

%                 outCropped = out(1:60,:);
%                 inCropped = in(1:60,:);
%                 Nmr = (outCropped-inCropped).^2;
% %                 Dnr = inCropped-mean2(inCropped).^2;
%                 mseValCropped(ang,lamb) = sqrt(sum(Nmr(:))/length(Nmr(:))); % computing relative MSE value.
%                 mseValCropped(ang,lamb) = sum(Nmr(:))/sum(Dnr(:)); % computing relative MSE value.

            %----------With a random prior + l1_ls-------------------------
% 
%             Afun = @(z) ARadon1(z,idx1,dim,numAngles,lambda1(lamb));
%             Atfun = @(z) AtRadon1(z,idx1,dim,numAngles,lambda1(lamb));      
% 
%             startPt = 1;
%             endPt = dim(1)*dim(2);
%             X = xArray(startPt:endPt);
%             X = reshape(X,[dim(1) dim(2)]);
%             X = idct2(X);
%             startPt = 1;
%             endPt = numAngles;
%             angles = idx1(startPt:endPt);
%             radProj = radon(X,angles);    
%             y = reshape(radProj,[size(radProj,1)*size(radProj,2) 1]); 
% 
%             priorIm = double(imread('random_prior.pgm'));
%             priorIm = priorIm(13:180,1:168);
%             priorIm = reshape(priorIm, [dim(1)*dim(2) 1]);
%             temp = lambda1(lamb)*priorIm;
%             y = cat(1,y,temp); 
% 
%             m = size(y,1);
%             n = size(xArray,1);         
% 
%             [x_hat,status,history]=l1_ls_modified(Afun,Atfun,m,n,y,lambda(la),rel_tol);
% 
%             startPt = 1;
%             endPt = size(x_hat,1);
%             xp = x_hat(startPt:endPt);
% 
%             output = reshape(xp, [dim(1) dim(2)]);
%             output = idct2(output);
% 
%             out = output;
%             out = out - minimum;
%             out = out./maximum;
% 
%             priorIm = reshape(priorIm,size(testIm));
%             priorIm = priorIm - minimum;
%             priorIm = priorIm./maximum;
% 
%             outfileName = sprintf('%s/%d_angles_random_prior_lambda1_%d_lambda_%d',folderName,numAngles,lamb,la); 
%             imwrite(out,outfileName,'png');   
% 
%             Nmr = (out-in).^2;
% %             Dnr = in-mean2(in).^2;
%             mseVal_PICCS(ang,lamb) = sqrt(sum(Nmr(:))/length(Nmr(:))); % computing relative MSE value.
% 
%             outCropped = out(1:60,:);
%             Nmr = (outCropped-inCropped).^2;
% %             Dnr = inCropped-mean2(inCropped).^2;
%             mseValCropped_PICCS(ang,lamb) = sqrt(sum(Nmr(:))/length(Nmr(:))); % computing relative MSE value only withing a sub-region



      %----------------------------------------------      
        end
    end  
end 
toc;

% for lamb = 1:length(lambda1)
%     mse05 = [mseValCropped_CS(1),mseValCropped_PICCS(1,lamb),mseValCropped(1,lamb)];  
%     outfileName = sprintf('%s/%d_mseVal05.mat',folderName,lamb); 
%     save(outfileName,'mse05');
%     mse10 = [mseValCropped_CS(2),mseValCropped_PICCS(2,lamb),mseValCropped(2,lamb)];
%     outfileName = sprintf('%s/%d_mseVal10.mat',folderName,lamb); 
%     save(outfileName,'mse10');
%     mse40 = [mseValCropped_CS(3),mseValCropped_PICCS(3,lamb),mseValCropped(3,lamb)];  
%     outfileName = sprintf('%s/%d_mseVal40.mat',folderName,lamb); 
%     save(outfileName,'mse40');
% end




