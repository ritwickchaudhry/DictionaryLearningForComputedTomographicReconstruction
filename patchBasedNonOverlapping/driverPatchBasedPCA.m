clear all;
close all;

% Load the Eigen Space information learned from the templates.

es = load('brainPatchBasedEigenSpace.mat');
eigenVecs = es.V;
% eigenVals = es.eigenVals;
numDims = size(eigenVecs,2);
meanTemplate = es.meanPatch;
patchSize = es.patchSize;
minimum = es.minimum;
maximum = es.maximum;
% minimum = es.minimum;
% maximum = es.maximum;
folderName = 'testPatchBasedPCA';
mkdir(folderName);

% Now read a new test image and find its weights

% testIm = double(imread('test_1.pgm'));
testIm = double(imread('testBrain.png'));
% testIm = testIm(13:180,1:168);
temp = testIm;
testIm = testIm(:);
% alphaTest = eigenVecs'*(testIm-meanTemplate);
 
%-----The actual experiment starts...-----------------------------------------------------------

testIm = reshape(testIm,size(temp));
input = testIm;
load('dir_vectors_3668.mat');
id = 1:40;
dim = size(input);

lambda1 = 0.1;
lambda2List = [0.1]; 
lambda3List = [0.1];
%lambda2List = [0.1 0.2]; 
%lambda3List = [0.1 0.2];
rel_tol = 0.01; % relative target duality gap
angleSet = [5 10 40];
numCycles = 5;  

mseVal = zeros(length(angleSet),length(lambda2List),length(lambda3List));
% mseValCropped = zeros(length(angleSet),length(lambda2List),length(lambda3List));

%%
for ang = 1:length(angleSet)
    for lambda2 = 1:length(lambda2List)
        tic;
        for lambda3 = 1:length(lambda3List)
            numAngles = angleSet(ang);
            idx = id(1:numAngles);

            CompleteAngleSet  = mtt(idx,:);
            idx1 = (atan(CompleteAngleSet(:,2)./CompleteAngleSet(:,1)))*180/pi;

            x = dct2(input);
            x = reshape(x, [dim(1)*dim(2) 1]);
    %         DCT Coefficients of the image 
            xArray = x;

    %%         Dictionary Learning Part Starts  
            % assume initial set of alphas to be all zeros
             la2 = lambda2List(lambda2);
             la3 = lambda3List(lambda3);

            Afun = @(z) ARadon5(z,idx1,dim,patchSize,numAngles,la2);
            Atfun = @(z) AtRadon5(z,idx1,dim,patchSize,numAngles,la2);

            Afun2 = @(z) ARadon6(z,eigenVecs);
            Atfun2 = @(z) AtRadon6(z,eigenVecs);

            H = dim(1);
            W = dim(2);
            numPatches = ((H*W)/(patchSize*patchSize))
            
            for numIter = 1:numCycles         
                disp "---------------------------------------";
                numIter
                disp "---------------------------------------";
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
    %                     Zero Alphas For all patches (That's why just mean template)                
%                     disp Here1
                    vec = (la2/numPatches)*meanTemplate;
                    vec = repmat(vec,numPatches,1);
%                     size(vec)
                    y = cat(1,y,vec);
                    size(y)
%                     disp Here2

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
%                     result  = result(:);

    %                      Code to find the next set of optimum alphas and
    %                      concatenate the new patches in the y vector
                    count = 0;
                    for j=1:(H/patchSize)
                        for k=1:(W/patchSize)
                            dimH = (j-1)*patchSize + 1;
                            dimW = (k-1)*patchSize + 1;
%                             Extract the patch from the result till yet
                            patch = reshape(result(dimH:dimH + patchSize - 1,dimW:dimW + patchSize - 1),[patchSize*patchSize 1]);
                            patch = patch - meanTemplate;
                            m = size(patch,1);
                            n = numDims;
%                             Solve for alphas for every patch
                            
                            disp "-------------PatchCount---------------------";
                            count
                            disp "--------------------------------------------";
                            
                            [x_hat,status,history]=l1_ls_modified(Afun2,Atfun2,m,n,patch,la3/la2,rel_tol,'quiet');                    
                            newPatch=(eigenVecs*x_hat(:)) + meanTemplate;
%                             Append into the y yector so as to solve for
%                             thetas later
                            y = cat(1,y,(la2/numPatches)*newPatch);
                            count = count + 1;
                        end
                    end
                end                         

                
                m = size(y,1);
                n = size(xArray,1);         
    %             Solve for theta given the eigencoefficients of the patches
                [x_hat,status,history]=l1_ls_modified(Afun,Atfun,m,n,y,lambda1,rel_tol);

                startPt = 1;
                endPt = size(x_hat,1);
                xp = x_hat(startPt:endPt);

                output = reshape(xp, [dim(1) dim(2)]);
    %             Take the IDCT to get the image from the theta
                output = idct2(output);          

            end
		in = input;
		out = output;
                Nmr = (out-in).^2;
    %                 Dnr = in-mean2(in).^2;
                mseVal(ang,lambda2,lambda3) = sqrt(sum(Nmr(:))/length(Nmr(:))); % computing relative MSE value.
 
                in = in - minimum;
                in = in./maximum;

                out = out - minimum;
                out = out./maximum;


                outfileName = sprintf('%s/%d_angles_PatchBasedPCA_lambda2_%d_lambda3_%d',folderName,numAngles,la2,la3); 
                imwrite(out,outfileName,'png');

    %                 mseVal(ang,lamb) = sum(Nmr(:))/sum(Dnr(:)); % computing relative MSE value.
          %----------------------------------------------      
        end
    end 
end
toc;

save('errors','mseVal');

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
