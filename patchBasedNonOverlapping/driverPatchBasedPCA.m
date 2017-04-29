clear all;
close all;
outDirectory = 'results';
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
misAlign=0;
sliceNumber  = 80;
% Now read a new test image and find its weights

% ------------------------------Now read a new test image and find its weights
fnm = sprintf('t1_ai_msles2_1mm_pn0_rf40.rawb');
dim = [181 217 181];
fid = fopen(fnm);
data = fread(fid, prod(dim), 'uint8');
img = reshape(data, dim);
testImData = zeros(280,280);
testIm = zeros(280,280);
testImData(51:50+180,51:50+180) = img(1:180,21:200,sliceNumber); % taking the (sliceNumber)th slice of every volume for training
if misAlign==1
    testImData = imrotate(testImData,rotAngle,'bilinear','crop');    
    testIm(1:end-transY,1:end-transX) = testImData(transY+1:end,transX+1:end);    
else
    testIm = testImData;
end


fnm = sprintf('t1_ai_msles2_1mm_pn0_rf0.rawb');
dim = [181 217 181];
fid = fopen(fnm);
data = fread(fid, prod(dim), 'uint8');
img = reshape(data, dim);
groundTruthImData = zeros(280,280);
groundTruthIm = zeros(280,280);
groundTruthImData(51:50+180,51:50+180) = img(1:180,21:200,sliceNumber); % taking the 58th slice of every volume for training
if misAlign==1
    groundTruthImData = imrotate(groundTruthImData,rotAngle,'bilinear','crop');    
    groundTruthIm(1:end-transY,1:end-transX) = groundTruthImData(transY+1:end,transX+1:end);    
else
    groundTruthIm = groundTruthImData;
end

%testIm = double(imread('testBrain.png'));

temp = testIm;
testIm = testIm(:);
% alphaTest = eigenVecs'*(testIm-meanTemplate);
 
%-----The actual experiment starts...-----------------------------------------------------------

testIm = reshape(testIm,size(temp));
%input = testIm;
load('dir_vectors_3668.mat');
id = 1:40;
dim = size(testIm);

lambda1 = 0.1;
lambda2List = 20;%1:1:100; 
lambda3List = [0.1];
rel_tol = 0.001; % relative target duality gap
angleSet = 12;
numCycles = 5;  

totalNumIterations = length(lambda2List);
relMseVall2 = zeros(1,totalNumIterations);
ssiml2 = zeros(1,totalNumIterations);
primaryObj = zeros(1,totalNumIterations);

%%
tic;
parfor lambda2 = 1:length(lambda2List)
    

    numAngles = angleSet(1);
    idx = id(1:numAngles);
    output = zeros(size(testIm));

    CompleteAngleSet  = mtt(idx,:);
    idx1 = (atan(CompleteAngleSet(:,2)./CompleteAngleSet(:,1)))*180/pi;

    x = dct2(testIm);
    x = reshape(x, [dim(1)*dim(2) 1]);
%         DCT Coefficients of the image 
    xArray = x;

%%         Dictionary Learning Part Starts  
    % assume initial set of alphas to be all zeros
     la2 = lambda2List(lambda2);
     la3 = lambda3List(1);

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

    relMseVall2(lambda2) = mean2((output- groundTruthIm).^2)/mean2(groundTruthIm.^2);
    ssiml2(lambda2) = ssim(output,groundTruthIm);
    primaryObj(lambda2) = history(2,end);



    name = sprintf('%s/result_lambda2_%.2f.png',outDirectory,la2);
    temp = output;
    temp = temp - min(temp(:));
    temp = temp/max(temp(:));
    imwrite(temp,name);
    %       
      %----------------------------------------------      

end 

toc;

%save('errors','mseVal');
