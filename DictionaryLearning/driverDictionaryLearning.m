clear all;
close all;

misAlign=1;
rotAngle = 40;
sliceNumber  = 80;
outDirectory = 'results/brainweb';

% Load the Eigen Space information learned from the templates.

parameters = load('dictionary_brainweb_patch_30.mat');
dict = parameters.dict;
numDims = size(dict,2);
meanTemplate = parameters.meanPatch;
patchSize = parameters.patchSize;
minimum = parameters.minimum;
maximum = parameters.maximum;
% minimum = es.minimum;
% maximum = es.maximum;
folderName = 'testDictionaryBasedPrior';
mkdir(folderName);

% Now read a new test image and find its weights

% ------------------------------Now read a new test image and find its weights
fnm = sprintf('t1_ai_msles2_1mm_pn0_rf40.rawb');
dim = [181 217 181];
fid = fopen(fnm);
data = fread(fid, prod(dim), 'uint8');
img = reshape(data, dim);
testImData = zeros(300,300);
testImData(61:60+181,41:40+217) = img(1:181,1:217,sliceNumber);  % taking the (sliceNumber)th slice of every volume for training
if misAlign==1
    testIm= imrotate(testImData,rotAngle,'bilinear','crop');    
    %testIm(1:end-transY,1:end-transX) = testImData(transY+1:end,transX+1:end);    
else
    testIm = testImData;
end


fnm = sprintf('t1_ai_msles2_1mm_pn0_rf0.rawb');
dim = [181 217 181];
fid = fopen(fnm);
data = fread(fid, prod(dim), 'uint8');
img = reshape(data, dim);
groundTruthImData = zeros(300,300);
groundTruthImData(61:60+181,41:40+217) = img(1:181,1:217,sliceNumber);  % taking the (sliceNumber)th slice of every volume for training
if misAlign==1
    groundTruthIm = imrotate(groundTruthImData,rotAngle,'bilinear','crop');    
    %groundTruthIm(1:end-transY,1:end-transX) = groundTruthImData(transY+1:end,transX+1:end);    
else
    groundTruthIm = groundTruthImData;
end

figure;imshow([groundTruthIm testIm],[]);
%-------------------------------------------------------------------
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
aa = 0:0.1:2;
bb = 3:1:40;
lambda2List = cat(2,aa,bb)
lambda3List = [0.1];
%lambda2List = [0.1 0.2]; 
%lambda3List = [0.1 0.2];
rel_tol = 0.001; % relative target duality gap
angleSet = 12;
numCycles = 5;  

totalNumIterations = length(lambda2List);
relMseVall2 = zeros(1,totalNumIterations);
ssiml2 = zeros(1,totalNumIterations);
primaryObj = zeros(1,totalNumIterations);


%---------------Compute noise to be added--------------------------
numAngles = angleSet(1);
idx = id(1:numAngles);
CompleteAngleSet  = mtt(idx,:);
idx1 = (atan(CompleteAngleSet(:,2)./CompleteAngleSet(:,1)))*180/pi;

startPt = 1;
endPt = numAngles;
angles = idx1(startPt:endPt);

radProj = radon(testIm,angles);
y = reshape(radProj,[size(radProj,1)*size(radProj,2) 1]); 
noiseMean = 0;
noiseSD = 0.02*mean(y);
noise = noiseMean + noiseSD*randn(size(y));

%-----------------------------------------------------------------

x = dct2(testIm);
x = reshape(x, [dim(1)*dim(2) 1]);
%         DCT Coefficients of the image 
xArray = x;
la3 = lambda3List(1);

%%
tic;

parfor lambda2 = 1:length(lambda2List)
        output = zeros(size(testIm));


%%         Dictionary Learning Part Starts  
        % assume initial set of alphas to be all zeros
         la2 = lambda2List(lambda2);
         

        Afun = @(z) ARadon5(z,idx1,dim,patchSize,numAngles,la2);
        Atfun = @(z) AtRadon5(z,idx1,dim,patchSize,numAngles,la2);

        Afun2 = @(z) ARadon6(z,dict);
        Atfun2 = @(z) AtRadon6(z,dict);

        H = dim(1);
        W = dim(2);
        numPatches = ((H*W)/(patchSize*patchSize))

        for numIter = 1:numCycles         
            disp '---------------------------------------';
            numIter
            disp '---------------------------------------';
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

%                     Actual Measurements - y
                y = reshape(radProj,[size(radProj,1)*size(radProj,2) 1]); 
                y = y + noise;
%                     Zero Alphas For all patches (That's why just mean template)                
%                     disp Here1
                vec = (la2/numPatches)*meanTemplate;
                vec = repmat(vec,numPatches,1);
%                     size(vec)
                y = cat(1,y,vec);
                size(y);
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
                y = y+ noise;

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

                        disp '-------------PatchCount---------------------';
                        count
                        disp '--------------------------------------------';

%                             Solve for alphas for every patch
                        [x_hat,status,history]=l1_ls_modified(Afun2,Atfun2,m,n,patch,la3/la2,rel_tol);
                        newPatch=(dict*x_hat(:)) + meanTemplate;
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
    
    fname = sprintf('%s/result_lambda2_%.2f.mat',outDirectory,la2);
    parsave(fname,output);

end 

toc;

%save('errors','mseVal');
fname = sprintf('%s/groundTruth.mat',outDirectory);
parsave(fname,groundTruthIm);

