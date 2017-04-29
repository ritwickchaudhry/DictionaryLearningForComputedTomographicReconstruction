clear all;
close all;

outDirectory = 'results/pelvis';

% Load the Eigen Space information learned from the templates.

parameters = load('dictionary1.mat');
dict = parameters.dict;
numDims = size(dict,2);
meanTemplate = parameters.meanPatch;
patchSize = parameters.patchSize;
minimum = parameters.minimum;
maximum = parameters.maximum;
% minimum = es.minimum;
% maximum = es.maximum;


% Now read a new test image and find its weights

% ------------------------------Now read a new test image and find its weights
templateDirectory = '/net/voxel03/misc/me/preetig/Documents/Code/mPICCS/2D/templates/colon/subject03';
dirInfo = dir(templateDirectory);
numSlices = length(dirInfo)-2;

fileName = dirInfo(38).name;
filePath = sprintf('%s/%s',templateDirectory,fileName);
testImData = double(dicomread(filePath));
groundTruthImData = testImData;
 
%outDirectory = sprintf('/net/voxel03/misc/me/preetig/Documents/Code/mPICCS/2D/single_slice/l1_norm/results/%s',dataset');
misAlign = 0;
rotAngle = 33;
transX = 35;  % translating along left is positive
transY = 20;  % translating along  upper direction is positive


if misAlign==1
    testImData = imrotate(testImData,rotAngle,'bilinear','crop');    
    testIm(1:end-transY,1:end-transX) = testImData(transY+1:end,transX+1:end);    
else
    testIm = testImData;
end


if misAlign==1
    groundTruthImData = imrotate(groundTruthImData,rotAngle,'bilinear','crop');    
    groundTruthIm(1:end-transY,1:end-transX) = groundTruthImData(transY+1:end,transX+1:end);    
else
    groundTruthIm = groundTruthImData;
end


figure;imshow([testIm groundTruthIm],[]);

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
lambda2List = 1:1:10; 
lambda3List = [0.1];
rel_tol = 0.001; % relative target duality gap
angleSet = 15;
numCycles = 5;  

totalNumIterations = length(lambda2List);
relMseVall2 = zeros(1,totalNumIterations);
ssiml2 = zeros(1,totalNumIterations);
primaryObj = zeros(1,totalNumIterations);

%--------------------compute noise level-------------------------------------------
numAngles = angleSet(1);
idx = id(1:numAngles);
CompleteAngleSet  = mtt(idx,:);
idx1 = (atan(CompleteAngleSet(:,2)./CompleteAngleSet(:,1)))*180/pi;

Afun = @(z) ARadon(z,idx1,dim,numAngles);
Atfun = @(z) AtRadon(z,idx1,dim,numAngles);


x = dct2(testIm);
xArray = reshape(x, [dim(1)*dim(2) 1]);
y = Afun(xArray); 
% adding Gaussian noise to the projections
noiseMean = 0;
noiseSD = 0.02*mean(y);
noise = noiseMean + noiseSD*randn(size(y));

%%
tic;
parfor lambda2 = 1:length(lambda2List)
    
        output = zeros(size(testIm));
        


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
                y = y+ noise;
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
    fname = sprintf('%s/result_lambda2_%.2f.mat',outDirectory,la2);
    parsave(fname,output);
    relMseVall2(lambda2) = mean2((output- groundTruthIm).^2)/mean2(groundTruthIm.^2);
    ssiml2(lambda2) = ssim(output,groundTruthIm);
    primaryObj(lambda2) = history(2,end);



    name = sprintf('%s/result_lambda2_%.2f.png',outDirectory,la2);
    temp = output;
    temp = temp - min(temp(:));
    temp = temp/max(temp(:));
    parsaveImg(name,temp);
   

end 

toc;

figure4 = figure;
plot(noise); xlabel('bin number'); ylabel('Intensity of noise'); title('noise added to the projections');
saveas(figure4,'noise.png');

figure;plot(lambda2List,primaryObj,'b*-','lineWidth',3);xlabel('lambda2 value'); ylabel('Objective function value');

figure1 = figure;
plot(lambda2List,relMseVall2,'m*-','lineWidth',3);hold on;
legend('reconstruction with l2 prior(full image)','Location','NorthOutside');
ylabel('relative MSE');xlabel('lambda2 value');
set(gca,'XGrid','on','YGrid','on','XMinorGrid','on'); 
saveas(figure1,'rMSEplot.png') ; % here you save the figure
 
 
figure2 = figure;
plot(lambda2List,ssiml2,'m*-','lineWidth',3);hold on;
ylabel('SSIM');xlabel('lambda2 value');
legend('reconstruction with l2 prior(full image)','Location','NorthOutside');
set(gca,'XGrid','on','YGrid','on','XMinorGrid','on');   
saveas(figure2,'SSIMplot.png');  % here you save the figure 


fname = sprintf('%s/testImage.mat',outDirectory);
save(fname,'testIm');

