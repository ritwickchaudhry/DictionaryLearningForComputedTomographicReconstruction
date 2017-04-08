close all;
clear all;

templateDirectory = './templatesBrain/';
dirInfo = dir(templateDirectory);
templateIm = cell(1,length(dirInfo)-2);
figure;
numTemplates = length(dirInfo)-2;

% Get the templates and compute their mean----------------------------------------------------------

for i = 1:numTemplates
    name = sprintf('%s%s',templateDirectory,dirInfo(i+2).name)
    temp = double(imread(name));
%     temp = temp(13:180,1:168);
    imshow(temp,[]); title(i); pause(0.1);    
    if i==1
        templates = zeros((size(temp,1)*size(temp,2)), numTemplates);
        templates(:,1) = temp(:);
        sumI = zeros(size(templates(:,i)));
        
        tt = temp;
        minimum = min(tt(:));
        tt = tt - minimum;
        maximum = max(tt(:));
    end
    templates(:,i) = temp(:);
    sumI = sumI + templates(:,i);
    
    temp = temp - minimum;
    temp = temp./maximum;
    templateIm{i} = temp;
end

meanTemplate = sumI./numTemplates;
figure;imshow(reshape(meanTemplate,size(temp)),[]);title('Mean image');
templatesFig = [templateIm{1} templateIm{2} templateIm{3};...
                templateIm{4} templateIm{5} templateIm{6};...
                templateIm{7} templateIm{8} templateIm{9};...
                templateIm{10} templateIm{11} templateIm{12}];
            
 imwrite(templatesFig,'templatesBrain.png');


% Compute the Covariance Matrix--------------------------------------------------------------------
templates = templates - meanTemplate(:,ones(1,numTemplates));
L = templates'*templates;
[W,D] = eig(L);
V = templates*W;
V = normc(V);
[m n] = size(V);

% picking top k eigen values and their corresponding vectors-----------------------------------------------------
% This forms the eigen space of the covariance matrix of the templates-----------------                  

numDim = 11;
eigenVals = zeros(1,numDim);
eigenVecs = zeros(m,numDim);
figure;
for j = 1:numDim    
    eigenVals(j) = D(n-j+1,n-j+1);
    eigenVecs(:,j) = V(:,n-j+1);
    imshow(reshape(eigenVecs(:,j),size(temp)),[]);title(j);pause(0.1);
end

save('EigenSpaceBrain.mat','eigenVals','eigenVecs','meanTemplate','minimum','maximum');

%-------------------------------------------------------------------------
%-------------Testing quality of Eigen space constructed---------------------------
%------------------------------------------------------------------------
% Compute the weights ('alpha') for each of the templates (you're taking the
% projection (dot product) of each template onto each eigenvector
% Note: Here- cols of eigenVecs are eigenvectors.

alpha = cell(1,numTemplates);

for i = 1:numTemplates
    alpha{i} = eigenVecs'*(templates(:,i));  
end

% Reconstruct each template back from its projection coefficients on the
% eigen vectors

for i = 1:numTemplates
    coeff = alpha{i};
    recon = zeros(size(meanTemplate));
    for j = 1:numDim
        recon = recon + (coeff(j)*eigenVecs(:,j));
    end
    recon = recon + meanTemplate;
    imshow([reshape(templates(:,i),size(temp)) reshape(recon,size(temp))],[]);title(i);pause(.1)
end