dataDirectory = './T1TrainingBrainData';
templateDirectory = './T1TestingData';

dirInfo = dir(dataDirectory);

for i = 3:length(dirInfo)
    fnm = sprintf('%s/%s',dataDirectory,dirInfo(i).name);
    
    dim = [181 217 181];
    fid = fopen(fnm);
    data = fread(fid, prod(dim), 'uint8');
%     size(data)
    img = reshape(data, dim);    
    
    template = img(1:180,21:200,10);
    template = template - min(template(:));
    template = template./max(template(:));
    
    name = sprintf('%s.png',dirInfo(i).name);
    imwrite(template,name);

end

% 
% 
% clear all;
% close all;
%     
%     fnm = 't1_icbm_normal_1mm_pn3_rf20.rawb';
%     dim = [181 217 181];
%     fid1 = fopen(fnm);
%     data1 = fread(fid1, prod(dim), 'uint8');
%     img1 = reshape(data1, dim);    
%     
%     
%     fnm = 't1_ai_msles2_1mm_pn3_rf20.rawb';
%     dim = [181 217 181];
%     fid2 = fopen(fnm);
%     data2 = fread(fid2, prod(dim), 'uint8');
%     img2 = reshape(data2, dim); 
%     
%     dataIm = zeros(dim(1), 2*dim(2), dim(3));
%     diffIm = zeros(dim);
%  
%     
% for i = 1:dim(3)
%     temp = [img1(:,:,i) img2(:,:,i)];
%     temp = temp - min(temp(:));
%     temp = temp./max(temp(:));
%     dataIm(:,:,i)  = temp;    
%     
%     temp = abs(img1(:,:,i) - img2(:,:,i));
%     temp = temp - min(temp(:));
%     temp = temp./max(temp(:));
%     diffIm(:,:,i) = temp;
% end
% 
% writevideo ('data.avi', dataIm, 1);
% writevideo ('diffIm.avi', diffIm, 1);