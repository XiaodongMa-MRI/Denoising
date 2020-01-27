%%% Simulation to mimic 10 averages
%%% racian & gaussian


%% ===============racian noise=======================

%% Load nii file
clear all;clc;close all
% img=load_nii('Data_LinearCSM/b2000_StickTensorBallBall_RELAX.nii');
img=load_nii('Data_ConstCSM/b2000_StickTensorBallBall_RELAX.nii');

%% parameters
bRacianNoise = 0; % 1-racian; 0-gaussian
Naverages = 10; % number of averages

%% reform and insert b0 images
dwi1 = double(img.img);
% dwi1 = permute(flip(dwi1,2),[2 1 3 4]); % flip and transpose
dwi1 = permute(flipdim(dwi1,2),[2 1 3 4]); % flip and transpose
dwi1= dwi1/max(dwi1(:)); % normalize to 1

bvals_orig = [0,1000*ones(1,30),2000*ones(1,30)];

%insert b0 images every after 9th volumes
ninterv_b0 = 9;
nvol_tot = size(dwi1,4) + floor((size(dwi1,4)-1)/9);


dwi_tmp = dwi1;
dwi1 = zeros(size(dwi_tmp,1),size(dwi_tmp,2),size(dwi_tmp,3),nvol_tot);
bvals = zeros(1,nvol_tot);

idx_b0 = 1 : (ninterv_b0+1) : nvol_tot;
idx_hb = 1:nvol_tot;
idx_hb(idx_b0) = [];
dwi1(:,:,:,idx_hb) = dwi_tmp(:,:,:,2:end);
dwi1(:,:,:,idx_b0) = repmat(dwi_tmp(:,:,:,1),[1 1 1 numel(idx_b0)]);
bvals(1,idx_hb) = bvals_orig(:,2:end);
bvals(1,idx_b0) = 0.;

% repeat 10 times
dwi1 = repmat(dwi1,[1 1 1 Naverages]);
bvals = repmat(bvals,[1 Naverages]);

Mask1 = (dwi1(:,:,:,1) > 0.01*max(max(max(dwi1(:,:,:,1)))));
%% create noisy data with 2 shell b1000 and b2000

ksize=5;%5 to be consistent with the manucsript;

% remove background to save computation power
[i1,i2,i3]= ind2sub(size(Mask1),find(Mask1));

[nx0,ny0,nz0] = size(Mask1);
ind1_start = max(min(i1)-ksize,1);
ind1_end   = min(max(i1)+ksize,nx0);
ind2_start = max(min(i2)-ksize,1);
ind2_end   = min(max(i2)+ksize,ny0);
ind3_start = max(min(i3)-ksize,1);
ind3_end   = min(max(i3)+ksize,nz0);

% this is a ugly trick
Mask1 = (dwi1(:,:,:,1) > 0.1*max(max(max(dwi1(:,:,:,1)))));
mask = Mask1(ind1_start:ind1_end,ind2_start:ind2_end,ind3_start:ind3_end);
dwi   = dwi1 (ind1_start:ind1_end,ind2_start:ind2_end,ind3_start:ind3_end,:); % full fov but with reduced background.

nz = size(dwi,3);
nz_center = 45;
mask0 = mask(:,:,nz_center);
figure, myimagesc(dwi(:,:,nz_center,1),mask0)

s= rng;
dwi00= squeeze(dwi(:,:,nz_center,:));
% estimate spatially varying nois
% spatial modulation
% - fast variation (gaussian + sine wave modulation)
%clear Sigma_MPPCA* Sigma_VST*

% sm1= customgauss([size(dwi,1),size(dwi,2)], round(0.5*size(dwi,2)), round(0.5*size(dwi,2)), 0, 0.2, 1, [1 1]);
% sm2= sin(repmat(linspace(-5*pi,5*pi,size(dwi,2)), size(dwi,1),1));
% %figure, myimagesc(0.1*sm2+sm1)
% 
% sm= sm1+ 0.1*sm2;
sm1= customgauss([size(dwi,1),size(dwi,2)], round(0.5*size(dwi,2)), round(0.5*size(dwi,2)), 0, 0.2, 1, [1 1]);
sm1_z= customgauss([size(dwi,1),size(dwi,3)], round(0.7*size(dwi,1)), round(0.7*size(dwi,1)), 0, 0.1, 1, [1 1]);
sm1_z = sm1_z(floor(size(sm1_z,1)/2),:);

sm1 = repmat(sm1,[1 1 size(dwi,3)]);
sm1 = sm1.*repmat(reshape(sm1_z,[1 1 size(dwi,3)]),[size(dwi,1),size(dwi,2)]);

sm2= sin(repmat(linspace(-5*pi,5*pi,size(dwi,2)), size(dwi,1),1));
%figure, myimagesc(0.1*sm2+sm1)

sm= sm1+ 0.1*repmat(sm2,[1 1 size(dwi,3)]);

sm= sm./max(sm(:));
figure, myimagesc(sm(:,:,nz_center),mask0)

%save data_2shell_trimmed dwi dwi00 sm mask nz ksize
%
%
levels = [4];% percent
% dwi = single(dwi);
for idx = 1:length(levels)
    level = levels(idx);
    % im_r is the simulated noisy data with varying noise level
    rng(s);
    
    noisemap= level*randn(size(dwi))/100 .* repmat(sm, [1 1 1 size(dwi,4)]);    
    noisemap1= level*randn(size(dwi))/100 .* repmat(sm, [1 1 1 size(dwi,4)]);
    
    if ~bRacianNoise
        noisemap1 = noisemap1.*0;
    end
    
%     Sigma0(:,:,:,idx)= 0.01* level* sm;
%     Sigma1(:,:,:,idx)= 0.5*(std(noisemap(:,:,:,:),0,4)+std(noisemap1(:,:,:,:),0,4) );

    im_r0=sqrt((dwi+noisemap).^2+(noisemap1).^2);
    IM_R(:,:,:,:,idx)= im_r0; % save this for denoising

end


disp('->done..')
IM_R = single(IM_R);
% if bRacianNoise
%     save -v7.3 data_2shell_brain_noisy3D_10AvgRacian dwi dwi00 sm mask nz ksize IM_R levels bvals
% else
%     save -v7.3 data_2shell_brain_noisy3D_10AvgGaussian dwi dwi00 sm mask nz ksize IM_R levels bvals
% end

if bRacianNoise
    save -v7.3 data_2shell_brain_noisy3D_10AvgRacian dwi dwi00 sm mask nz ksize IM_R levels bvals
else
    save -v7.3 data_2shell_brain_noisy3D_10AvgGaussian dwi dwi00 sm mask nz ksize IM_R levels bvals
end