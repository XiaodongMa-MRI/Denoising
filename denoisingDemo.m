%%% A script demonstrating how to use the proposed method to denoise
%%% magnitude diffusion images. 
%%% Two denoising methods were tested:
%%%     (1) proposed framework combining VST and optimal shrinkage;
%%%     (2) MPPCA
%%% Referenece:  Xiaodong Ma, Xiaoping Wu, Kamil Ugurbil, NeuroImage 2020
%%%
%%% Author:      Xiaoping Wu, Xiaodong Ma, 04-2020

clear all; close all;
addpath(genpath('Utils'));
addpath(genpath('RiceOptVST'));

%% Simulate noisy data with 2 shell b1000 and b2000
load data_2shell

% noise level (percent)
level = 4;

GenerateNoisyData;

%% parameter setting
ks = 5; % kernel size for noise estimation
ws = 5; % window size for denosing
indz4test = 41:49; % here a few slices (no smaller than 2*ks-1) are denoised for testing
VST_ABC='B'; % choose the function for noise estimation; VST-B recommended

% parallel computation
if isempty(gcp)
    mypool= parpool(8);
end

%% Preprocess data by removing the backgrounds to accelerate computation
% remove background to save computation power
[isub{1},isub{2},isub{3}]= ind2sub(size(mask0),find(mask0));
size_mask0 = size(mask0);
for ndim = 1:3
    ind_start{ndim} = max(min(isub{ndim})-ks,1);
    ind_end{ndim}   = min(max(isub{ndim})+ks,size_mask0(ndim));
end
% full fov but with reduced background.
mask = mask0(ind_start{1}:ind_end{1},ind_start{2}:ind_end{2},ind_start{3}:ind_end{3});
dwi  = dwi0 (ind_start{1}:ind_end{1},ind_start{2}:ind_end{2},ind_start{3}:ind_end{3},:); 
dwi_noisy = dwi0_noisy(ind_start{1}:ind_end{1},ind_start{2}:ind_end{2},ind_start{3}:ind_end{3},:);

mask = mask(:,:,indz4test);
dwi = dwi(:,:,indz4test,:);
dwi_noisy = dwi_noisy(:,:,indz4test,:);

%% estimate noise
im_r0 = dwi_noisy;
im_r  = im_r0(:,:,:,bvals0>500&bvals0<1500); %b1000
Sigma_VST2_b1k = estimate_noise_vst3(im_r,ks,VST_ABC) ; 

%% denoise with proposed framework
sig = Sigma_VST2_b1k;
im_r = dwi_noisy;

% VST
rimavst= perform_riceVST3(im_r,sig,ws,VST_ABC) ; 

% estimate noise from images after VST
[IMVSTd_mppca,sig_med] = denoise_mppca3(rimavst,ws);

% denoise using optimal shrinkage
stepsize = 2;
method = 'shrink'; % other options: 'tsvd', 'soft', 'hard'

[IMVSTd_shrink,rankmap] = denoise_svs3(rimavst,ks,stepsize,sig_med,method);

% EUI VST
IMVSTd_shrink_EUIVST = perform_riceVST_EUI3(IMVSTd_shrink,sig,ws,VST_ABC);
        
%% denoise with mppca
[IMd_mppca,Sigma_mppca] = MPdenoising(im_r,[],ks,'full');

%% Results evaluation and display
nzToShow = round(size(dwi_noisy,3)/2);

% noise std display
figure, myimagesc(Sigma0(:,:,nzToShow)); caxis([0 0.01*level]); title('Ground-truth noise std');
figure, myimagesc(Sigma_VST2_b1k(:,:,nzToShow)), caxis([0 0.01*level]); title('Estimated noise std with VST');
figure, myimagesc(Sigma_mppca(:,:,nzToShow)), caxis([0 0.01*level]); title('Estimated noise std with MPPCA');

% extract the central slice for evaluation and display
IMs_r = squeeze(dwi_noisy(:,:,nzToShow,:));
dwi00 = squeeze(dwi(:,:,nzToShow,:));
IMd_mppca = squeeze(IMd_mppca(:,:,nzToShow,:));
IMVSTd_shrink_EUIVST = squeeze(IMVSTd_shrink_EUIVST(:,:,nzToShow,:));
mask = mask(:,:,nzToShow);

% PSNR (peak SNR) calculation
for ind = 1:size(dwi00,3)
    PSNR_noisy{ind} = PSNR(IMs_r(:,:,ind),dwi00(:,:,ind));
    PSNR_MPPCA{ind} = PSNR(IMd_mppca(:,:,ind),dwi00(:,:,ind));
    PSNR_proposed{ind} = PSNR(IMVSTd_shrink_EUIVST(:,:,ind),dwi00(:,:,ind));
end

% results display
mystr={'Reference','Noisy','MPPCA','Proposed'};

% b0
ind=1;
ims{1}= dwi00(:,:,ind);
ims{2}= IMs_r(:,:,ind);% noisy
ims{3}= IMd_mppca(:,:,ind); % mppca
ims{4}= IMVSTd_shrink_EUIVST(:,:,ind); % shrink

figure, position_plots(ims,[1 4],[0 1],[],mask,mystr,'y','gray',1)

% b1000
ind=2;
ims{1}= dwi00(:,:,ind);
ims{2}= IMs_r(:,:,ind);% noisy
ims{3}= IMd_mppca(:,:,ind); % mppca
ims{4}= IMVSTd_shrink_EUIVST(:,:,ind); % shrink

figure, position_plots(ims,[1 4],[0 .3],[],mask,mystr,'y','gray',1)

% b2000
ind=35;
ims{1}= dwi00(:,:,ind);
ims{2}= IMs_r(:,:,ind);% noisy
ims{3}= IMd_mppca(:,:,ind); % mppca
ims{4}= IMVSTd_shrink_EUIVST(:,:,ind); % shrink

figure, position_plots(ims,[1 4],[0 .2],[],mask,mystr,'y','gray',1)
