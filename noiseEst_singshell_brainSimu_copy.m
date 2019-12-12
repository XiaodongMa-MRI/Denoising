%%% to compare different methods for estimating spatially varying noise,
%%% when considering 2-shell diffusion data. this was work after learning
%%% from what had been done using noiseEst.m.
%%% this was motivated by an observation that for the human 0.7 mm data,
%%% using only b=1k, or only b=2k, or both b=1k and 2k
%%% dwi, or all images including b0 led to different noise estimation
%%% Including b0 images resulted in noise estimation that is biased by high
%%% CSF signals (therefore ignored). The noise level estimated
%%% using b=1k only turned out to be highest,
%%% and the noise level estimated by using only b=2k lowest, showing that
%%% the noise estimatioin is snr dependent.
%%% I compared denoised images resulting from noise estimation using only
%%% b1k and both b1k and b2k and found that
%%% the difference in noise estimation
%%% led to different denoising performances. Visually, the denoised dwi
%%% resulting from using only b1k dwi for noise estimation
%%% appeared to over do the denoising since the average CSF signal for b2k
%%% was null (note Veraart et al MRM 2016 paper shows that CSF signal nulls
%%% at b3k). This observation suggests that noise estimation considering
%%% all dwi volumes (both 1k and 2k) should be used.
%%% I was wondering if this observation could be verified by simulation.
%%%
%%% xiaodong ma 12/2019

clear all;clc;close all
% addpath('./tensor_fit/');
addpath('./RiceOptVST/');
% addpath('./GL-HOSVD/');
% addpath('./HOSVD/');
% addpath('./data/simulation/');

load data_2shell_brain_noisy.mat % created in noisyDataCreation.m
%%
myconfig=1
switch myconfig
    case 1
        ks=5; % kernel size
        fn='sigEst_singshell_fullFOV_B_ws5_noiseLevel42';VST_ABC='B';
    case 2
        ks=7;
        fn='sigEst_singshell_fullFOV_B_ws7_noiseLevel42';VST_ABC='B';
    otherwise
end

%
if isempty(gcp)
    mypool= parpool(length(levels));
end

%% select noise levels and slices for noise estimation
nlevel_idx = [4 2];
% nz_idx = 41:41+8; % choose nz=45 as center slice
IM_R = IM_R(:,:,:,:,nlevel_idx);
levels = levels(nlevel_idx);
Sigma0 = Sigma0(:,:,nlevel_idx);
Sigma1 = Sigma1(:,:,nlevel_idx);

nzToShow_idx = round(size(IM_R,3)/2);
%%
parfor idx=1:numel(levels)
    im_r0= IM_R(:,:,:,:,idx);
    
%     im_r= im_r0;
%     sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; % 
%     Sigma_VST2_all(:,:,idx)= sigma_vst(:,:,nzToShow_idx);
    
    im_r= im_r0(:,:,:,bvals0>500&bvals0<1500);
    sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; % 
    Sigma_VST2_b1k(:,:,idx)= sigma_vst(:,:,nzToShow_idx);
    
%     im_r= im_r0(:,:,:,bvals0>1500);
%     sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; % 
%     Sigma_VST2_b2k(:,:,idx)= sigma_vst(:,:,nzToShow_idx);
%     
%     im_r= im_r0(:,:,:,bvals0>500);
%     sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; %
%     Sigma_VST2_b1k2k(:,:,idx)= sigma_vst(:,:,nzToShow_idx);
    
end
%
save(fn, '-v7.3', 'Sigma_VST2_*')

% fn='sigEst_multishell_fullFOV_new';
% save(fn, '-v7.3', 'Sigma_VST2_all_ws5', 'Sigma_VST2_b1k', ...
%     Sigma_VST2_b2k Sigma_VST2_b1k2k_ws5 ...
%     Sigma_VST2_all_ws7 Sigma_VST2_b1k_ws7 ...
%     Sigma_VST2_b2k_ws7 Sigma_VST2_b1k2k_ws7 ...
%     Sigma0 Sigma1 levels sm IM_R mask dwi00
%% 
%%