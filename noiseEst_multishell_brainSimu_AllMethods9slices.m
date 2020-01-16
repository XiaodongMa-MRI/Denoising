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
%%% Estimate whole-brain noise map

clear all;clc;close all
% addpath('./tensor_fit/');
% addpath('./RiceOptVST/');
% addpath('./GL-HOSVD/');
% addpath('./HOSVD/');
% addpath('./data/simulation/');

load data_2shell_brain_noisy_3DNoiseMap.mat % created in noisyDataCreation.m
%%
myconfig=1
switch myconfig
    case 1
        ks=5; % kernel size
        fn='sigEst_multishell_fullFOV_B_ws5_WholeBrain_AllMethods';VST_ABC='B';
    case 2
        ks=7;
        fn='sigEst_multishell_fullFOV_B_ws7_WholeBrain_AllMethods';VST_ABC='B';
    otherwise
end

%
if isempty(gcp)
    mypool= parpool(length(levels));
end

%% select noise levels and slices for noise estimation
nlevel_idx = 1:10;
nz_idx = 41:41+8; % choose nz=45 as center slice
IM_R = IM_R(:,:,nz_idx,:,nlevel_idx);
Sigma0 = Sigma0(:,:,nz_idx,nlevel_idx);
Sigma1 = Sigma1(:,:,nz_idx,nlevel_idx);

nz_center = 45;
mask = mask(:,:,nz_center);
% 
nzToShow_idx = round(size(IM_R,3)/2);
%%
% parfor idx=1:numel(levels)
for idx=1:numel(levels)
    im_r0= IM_R(:,:,:,:,idx);
    
    im_r= im_r0;
    sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; % 
    Sigma_VST2_all(:,:,:,idx)= sigma_vst;
    
    im_r= im_r0(:,:,:,bvals0>500&bvals0<1500);
    sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; % 
    Sigma_VST2_b1k(:,:,:,idx)= sigma_vst;
    
    im_r= im_r0(:,:,:,bvals0>1500);
    sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; % 
    Sigma_VST2_b2k(:,:,:,idx)= sigma_vst;
%     
    im_r= im_r0(:,:,:,bvals0>500);
    sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; %
    Sigma_VST2_b1k2k(:,:,:,idx)= sigma_vst;
    
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
clear Rmse_VST2*
parfor idx=1:numel(levels)
    isigma0= Sigma0(:,:,nzToShow_idx,idx);
    isigma= Sigma1(:,:,nzToShow_idx,idx);
    Rmse_Sigma1(idx)= RMSE(isigma(mask),isigma0(mask));
    
    isigma= Sigma_VST2_all(:,:,nzToShow_idx,idx);
    Rmse_VST2_all(idx)= RMSE(isigma(mask),isigma0(mask));
    
    isigma= Sigma_VST2_b1k(:,:,nzToShow_idx,idx);
    Rmse_VST2_b1k(idx)= RMSE(isigma(mask),isigma0(mask));
    
    isigma= Sigma_VST2_b2k(:,:,nzToShow_idx,idx);
    Rmse_VST2_b2k(idx)= RMSE(isigma(mask),isigma0(mask));
    
    isigma= Sigma_VST2_b1k2k(:,:,nzToShow_idx,idx);
    Rmse_VST2_b1k2k(idx)= RMSE(isigma(mask),isigma0(mask));
    
    isigma= 0.5*(Sigma_VST2_b1k(:,:,nzToShow_idx,idx)+ Sigma_VST2_b2k(:,:,nzToShow_idx,idx));
    Rmse_VST2_b1k2k_ave(idx)= RMSE(isigma(mask),isigma0(mask));
    
end
% 
% %%
save Rmse_2shell_brainSimu_3DNoiseMap  levels Rmse_Sigma1 ...
    Rmse_VST2_all Rmse_VST2_b1k2k Rmse_VST2_b1k ...
    Rmse_VST2_b2k Rmse_VST2_b1k2k_ave


figure, plot(levels,[Rmse_Sigma1.' Rmse_VST2_all.' Rmse_VST2_b1k2k.' ... 
    Rmse_VST2_b1k.' ...
    Rmse_VST2_b2k.'  Rmse_VST2_b1k2k_ave.'],'x-')
legend('sampled noise','all','b1k2k','b1k','b2k','mean(b1k+b2k)')
title('Noise estimate (fast spatially varying noise)')
ylabel('RMSE')
xlabel('Noise level (%)')
% 
% figure, plot(levels,[Rmse_VST2_all_ws7.' Rmse_VST2_b1k2k_ws7.' ... 
%     Rmse_VST2_b1k_ws7.' ...
%     Rmse_VST2_b2k_ws7.'  Rmse_VST2_b1k2k_ave_ws7.'],'x-')
% legend('all','b1k2k','b1k','b2k','mean(b1k+b2k)')
% title('Noise estimate (fast spatially varying noise)')
% ylabel('RMSE')
% xlabel('Noise level (%)')
% 
% %%% found that use of only b1k led to best noise estimation across the
% %%% noise levels for both kernel sizes (5 and 7) (which is different than previous observation that averaging noise estimations by only b1k and only b2k was most robust when the same noise was added to both real and imaginary channels before synthesizing the magnitude images)
% %%% and that use of kernel size 7 vs 5 gave rise to better noise estimation.
%% show images
ind=1;
sigs{1}= Sigma0(:,:,nzToShow_idx,ind);
sigs{2}= Sigma1(:,:,nzToShow_idx,ind);
sigs{3}= Sigma_VST2_all(:,:,nzToShow_idx,ind);
sigs{4}= Sigma_VST2_b1k2k(:,:,nzToShow_idx,ind);
sigs{5}= Sigma_VST2_b1k(:,:,nzToShow_idx,ind);
sigs{6}= Sigma_VST2_b2k(:,:,nzToShow_idx,ind);
%sigs{6}= 0.5*(sigs{4}+ sigs{5});
figure, position_plots(sigs,[1 length(sigs)],[0 levels(ind)/100],[],mask)

%% 
clear opt
opt.Markers={'.','v','+','o','x','^'};
opt.XLabel='Noise level (%)';
opt.YLabel='RMSE (%)';
opt.YLim=[0 0.7];
X{1}= levels;
X{2}= levels;
X{3}= levels;
X{4}= levels;
X{5}= levels;
%X{6}= levels;
Y{1}= 100*Rmse_Sigma1;
Y{2}= 100*Rmse_VST2_all;
Y{3}= 100*Rmse_VST2_b1k2k;
Y{4}= 100*Rmse_VST2_b1k;
Y{5}= 100*Rmse_VST2_b2k;
%Y{6}= 100*Rmse_VST2_b1k2k_ave;
opt.Legend= {'Sampled noise','All','b1k + b2k','b1k','b2k'};
opt.LegendLoc= 'NorthEast';

opt.FileName='rmse_vs_noise_2shell.png';
maxBoxDim=5;
figplot

% %
% sigs{1}= Sigma1(:,:,ind);
% sigs{2}= Sigma_VST2_all_ws7(:,:,ind);
% sigs{3}= Sigma_VST2_b1k2k_ws7(:,:,ind);
% sigs{4}= Sigma_VST2_b1k_ws7(:,:,ind);
% sigs{5}= Sigma_VST2_b2k_ws7(:,:,ind);
% sigs{6}= 0.5*(sigs{4}+ sigs{5});
% figure, position_plots(sigs(1:end),[1 6],[0 levels(ind)/100],[],mask)
