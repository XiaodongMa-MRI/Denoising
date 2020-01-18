%%% Calculate SNR of simulated brain images
%%% Two strategies were uses:
%%% ====Strategy #1=====
%%% estimate the average SNR values using simulated Rician data. 
%%% For each noise level, you need to average the noisy magnitude images
%%% for a given b value and consider the estimated noise map (ie, the one 
%%% you obtained using the VST method) for evaluating the SNR map. In this 
%%% case, the histogram of this SNR map is expected to present two peaks. 
%%% This strategy has been used to estimate SNR values for the real data.
%%% ====Strategy #2=====
%%% estimate the average SNR values using ground truth data. For each noise
%%% level, you need to average the ground truth images for a given b value 
%%% and consider the ground truth noise map (ie, the one you created to 
%%% generate spatially varying Gaussian noises) for evaluating the SNR map.
%%% In this case, the histogram of this SNR map is expected to present only
%%% one peak. The results of this strategy can serve as a reference, 
%%% against which the results of strategy 1 can be compared so as to see 
%%% how accurate it would be when estimating average SNR using Rician data 
%%% and estimated noises.

%% load data
% load data_2shell_brain_noisy_3DNoiseMap.mat % created in noisyDataCreation.m
load data_2shell_noisy_3DNoiseMap_ConstCSM.mat % created in noisyDataCreation.m

%% Strategy #1
%% calculate the mean image for each b value
for idx=1:length(imgs_r)
    imgave_b0{idx}= nanmean(imgs_r{idx}(:,:,:,bvals{idx}<500),4);
    imgave_b1k{idx}= nanmean(imgs_r{idx}(:,:,:,bvals{idx}>500& bvals{idx}<1500),4);
    imgave_b2k{idx}= nanmean(imgs_r{idx}(:,:,:,bvals{idx}>1500),4);
end
figure, position_plots(imgave_b0{1}(:,:,33:6:99),[3,4])
figure, position_plots(imgave_b1k{1}(:,:,33:6:99),[3,4])
figure, position_plots(imgave_b2k{1}(:,:,33:6:99),[3,4])
figure, position_plots(sigmaVST{1}.sigma_vst(:,:,33:6:99),[3,4])

x= imgave_b1k{1}(:);
x= imgave_b0{1}(:);
x= imgave_b2k{1}(:);
x= sigmaVST{1}.sigma_vst(:);
figure, histogram(x(x>prctile(x,0))) % even for b2k, the mean value can be viewed as being large enough (relative to the mean sigma) that the mean image (ie, expectation of Rician data) can be used as good approximate for the signal mean (ie, mean of underlying Gaussian data). 

%% calculate snr
for idx=1:length(imgave_b0)
    snr_b0{idx}= imgave_b0{idx}./sigmaVST{idx}.sigma_vst;
    snr_b1k{idx}= imgave_b1k{idx}./sigmaVST{idx}.sigma_vst;
    snr_b2k{idx}= imgave_b2k{idx}./sigmaVST{idx}.sigma_vst;
end
figure, position_plots(snr_b0{1}(:,:,33:6:99),[3,4])
figure, position_plots(snr_b1k{1}(:,:,33:6:99),[3,4])
figure, position_plots(snr_b2k{1}(:,:,33:6:99),[3,4])

% hist
x= snr_b1k{1}(:);
x= snr_b0{1}(:);
x= snr_b2k{1}(:);
figure, histogram(x(x>prctile(x,0))) % check histogram to make sure there are two peaks



%% Strategy #2: 
mdwi_b0 = squeeze(mean(dwi(:,:,:,(bvals0<50)),4));
mdwi_b1k = squeeze(mean(dwi(:,:,:,(bvals0>50)&(bvals0<1500)),4));
mdwi_b2k = squeeze(mean(dwi(:,:,:,(bvals0>1500)),4));
% load FA
Dir0 = '/home/naxos2-raid1/maxiao/Projects/DiffusionAnalysis/diffusion-analysis/BrainSimu_2Shell/T1w/';
dwiName={ 'im_2shelll_ground_truth'};
dir_data = [Dir0,dwiName{1},filesep,'dti/'];
mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_FA.nii.gz ',...
    dir_data,'dti_FA_ch.nii'];
system(mycmd1)
FA = load_nii([dir_data,'dti_FA_ch.nii']);
FA = FA.img;
FA = permute(flip(FA,2),[2 1 3]);

mask_wm = FA>0.25;

dwi = single(dwi);
Sigma0 = single(Sigma0);
SNR_map_b0 = repmat(mdwi_b0,[1 1 1 size(IM_R,5)])./Sigma0;
SNR_map_b1k = repmat(mdwi_b1k,[1 1 1 size(IM_R,5)])./Sigma0;
SNR_map_b2k = repmat(mdwi_b2k,[1 1 1 size(IM_R,5)])./Sigma0;

% display SNR maps
% % b0 [0 25]
% figure;myimagesc(SNR_map_All(:,:,45,1,5));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(:,40,:,1,5))));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(60,:,:,1,5))));colorbar;imcontrast
% %b1k [0 7]
% figure;myimagesc(SNR_map_All(:,:,45,3,5));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(:,40,:,3,5))));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(60,:,:,3,5))));colorbar;imcontrast
% %b2k [0 5]
% figure;myimagesc(SNR_map_All(:,:,45,36,5));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(:,40,:,36,5))));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(60,:,:,36,5))));colorbar;imcontrast

%% average SNR across whole brain
for idx = 1:numel(levels)
    SNR_ij = SNR_map_b0(:,:,:,idx);
    SNR_wb_b0(idx) = mean(mean(SNR_ij(mask)));
    SNR_ij = SNR_map_b1k(:,:,:,idx);
    SNR_wb_b1k(idx) = mean(mean(SNR_ij(mask)));
    SNR_ij = SNR_map_b2k(:,:,:,idx);
    SNR_wb_b2k(idx) = mean(mean(SNR_ij(mask)));
end

clear opt
opt.Markers={'v','+','o'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
opt.YLabel='SNR Whole Brain';
opt.FileName='SNR_wholeBrain';
Y{1} = SNR_wb_b0;
Y{2} = SNR_wb_b1k;
Y{3} = SNR_wb_b2k;
opt.Colors=[0,0,0;0,0,1;1,0,0];
opt.Legend= {'b0','b1k','b2k'};
opt.LegendLoc= 'NorthEast';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=5;
figplot

%% average SNR across white matter
for idx = 1:numel(levels)
    SNR_ij = SNR_map_b0(:,:,:,idx);
    SNR_wm_b0(idx) = mean(mean(SNR_ij(mask_wm)));
    SNR_ij = SNR_map_b1k(:,:,:,idx);
    SNR_wm_b1k(idx) = mean(mean(SNR_ij(mask_wm)));
    SNR_ij = SNR_map_b2k(:,:,:,idx);
    SNR_wm_b2k(idx) = mean(mean(SNR_ij(mask_wm)));
end

clear opt
opt.Markers={'v','+','o'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
opt.YLabel='SNR White Matter';
opt.FileName='SNR_WhiteMatter';
Y{1} = SNR_wm_b0;
Y{2} = SNR_wm_b1k;
Y{3} = SNR_wm_b2k;
opt.Colors=[0,0,0;0,0,1;1,0,0];
opt.Legend= {'b0','b1k','b2k'};
opt.LegendLoc= 'NorthEast';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=5;
figplot






