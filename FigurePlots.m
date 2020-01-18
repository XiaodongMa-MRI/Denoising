%% Figure drawing

%% Fig.2
clear all; clc; close all
load data_2shell_brain_noisy dwi mask bvals0 sm
nSliceToShow = 45;
mask0 = mask(:,:,nSliceToShow);

figure, myimagesc(dwi(:,:,nSliceToShow,1),mask0,'y',2);
colormap(gray);colorbar;imcontrast % scale to 1
figure, myimagesc(dwi(:,:,nSliceToShow,3))
colormap(gray);colorbar;imcontrast % scale to 0.25
figure, myimagesc(dwi(:,:,nSliceToShow,36))
colormap(gray);colorbar;imcontrast % scale to 0.2

figure; myimagesc(sm);colorbar;
%% Fig.3
clear all; clc; close all
load sigEst_multishell_fullFOV_B_ws5
load data_2shell_brain_noisy Sigma0 Sigma1 mask

nlevel_idx = 10:-2:2;
Sigma0 = Sigma0(:,:,nlevel_idx);
Sigma1 = Sigma1(:,:,nlevel_idx);

nSliceToShow = 45;
mask0 = mask(:,:,nSliceToShow);

ind=5; % level=2
sigs{1}= Sigma0(:,:,ind);
sigs{2}= Sigma1(:,:,ind);
sigs{3}= Sigma_VST2_all(:,:,ind);
sigs{4}= Sigma_VST2_b1k2k(:,:,ind);
sigs{5}= Sigma_VST2_b1k(:,:,ind);
sigs{6}= Sigma_VST2_b2k(:,:,ind);
%sigs{6}= 0.5*(sigs{4}+ sigs{5});
figure, position_plots(sigs,[1 length(sigs)],[0 levels(ind)/100],[],mask0)

ind=3; % level=6
sigs{1}= Sigma0(:,:,ind);
sigs{2}= Sigma1(:,:,ind);
sigs{3}= Sigma_VST2_all(:,:,ind);
sigs{4}= Sigma_VST2_b1k2k(:,:,ind);
sigs{5}= Sigma_VST2_b1k(:,:,ind);
sigs{6}= Sigma_VST2_b2k(:,:,ind);
%sigs{6}= 0.5*(sigs{4}+ sigs{5});
figure, position_plots(sigs,[1 length(sigs)],[0 levels(ind)/100],[],mask0)

ind=1; % level=10
sigs{1}= Sigma0(:,:,ind);
sigs{2}= Sigma1(:,:,ind);
sigs{3}= Sigma_VST2_all(:,:,ind);
sigs{4}= Sigma_VST2_b1k2k(:,:,ind);
sigs{5}= Sigma_VST2_b1k(:,:,ind);
sigs{6}= Sigma_VST2_b2k(:,:,ind);
%sigs{6}= 0.5*(sigs{4}+ sigs{5});
figure, position_plots(sigs,[1 length(sigs)],[0 levels(ind)/100],[],mask0)

% RMSE of estimated noise map
load Rmse_2shell_brainSimu
idx_levels = 10:-2:2;
levels_all = zeros(1,10);
Rmse_Sigma1_allLevels = zeros(1,10);
Rmse_VST2_all_allLevels = zeros(1,10);
Rmse_VST2_b1k2k_allLevels = zeros(1,10);
Rmse_VST2_b1k_allLevels = zeros(1,10);
Rmse_VST2_b2k_allLevels = zeros(1,10);

levels_all(idx_levels) = levels;
Rmse_Sigma1_allLevels(idx_levels) = Rmse_Sigma1;
Rmse_VST2_all_allLevels(idx_levels) = Rmse_VST2_all;
Rmse_VST2_b1k2k_allLevels(idx_levels) = Rmse_VST2_b1k2k;
Rmse_VST2_b1k_allLevels(idx_levels) = Rmse_VST2_b1k;
Rmse_VST2_b2k_allLevels(idx_levels) = Rmse_VST2_b2k;


load Rmse_2shell_brainSimu_OddLevels
idx_levels = 9:-2:1;

levels_all(idx_levels) = levels;
Rmse_Sigma1_allLevels(idx_levels) = Rmse_Sigma1;
Rmse_VST2_all_allLevels(idx_levels) = Rmse_VST2_all;
Rmse_VST2_b1k2k_allLevels(idx_levels) = Rmse_VST2_b1k2k;
Rmse_VST2_b1k_allLevels(idx_levels) = Rmse_VST2_b1k;
Rmse_VST2_b2k_allLevels(idx_levels) = Rmse_VST2_b2k;

%
clear opt
opt.Markers={'.','v','+','o','x','^'};
opt.XLabel='Noise level (%)';
opt.YLabel='RMSE (%)';
opt.YLim=[0 0.7];
X{1}= levels_all;
X{2}= levels_all;
X{3}= levels_all;
X{4}= levels_all;
X{5}= levels_all;
%X{6}= levels;
Y{1}= 100*Rmse_Sigma1_allLevels;
Y{2}= 100*Rmse_VST2_all_allLevels;
Y{3}= 100*Rmse_VST2_b1k2k_allLevels;
Y{4}= 100*Rmse_VST2_b1k2k_allLevels;
Y{5}= 100*Rmse_VST2_b2k_allLevels;
%Y{6}= 100*Rmse_VST2_b1k2k_ave;
opt.Legend= {'Sampled noise','All','b1k + b2k','b1k','b2k'};
opt.LegendLoc= 'NorthWest';

opt.FileName='rmse_vs_noise_2shell.png';
maxBoxDim=5;
figplot
%% Fig.4 PSNR of denoised images of whole volume
clear all; clc; close all

load data_2shell_brain_noisy_3DNoiseMap IM_R dwi levels mask dwi00 bvals0
levels_all = levels;

nzToShow_idx = 1:size(dwi,3);

% load and resort mppca images
load IMd_mppca_2shell_mrtrix3 IMd_mppca
IMs_denoised2 = squeeze(double(IMd_mppca(:,:,nzToShow_idx,:,:)));
clear IMd_mppca

% load and resort mppca images
load IMVSTd_EUIVST_2shell_3DNoiseMapAllSlcs_New IMVSTd_shrink_EUIVST
IMs_denoised1 = squeeze(IMVSTd_shrink_EUIVST(:,:,nzToShow_idx,:,:));
clear IMVSTd_shrink_EUIVST

IMs_r = squeeze(IM_R(:,:,nzToShow_idx,:,:));
dwi00 = dwi(:,:,nzToShow_idx,:);
mask = mask(:,:,nzToShow_idx);

for idx=1:size(IMs_r,5)
    
    % proposed
    ims_denoised1= IMs_denoised1(:,:,:,:,idx);
    
    ii= ims_denoised1;
    ii0= dwi00;
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_denoised1_Overall(idx) = psnr(ii(tmp),ii0(tmp));
    
    ii= ims_denoised1(:,:,:,bvals0<500);
    ii0= dwi00(:,:,:,bvals0<500);
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_denoised1_b0(idx) = psnr(ii(tmp),ii0(tmp));
    
    ii= ims_denoised1(:,:,:,bvals0>500& bvals0<1500);
    ii0= dwi00(:,:,:,bvals0>500& bvals0<1500);
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_denoised1_b1k(idx) = psnr(ii(tmp),ii0(tmp));
    
    ii= ims_denoised1(:,:,:,bvals0>1500);
    ii0= dwi00(:,:,:,bvals0>1500);
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_denoised1_b2k(idx) = psnr(ii(tmp),ii0(tmp));
    
    % mppca
    ims_denoised2= IMs_denoised2(:,:,:,:,idx);
    
    ii= ims_denoised2;
    ii0= dwi00;
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_denoised2_Overall(idx) = psnr(ii(tmp&~~ii),ii0(tmp&~~ii));
    
    ii= ims_denoised2(:,:,:,bvals0<500);
    ii0= dwi00(:,:,:,bvals0<500);
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_denoised2_b0(idx) = psnr(ii(tmp&~~ii),ii0(tmp&~~ii));
    
    ii= ims_denoised2(:,:,:,bvals0>500& bvals0<1500);
    ii0= dwi00(:,:,:,bvals0>500& bvals0<1500);
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_denoised2_b1k(idx) = psnr(ii(tmp&~~ii),ii0(tmp&~~ii));
    
    ii= ims_denoised2(:,:,:,bvals0>1500);
    ii0= dwi00(:,:,:,bvals0>1500);
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_denoised2_b2k(idx) = psnr(ii(tmp&~~ii),ii0(tmp&~~ii));

    % noisy
    im_r00= IMs_r(:,:,:,:,idx);
    
    ii= im_r00;
    ii0= dwi00;
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_noisy_Overall(idx)= psnr(ii(tmp),ii0(tmp));
    
    ii= im_r00(:,:,:,bvals0<500);
    ii0= dwi00(:,:,:,bvals0<500);
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_noisy_b0(idx)= psnr(ii(tmp),ii0(tmp));
    
    ii= im_r00(:,:,:,bvals0>500 &bvals0<1500);
    ii0= dwi00(:,:,:,bvals0>500 &bvals0<1500);
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_noisy_b1k(idx)= psnr(ii(tmp),ii0(tmp));
    
    ii= im_r00(:,:,:,bvals0>1500);
    ii0= dwi00(:,:,:,bvals0>1500);
    tmp=repmat(mask,[1 1 1 size(ii,4)]);
    PSNR_noisy_b2k(idx)= psnr(ii(tmp),ii0(tmp));
    
end



clear opt
opt.Markers={'v','+','o'};
opt.XLabel='Noise level (%)';
opt.Colors = [0,0,0;0,0,1;1,0,0];
opt.YLabel='PSNR';
opt.XLim=[0 11];
opt.YLim=[21 55];
X{1}= levels_all;
X{2}= levels_all;
X{3}= levels_all;
Y{1}= PSNR_noisy_Overall;
Y{2}= PSNR_denoised2_Overall;
Y{3}= PSNR_denoised1_Overall;
opt.Legend= {'Noisy','MPPCA','Proposed'};
opt.LegendLoc= 'NorthEast';
opt.FileName='PSNR_vs_noise_NoisyMPPCAProposed_Overall.png';
maxBoxDim=5;
figplot

clear opt
opt.Markers={'v','+','o'};
opt.XLabel='Noise level (%)';
opt.Colors = [0,0,0;0,0,1;1,0,0];
opt.YLabel='PSNR';
opt.XLim=[0 11];
opt.YLim=[21 55];
X{1}= levels_all;
X{2}= levels_all;
X{3}= levels_all;
Y{1}= PSNR_noisy_b0;
Y{2}= PSNR_denoised2_b0;
Y{3}= PSNR_denoised1_b0;
opt.Legend= {'Noisy','MPPCA','Proposed'};
opt.LegendLoc= 'NorthEast';
opt.FileName='PSNR_vs_noise_NoisyMPPCAProposed_b0.png';
maxBoxDim=5;
figplot


clear opt
opt.Markers={'v','+','o'};
opt.XLabel='Noise level (%)';
% opt.Colors={'k','b','r'};
opt.YLabel='PSNR';
opt.Colors = [0,0,0;0,0,1;1,0,0];
opt.YLabel='PSNR';
opt.XLim=[0 11];
opt.YLim=[21 55];
X{1}= levels_all;
X{2}= levels_all;
X{3}= levels_all;
Y{1}= PSNR_noisy_b1k;
Y{2}= PSNR_denoised2_b1k;
Y{3}= PSNR_denoised1_b1k;
opt.Legend= {'Noisy','MPPCA','Proposed'};
opt.LegendLoc= 'NorthEast';
opt.FileName='PSNR_vs_noise_NoisyMPPCAProposed_b1k.png';
maxBoxDim=5;
figplot


clear opt
opt.Markers={'v','+','o'};
opt.XLabel='Noise level (%)';
% opt.Colors={'k','b','r'};
opt.YLabel='PSNR';
opt.Colors = [0,0,0;0,0,1;1,0,0];
opt.YLabel='PSNR';
opt.XLim=[0 11];
opt.YLim=[21 55];
X{1}= levels_all;
X{2}= levels_all;
X{3}= levels_all;
Y{1}= PSNR_noisy_b2k;
Y{2}= PSNR_denoised2_b2k;
Y{3}= PSNR_denoised1_b2k;
opt.Legend= {'Noisy','MPPCA','Proposed'};
opt.LegendLoc= 'NorthEast';
opt.FileName='PSNR_vs_noise_NoisyMPPCAProposed_b2k.png';
maxBoxDim=5;
figplot
%% Fig.5
clear all;clc;close all
% create dwi and difference images

level_idx = 5;
nzToShow_idx = 45;
load data_2shell_brain_noisy_3DNoiseMap IM_R dwi mask dwi00 bvals0

% load and resort mppca images
load IMd_mppca_2shell_mrtrix3 IMd_mppca
IMs_denoised2 = squeeze(IMd_mppca(:,:,nzToShow_idx,:,level_idx));
clear IMd_mppca;

% load and resort mppca images
load IMVSTd_EUIVST_2shell_3DNoiseMapAllSlcs_Level5 IMVSTd_shrink_EUIVST
IMs_denoised1 = squeeze(IMVSTd_shrink_EUIVST(:,:,nzToShow_idx,:));
clear IMVSTd_shrink_EUIVST

IMs_r = squeeze(IM_R(:,:,nzToShow_idx,:,level_idx));
mask=mask(:,:,nzToShow_idx);

% b0
ind=1;
sf=20;
fig_dwi_BrainSimuResults;
figure, position_plots(ims2,[1 4],[0 1],[],mask,'','y','gray',2)


% b1000
ind=3;
sf=10;
fig_dwi_BrainSimuResults;
figure, position_plots(ims2,[1 4],[0 0.3],[],mask,'','y','gray',2)

% b2000
ind=36;
sf=5;
fig_dwi_BrainSimuResults;
figure, position_plots(ims2,[1 4],[0 0.2],[],mask,'','y','gray',2)

% calculate psnr for each image

for idx=size(IMs_r,4)
    
    % proposed
    ims_denoised1= IMs_denoised1(:,:,:,idx);
    
    ii= ims_denoised1(:,:,1);
    ii0= dwi00(:,:,1);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposed_b0_1 = psnr(ii(tmp),ii0(tmp))
    
    ii= ims_denoised1(:,:,3);
    ii0= dwi00(:,:,3);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposed_b1k_1 = psnr(ii(tmp),ii0(tmp))
    
    ii= ims_denoised1(:,:,36);
    ii0= dwi00(:,:,36);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposed_b2k_1 = psnr(ii(tmp),ii0(tmp))
    
    % mppca
    ims_denoised2= double(IMs_denoised2(:,:,:,idx));
    
    ii= ims_denoised2(:,:,1);
    ii0= dwi00(:,:,1);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_mppca_b0_1 = psnr(ii(tmp&~~ii),ii0(tmp&~~ii))
    
    ii= ims_denoised2(:,:,3);
    ii0= dwi00(:,:,3);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_mppca_b1k_1 = psnr(ii(tmp&~~ii),ii0(tmp&~~ii))
    
    ii= ims_denoised2(:,:,36);
    ii0= dwi00(:,:,36);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_mppca_b2k_1 = psnr(ii(tmp&~~ii),ii0(tmp&~~ii))

    % noisy
    im_r00= IMs_r(:,:,:,idx);
    
    ii= im_r00(:,:,1);
    ii0= dwi00(:,:,1);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_noisy_b0_1= psnr(ii(tmp),ii0(tmp))
    
    ii= im_r00(:,:,3);
    ii0= dwi00(:,:,3);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_noisy_b1k_1= psnr(ii(tmp),ii0(tmp))
    
        ii= im_r00(:,:,36);
    ii0= dwi00(:,:,36);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_noisy_b2k_1= psnr(ii(tmp),ii0(tmp))
    
end
%% Fig.6 RMSE of MD and FA

clear all;clc;close all
% create dwi and difference images

nzToShow_idx = 45;
load data_2shell_brain_noisy_3DNoiseMap levels mask

% load FA and MD
Dir0 = '/home/naxos2-raid1/maxiao/Projects/DiffusionAnalysis/diffusion-analysis/BrainSimu_2Shell/T1w/';

% grould-truth
dwiName={ 'im_2shelll_ground_truth'};
dir_data = [Dir0,dwiName{1},filesep,'dti/'];

mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_MD.nii.gz ',...
    dir_data,'dti_MD_ch.nii'];
system(mycmd1)
MD_groundtruth = load_nii([dir_data,'dti_MD_ch.nii']);
mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_FA.nii.gz ',...
    dir_data,'dti_FA_ch.nii'];
system(mycmd1)
FA_groundtruth = load_nii([dir_data,'dti_FA_ch.nii']);

MD_groundtruth_AllSlc = MD_groundtruth.img;
FA_groundtruth_AllSlc = FA_groundtruth.img;

MD_groundtruth = MD_groundtruth.img(:,:,nzToShow_idx);
FA_groundtruth = FA_groundtruth.img(:,:,nzToShow_idx);

% noisy
dwiName={'im_2shelll_noisy_level1',...
        'im_2shelll_noisy_level2',...
        'im_2shelll_noisy_level3',...
        'im_2shelll_noisy_level4',...
        'im_2shelll_noisy_level5',...
        'im_2shelll_noisy_level6',...
        'im_2shelll_noisy_level7',...
        'im_2shelll_noisy_level8',...
        'im_2shelll_noisy_level9',...
        'im_2shelll_noisy_level10'};
for idx_level = 1:numel(levels)
    dir_data = [Dir0,dwiName{idx_level},filesep,'dti/'];
    mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_MD.nii.gz ',...
        dir_data,'dti_MD_ch.nii'];
    system(mycmd1)
    MD_tmp = load_nii([dir_data,'dti_MD_ch.nii']);
    mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_FA.nii.gz ',...
        dir_data,'dti_FA_ch.nii'];
    system(mycmd1)
    FA_tmp = load_nii([dir_data,'dti_FA_ch.nii']);
    
    MD_noisy_AllSlc(:,:,:,idx_level) = MD_tmp.img;
    FA_noisy_AllSlc(:,:,:,idx_level) = FA_tmp.img;
    
    MD_noisy(:,:,idx_level) = MD_tmp.img(:,:,nzToShow_idx);
    FA_noisy(:,:,idx_level) = FA_tmp.img(:,:,nzToShow_idx);
end

% proposed
dwiName={'im_2shelll_proposed_level1',...
    'im_2shelll_proposed_level2',...
    'im_2shelll_proposed_level3',...
    'im_2shelll_proposed_level4',...
    'im_2shelll_proposed_level5',...
    'im_2shelll_proposed_level6',...
    'im_2shelll_proposed_level7',...
    'im_2shelll_proposed_level8',...
    'im_2shelll_proposed_level9',...
    'im_2shelll_proposed_level10'};

for idx_level = 1:numel(levels)
    dir_data = [Dir0,dwiName{idx_level},filesep,'dti/'];
    mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_MD.nii.gz ',...
        dir_data,'dti_MD_ch.nii'];
    system(mycmd1)
    MD_tmp = load_nii([dir_data,'dti_MD_ch.nii']);
    mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_FA.nii.gz ',...
        dir_data,'dti_FA_ch.nii'];
    system(mycmd1)
    FA_tmp = load_nii([dir_data,'dti_FA_ch.nii']);
    
    MD_proposed_AllSlc(:,:,:,idx_level) = MD_tmp.img;
    FA_proposed_AllSlc(:,:,:,idx_level) = FA_tmp.img;
    
    MD_proposed(:,:,idx_level) = MD_tmp.img(:,:,nzToShow_idx);
    FA_proposed(:,:,idx_level) = FA_tmp.img(:,:,nzToShow_idx);
end
    
% mppca
dwiName={'im_2shelll_mppca_mrtrix3_level1',...
    'im_2shelll_mppca_mrtrix3_level2',...
    'im_2shelll_mppca_mrtrix3_level3',...
    'im_2shelll_mppca_mrtrix3_level4',...
    'im_2shelll_mppca_mrtrix3_level5',...
    'im_2shelll_mppca_mrtrix3_level6',...
    'im_2shelll_mppca_mrtrix3_level7',...
    'im_2shelll_mppca_mrtrix3_level8',...
    'im_2shelll_mppca_mrtrix3_level9',...
    'im_2shelll_mppca_mrtrix3_level10'};

for idx_level = 1:numel(levels)
    dir_data = [Dir0,dwiName{idx_level},filesep,'dti/'];
    mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_MD.nii.gz ',...
        dir_data,'dti_MD_ch.nii'];
    system(mycmd1)
    MD_tmp = load_nii([dir_data,'dti_MD_ch.nii']);
    mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_FA.nii.gz ',...
        dir_data,'dti_FA_ch.nii'];
    system(mycmd1)
    FA_tmp = load_nii([dir_data,'dti_FA_ch.nii']);
    
    MD_mppca_AllSlc(:,:,:,idx_level) = MD_tmp.img;
    FA_mppca_AllSlc(:,:,:,idx_level) = FA_tmp.img;
    
    MD_mppca(:,:,idx_level) = MD_tmp.img(:,:,nzToShow_idx);
    FA_mppca(:,:,idx_level) = FA_tmp.img(:,:,nzToShow_idx);
end
    
% RMSE calculation
Mask = mask;
for idx= 1:length(levels)
%     fa_gt = rot90(FA_groundtruth);
%     md_gt = rot90(MD_groundtruth);
    fa_gt = permute(flip(FA_groundtruth_AllSlc,2),[2 1 3]);
    md_gt = permute(flip(MD_groundtruth_AllSlc,2),[2 1 3]);
    %noisy
%     fa = rot90(FA_noisy(:,:,idx));
%     md = rot90(MD_noisy(:,:,idx));
    fa = permute(flip(FA_noisy_AllSlc(:,:,:,idx),2),[2 1 3]);
    md = permute(flip(MD_noisy_AllSlc(:,:,:,idx),2),[2 1 3]);
    err_FA_noisy(idx)=RMSE(fa_gt(Mask),fa(Mask));
    err_MD_noisy(idx)=RMSE(md_gt(Mask),md(Mask));
    %mppca
%     fa = rot90(FA_mppca(:,:,idx));
%     md = rot90(MD_mppca(:,:,idx));
    fa = permute(flip(FA_mppca_AllSlc(:,:,:,idx),2),[2 1 3]);
    md = permute(flip(MD_mppca_AllSlc(:,:,:,idx),2),[2 1 3]);
    err_FA_mppca(idx)=RMSE(fa_gt(Mask),fa(Mask));
    err_MD_mppca(idx)=RMSE(md_gt(Mask),md(Mask));
    %proposed
%     fa = rot90(FA_proposed(:,:,idx));
%     md = rot90(MD_proposed(:,:,idx));
    fa = permute(flip(FA_proposed_AllSlc(:,:,:,idx),2),[2 1 3]);
    md = permute(flip(MD_proposed_AllSlc(:,:,:,idx),2),[2 1 3]);
    err_FA_proposed(idx)=RMSE(fa_gt(Mask),fa(Mask));
    err_MD_proposed(idx)=RMSE(md_gt(Mask),md(Mask));
end

save rmse_MD_FA_2shell err_FA_noisy err_MD_noisy err_FA_mppca err_MD_mppca err_FA_proposed err_MD_proposed

% show RMSE of FA and MD
isFA=0;
clear opt
opt.Markers={'v','+','o'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
if isFA
    opt.YLabel='RMSE FA';
    opt.FileName='rmse_FA_2shell';
    Y{1} = err_FA_noisy;
    Y{2} = err_FA_mppca;
    Y{3} = err_FA_proposed;
else
    opt.YLabel='RMSE MD';
    opt.FileName='rmse_MD_2shell';
    Y{1} = err_MD_noisy;
    Y{2} = err_MD_mppca;
    Y{3} = err_MD_proposed;
end
opt.Colors=[0,0,0;0,0,1;1,0,0];
opt.Legend= {'Noisy','MPPCA','Proposed'};
opt.LegendLoc= 'NorthWest';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=5;
figplot


isFA=1;
clear opt
opt.Markers={'v','+','o'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
if isFA
    opt.YLabel='RMSE FA';
    opt.FileName='rmse_FA_2shell';
    Y{1} = err_FA_noisy;
    Y{2} = err_FA_mppca;
    Y{3} = err_FA_proposed;
else
    opt.YLabel='RMSE MD';
    opt.FileName='rmse_MD_2shell';
    Y{1} = err_MD_noisy;
    Y{2} = err_MD_mppca;
    Y{3} = err_MD_proposed;
end
opt.Colors=[0,0,0;0,0,1;1,0,0];
opt.Legend= {'Noisy','MPPCA','Proposed'};
opt.LegendLoc= 'NorthWest';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=5;
figplot


% display MD and FA
Mask = mask(:,:,nzToShow_idx);
idxn= 5;
ims{1}=rot90(FA_groundtruth).*Mask;
ims{2}=rot90(FA_noisy(:,:,idxn)).*Mask;
ims{3}=rot90(FA_mppca(:,:,idxn)).*Mask;
ims{4}=rot90(FA_proposed(:,:,idxn)).*Mask;
mystr=[];
figure, position_plots(ims, [1 4],[0 1],[],[],mystr,'w','jet',1)

% display MD and FA
idxn= 5;
ims{1}=rot90(MD_groundtruth).*Mask;
ims{2}=rot90(MD_noisy(:,:,idxn)).*Mask;
ims{3}=rot90(MD_mppca(:,:,idxn)).*Mask;
ims{4}=rot90(MD_proposed(:,:,idxn)).*Mask;
mystr=[];
figure, position_plots(ims, [1 4],[0 2*10^(-3)],[],[],mystr,'w','jet',1)

%% 10 averages

clear all;clc;close all
% create dwi and difference images

nzToShow_idx = 45;
load data_2shell_brain_noisy levels mask

% load FA and MD
Dir0 = '/home/naxos2-raid1/maxiao/Projects/DiffusionAnalysis/diffusion-analysis/BrainSimu_2Shell/T1w/';

% grould-truth
dwiName={ 'im_2shelll_ground_truth'};
dir_data = [Dir0,dwiName{1},filesep,'dti/'];

mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_MD.nii.gz ',...
    dir_data,'dti_MD_ch.nii'];
system(mycmd1)
MD_groundtruth = load_nii([dir_data,'dti_MD_ch.nii']);
mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_FA.nii.gz ',...
    dir_data,'dti_FA_ch.nii'];
system(mycmd1)
FA_groundtruth = load_nii([dir_data,'dti_FA_ch.nii']);

MD_groundtruth = MD_groundtruth.img(:,:,nzToShow_idx);
FA_groundtruth = FA_groundtruth.img(:,:,nzToShow_idx);

% 10 averages gaussian; level8
dwiName={ 'im_2shelll_noisy_10AvgGaussian_level8'};
dir_data = [Dir0,dwiName{1},filesep,'dti/'];

mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_MD.nii.gz ',...
    dir_data,'dti_MD_ch.nii'];
system(mycmd1)
MD_10AvgGaussianLevel8 = load_nii([dir_data,'dti_MD_ch.nii']);
mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_FA.nii.gz ',...
    dir_data,'dti_FA_ch.nii'];
system(mycmd1)
FA_10AvgGaussianLevel8 = load_nii([dir_data,'dti_FA_ch.nii']);

MD_10AvgGaussianLevel8 = MD_10AvgGaussianLevel8.img(:,:,nzToShow_idx);
FA_10AvgGaussianLevel8 = FA_10AvgGaussianLevel8.img(:,:,nzToShow_idx);
    
% 10 averages racian; level8
dwiName={ 'im_2shelll_noisy_10AvgRacian_level8'};
dir_data = [Dir0,dwiName{1},filesep,'dti/'];

mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_MD.nii.gz ',...
    dir_data,'dti_MD_ch.nii'];
system(mycmd1)
MD_10AvgRacianLevel8 = load_nii([dir_data,'dti_MD_ch.nii']);
mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_FA.nii.gz ',...
    dir_data,'dti_FA_ch.nii'];
system(mycmd1)
FA_10AvgRacianLevel8 = load_nii([dir_data,'dti_FA_ch.nii']);

MD_10AvgRacianLevel8 = MD_10AvgRacianLevel8.img(:,:,nzToShow_idx);
FA_10AvgRacianLevel8 = FA_10AvgRacianLevel8.img(:,:,nzToShow_idx);

% RMSE calculation
Mask = mask(:,:,nzToShow_idx);
fa_gt = rot90(FA_groundtruth);
md_gt = rot90(MD_groundtruth);
%10AvgGaussian
fa = rot90(FA_10AvgGaussianLevel8);
md = rot90(MD_10AvgGaussianLevel8);
err_FA_10AvgGaussian=RMSE(fa_gt(Mask),fa(Mask))
err_MD_10AvgGaussian=RMSE(md_gt(Mask),md(Mask))
%10AvgRacian
fa = rot90(FA_10AvgRacianLevel8);
md = rot90(MD_10AvgRacianLevel8);
err_FA_10AvgRacianLevel8=RMSE(fa_gt(Mask),fa(Mask))
err_MD_10AvgRacianLevel8=RMSE(md_gt(Mask),md(Mask))

% save rmse_MD_FA_2shell err_FA_noisy err_MD_noisy err_FA_mppca err_MD_mppca err_FA_proposed err_MD_proposed

% display MD and FA
ims{1}=rot90(FA_groundtruth).*Mask;
ims{2}=rot90(FA_10AvgGaussianLevel8).*Mask;
ims{3}=rot90(FA_10AvgRacianLevel8).*Mask;
mystr=[];
figure, position_plots(ims, [1 3],[0 1],[],[],mystr,'w','jet',1)

ims{1}=rot90(MD_groundtruth).*Mask;
ims{2}=rot90(MD_10AvgGaussianLevel8).*Mask;
ims{3}=rot90(MD_10AvgRacianLevel8).*Mask;
mystr=[];
figure, position_plots(ims, [1 3],[0 2*10^(-3)],[],[],mystr,'w','jet',1)