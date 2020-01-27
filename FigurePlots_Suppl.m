%% Display DW images and error maps
clear all;clc;close all
% create dwi and difference images

level_idx = 5;
nzToShow_idx = 45;
load data_2shell_brain_noisy_3DNoiseMap dwi mask dwi00 bvals0

% load and resort proposed images
load IMVSTd_EUIVST_2shell_3DNoiseMapAllSlcs IMVSTd_shrink_EUIVST
IMs_r = squeeze(IMVSTd_shrink_EUIVST(:,:,nzToShow_idx,:,level_idx));
clear IMVSTd_shrink_EUIVST

% VST-based method with Sigma0
load IMVSTd_EUIVST_2shell_VSTwithSigma0Level5
IMs_denoised2 = squeeze(IMVSTd_shrink_EUIVST(:,:,5,:));
clear IMVSTd_shrink_EUIVST;

% VST-based method with mppca
load IMVSTd_EUIVST_2shell_VSTwithSigmaMPPCALevel5
IMs_denoised1 = squeeze(IMVSTd_shrink_EUIVST(:,:,5,:));
clear IMVSTd_shrink_EUIVST;

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
    
    % VST-based method with mppca
    ims_denoised1= IMs_denoised1(:,:,:,idx);
    
    ii= ims_denoised1(:,:,1);
    ii0= dwi00(:,:,1);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposedMPPCA_b0_1 = psnr(ii(tmp),ii0(tmp))
    
    ii= ims_denoised1(:,:,3);
    ii0= dwi00(:,:,3);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposedMPPCA_b1k_1 = psnr(ii(tmp),ii0(tmp))
    
    ii= ims_denoised1(:,:,36);
    ii0= dwi00(:,:,36);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposedMPPCA_b2k_1 = psnr(ii(tmp),ii0(tmp))
    
    % VST-based method with Sigma0
    ims_denoised2= double(IMs_denoised2(:,:,:,idx));
    
    ii= ims_denoised2(:,:,1);
    ii0= dwi00(:,:,1);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposedSigma0_b0_1 = psnr(ii(tmp&~~ii),ii0(tmp&~~ii))
    
    ii= ims_denoised2(:,:,3);
    ii0= dwi00(:,:,3);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposedSigma0_b1k_1 = psnr(ii(tmp&~~ii),ii0(tmp&~~ii))
    
    ii= ims_denoised2(:,:,36);
    ii0= dwi00(:,:,36);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposedSigma0_b2k_1 = psnr(ii(tmp&~~ii),ii0(tmp&~~ii))

    % proposed images
    im_r00= IMs_r(:,:,:,idx);
    
    ii= im_r00(:,:,1);
    ii0= dwi00(:,:,1);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposed_b0_1= psnr(ii(tmp),ii0(tmp))
    
    ii= im_r00(:,:,3);
    ii0= dwi00(:,:,3);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposed_b1k_1= psnr(ii(tmp),ii0(tmp))
    
        ii= im_r00(:,:,36);
    ii0= dwi00(:,:,36);
    tmp=repmat(mask,[1 1 size(ii,3)]);
    PSNR_proposed_b2k_1= psnr(ii(tmp),ii0(tmp))
    
end
%% 10 averages (Level2)

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

MD_groundtruth = MD_groundtruth.img(:,:,nzToShow_idx);
FA_groundtruth = FA_groundtruth.img(:,:,nzToShow_idx);

% 10 averages gaussian; level8
dwiName={ 'im_2shelll_noisy_10AvgGaussian_level2'};
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
dwiName={ 'im_2shelll_noisy_10AvgRacian_level2'};
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


%% display DWI images of 10/20 averages



%% Fig.S4

load sigEst3D_fastvarying_fullFOV_ConstCSM Sigma_VST_A Sigma_VST_B Sigma_MPPCA ...
    Sigma0 Sigma1 levels mask

n=0;ind=2;
n=n+1;
ims{n}= Sigma0(:,:,ind);
n=n+1;
ims{n}= Sigma1(:,:,ind);
n=n+1;
ims{n}= Sigma_MPPCA(:,:,ind);
n=n+1;
ims{n}= Sigma_VST_A(:,:,ind);% highest noise
n=n+1;
ims{n}= Sigma_VST_B(:,:,ind); % mppca
figure, position_plots(ims,[1 length(ims)],[0 0.01*levels(ind)],[],mask,'','w','jet',2)

n=0;ind=6;
n=n+1;
ims{n}= Sigma0(:,:,ind);
n=n+1;
ims{n}= Sigma1(:,:,ind);
n=n+1;
ims{n}= Sigma_MPPCA(:,:,ind);
n=n+1;
ims{n}= Sigma_VST_A(:,:,ind);% highest noise
n=n+1;
ims{n}= Sigma_VST_B(:,:,ind); % mppca
figure, position_plots(ims,[1 length(ims)],[0 0.01*levels(ind)],[],mask,'','w','jet',2)

n=0;ind=10;
n=n+1;
ims{n}= Sigma0(:,:,ind);
n=n+1;
ims{n}= Sigma1(:,:,ind);
n=n+1;
ims{n}= Sigma_MPPCA(:,:,ind);
n=n+1;
ims{n}= Sigma_VST_A(:,:,ind);% highest noise
n=n+1;
ims{n}= Sigma_VST_B(:,:,ind); % mppca
figure, position_plots(ims,[1 length(ims)],[0 0.01*levels(ind)],[],mask,'','w','jet',2)

%%
for idx=1:length(levels)
    isigma0= Sigma0(:,:,idx);
    isigma= Sigma1(:,:,idx);
    Rmse_Sigma1(idx)= RMSE(isigma(mask),isigma0(mask));
    
    isigma= Sigma_VST_A(:,:,idx);
    Rmse_VST_A(idx)= RMSE(isigma(mask),isigma0(mask));
    
    isigma= Sigma_VST_B(:,:,idx);
    Rmse_VST_B(idx)= RMSE(isigma(mask),isigma0(mask));
    
    isigma= Sigma_MPPCA(:,:,idx);
    Rmse_MPPCA(idx)= RMSE(isigma(mask),isigma0(mask));
    
end

clear opt
opt.Markers={'.','x','o','^'};
opt.XLabel='Noise level (%)';
opt.YLabel='RMSE (%)';
opt.YLim=[0 1.2];
X{1}= levels;
X{2}= levels;
X{3}= levels;
X{4}= levels;
Y{1}= 100*Rmse_Sigma1;
Y{2}= 100*Rmse_MPPCA;
Y{3}= 100*Rmse_VST_A;
Y{4}= 100*Rmse_VST_B;
opt.Legend= {'Sampled noise','MPPCA','VST A','VST B'};
opt.LegendLoc= 'NorthWest';

% opt.FileName='rmse_vs_noise_whichmethod.png';
maxBoxDim=5;
figplot

%%
sig_A= load('IMVSTd_AvsB_A_3DNoise_ConstCSM','Sigma_MPPCA')
sig_B= load('IMVSTd_AvsB_B_3DNoise_ConstCSM','Sigma_MPPCA')
soi=8;
mean_A= zeros(1,length(levels));
mean_B= mean_A;
std_A= mean_A;
std_B= mean_A;
for idx=1:length(levels)
    
    iSigA=sig_A.Sigma_MPPCA{idx}(:,:,soi);
    mean_A(idx)= mean(iSigA(mask));
    std_A(idx)= std(iSigA(mask));
    
    iSigB=sig_B.Sigma_MPPCA{idx}(:,:,soi);
    mean_B(idx)= mean(iSigB(mask));
    std_B(idx)= std(iSigB(mask));
    
end

clear opt X Y
opt.Markers={'v','+','o','x','^','*'};
opt.XLabel='Noise level (%)';
opt.YLabel='Average of noise std';
%opt.XLim=[0.8 1.2];
X{1}= levels;
X{2}= levels;
Y{1}= mean_A;
Y{2}= mean_B;
opt.Legend= {'VST A','VST B'};
opt.LegendLoc= 'NorthWest';

opt.FileName=['VSTAvsB4VST_aveNoiseSTD.png'];
maxBoxDim=5;
figplot
%
clear opt X Y
opt.Markers={'v','+','o','x','^','*'};
opt.XLabel='Noise level (%)';
opt.YLabel='Std of noise std';
%opt.XLim=[0.8 1.2];
X{1}= levels;
X{2}= levels;
Y{1}= std_A;
Y{2}= std_B;
opt.Legend= {'VST A','VST B'};
opt.LegendLoc= 'NorthEast';

opt.FileName=['VSTAvsB4VST_stdNoiseSTD.png'];
maxBoxDim=5;
figplot
%%
ind=10;
sig{1}= sig_A.Sigma_MPPCA{ind}(:,:,soi);
sig{2}= sig_B.Sigma_MPPCA{ind}(:,:,soi);
figure, position_plots(sig,[1 length(sig)],[0.8 1.2],[],mask,...
    {'VST A','VST B'},'w','jet',2)

%
nbins=100;
[n1,e1]= histcounts(sig{1}(mask), nbins,'Normalization','probability');
[n2,e2]= histcounts(sig{2}(mask),nbins,'Normalization','probability');

clear opt X Y
%opt.Markers={'.','v','+','o','x','^','*'};
opt.XLabel='Noise std';
opt.YLabel='Probability (%)';
opt.XLim=[0.8 1.2];
X{1}= e1(1:end-1)+ diff(e1);
X{2}= e2(1:end-1)+ diff(e2);
Y{1}= 100*n1;
Y{2}= 100*n2;
opt.Legend= {'VST A','VST B'};
opt.LegendLoc= 'NorthWest';

opt.FileName=['VSTAvsB4VST_hist_',num2str(ind),'perc.png'];
maxBoxDim=5;
figplot


%% PSNR of various singular value manipulation methods

% load Results_LinearCSM/psnr_2shell_AllMethods.mat
load psnr_2shell_ConstCSM_AllMethods

clear opt X Y
opt.Markers={'.','v','+','o','x','^','*'};
opt.XLabel='Noise level (%)';
opt.YLabel='PSNR';
%opt.XLim=[0.8 1.2];
X{1}= levels;
X{2}= levels;
X{3}= levels;
X{4}= levels;
X{5}= levels;
X{6}= levels;
X{7}= levels;
Y{1}= PSNR_noisy;
Y{2}= PSNR_mppca;
Y{3}= PSNR_VSTd_mppca;
Y{4}= PSNR_VSTd_shrink;
Y{5}= PSNR_VSTd_hard;
Y{6}= PSNR_VSTd_soft;
Y{7}= PSNR_VSTd_tsvd;
opt.Legend= {'Noisy','MPPCA','MPPCA+','Shrink','Hard','Solft','TSVD'};
opt.LegendLoc= 'NorthEast';

opt.FileName=['PSNR_2shell_BrainSimu_ForSupp.png'];
maxBoxDim=5;
figplot


clear opt X Y
opt.Markers={'.','v','+','o','x','^','*'};
opt.XLabel='Noise level (%)';
opt.YLabel='PSNR';
opt.XLim=[3.5 4.5];
opt.YLim=[40 43];
X{1}= levels;
X{2}= levels;
X{3}= levels;
X{4}= levels;
X{5}= levels;
X{6}= levels;
X{7}= levels;
Y{1}= PSNR_noisy;
Y{2}= PSNR_mppca;
Y{3}= PSNR_VSTd_mppca;
Y{4}= PSNR_VSTd_shrink;
Y{5}= PSNR_VSTd_hard;
Y{6}= PSNR_VSTd_soft;
Y{7}= PSNR_VSTd_tsvd;

opt.FileName=['PSNR_2shell_BrainSimu_ForSupp_Zoomed.png'];
maxBoxDim=5;
figplot

% display images
% b0

% load Results_LinearCSM/data_2shell_brain_noisy_3DNoiseMap.mat IM_R dwi mask
% load Results_LinearCSM/IMd_mppca_2shell_AllMethods.mat IMd_mppca
% load Results_LinearCSM/IMVSTd_EUIVST_2shell_3DNoiseMap_AllMethods.mat

load data_2shell_noisy_3DNoiseMap_ConstCSM IM_R dwi mask
load IMd_mppca_2shell_ConstCSM_AllMethods IMd_mppca
load IMVSTd_EUIVST_2shell_3DNoiseMap_ConstCSM_AllMethods

nzToShow_idx = 45;
idx_noiseLevel = 4;

dwi00 = squeeze(dwi(:,:,nzToShow_idx,:));%extract b=0 and b1k

ind=1;
sf = 30;
fig_dwi_1;
figure, position_plots(ims2,[2 .5*length(ims2)],[0 1],[],mask(:,:,nzToShow_idx),mystr,'y','gray',1)
% b2000
ind=36;
sf=10;
fig_dwi_1;
figure, position_plots(ims2,[2 .5*length(ims2)],[0 .2],[],mask(:,:,nzToShow_idx),mystr,'y','gray',1)


