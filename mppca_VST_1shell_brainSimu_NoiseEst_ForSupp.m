%%% to compare different methods for estimating spatially varying noise,
%%% which is important in denoising.
%%% I decided to rerun this comparison using full fov images since I
%%% realized that perhaps it would be a good idea to publish a paper dedicated to
%%% the new denoising method per se (and leave the submillimeter diffusion
%%% to a separate paper). 
%%% please also refer to other noiseEst*.m files for more results on noise
%%% estimation
%%%
%%% NOTE: 
%%% 1) found that use of B vs A led to better noise estimation especially for higher noise levels (>4%). 
%%% 2) found that averaging along the third dimension had some effect on
%%% the final noise estimation performance compared to single average in the third dimension. 
%%% This is because the addition of noise to the images introduced spatial variation of image intensities in the third dimension, 
%%% in which case averaging with a patch based approach did help, 
%%% despite that the third dimension of the ground truth images per se
%%% was constructed by duplicating the 2D images. 

clear all;clc;close all
% addpath('./tensor_fit/');
% addpath('./RiceOptVST/');
% addpath('./GL-HOSVD/');
% addpath('./HOSVD/');
% addpath('./data/simulation/');

img=load_nii('Data/b2000_StickTensorBallBall_RELAX.nii');

%% reform and insert b0 images
dwi1 = double(img.img);
dwi1 = permute(flip(dwi1,2),[2 1 3 4]); % flip and transpose
dwi1= dwi1/max(dwi1(:)); % normalize to 1

bvals0_orig = [0,1000*ones(1,30),2000*ones(1,30)];

%insert b0 images every after 9th volumes
ninterv_b0 = 9;
nvol_tot = size(dwi1,4) + floor((size(dwi1,4)-1)/9);


dwi_tmp = dwi1;
dwi1 = zeros(size(dwi_tmp,1),size(dwi_tmp,2),size(dwi_tmp,3),nvol_tot);
bvals0 = zeros(1,nvol_tot);

idx_b0 = 1 : (ninterv_b0+1) : nvol_tot;
idx_hb = 1:nvol_tot;
idx_hb(idx_b0) = [];
dwi1(:,:,:,idx_hb) = dwi_tmp(:,:,:,2:end);
dwi1(:,:,:,idx_b0) = repmat(dwi_tmp(:,:,:,1),[1 1 1 numel(idx_b0)]);
bvals0(1,idx_hb) = bvals0_orig(:,2:end);
bvals0(1,idx_b0) = 0.;

% extract single shell of b1k
dwi1 = dwi1(:,:,:,1:34);
bvals0 = bvals0(1:34);

Mask1 = (dwi1(:,:,:,1) > 0.01*max(max(max(dwi1(:,:,:,1)))));

% remove background to save computation power
ksize=5;%5 to be consistent with the manucsript;

[i1,i2,i3]= ind2sub(size(Mask1),find(Mask1));

[nx0,ny0,nz0] = size(Mask1);
ind1_start = max(min(i1)-ksize,1);
ind1_end   = min(max(i1)+ksize,nx0);
ind2_start = max(min(i2)-ksize,1);
ind2_end   = min(max(i2)+ksize,ny0);
ind3_start = max(min(i3)-ksize,1);
ind3_end   = min(max(i3)+ksize,nz0);
mask = Mask1(ind1_start:ind1_end,ind2_start:ind2_end,ind3_start:ind3_end);
dwi   = dwi1 (ind1_start:ind1_end,ind2_start:ind2_end,ind3_start:ind3_end,:);
%%
% dwi0= dwi;
% kernelsize_max=7;
% nz= 2*kernelsize_max+1;% 15
% if numel(size(dwi0))<4
%     dwi1= repmat(reshape(dwi0,size(dwi0,1),size(dwi0,2),1,size(dwi0,3)),1,1,nz,1);
%     Mask1= repmat(Mask0,1,1,nz);
% end
% numDWI= size(dwi1,4);

% figure, myimagesc(dwi(:,:,round(nz/2),1),mask0)

s= rng;
%tmp=repmat(Mask,[1 1 1 numDWI]);

nz_idx = 41:41+8; % choose nz=45 as center slice
nz_center = 45;
dwi00= squeeze(dwi(:,:,nz_center,:));
mask = mask(:,:,nz_center);

dwi = dwi(:,:,nz_idx,:); % extract 9 slices

nz = size(dwi,3);
% estimate spatially varying nois
% spatial modulation
%% - fast variation (gaussian + sine wave modulation)
%clear Sigma_MPPCA* Sigma_VST*

sm1= customgauss([size(dwi,1),size(dwi,2)], round(0.5*size(dwi,2)), round(0.5*size(dwi,2)), 0, 0.2, 1, [1 1]);
sm2= sin(repmat(linspace(-5*pi,5*pi,size(dwi,2)), size(dwi,1),1));
%figure, myimagesc(0.1*sm2+sm1)

sm= sm1+ 0.1*sm2;
sm= sm./max(sm(:));
figure, myimagesc(sm,mask)

%% VST A vs B for noise estimation (use b1k only)
levels=2:2:10;% percent
ks= 5;
IM_R={};
parfor idx=1:length(levels)
    level= levels(idx);
    % im_r is the simulated noisy data with varying noise level
    rng(s);
    
    noisemap= level*randn(size(dwi))/100 .* repmat(sm, [1 1 size(dwi,3) size(dwi,4)]);
    noisemap1= level*randn(size(dwi))/100 .* repmat(sm, [1 1 size(dwi,3) size(dwi,4)]);
    
    im_r0=sqrt((dwi+noisemap).^2+(noisemap1).^2);
    IM_R{idx}= im_r0; 
    im_r= im_r0(:,:,:,bvals0>500&bvals0<1500);
    
    Sigma0(:,:,idx)= 0.01* level* sm;
    Sigma1(:,:,idx)= std(noisemap(:,:,:),0,3);
    
    sigma_vst= estimate_noise_vst2(im_r,ks,'A') ; % found use of 'A' gives better result than use of the default 'B'.
    Sigma_VST_A(:,:,idx)= sigma_vst(:,:,round(nz/2));
    
    sigma_vst= estimate_noise_vst2(im_r,ks,'B') ; % found use of 'A' gives better result than use of the default 'B'.
    Sigma_VST_B(:,:,idx)= sigma_vst(:,:,round(nz/2));
    
    [~,sigma_mppca]= denoise_mppca(im_r,ks);
    Sigma_MPPCA(:,:,idx)= sigma_mppca(:,:,round(nz/2));
%     
end
%
save -v7.3 sigEst_fastvarying_fullFOV_new Sigma_VST_A Sigma_VST_B ...
    Sigma_MPPCA Sigma0 Sigma1 levels sm mask IM_R dwi00

%%
load sigEst_fastvarying_fullFOV_new Sigma_VST_A Sigma_VST_B Sigma_MPPCA ...
    Sigma0 Sigma1 levels mask
%%
n=0;ind=numel(levels);
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
figure, position_plots(ims,[1 length(ims)],[0 0.01*levels(ind)],[],mask,'','w','jet',1)

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

% %
% figure, plot(levels,Rmse_MPPCA,'x-')
% hold on
% plot(levels,Rmse_VST_A,'o-')
% plot(levels,Rmse_VST_B,'^-')
% hold off
% xlim([0 11])
% legend('mppca','vst A','vst B','location','northwest')
% title('Noise estimate (fast spatially varying noise)')
% ylabel('RMSE')
% xlabel('Noise level (%)')
% 
figure, plot(levels,100*Rmse_Sigma1,'.-')
hold on
plot(levels,100*Rmse_MPPCA,'x-')
plot(levels,100*Rmse_VST_A,'o-')
plot(levels,100*Rmse_VST_B,'^-')
hold off
xlim([0 11])
legend('sampled noise','mppca','vst A','vst B','location','northwest')
title('Noise estimate (fast spatially varying noise)')
ylabel('RMSE (%)')
xlabel('Noise level (%)')
%%
clear opt
opt.Markers={'.','x','o','^'};
opt.XLabel='Noise level (%)';
opt.YLabel='RMSE (%)';
opt.YLim=[0 1.6];
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

%%% to determine which function (VST A vs B) should be used for VST.
% noise estimation was done in noiseEst_fullFOV_new.m

clear all;clc;close all
% addpath('./tensor_fit/');
% addpath('./RiceOptVST/');
% addpath('./GL-HOSVD/');
% addpath('./HOSVD/');
% addpath('./data/simulation/');

% % load 'TestData_Rician_SingleShellb2000.mat'; %%% This dataset can be downloaded from the homepage of Fan Lam
% % load Mask_simu_b2000.mat
% % %[N1,N2,numDWI] = size(dwi);
% % dwi0= dwi;
% % Mask0= Mask;
% % nz= 7;
% % if numel(size(dwi0))<4
% %     dwi1= repmat(reshape(dwi0,size(dwi0,1),size(dwi0,2),1,size(dwi0,3)),1,1,nz,1);
% %     Mask1= repmat(Mask0,1,1,nz);
% % end
% % numDWI= size(dwi1,4);
% % 
% % bs= 50;
% % inds= round(size(dwi1,1)/2)-round(bs/2): round(size(dwi1,1)/2)+round(bs/2);
% % dwi= dwi1(inds,inds,:,:);% a small fov at the center for quick check
% % mask= Mask1(inds,inds,round(nz/2));
% % if 0
% % dwi= dwi1; % full fov
% % 
% % %Mask= Mask1(inds,inds,1);
% % %mask= Mask1(:,:,round(nz/2));
% % 
% % ksize=5;
% % [i1,i2,i3]= ind2sub(size(Mask1),find(Mask1));
% % dwi= dwi1(min(i1):max(i1)+ksize,min(i2)-ksize:max(i2)+ksize,:,:); % full fov but with reduced background.
% % mask= Mask1(min(i1):max(i1)+ksize,min(i2)-ksize:max(i2)+ksize,round(nz/2));
% % figure, myimagesc(dwi(:,:,1,1),mask)
% % end
% % %
% % s= rng;
% % levels=2:2:10;% percent
% tmp=repmat(mask,[1 1 numDWI]);
% %% estimate spatially varying nois
% % spatial modulation
% %% - fast variation (gaussian + sine wave modulation)
% sm1= customgauss([size(dwi,1),size(dwi,2)], round(0.5*size(dwi,2)), round(0.5*size(dwi,2)), 0, 0.2, 1, [1 1]);
% sm2= sin(repmat(linspace(-5*pi,5*pi,size(dwi,2)), size(dwi,1),1));
% 
% sm= sm1+ 0.1*sm2;
% sm= sm./max(sm(:));
% figure, myimagesc(sm,mask)
% 
% dwi00= squeeze(dwi(:,:,1,:));
% clear PSNR* error_FA*
%% VST A vs VST B for VST
 load sigEst_fastvarying_fullFOV_new Sigma_VST_B mask IM_R dwi00 levels
 Sigma_VST= Sigma_VST_B; % found use of VST B vs A led to better noise estimation
 nz= size(IM_R{1},3)
 tmp=repmat(mask,[1 1 size(IM_R{1},4)]);
 
%  load data_1shell2000_noisy
%  load noiseEst_B_ws5.mat % 
 %nz= size(IM_R,3);
 

 
%% IMVSTd_AvsB_A
 ws = 5;       ksize=5;
 myconfig=1;
switch myconfig
    case 1
        VST_ABC='A';
        fn= 'IMVSTd_AvsB_A';
    case 2
        VST_ABC='B';
        fn= 'IMVSTd_AvsB_B';
    otherwise
end

 if isempty(gcp)
 myPool= parpool(length(levels));
 end
 
 %
parfor idx=1:length(levels)
    
    im_r= IM_R{idx};
    %im_r= IM_R(:,:,:,:,idx);
    
    sig= repmat(Sigma_VST(:,:,idx),[1 1 size(im_r,3)]);

    rimavst= perform_riceVST(im_r,sig, ws,VST_ABC) ; % 
    IMVST(:,:,:,:,idx)= rimavst;
    
    [im_denoised2,sig_mppca]= denoise_mppca(rimavst,ksize);
    Sigma_MPPCA{idx}= sig_mppca;
    IMVSTd(:,:,:,idx)= squeeze(im_denoised2(:,:,round(nz/2),:));
     
     im_denoised2= perform_riceVST_EUI(im_denoised2,sig,ws,VST_ABC);
    ims_denoised3= squeeze(im_denoised2(:,:,round(nz/2),:));
    IMVSTd_EUIVST(:,:,:,idx)= ims_denoised3;
    PSNR(idx) = psnr(ims_denoised3(tmp),dwi00(tmp));
    
end

save(fn,'-v7.3','IMVST','Sigma_MPPCA','IMVSTd','IMVSTd_EUIVST','PSNR')

%% IMVSTd_AvsB_B
 ws = 5;       ksize=5;
 myconfig=2;
switch myconfig
    case 1
        VST_ABC='A';
        fn= 'IMVSTd_AvsB_A';
    case 2
        VST_ABC='B';
        fn= 'IMVSTd_AvsB_B';
    otherwise
end

 if isempty(gcp)
 myPool= parpool(length(levels));
 end
 
 %
parfor idx=1:length(levels)
    
    im_r= IM_R{idx};
    %im_r= IM_R(:,:,:,:,idx);
    
    sig= repmat(Sigma_VST(:,:,idx),[1 1 size(im_r,3)]);

    rimavst= perform_riceVST(im_r,sig, ws,VST_ABC) ; % 
    IMVST(:,:,:,:,idx)= rimavst;
    
    [im_denoised2,sig_mppca]= denoise_mppca(rimavst,ksize);
    Sigma_MPPCA{idx}= sig_mppca;
    IMVSTd(:,:,:,idx)= squeeze(im_denoised2(:,:,round(nz/2),:));
     
     im_denoised2= perform_riceVST_EUI(im_denoised2,sig,ws,VST_ABC);
    ims_denoised3= squeeze(im_denoised2(:,:,round(nz/2),:));
    IMVSTd_EUIVST(:,:,:,idx)= ims_denoised3;
    PSNR(idx) = psnr(ims_denoised3(tmp),dwi00(tmp));
    
end

save(fn,'-v7.3','IMVST','Sigma_MPPCA','IMVSTd','IMVSTd_EUIVST','PSNR')

%%
PSNR_A= load('IMVSTd_AvsB_A.mat','PSNR')
PSNR_B= load('IMVSTd_AvsB_B.mat','PSNR')
figure, plot(levels, [PSNR_A.PSNR.' PSNR_B.PSNR.'],'x-')
legend('VST A','VST B')
%%% comparable psnr was observed for VST A vs VST B, across the noise
%%% levels considered here. 

%%
sig_A= load('IMVSTd_AvsB_A.mat','Sigma_MPPCA')
sig_B= load('IMVSTd_AvsB_B.mat','Sigma_MPPCA')
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
%figure, plot(levels,[mean_A.' mean_B.'])
%%
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
ind=2;
sig{1}= sig_A.Sigma_MPPCA{ind}(:,:,soi);
sig{2}= sig_B.Sigma_MPPCA{ind}(:,:,soi);
figure, position_plots(sig,[1 length(sig)],[0.8 1.2],[],mask,...
    {'VST A','VST B'},'w','jet',2)

%%
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


% save denoiseVST_A_vs_B_new PSNR_denoised3_A PSNR_denoised3_B ...
%     IMVSTd_EUIVST IMs_denoised3_B IMVST_A IMVST_B levels ...
%     IMVST_denoised_A IMVST_denoised_B ...
%     Sigma_MPPCA_A Sigma_MPPCA_B
%%% NOTE: compared vst using patch average noise level vs position
%%% specific noise level and found that the latter led to slightly
%%% better (<0.2% increase) psnr only at 10% noise level and to slightly
%%% worse (<0.1% decrease) psnr for lower noise levels, which may be
%%% explained by the fact that using patch based averaging is less prone to
%%% errors due to inaccuracy of noise estimation than is position specific
%%% approach (where there is no averaging). It would be interesting to see
%%% whether the position specific approach would do better if we use the
%%% known noise level (instead of using the estimated one). 
%%% In both VST methods, use of VST A vs B for VST and VST EUI was found to provide higher psnr especially for higher noise levels.
%%%

%%
%save mppca_nonstationaryNoise IMs_denoised1 IMs_denoised2 sm levels Sigma_VST  PSNR_denoised1 PSNR_denoised2 PSNR_noisy
% save mppca_nonstationaryNoise IMs_denoised1 IMs_denoised2 ...
%     sm levels Sigma_VST  Sigma_MPPCA PSNR_denoised1 ...
%     PSNR_denoised2 PSNR_noisy dwi00 IMs_r mask

% figure, plot(levels, [PSNR_noisy.' PSNR_denoised1.'...
%     PSNR_denoised2.' PSNR_denoised3.' PSNR_denoised4.' PSNR_denoised5.'],'x-')
% legend('noisy','mppca+','optimal hard','optimal shrinkage','soft','tsvd')
% title('PSNR (fast spatially varying noise)')
% ylabel('PSNR')
% xlabel('Noise level (%)')

%%% the improvement in noise estimation using MPPCA vs median based noise estimation 
% was found to translate into improvement in denoising with optimal shrinkage. 
% however, the improvement in psnr was little (<0.1%).

% %% 
% for idx=1:length(levels)
%     level= levels(idx);
%     
%     % im_r is the simulated noisy data with varying noise level
%     rng(s);
%     
%     noisemap= level*randn(size(dwi))/100 .* repmat(sm, [1 1 size(dwi,3) size(dwi,4)]);
%     
%     im_r=sqrt((dwi+noisemap).^2+(noisemap).^2);
%     
%     % noise estimation
%     ws = 3;
%     sigma_vst= estimate_noise_vst2(im_r,ws,'A') ; 
%     Sigma_VST(:,:,idx)= sigma_vst(:,:,round(nz/2));
%     sig= repmat(Sigma_VST(:,:,idx),[1 1 size(im_r,3)]);
%     
%     rimavst= perform_riceVST(im_r,sig,ws,'A') ; % found use of 'A' gives better result than use of the default 'B'.
%     [im_denoised,sig_mppca]= denoise_mppca(rimavst,ksize);
%     im_denoised= perform_riceVST_EUI(im_denoised,sig,ws,'A');
%     ims_denoised1= squeeze(im_denoised(:,:,round(nz/2),:));
%     IMs_denoised1(:,:,:,idx)= ims_denoised1;
%     
%     %
%     [im_denoised,sigma_mppca]= MPdenoising(im_r,[],ksize,'full');
%     ims_denoised2= squeeze(im_denoised(:,:,round(nz/2),:));
%     IMs_denoised2(:,:,:,idx)= ims_denoised2;
%     Sigma_MPPCA(:,:,idx)= sigma_mppca(:,:,round(nz/2));
%     
%     %
%     im_r00= squeeze(im_r(:,:,1,:));
%     IMs_r(:,:,:,idx)= im_r00;
%     PSNR_noisy(idx)= psnr(im_r00(tmp),dwi00(tmp));
%     PSNR_denoised1(idx) = psnr(ims_denoised1(tmp),dwi00(tmp));
%     PSNR_denoised2(idx) = psnr(ims_denoised2(tmp&~~ims_denoised2),dwi00(tmp&~~ims_denoised2));
%     
% %     %  FA estimation and FA-RMSE calculation
% %     bacq      = 2000;
% %     display   = 0;
% %     
% %     [FA_noisefree, RGB_noisefree, tensors_noisefree, MD_noisefree] = ...
% %         tensor_est(dwi00,gradientDirections,bVal,bacq,display,Mask);
% %     
% %     [FA_noisy, RGB_noisy, tensors_noisy, MD_noisy] = ...
% %         tensor_est(im_r00,gradientDirections,bVal,bacq,display,Mask);
% %     error_FA_noisy(idx)=RMSE(FA_noisefree(Mask),FA_noisy(Mask));
% %     
% %     [FA_denoised1, RGB_denoised, tensors_denoised, MD_denoised] = ...
% %         tensor_est(ims_denoised1,gradientDirections,bVal,bacq,display,Mask);
% %     error_FA_denoised1(idx)=RMSE(FA_noisefree(Mask),FA_denoised1(Mask));
% %     
% %     [FA_denoised2, RGB_denoised, tensors_denoised, MD_denoised] = ...
% %         tensor_est(ims_denoised2,gradientDirections,bVal,bacq,display,Mask);
% %     error_FA_denoised2(idx)=RMSE(FA_noisefree(Mask),FA_denoised2(Mask));
%     
% end
% %%
% %save mppca_nonstationaryNoise IMs_denoised1 IMs_denoised2 sm levels Sigma_VST  PSNR_denoised1 PSNR_denoised2 PSNR_noisy
% save mppca_nonstationaryNoise_fullfov IMs_denoised1 IMs_denoised2 ...
%     sm levels Sigma_VST  Sigma_MPPCA PSNR_denoised1 ...
%     PSNR_denoised2 PSNR_noisy dwi00 IMs_r mask
% 
% figure, plot(levels, [PSNR_noisy.' PSNR_denoised2.' PSNR_denoised1.'],'x-')
% legend('noisy','mppca','new mppca')
% title('PSNR (fast spatially varying noise)')
% ylabel('PSNR')
% xlabel('Noise level (%)')
% % figure, plot(levels, [error_FA_noisy.' error_FA_denoised2.' error_FA_denoised1.'])
% % legend('noisy','mppca','new mppca')
% % title('FA error (fast spatially varying noise)')
% % ylabel('RMSE')
% % xlabel('Noise level (%)')
