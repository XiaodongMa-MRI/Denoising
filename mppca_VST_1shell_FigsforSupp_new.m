%%% mainly to investigate which standard denoising algorithm would best denoise the VST'ed data.
%%% the noisy data and noise estimation for various noise levels were done
%%% in noiseEst_fullFOV_new.m
%%% at this point, it has been determined that VST B should be used for
%%% both noise estimation as well as for VST.

clear all;clc;close all

load data_2shell_brain_noisy.mat

%% load estimated noise maps

load sigEst_singshell_fullFOV_B_ws5.mat
Sigma_VST2_b1k_all = Sigma_VST2_b1k;

load sigEst_singshell_fullFOV_B_ws5_noiseLevel42.mat
Sigma_VST2_b1k_all = cat(3,Sigma_VST2_b1k_all,Sigma_VST2_b1k);

Sigma_VST = Sigma_VST2_b1k_all;

nlevel_idx = [10 8 6 4 2];
nz_idx = 41:41+8; % choose nz=45 as center slice
IM_R = IM_R(:,:,nz_idx,1:34,nlevel_idx); % extract b=1k
levels = levels(nlevel_idx);
Sigma0 = Sigma0(:,:,nlevel_idx);
Sigma1 = Sigma1(:,:,nlevel_idx);

slc_idx=45;
nzToShow_idx = 5; % !!!!!!!!!!!!

tmp=repmat(mask(:,:,slc_idx),[1 1 size(IM_R,4)]); % !!!!
%
%% config
myconfig=1;
VST_ABC='B';
switch myconfig
    case 1
        ws= 5;% kernel size for VST
        %Sigma_VST= Sigma_VST2_b1k_ws5;
        ksize=5;% kernel size for denoising
        fn1= 'IMVST_1shell.mat'; %
        %fn= 'denoiseVST_nonstationaryNoise_fullfov_multishell_ws5_new';
        fn2= 'IMVSTd_1shell.mat';
        fn3='IMVSTd_EUIVST_1shell.mat';
        fn_mppca='IMd_mppca_1shell.mat';
        fn_psnr='psnr_1shell.mat';
%     case 2
%         ws= 7;Sigma_VST= Sigma_VST2_b1k_ws7;ksize=7;
%         fn= 'denoiseVST_nonstationaryNoise_fullfov_multishell_ws7_new';
%         fn1= 'IMVST_ws7_7.mat'; %
%         fn2= 'IMVSTd_ws7_7.mat';
%         fn3='IMVSTd_EUIVST_ws7_7.mat';
%         fn_mppca='IMd_mppca_ws7.mat';
%     case 3
%         ws= 5;Sigma_VST= Sigma_VST2_b1k_ws7;ksize=5;
%         fn= 'denoiseVST_nonstationaryNoise_fullfov_multishell_ws7_5_new';
%         fn1= 'IMVST_ws7_5.mat'; %
%         fn2= 'IMVSTd_ws7_5.mat';
%         fn3='IMVSTd_EUIVST_ws7_5.mat';
%         fn_mppca='IMd_mppca_ws5.mat';
    otherwise
end

if isempty(gcp)
    myPool= parpool(size(IM_R,5));
end
clear IMVST
%% - VST
if ~exist(fn1,'file')
    disp('-> starting to conduct VST of the noisy data...')
    
    IMVST= IM_R;
    IMVST(:)=0;
    parfor idx=1:size(IM_R,5)
        
        im_r= IM_R(:,:,:,:,idx);
        sig= repmat(Sigma_VST(:,:,idx),[1 1 size(im_r,3)]);
        
        rimavst= perform_riceVST(im_r,sig,ws,VST_ABC) ; % 
        IMVST(:,:,:,:,idx)= rimavst;
        
        [im_denoised,sig_mppca]= denoise_mppca(rimavst,ws);
       IMVSTd_mppca(:,:,:,:,idx)= im_denoised;
       Sigma_MPPCA{idx}= sig_mppca;
    end
    save(fn1,'-v7.3','IMVST', 'Sigma_VST', ...
        'Sigma_MPPCA', 'IMVSTd_mppca')
else
    if ~exist('IMVST','var')
        disp('-> loading transformed images after VST...')
        load(fn1,'IMVST','Sigma_MPPCA','IMVSTd_mppca')
    end
end

%% denoising
if ~exist(fn2,'file')
    disp('-> starting to denoise using standard methods...')
    parfor idx=1:size(IM_R,5)
        %im_r= IM_R(:,:,:,:,idx);
        % sig= repmat(Sigma_VST(:,:,idx),[1 1 nz]);
        
        %rimavst= perform_riceVST(im_r,sig,ws,'A') ; % found use of 'A' gives better result than use of the default 'B'.
        %IMVST{idx}= rimavst;
        rimavst= IMVST(:,:,:,:,idx);
        
        % mppca+
        % [im_denoised,sig_mppca]= denoise_mppca(rimavst,ksize);
        %IMVST_denoised1(:,:,:,idx)= squeeze(im_denoised(:,:,round(nz/2),:));
        % IMVSTd_mppca(:,:,:,:,idx)= im_denoised;
        
        %Sigma_MPPCA{idx}= sig_mppca;
        sig_med= Sigma_MPPCA{idx};
        
        % optimal shrinkage
        im_denoised= denoise_svs(rimavst,ksize,1,sig_med,'shrink');
        IMVSTd_shrink(:,:,:,:,idx)= im_denoised;
        
        % optimal hard threshold
        im_denoised= denoise_svs(rimavst,ksize,1,sig_med,'hard');
        IMVSTd_hard(:,:,:,:,idx)= im_denoised;
        
        % optimal soft threshold
        im_denoised= denoise_svs(rimavst,ksize,1,sig_med,'soft');
        IMVSTd_soft(:,:,:,:,idx)= im_denoised;
        
        % tsvd
        im_denoised= denoise_svs(rimavst,ksize,1,sig_med,'tsvd');
        IMVSTd_tsvd(:,:,:,:,idx)= im_denoised;
    end
    save(fn2,'IMVSTd_*')
else
    if ~exist('IMVSTd_shrink','var')
        disp('-> loading denoised images before EUIVST...')
        load(fn2)
    end
end

%% EUIVST
% if ~exist(fn3,'file')
    
    disp('-> starting to conduct EUIVST of denoised data...')
    
    parfor idx=1:size(IM_R,5)
        sig= repmat(Sigma_VST(:,:,idx),[1 1 length(nz_idx)]);
        
        % mppca+
        im_denoised= IMVSTd_mppca(:,:,nz_idx,:,idx);
        im_denoised= perform_riceVST_EUI(im_denoised,sig,ws,VST_ABC);
        IMVSTd_mppca_EUIVST(:,:,:,idx)= squeeze(im_denoised(:,:,nzToShow_idx,:));
        
        % optimal shrinkage
        im_denoised= IMVSTd_shrink(:,:,nz_idx,:,idx);
        im_denoised= perform_riceVST_EUI(im_denoised,sig,ws,VST_ABC);
        IMVSTd_shrink_EUIVST(:,:,:,idx)= squeeze(im_denoised(:,:,nzToShow_idx,:));
        
        % optimal hard threshold
        im_denoised= IMVSTd_hard(:,:,nz_idx,:,idx);
        im_denoised= perform_riceVST_EUI(im_denoised,sig,ws,VST_ABC);
        IMVSTd_hard_EUIVST(:,:,:,idx)= squeeze(im_denoised(:,:,nzToShow_idx,:));
        
        % optimal soft threshold
        im_denoised= IMVSTd_soft(:,:,nz_idx,:,idx);
        im_denoised= perform_riceVST_EUI(im_denoised,sig,ws,VST_ABC);
        IMVSTd_soft_EUIVST(:,:,:,idx)= squeeze(im_denoised(:,:,nzToShow_idx,:));
        
        % tsvd
        im_denoised= IMVSTd_tsvd(:,:,nz_idx,:,idx);
        im_denoised= perform_riceVST_EUI(im_denoised,sig,ws,VST_ABC);
        IMVSTd_tsvd_EUIVST(:,:,:,idx)= squeeze(im_denoised(:,:,nzToShow_idx,:));
    end
    save(fn3,'IMVSTd_*_EUIVST')
% else
%     if ~exist('IMVSTd_shrink_EUIVST','var')
%         disp('-> loading denoised images with VST based denoising...')
%         load(fn3)
%     end
% end

%% MPPCA
% if ~exist(fn_mppca,'file')
    
    disp('-> starting to denoise with MPPCA...')
    
    parfor idx=1:size(IM_R,5)
        
        im_r= IM_R(:,:,:,:,idx);
        [im_denoised,sigma_mppca]= MPdenoising(im_r,[],ksize,'full');
        IMd_mppca(:,:,:,idx)= squeeze(im_denoised(:,:,nzToShow_idx,:));
        Sigma_mppca(:,:,idx)= sigma_mppca(:,:,nzToShow_idx);
        
    end
    save(fn_mppca,'IMd_mppca','Sigma_mppca');
% else
%     disp('-> loading denoised images with mppca...')
%     load(fn_mppca)
% endee

%% calc psnr
IM_R1= squeeze(IM_R(:,:,nzToShow_idx,:,:));
dwi00 = dwi(:,:,slc_idx,1:34);%extract b=0 and b1k
parfor idx=1:length(levels)
    im_r00= IM_R1(:,:,:,idx);
    %IMs_r(:,:,:,idx)= im_r00;
    PSNR_noisy(idx)= psnr(im_r00(tmp),dwi00(tmp));
    imd= IMVSTd_mppca_EUIVST(:,:,:,idx);
    PSNR_VSTd_mppca(idx) = psnr(imd(tmp),dwi00(tmp));
    
    imd=IMVSTd_shrink_EUIVST(:,:,:,idx);
    PSNR_VSTd_shrink(idx) = psnr(imd(tmp),dwi00(tmp));
    
    imd=IMVSTd_hard_EUIVST(:,:,:,idx);
    PSNR_VSTd_hard(idx) = psnr(imd(tmp),dwi00(tmp));
    
    imd=IMVSTd_soft_EUIVST(:,:,:,idx);
    PSNR_VSTd_soft(idx) = psnr(imd(tmp),dwi00(tmp));
    
    imd=IMVSTd_tsvd_EUIVST(:,:,:,idx);
    PSNR_VSTd_tsvd(idx) = psnr(imd(tmp),dwi00(tmp));
    
    imd=IMd_mppca(:,:,:,idx);
    PSNR_mppca(idx) = psnr(imd(tmp&~~imd),dwi00(tmp&~~imd));
    
end

save(fn_psnr,'PSNR_*','levels')
%%
% %save mppca_nonstationaryNoise IMs_denoised1 IMs_denoised2 sm levels Sigma_VST  PSNR_denoised1 PSNR_denoised2 PSNR_noisy
% save(fn,'-v7.3', 'IMs_denoised1', 'IMs_denoised2', 'IMs_denoised22', ...
%     'IMs_denoised33', 'IMs_denoised44', 'IMs_denoised55', ...
%     'sm', 'levels', 'Sigma_VST',  'Sigma_MPPCA', 'Sigma_MPPCA1', ...
%     'IMVST', 'PSNR_denoised1', 'PSNR_denoised2', ...
%     'PSNR_denoised22', 'PSNR_denoised33', 'PSNR_denoised44', ...
%     'PSNR_denoised55','PSNR_noisy', 'dwi00', 'IMs_r', 'mask', 'bvals0',...
%     'Sigma0', 'Sigma1', 'IMVST_de*')
%% psnr


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

opt.FileName=['PSNR_1shell_BrainSimu_ForSupp.png'];
maxBoxDim=5;
figplot


%% display images
%% b0
% clear all;
% load data_2shell_brain_noisy.mat
% load IMVSTd_EUIVST_1shell.mat
% load IMd_mppca_1shell.mat

dwi00 = squeeze(dwi(:,:,slc_idx,1:34));%extract b=0 and b1k

ind=1;
sf = 10;
fig_dwi_1;
figure, position_plots(ims2,[2 .5*length(ims2)],[0 1],[],mask(:,:,slc_idx),mystr,'y','gray',1)
% b1000
ind=3;
sf=3;
fig_dwi_1;
figure, position_plots(ims2,[2 .5*length(ims2)],[0 .3],[],mask(:,:,slc_idx),mystr,'y','gray',1)


