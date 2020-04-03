%%% VST-based denosing with ground-truth noise model, or with noise
%%% estimated from MPPCA

clear all;clc;close all
warning off

% load data_2shell_brain_noisy.mat
load data_2shell_noisy_3DNoiseMap_ConstCSM
load IMd_mppca_2shell_mrtrix3_ConstCSM Sigma_mppca
%% denosing with ground-truth noise model

nlevel_idx = [10];

nz_idx = 41:41+8; % choose nz=45 as center slice
% nz_idx = 45; % choose nz=45 as center slice
IM_R = IM_R(:,:,nz_idx,:,nlevel_idx); % extract both b=1k and b2k
levels = levels(nlevel_idx);
Sigma0 = Sigma0(:,:,nz_idx,nlevel_idx);
% Sigma1 = Sigma1(:,:,:,nlevel_idx);

% nzToShow_idx = round(size(IM_R,3)/2);
% nzToShow_idx = 45;
% tmp=repmat(mask(:,:,nz_idx),[1 1 size(IM_R,4)]);

Sigma_mppca_mrtrix3 = Sigma_mppca(:,:,nz_idx,nlevel_idx);
Sigma_mppca_mrtrix3(isnan(Sigma_mppca_mrtrix3)) = 1e-6;
Sigma_mppca_mrtrix3(isinf(Sigma_mppca_mrtrix3)) = 1e-6;
%%
FlagSigma = 1; % 1 for ground-truth; 0 for MPPCA
%
%% config
myconfig=1;
VST_ABC='B';
switch myconfig
    case 1
        ws= 5;% kernel size for VST
        %Sigma_VST= Sigma_VST2_b1k_ws5;
        ksize=5;% kernel size for denoising
        if FlagSigma
            fn1='IMVSTd_EUIVST_2shell_VSTwithSigma0_ConstCSM_Level10.mat';
        else
            fn1='IMVSTd_EUIVST_2shell_VSTwithSigmaMPPCA_ConstCSM_Level10.mat';
        end
    otherwise
end

% if isempty(gcp)
%     myPool= parpool(size(IM_R,5));
% end
clear IMVST
%% - VST
if 1
    disp('-> starting to conduct VST of the noisy data...')
    
    IMVST= IM_R;
    IMVST(:)=0;
%     parfor idx=1:size(IM_R,5)
    for idx=1:size(IM_R,5)
        
        im_r= IM_R(:,:,:,:,idx);
%         sig= repmat(Sigma_VST(:,:,idx),[1 1 size(im_r,3)]);
        if FlagSigma
            sig= Sigma0(:,:,:,idx);
        else
            sig= Sigma_mppca_mrtrix3(:,:,:,idx);
        end
        
        rimavst= perform_riceVST3(im_r,sig,ws,VST_ABC) ; % 
%         rimavst= perform_riceVST(im_r,sig,ws,VST_ABC) ; % 
        IMVST(:,:,:,:,idx)= rimavst;
        
        [im_denoised,sig_mppca]= denoise_mppca(rimavst,ws);
       IMVSTd_mppca(:,:,:,:,idx)= im_denoised;
       Sigma_MPPCA{idx}= sig_mppca;
    end
%     save(fn1,'-v7.3','IMVST', 'Sigma0')
else
%     if ~exist('IMVST','var')
%         disp('-> loading transformed images after VST...')
%         load(fn1,'IMVST','Sigma_MPPCA','IMVSTd_mppca')
%     end
end

%% denoising
% if ~exist(fn2,'file')
if 1
    disp('-> starting to denoise using standard methods...')
%     parfor idx=1:size(IM_R,5)
    for idx=1:size(IM_R,5)

        rimavst= IMVST(:,:,:,:,idx);
        
        sig_med= Sigma_MPPCA{idx};
        
        
        % optimal shrinkage
        im_denoised= denoise_svs(rimavst,ksize,1,sig_med,'shrink');
        IMVSTd_shrink(:,:,:,:,idx)= im_denoised;
        
    end
%     save(fn2,'IMVSTd_*')
else
%     if ~exist('IMVSTd_shrink','var')
%         disp('-> loading denoised images before EUIVST...')
%         load(fn2)
%     end
end

%% EUIVST for proposed
disp('-> starting to conduct EUIVST of denoised data...')
% if ~exist(fn_proposed,'file')
if 1
    for idx=1:size(IM_R,5)
%     for idx=1:size(IMVSTd_shrink,5)
%         sig=Sigma_VST;

        if FlagSigma
            sig= Sigma0(:,:,:,idx);
        else
            sig= Sigma_mppca_mrtrix3(:,:,:,idx);
        end

        % optimal shrinkage
        im_denoised= IMVSTd_shrink(:,:,:,:,idx);
        im_denoised= perform_riceVST_EUI(im_denoised,sig,ws,VST_ABC);
        IMVSTd_shrink_EUIVST(:,:,:,:,idx)= im_denoised;
    end
    save(fn1,'IMVSTd_*_EUIVST','-v7.3')
    clear im_denoised
end


