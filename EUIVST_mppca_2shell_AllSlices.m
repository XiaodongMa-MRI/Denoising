%%% Reconstruct denoised results for all slices
%%% (1) EUIVST from pre-reconstructed IMVStd result for Proposed method
%%% (2) mppca
%%
clear all;clc;close all

% parameters for reconstruct denoised images of all slices
VST_ABC='B';
ws= 5;% kernel size for VST
ksize=5;% kernel size for denoising

fn_proposed='IMVSTd_EUIVST_2shell_AllSlcs.mat';
fn_mppca='IMd_mppca_2shell_AllSlcs.mat';

if isempty(gcp)
    myPool= parpool(size(IM_R,5));
end
%% Denoised images using proposed method
levels = 1:10;
load IMVSTd_2shell.mat IMVSTd_shrink
im_all(:,:,:,:,[10:-2:2]) = IMVSTd_shrink;
load IMVSTd_2shell_OddLevels.mat IMVSTd_shrink
im_all(:,:,:,:,[9:-2:1]) = IMVSTd_shrink;

% EUIVST
% load estimated noise maps
load sigEst_singshell_fullFOV_B_ws5.mat
Sigma_VST2_b1k_all = Sigma_VST2_b1k;

load sigEst_singshell_fullFOV_B_ws5_noiseLevel42.mat
Sigma_VST2_b1k_all = cat(3,Sigma_VST2_b1k_all,Sigma_VST2_b1k);

Sigma_VST(:,:,[10:-2:2]) = Sigma_VST2_b1k_all;

load sigEst_singshell_fullFOV_B_ws5_OddLevels.mat
Sigma_VST(:,:,[9:-2:1]) = Sigma_VST2_b1k;


disp('-> starting to conduct EUIVST of denoised data...')
if ~exist(fn_proposed,'file')
    parfor idx=1:size(im_all,5)
        sig= repmat(Sigma_VST(:,:,idx),[1 1 size(im_all,3)]);

        % optimal shrinkage
        im_denoised= im_all(:,:,:,:,idx);
        im_denoised= perform_riceVST_EUI(im_denoised,sig,ws,VST_ABC);
        IMVSTd_shrink_EUIVST(:,:,:,:,idx)= im_denoised;
    end
    save(fn_proposed,'IMVSTd_*_EUIVST','-v7.3')
    clear im_denoised
end

clear im_all
%% MPPCA
load data_2shell_brain_noisy.mat IM_R

disp('-> starting to denoise with MPPCA...')

if ~exist(fn_mppca,'file')
%     parfor idx=1:size(IM_R,5)
    for idx=1:size(IM_R,5)


        im_r= IM_R(:,:,:,:,idx);
        [im_denoised,sigma_mppca]= MPdenoising(im_r,[],ksize,'full');
        IMd_mppca(:,:,:,:,idx)= im_denoised;
        Sigma_mppca(:,:,:,idx)= sigma_mppca;

    end
    save(fn_mppca,'IMd_mppca','Sigma_mppca','-v7.3');
    clear im_denoised
end