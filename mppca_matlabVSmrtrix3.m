%%% compare MPPCA in matlab(i.e., MPdenoising.m) VS mppca in mrtrix3 (i.e. dwidenoise)
%% mppca using mrtrix3
clear all;close all; clc

dataDir0='/home/naxos2-raid1/maxiao/Projects/Data/Denoising/';
% subjID={'BrainSimu_1Shell'};%{'1009','1013'};
subjID={'BrainSimu_2Shell'};%{'1009','1013'};

dwiName={'im_2shelll_noisy_level8'};

for ind=1:length(subjID)
    for jnd=1:length(dwiName)


        if ~exist(fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd}),'dir')
            error('Data dir does not exist...')
        end

    %         ijdatadir= fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd},'data')
        ijdatadir= fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd})
        ijmrtrixdir= fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd},'mrtrix')

        if ~exist(ijmrtrixdir,'dir')
            mkdir(ijmrtrixdir)
        end

    %     %% convert nift to mif
    %     if ~exist(fullfile(ijmrtrixdir,'dwi.mif'),'file')
    %         mycmd=['mrconvert ', fullfile(ijdatadir,'data.nii.gz'),...
    %             ' -fslgrad ',fullfile(ijdatadir,'bvecs'),' ',fullfile(ijdatadir,'bvals'),...
    %             ' -datatype  float32  -stride 0,0,0,1  -force  ',fullfile(ijmrtrixdir,'dwi.mif')]
    %         system(mycmd)
    %     end
    % 
    %     if ~exist(fullfile(ijmrtrixdir,'a_dwi.mif'),'file')
    %                                 mycmd=['mrconvert ', fullfile(ijmrtrixdir,'dwi.mif'),...
    %                                     '  -coord 3 0:2  ',fullfile(ijmrtrixdir,'a_dwi.mif')]
    %                                 system(mycmd)
    %     end
        mycmd=['dwidenoise -noise ',fullfile(ijmrtrixdir,'noise_map.nii.gz'),...
            ' , -force ', fullfile(ijdatadir,'data.nii.gz'),...
            ' ',fullfile(ijmrtrixdir,'data_denoise.nii.gz')];
        system(mycmd)
    end
end


%% load

% load denoised images by mppca in mrtrix3
dir_data= fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd},'mrtrix');

mycmd1=['fslchfiletype NIFTI ',fullfile(dir_data,'data_denoise.nii.gz'),' ',...
    fullfile(dir_data,'data_denoise_load.nii')];
system(mycmd1)
image_mppca_mrtrix3 = load_nii(fullfile(dir_data,'data_denoise_load.nii'));

% load and resort mppca images
load IMd_mppca_2shell IMd_mppca
nlevel_idx = 2; % level=8
S_all = size(IMd_mppca);
image_mppca_matlab = IMd_mppca(:,:,:,nlevel_idx);

%% compare image
load data_2shell_brain_noisy.mat mask
nz_ToShow = 45;
Mask = mask(:,:,nz_ToShow);

% normalize
image_mppca_mrtrix3 = image_mppca_mrtrix3.img./200;

% b0
ims{1}=image_mppca_matlab(:,:,1);
ims{2}=rot90(image_mppca_mrtrix3(:,:,nz_ToShow,1));
mystr=[];
figure, position_plots(ims, [2 1],[0 1],[],[],mystr,'w','jet',1);colormap(gray)
err_twomethods_b0 = RMSE(ims{1}(Mask),ims{2}(Mask))

% b1k
ims{1}=image_mppca_matlab(:,:,3);
ims{2}=rot90(image_mppca_mrtrix3(:,:,nz_ToShow,3));
mystr=[];
figure, position_plots(ims, [2 1],[0 0.25],[],[],mystr,'w','jet',1);colormap(gray)
err_twomethods_b1k = RMSE(ims{1}(Mask),ims{2}(Mask))


% b1k
ims{1}=image_mppca_matlab(:,:,36);
ims{2}=rot90(image_mppca_mrtrix3(:,:,nz_ToShow,36));
mystr=[];
figure, position_plots(ims, [2 1],[0 0.25],[],[],mystr,'w','jet',1);colormap(gray)
err_twomethods_b2k = RMSE(ims{1}(Mask),ims{2}(Mask))

