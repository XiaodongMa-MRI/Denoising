%%% mppca in mrtrix3 (i.e. dwidenoise)
%% mppca using mrtrix3
clear all;close all; clc

dataDir0='/home/naxos2-raid1/maxiao/Projects/Data/Denoising/';
% subjID={'BrainSimu_1Shell'};%{'1009','1013'};
subjID={'BrainSimu_2Shell'};%{'1009','1013'};

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

for ind=1:length(subjID)
    for jnd=1:length(dwiName)

        if ~exist(fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd}),'dir')
            error('Data dir does not exist...')
        end

        ijdatadir= fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd})
        ijmrtrixdir= fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd},'mrtrix')

        if ~exist(ijmrtrixdir,'dir')
            mkdir(ijmrtrixdir)
        end

        mycmd=['dwidenoise -noise ',fullfile(ijmrtrixdir,'noise_map.nii.gz'),...
            ' -force ', fullfile(ijdatadir,'data.nii.gz'),...
            ' ',fullfile(ijmrtrixdir,'data_denoise.nii.gz')];
        system(mycmd)
    end
end

%% load the denoised images (nii.gz files)
%% load
for ind=1:length(subjID)
    for jnd=1:length(dwiName)

        if ~exist(fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd}),'dir')
            error('Data dir does not exist...')
        end

        % load denoised images by mppca in mrtrix3
        dir_data= fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd},'mrtrix');

        mycmd1=['fslchfiletype NIFTI ',fullfile(dir_data,'data_denoise.nii.gz'),' ',...
            fullfile(dir_data,'data_denoise_load.nii')];
        system(mycmd1)
        image_mppca_mrtrix3 = load_nii(fullfile(dir_data,'data_denoise_load.nii'));
        IMd_mppca(:,:,:,:,jnd) = flip(permute(image_mppca_mrtrix3.img,[2 1 3 4]),1);
        
        % load denoised images by mppca in mrtrix3
        dir_data= fullfile(dataDir0,subjID{ind},'T1w',dwiName{jnd},'mrtrix');

        mycmd1=['fslchfiletype NIFTI ',fullfile(dir_data,'noise_map.nii.gz'),' ',...
            fullfile(dir_data,'noise_map_load.nii')];
        system(mycmd1)
        Sigma_mppca_mrtrix3 = load_nii(fullfile(dir_data,'noise_map_load.nii'));
        (:,:,:,jnd) = flip(permute(Sigma_mppca_mrtrix3.img,[2 1 3]),1);
    end
end

IMd_mppca = double(IMd_mppca./200.0);
Sigma_mppca = double(Sigma_mppca./200.0);

save('IMd_mppca_2shell_mrtrix3.mat','IMd_mppca','Sigma_mppca','-v7.3');