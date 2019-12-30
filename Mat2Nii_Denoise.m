%%% Transfer *.mat images into *.nii

%%
% fs_home = '/opt/local/hcp-5.3.0/freesurfer/';
% mrtrix_home = '/opt/local/mrtrix3.20180307/';
% 
% % setenv('FSLDIR', '/opt/local/fsl-5.0.9/fsl');
% 
% %% brain simulation data (1 shell)
% % ground-truth and noisy images
% load data_2shell_brain_noisy.mat dwi IM_R levels mask
% 
% Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_1Shell/T1w/';
% for idx_level = 1:length(levels)
%     
%     dwi_name = ['im_1shelll_noisy_level',num2str(levels(idx_level))] ;
%     im_noisy_1shell = IM_R(:,:,:,1:34,idx_level) .* 200 ;
%     im_noisy_1shell = uint8(im_noisy_1shell);
%     im_noisy_1shell = flip(permute(im_noisy_1shell,[2 1 3 4]),2);
%     
%     savepath = [Dir0,dwi_name]; 
%     if ~exist(savepath,'dir')
%         eval(['mkdir ',savepath]);
%     end
%     
%     img = make_nii( im_noisy_1shell , [2 2 2] );
%     save_nii(img,...
%         [savepath,filesep,'data.nii']);
% 
%     mycmd1=['fslchfiletype NIFTI_GZ ',savepath,filesep,'data.nii'];
% %     mycmd1=['/opt/local/fsl-5.0.9/fsl/bin/fslchfiletype_exe ',...
% %         'NIFTI_GZ ', savepath,filesep,'data.nii'];
%     
%     system(mycmd1)
% end
% 
% 
% fall = dir(Dir0);
% for idx_level = 3:length(fall)    
%     if fall(idx_level).isdir
%         cppath = [Dir0,fall(idx_level).name]; 
%         eval(['copyfile ',Dir0,'nodif_brain_mask.nii.gz ',cppath]);
%     end
% end
% 
% mycmd1 = ['bet ', Dir0, 'T1w ', Dir0, 't1w_brain'];
% system(mycmd1)


%%

%% brain simulation data (2 shell)
% ground-truth and noisy images
load data_2shell_brain_noisy.mat dwi IM_R levels mask

Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell/T1w/';
for idx_level = 1:length(levels)
    
    dwi_name = ['im_2shelll_noisy_level',num2str(levels(idx_level))] ;
    im_noisy_1shell = IM_R(:,:,:,:,idx_level) .* 200 ;
    im_noisy_1shell = uint8(im_noisy_1shell);
    im_noisy_1shell = flip(permute(im_noisy_1shell,[2 1 3 4]),2);
    
    savepath = [Dir0,dwi_name]; 
    if ~exist(savepath,'dir')
        eval(['mkdir ',savepath]);
    end
    
    img = make_nii( im_noisy_1shell , [2 2 2] );
    save_nii(img,...
        [savepath,filesep,'data.nii']);

    mycmd1=['fslchfiletype NIFTI_GZ ',savepath,filesep,'data.nii'];
%     mycmd1=['/opt/local/fsl-5.0.9/fsl/bin/fslchfiletype_exe ',...
%         'NIFTI_GZ ', savepath,filesep,'data.nii'];

    system(mycmd1)

    mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
        'nodif_brain -m -n -f <0.05>'];
    system(mycmd1)
end

% mycmd1 = ['bet ', Dir0, 'data ', Dir0, 'nodif_brain -m -n'];
% system(mycmd1)

% fall = dir(Dir0);
% for idx_level = 3:length(fall)    
%     if fall(idx_level).isdir
%         cppath = [Dir0,fall(idx_level).name]; 
%         eval(['copyfile ',Dir0,'nodif_brain_mask.nii.gz ',cppath]);
%     end
% end

mycmd1 = ['bet ', Dir0, 't1w ', Dir0, 't1w_brain'];
system(mycmd1)
