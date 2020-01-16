%%% Transfer *.mat images into *.nii
%%% Prior steps: 
%%% (1) t1w.nii.gz should be in "T1w" folder
%%% (2) bvals and bvecs in "T1w" folder
%% brain simulation data (1 shell)
%% generate mask for T1w
Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_1Shell/T1w/';

mycmd1 = ['bet ', Dir0, 't1w ', Dir0, 't1w_brain'];
system(mycmd1)

%% noisy images
load data_2shell_brain_noisy.mat IM_R levels

for idx_level = 1:length(levels)
    
    dwi_name = ['im_2shelll_noisy_level',num2str(levels(idx_level))] ;
    im2save = IM_R(:,:,:,:,idx_level) .* 200 ;
    im2save = uint8(im2save);
    im2save = flip(permute(im2save,[2 1 3 4]),2);
    
    savepath = [Dir0,dwi_name]; 
    if ~exist(savepath,'dir')
        eval(['mkdir ',savepath]);
    end
    
    img = make_nii( im2save , [2 2 2] );
    save_nii(img,...
        [savepath,filesep,'data.nii']);

    % convert to nii.gz
    mycmd1=['fslchfiletype NIFTI_GZ ',savepath,filesep,'data.nii'];
    system(mycmd1)

    % generate mask using bet
    mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
        'nodif_brain -m -n -f 0.05'];
    system(mycmd1)
    
    % copy bvals and bvecs from parent folder
    copyfile([Dir0,'bvals'],savepath);
    copyfile([Dir0,'bvecs'],savepath);
end
clear IM_R

%% ground-truth
load data_2shell_brain_noisy.mat dwi
% ground-truth
dwi_name = 'im_2shelll_ground_truth';
im2save = dwi .* 200 ;
im2save = uint8(im2save);
im2save = flip(permute(im2save,[2 1 3 4]),2);

savepath = [Dir0,dwi_name]; 
if ~exist(savepath,'dir')
    eval(['mkdir ',savepath]);
end

img = make_nii( im2save , [2 2 2] );
save_nii(img,...
    [savepath,filesep,'data.nii']);

% convert to nii.gz
mycmd1=['fslchfiletype NIFTI_GZ ',savepath,filesep,'data.nii'];
system(mycmd1)

% generate mask using bet
mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
    'nodif_brain -m -n -f 0.05'];
system(mycmd1)

% copy bvals and bvecs from parent folder
copyfile([Dir0,'bvals'],savepath);
copyfile([Dir0,'bvecs'],savepath);


%% denoised images - proposed method
load IMVSTd_EUIVST_2shell_AllSlcs.mat IMVSTd_shrink_EUIVST
% Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell/T1w/';

levels = 1:10;

for idx_level = 1:length(levels)
    
    dwi_name = ['im_2shelll_proposed_level',num2str(levels(idx_level))] ;
    im2save = IMVSTd_shrink_EUIVST(:,:,:,:,idx_level) .* 200 ;
    im2save = uint8(im2save);
    im2save = flip(permute(im2save,[2 1 3 4]),2);
    
    savepath = [Dir0,dwi_name]; 
    if ~exist(savepath,'dir')
        eval(['mkdir ',savepath]);
    end
    
    img = make_nii( im2save , [2 2 2] );
    save_nii(img,...
        [savepath,filesep,'data.nii']);

    % convert to nii.gz
    mycmd1=['fslchfiletype NIFTI_GZ ',savepath,filesep,'data.nii'];
    system(mycmd1)

    % generate mask using bet
    mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
        'nodif_brain -m -n -f 0.05'];
    system(mycmd1)
    
    % copy bvals and bvecs from parent folder
    copyfile([Dir0,'bvals'],savepath);
    copyfile([Dir0,'bvecs'],savepath);
end
clear IMVSTd_shrink_EUIVST


%% denoised images - mppca
load IMd_mppca_2shell_AllSlcs.mat IMd_mppca
% Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell/T1w/';

levels = 1:10;

for idx_level = 1:length(levels)
    
    dwi_name = ['im_2shelll_mppca_level',num2str(levels(idx_level))] ;
    im2save = IMd_mppca(:,:,:,:,idx_level) .* 200 ;
    im2save = uint8(im2save);
    im2save = flip(permute(im2save,[2 1 3 4]),2);
    
    savepath = [Dir0,dwi_name]; 
    if ~exist(savepath,'dir')
        eval(['mkdir ',savepath]);
    end
    
    img = make_nii( im2save , [2 2 2] );
    save_nii(img,...
        [savepath,filesep,'data.nii']);

    % convert to nii.gz
    mycmd1=['fslchfiletype NIFTI_GZ ',savepath,filesep,'data.nii'];
    system(mycmd1)

    % generate mask using bet
    mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
        'nodif_brain -m -n -f 0.05'];
    system(mycmd1)
    
    % copy bvals and bvecs from parent folder
    copyfile([Dir0,'bvals'],savepath);
    copyfile([Dir0,'bvecs'],savepath);
end
clear IMd_mppca


%% noisy images of 10-average simulation (Gaussian)
load data_2shell_brain_noisy_10AvgGaussian IM_R levels
S_all = size(IM_R);
IM_R = reshape(IM_R,S_all(1),S_all(2),S_all(3),S_all(4)/10,10,S_all(5));
IM_R = squeeze(mean(IM_R,5));
% Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell/T1w/';
for idx_level = 1:length(levels)
    
    dwi_name = ['im_2shelll_noisy_10AvgGaussian_level',num2str(levels(idx_level))] ;
    im2save = IM_R(:,:,:,:,idx_level) .* 200 ;
    im2save = uint8(im2save);
    im2save = flip(permute(im2save,[2 1 3 4]),2);
    
    savepath = [Dir0,dwi_name]; 
    if ~exist(savepath,'dir')
        eval(['mkdir ',savepath]);
    end
    
    img = make_nii( im2save , [2 2 2] );
    save_nii(img,...
        [savepath,filesep,'data.nii']);

    % convert to nii.gz
    mycmd1=['fslchfiletype NIFTI_GZ ',savepath,filesep,'data.nii'];
    system(mycmd1)

    % generate mask using bet
    mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
        'nodif_brain -m -n -f 0.05'];
    system(mycmd1)
    
    % copy bvals and bvecs from parent folder
    copyfile([Dir0,'bvals'],savepath);
    copyfile([Dir0,'bvecs'],savepath);
end
clear IM_R


%% noisy images of 10-average simulation (Racian)
load data_2shell_brain_noisy_10AvgRacian IM_R levels
% S_all = size(IM_R);
% Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell/T1w/';
for idx_level = 1:length(levels)
    
    dwi_name = ['im_2shelll_noisy_10AvgRacian_level',num2str(levels(idx_level))] ;
    im2save = IM_R(:,:,:,:,idx_level) .* 200 ;
    im2save = uint8(im2save);
    im2save = flip(permute(im2save,[2 1 3 4]),2);
    
    savepath = [Dir0,dwi_name]; 
    if ~exist(savepath,'dir')
        eval(['mkdir ',savepath]);
    end
    
    img = make_nii( im2save , [2 2 2] );
    save_nii(img,...
        [savepath,filesep,'data.nii']);

    % convert to nii.gz
    mycmd1=['fslchfiletype NIFTI_GZ ',savepath,filesep,'data.nii'];
    system(mycmd1)

    % generate mask using bet
    mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
        'nodif_brain -m -n -f 0.05'];
    system(mycmd1)
    
end
clear IM_R

% generate new bvals and bvecs
for idx_level = 1:length(levels)
    
    dwi_name = ['im_2shelll_noisy_10AvgRacian_level',num2str(levels(idx_level))] ;
    
    savepath = [Dir0,dwi_name]; 
    
    % copy bvals and bvecs from parent folder
    copyfile([Dir0,'bvals'],[savepath,filesep,'bvals_tmp']);
    copyfile([Dir0,'bvecs'],[savepath,filesep,'bvecs_tmp']);
    
    % write new bvals for 10 averages
    fid1 = fopen([savepath,filesep,'bvals_tmp']);
    bvals_all = textscan(fid1,'%u');
    fclose(fid1);

    fid1 = fopen([savepath,filesep,'bvals'],'wb');
    bvals_10avg = repmat(bvals_all{1},[10 1]);
    for idx_b = 1:numel(bvals_10avg)
        fprintf(fid1,'%u',bvals_10avg(idx_b));
        fprintf(fid1,'%s',' ');
    end
    fclose(fid1);
    
    % write new bvecs for 10 averages
    fid1 = fopen([savepath,filesep,'bvecs_tmp']);
    bvecs_all = textscan(fid1,'%f');
    fclose(fid1);

    fid1 = fopen([savepath,filesep,'bvecs'],'wb');
    bvecs_10avg = reshape(bvecs_all{1},[numel(bvecs_all{1})/3,3]);
    
    for idx_avg = 1:10
        for idx_b = 1:size(bvecs_10avg,1)
            fprintf(fid1,'%f',bvecs_10avg(idx_b,1));
            fprintf(fid1,'%s',' ');
        end
    end
    fprintf(fid1,'\n');
    
    for idx_avg = 1:10
        for idx_b = 1:size(bvecs_10avg,1)
            fprintf(fid1,'%f',bvecs_10avg(idx_b,2));
            fprintf(fid1,'%s',' ');
        end
    end
    fprintf(fid1,'\n');
    
    for idx_avg = 1:10
        for idx_b = 1:size(bvecs_10avg,1)
            fprintf(fid1,'%f',bvecs_10avg(idx_b,2));
            fprintf(fid1,'%s',' ');
        end
    end
    
    fclose(fid1);
    
    delete([savepath,filesep,'*_tmp']);
    
end