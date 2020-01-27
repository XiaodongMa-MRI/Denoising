%%% Transfer *.mat images into *.nii
%%% Prior steps: 
%%% (1) t1w.nii.gz should be in "T1w" folder
%%% (2) bvals and bvecs in "T1w" folder
%% brain simulation data (2 shell)
%% generate mask for T1w
% Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell/T1w/';
Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell_ConstCSM/T1w/';

mycmd1 = ['bet ', Dir0, 't1w ', Dir0, 't1w_brain'];
system(mycmd1)


%% mask generation for dwi data

% load IMd_mppca_2shell_AllSlcs.mat IMd_mppca
load data_2shell_noisy_3DNoiseMap_ConstCSM mask
% Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell/T1w/';

im2save = mask;
im2save = uint8(im2save);
% im2save = flip(flip(permute(im2save,[2 1 3]),2),1);
im2save = flip(permute(im2save,[2 1 3]),2);

img = make_nii( im2save , [2 2 2] );
save_nii(img,...
    [Dir0,filesep,'nodif_brain_mask.nii']);

% convert to nii.gz
mycmd1=['fslchfiletype NIFTI_GZ ',Dir0,filesep,'nodif_brain_mask.nii'];
system(mycmd1)

%% noisy images
load Results_LinearCSM/data_2shell_brain_noisy_3DNoiseMap.mat IM_R levels
% load data_2shell_noisy_3DNoiseMap_ConstCSM.mat IM_R levels

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

%     % generate mask using bet
%     mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
%         'nodif_brain -m -n -f 0.05'];
%     system(mycmd1)
    
    copyfile([Dir0,'nodif_brain_mask.nii.gz'],savepath);
    
    % copy bvals and bvecs from parent folder
    copyfile([Dir0,'bvals'],savepath);
    copyfile([Dir0,'bvecs'],savepath);
end
clear IM_R

%% ground-truth
load data_2shell_noisy_3DNoiseMap_ConstCSM.mat dwi
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

% % generate mask using bet
% mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
%     'nodif_brain -m -n -f 0.05'];
% system(mycmd1)

copyfile([Dir0,'nodif_brain_mask.nii.gz'],savepath);

% copy bvals and bvecs from parent folder
copyfile([Dir0,'bvals'],savepath);
copyfile([Dir0,'bvecs'],savepath);


%% denoised images - proposed method
load IMVSTd_EUIVST_2shell_3DNoiseMapAllSlcs IMVSTd_shrink_EUIVST
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

%     % generate mask using bet
%     mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
%         'nodif_brain -m -n -f 0.05'];
%     system(mycmd1)
    
    copyfile([Dir0,'nodif_brain_mask.nii.gz'],savepath);

    % copy bvals and bvecs from parent folder
    copyfile([Dir0,'bvals'],savepath);
    copyfile([Dir0,'bvecs'],savepath);
end
clear IMVSTd_shrink_EUIVST


%% denoised images - mppca
load Results_LinearCSM/IMd_mppca_2shell_mrtrix3.mat IMd_mppca
% load IMd_mppca_2shell_mrtrix3.mat IMd_mppca
% Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell/T1w/';

levels = 1:10;

for idx_level = 1:length(levels)
    
    dwi_name = ['im_2shelll_mppca_mrtrix3_level',num2str(levels(idx_level))] ;
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

%     % generate mask using bet
%     mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
%         'nodif_brain -m -n -f 0.05'];
%     system(mycmd1)
    
    copyfile([Dir0,'nodif_brain_mask.nii.gz'],savepath);

    % copy bvals and bvecs from parent folder
    copyfile([Dir0,'bvals'],savepath);
    copyfile([Dir0,'bvecs'],savepath);
end
clear IMd_mppca

%% noisy images of 10-average simulation (Gaussian)
load data_2shell_brain_noisy3D_20AvgGaussian_Level3 IM_R levels
Navg = 20;
dwi_name_prefix = 'im_2shelll_noisy_20AvgGaussian_level';
Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell/T1w/';

S_all = size(IM_R);
if length(S_all)>4
    IM_R = reshape(IM_R,S_all(1),S_all(2),S_all(3),S_all(4)/Navg,Navg,S_all(5));
else
    IM_R = reshape(IM_R,S_all(1),S_all(2),S_all(3),S_all(4)/Navg,Navg);
end
IM_R = squeeze(mean(IM_R,5));
for idx_level = 1:length(levels)
    
    dwi_name = [dwi_name_prefix,num2str(levels(idx_level))] ;
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

%     % generate mask using bet
%     mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
%         'nodif_brain -m -n -f 0.05'];
%     system(mycmd1)
    
    copyfile([Dir0,'nodif_brain_mask.nii.gz'],savepath);
    % copy bvals and bvecs from parent folder
    copyfile([Dir0,'bvals'],savepath);
    copyfile([Dir0,'bvecs'],savepath);
end
clear IM_R


%% noisy images of 10-average simulation (Racian)
load data_2shell_brain_noisy3D_20AvgRacian_Level3 IM_R levels
Navg = 20;
dwi_name_prefix = 'im_2shelll_noisy_20AvgRacian_level';
Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell/T1w/';
% S_all = size(IM_R);
for idx_level = 1:length(levels)
    
    dwi_name = [dwi_name_prefix,num2str(levels(idx_level))] ;
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

%     % generate mask using bet
%     mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
%         'nodif_brain -m -n -f 0.05'];
%     system(mycmd1)
    
    copyfile([Dir0,'nodif_brain_mask.nii.gz'],savepath);
end
clear IM_R

% generate new bvals and bvecs
for idx_level = 1:length(levels)
    
    dwi_name = [dwi_name_prefix,num2str(levels(idx_level))] ;
    
    savepath = [Dir0,dwi_name]; 
    
    % copy bvals and bvecs from parent folder
    copyfile([Dir0,'bvals'],[savepath,filesep,'bvals_tmp']);
    copyfile([Dir0,'bvecs'],[savepath,filesep,'bvecs_tmp']);
    
    % write new bvals for 10 averages
    fid1 = fopen([savepath,filesep,'bvals_tmp']);
    bvals_all = textscan(fid1,'%u');
    fclose(fid1);

    fid1 = fopen([savepath,filesep,'bvals'],'wb');
    bvals_avg = repmat(bvals_all{1},[Navg 1]);
    for idx_b = 1:numel(bvals_avg)
        fprintf(fid1,'%u',bvals_avg(idx_b));
        fprintf(fid1,'%s',' ');
    end
    fclose(fid1);
    
    % write new bvecs for 10 averages
    fid1 = fopen([savepath,filesep,'bvecs_tmp']);
    bvecs_all = textscan(fid1,'%f');
    fclose(fid1);

    fid1 = fopen([savepath,filesep,'bvecs'],'wb');
    bvecs_avg = reshape(bvecs_all{1},[numel(bvecs_all{1})/3,3]);
    
    for idx_avg = 1:Navg
        for idx_b = 1:size(bvecs_avg,1)
            fprintf(fid1,'%f',bvecs_avg(idx_b,1));
            fprintf(fid1,'%s',' ');
        end
    end
    fprintf(fid1,'\n');
    
    for idx_avg = 1:Navg
        for idx_b = 1:size(bvecs_avg,1)
            fprintf(fid1,'%f',bvecs_avg(idx_b,2));
            fprintf(fid1,'%s',' ');
        end
    end
    fprintf(fid1,'\n');
    
    for idx_avg = 1:Navg
        for idx_b = 1:size(bvecs_avg,1)
            fprintf(fid1,'%f',bvecs_avg(idx_b,3));
            fprintf(fid1,'%s',' ');
        end
    end
    
    fclose(fid1);
    
    delete([savepath,filesep,'*_tmp']);
    
end

%% Ground-truth of 1 shell
load data_2shell_noisy_3DNoiseMap_ConstCSM dwi
Dir0 = '/home/naxos2-raid1/maxiao/Projects/Data/Denoising/BrainSimu_2Shell_ConstCSM/T1w/';
% S_all = size(IM_R);
dwi_name = 'im_1shelll_ground_truth';
im2save = dwi(:,:,:,1:34) .* 200 ;
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

% % generate mask using bet
% mycmd1 = ['bet ', savepath,filesep, 'data ', savepath,filesep, ...
%     'nodif_brain -m -n -f 0.05'];
% system(mycmd1)

copyfile([Dir0,'nodif_brain_mask.nii.gz'],savepath);



savepath = [Dir0,dwi_name]; 

% copy bvals and bvecs from parent folder
copyfile([Dir0,'bvals'],[savepath,filesep,'bvals_tmp']);
copyfile([Dir0,'bvecs'],[savepath,filesep,'bvecs_tmp']);

% write new bvals for 10 averages
fid1 = fopen([savepath,filesep,'bvals_tmp']);
bvals_all = textscan(fid1,'%u');
fclose(fid1);

fid1 = fopen([savepath,filesep,'bvals'],'wb');
for idx_b = 1:34
    fprintf(fid1,'%u',bvals_all{1}(idx_b));
    fprintf(fid1,'%s',' ');
end
fclose(fid1);

% write new bvecs for 10 averages
fid1 = fopen([savepath,filesep,'bvecs_tmp']);
bvecs_all = textscan(fid1,'%f');
fclose(fid1);

fid1 = fopen([savepath,filesep,'bvecs'],'wb');
bvecs_new = reshape(bvecs_all{1},[numel(bvecs_all{1})/3,3]);

% for idx_avg = 1:Navg
    for idx_b = 1:34
        fprintf(fid1,'%f',bvecs_new(idx_b,1));
        fprintf(fid1,'%s',' ');
    end
% end
fprintf(fid1,'\n');

% for idx_avg = 1:Navg
    for idx_b = 1:34
        fprintf(fid1,'%f',bvecs_new(idx_b,2));
        fprintf(fid1,'%s',' ');
    end
% end
fprintf(fid1,'\n');

% for idx_avg = 1:Navg
    for idx_b = 1:34
        fprintf(fid1,'%f',bvecs_new(idx_b,3));
        fprintf(fid1,'%s',' ');
    end
% end

fclose(fid1);

delete([savepath,filesep,'*_tmp']);
