%%% Calculate SNR of simulated brain images
%%% Two strategies were uses:
%%% ====Strategy #1=====
%%% estimate the average SNR values using simulated Rician data. 
%%% For each noise level, you need to average the noisy magnitude images
%%% for a given b value and consider the estimated noise map (ie, the one 
%%% you obtained using the VST method) for evaluating the SNR map. In this 
%%% case, the histogram of this SNR map is expected to present two peaks. 
%%% This strategy has been used to estimate SNR values for the real data.
%%% ====Strategy #2=====
%%% estimate the average SNR values using ground truth data. For each noise
%%% level, you need to average the ground truth images for a given b value 
%%% and consider the ground truth noise map (ie, the one you created to 
%%% generate spatially varying Gaussian noises) for evaluating the SNR map.
%%% In this case, the histogram of this SNR map is expected to present only
%%% one peak. The results of this strategy can serve as a reference, 
%%% against which the results of strategy 1 can be compared so as to see 
%%% how accurate it would be when estimating average SNR using Rician data 
%%% and estimated noises.

%% load data
% load Results_LinearCSM/data_2shell_brain_noisy_3DNoiseMap.mat % created in noisyDataCreation.m
% % load data_2shell_noisy_3DNoiseMap_ConstCSM.mat % created in noisyDataCreation.m
% load Results_LinearCSM/sigEst_multishell_fullFOV_B_ws5_WholeBrain.mat % created in noisyDataCreation.m

load data_2shell_noisy_3DNoiseMap_ConstCSM.mat % created in noisyDataCreation.m
% load data_2shell_noisy_3DNoiseMap_ConstCSM.mat % created in noisyDataCreation.m
load sigEst_multishell_fullFOV_B_ws5_WholeBrain_ConstCSM.mat % created in noisyDataCreation.m

%% Strategy #1
%% calculate the mean image for each b value
for idx=1:size(IM_R,5)
    imgave_b0{idx}= nanmean(IM_R(:,:,:,bvals0<500,idx),4);
    imgave_b1k{idx}= nanmean(IM_R(:,:,:,bvals0>500& bvals0<1500,idx),4);
    imgave_b2k{idx}= nanmean(IM_R(:,:,:,bvals0>1500,idx),4);
end
sigmaVST=cell(1,size(Sigma_VST2_b1k,4));
for idx=1:size(Sigma_VST2_b1k,4)
    sigmaVST{idx}.sigma_vst = Sigma_VST2_b1k(:,:,:,idx);
end

figure, position_plots(imgave_b0{1}(:,:,3:6:69),[3,4],[0 1])
figure, position_plots(imgave_b1k{1}(:,:,3:6:69),[3,4],[0 0.5])
figure, position_plots(imgave_b2k{1}(:,:,3:6:69),[3,4],[0 0.3])
figure, position_plots(sigmaVST{1}.sigma_vst(:,:,3:6:69),[3,4])

x= imgave_b1k{1}(:);
x= imgave_b0{1}(:);
x= imgave_b2k{1}(:);
x= sigmaVST{1}.sigma_vst(:);
figure, histogram(x(x>prctile(x,0))) % even for b2k, the mean value can be viewed as being large enough (relative to the mean sigma) that the mean image (ie, expectation of Rician data) can be used as good approximate for the signal mean (ie, mean of underlying Gaussian data). 

%% calculate snr
for idx=1:length(imgave_b0)
    snr_b0{idx}= imgave_b0{idx}./sigmaVST{idx}.sigma_vst;
    snr_b1k{idx}= imgave_b1k{idx}./sigmaVST{idx}.sigma_vst;
    snr_b2k{idx}= imgave_b2k{idx}./sigmaVST{idx}.sigma_vst;
end
figure, position_plots(snr_b0{1}(:,:,3:6:69),[3,4])
figure, position_plots(snr_b1k{1}(:,:,3:6:69),[3,4])
figure, position_plots(snr_b2k{1}(:,:,3:6:69),[3,4])

% hist
x= snr_b0{1}(:);
x= snr_b1k{1}(:);
x= snr_b2k{1}(:);
figure, histogram(x(x>prctile(x,0))) % check histogram to make sure there are two peaks

%% derive average snr for each b value by locating the second peak in the histogram of snr
%%% 
nbins=100;
mpp=1e-4;
for idx=1:length(snr_b0)
    
    % b0
    x= snr_b0{idx}(:);
    [p,edge]=histcounts(x(x>prctile(x,2.5)&x<prctile(x,97.5)),nbins,'Normalization','probability');
    xx= edge(1:end-1)+ 0.5*diff(edge);
    %figure, plot(xx,p)
    %[pks,locs]=findpeaks(p,xx,'MinPeakHeight',1e-3,'MinPeakWidth',1);
    [pks,locs]=findpeaks(p,xx,'MinPeakProminence',mpp,...
        'SortStr','descend','NPeaks',2);
    snrave_b0(idx)=locs(2);
    
    % b1k
    x= snr_b1k{idx}(:);
    [p,edge]=histcounts(x(x>prctile(x,2.5)&x<prctile(x,97.5)),nbins,'Normalization','probability');
    xx= edge(1:end-1)+ 0.5*diff(edge);
    %figure, plot(xx,p)
    %[pks,locs]=findpeaks(p,xx,'MinPeakHeight',1e-3,'MinPeakWidth',1);
    [pks,locs]=findpeaks(p,xx,'MinPeakProminence',mpp,...
        'SortStr','descend','NPeaks',2);
%     snrave_b1k(idx)=locs(2);
    if numel(locs)>1
        snrave_b1k(idx)=locs(2);
    else
    	snrave_b1k(idx)=locs;
    end
    
    % b2k
    x= snr_b2k{idx}(:);
    [p,edge]=histcounts(x(x>prctile(x,2.5)&x<prctile(x,97.5)),nbins,'Normalization','probability');
    xx= edge(1:end-1)+ 0.5*diff(edge);
    %figure, plot(xx,p)
    %[pks,locs]=findpeaks(p,xx,'MinPeakHeight',1e-3,'MinPeakWidth',1);
    [pks,locs]=findpeaks(p,xx,'MinPeakProminence',mpp,...
        'SortStr','descend','NPeaks',2);
%     snrave_b2k(idx)=locs(2);
    if numel(locs)>1
        snrave_b2k(idx)=locs(2);
    else
    	snrave_b2k(idx)=locs;
    end
end
% mean(snrave_b0) % 6.7
% mean(snrave_b1k) % 3.9
% mean(snrave_b2k) % 2.5
%% display

clear opt
opt.Markers={'v','+','o'};
opt.LineStyle={'-','-','-'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
opt.YLabel='Estimated SNR';
opt.FileName='SNR_wholeBrain_Strategy1';
Y{1} = snrave_b0;
Y{2} = snrave_b1k;
Y{3} = snrave_b2k;
opt.Colors=[0,0,0;0,0,1;1,0,0];
opt.Legend= {'b0','b1k','b2k'};
opt.LegendLoc= 'NorthEast';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=5;
figplot



%% Strategy #2
%% calculate the mean image for each b value
for idx=1:size(IM_R,5)
    imgave_b0{idx}= nanmean(dwi(:,:,:,bvals0<500),4);
    imgave_b1k{idx}= nanmean(dwi(:,:,:,bvals0>500& bvals0<1500),4);
    imgave_b2k{idx}= nanmean(dwi(:,:,:,bvals0>1500),4);
end
sigmaVST=cell(1,size(Sigma_VST2_b1k,4));
for idx=1:size(Sigma_VST2_b1k,4)
    sigmaVST{idx}.sigma_vst = Sigma0(:,:,:,idx);
end

figure, position_plots(imgave_b0{1}(:,:,3:6:69),[3,4],[0 1])
figure, position_plots(imgave_b1k{1}(:,:,3:6:69),[3,4],[0 0.5])
figure, position_plots(imgave_b2k{1}(:,:,3:6:69),[3,4],[0 0.3])
figure, position_plots(sigmaVST{1}.sigma_vst(:,:,3:6:69),[3,4])

x= imgave_b1k{1}(:);
x= imgave_b0{1}(:);
x= imgave_b2k{1}(:);
x= sigmaVST{1}.sigma_vst(:);
figure, histogram(x(x>prctile(x,0))) % even for b2k, the mean value can be viewed as being large enough (relative to the mean sigma) that the mean image (ie, expectation of Rician data) can be used as good approximate for the signal mean (ie, mean of underlying Gaussian data). 

%% calculate snr
for idx=1:length(imgave_b0)
    snr_b0{idx}= imgave_b0{idx}./sigmaVST{idx}.sigma_vst;
    snr_b1k{idx}= imgave_b1k{idx}./sigmaVST{idx}.sigma_vst;
    snr_b2k{idx}= imgave_b2k{idx}./sigmaVST{idx}.sigma_vst;
end
figure, position_plots(snr_b0{1}(:,:,3:6:69),[3,4])
figure, position_plots(snr_b1k{1}(:,:,3:6:69),[3,4])
figure, position_plots(snr_b2k{1}(:,:,3:6:69),[3,4])

% hist
x= snr_b0{1}(:);
x= snr_b1k{1}(:);
x= snr_b2k{1}(:);
figure, histogram(x(x>prctile(x,0))) % check histogram to make sure there are two peaks

%% derive average snr for each b value by locating the second peak in the histogram of snr
%%% 
nbins=100;
mpp=1e-4;
for idx=1:length(snr_b0)
    
    % b0
    x= snr_b0{idx}(:);
    [p,edge]=histcounts(x(x>prctile(x,2.5)&x<prctile(x,97.5)),nbins,'Normalization','probability');
    xx= edge(1:end-1)+ 0.5*diff(edge);
    %figure, plot(xx,p)
    %[pks,locs]=findpeaks(p,xx,'MinPeakHeight',1e-3,'MinPeakWidth',1);
    [pks,locs]=findpeaks(p,xx,'MinPeakProminence',mpp,...
        'SortStr','descend','NPeaks',2);
    snrave_b0_Stg2(idx)=locs(2);
    
    % b1k
    x= snr_b1k{idx}(:);
    [p,edge]=histcounts(x(x>prctile(x,2.5)&x<prctile(x,97.5)),nbins,'Normalization','probability');
    xx= edge(1:end-1)+ 0.5*diff(edge);
    %figure, plot(xx,p)
    %[pks,locs]=findpeaks(p,xx,'MinPeakHeight',1e-3,'MinPeakWidth',1);
    [pks,locs]=findpeaks(p,xx,'MinPeakProminence',mpp,...
        'SortStr','descend','NPeaks',2);
%     snrave_b1k_Stg2(idx)=locs(2);
    if numel(locs)>1
        snrave_b1k_Stg2(idx)=locs(2);
    else
    snrave_b1k_Stg2(idx)=locs;
    end
    
    % b2k
    x= snr_b2k{idx}(:);
    [p,edge]=histcounts(x(x>prctile(x,2.5)&x<prctile(x,97.5)),nbins,'Normalization','probability');
    xx= edge(1:end-1)+ 0.5*diff(edge);
    %figure, plot(xx,p)
    %[pks,locs]=findpeaks(p,xx,'MinPeakHeight',1e-3,'MinPeakWidth',1);
    [pks,locs]=findpeaks(p,xx,'MinPeakProminence',mpp,...
        'SortStr','descend','NPeaks',2);
%     snrave_b2k_Stg2(idx)=locs(2);
    if numel(locs)>1
        snrave_b2k_Stg2(idx)=locs(2);
    else
    snrave_b2k_Stg2(idx)=locs;
    end
end
% mean(snrave_b0) % 6.7
% mean(snrave_b1k) % 3.9
% mean(snrave_b2k) % 2.5
%% display
clear opt
opt.Markers={'v','+','o'};
opt.LineStyle={'-','-','-'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
opt.YLabel='SNR Whole Brain';
opt.FileName='SNR_wholeBrain_Strategy1';
Y{1} = snrave_b0_Stg2;
Y{2} = snrave_b1k_Stg2;
Y{3} = snrave_b2k_Stg2;
opt.Colors=[0,0,0;0,0,1;1,0,0];
opt.Legend= {'b0','b1k','b2k'};
opt.LegendLoc= 'NorthEast';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=5;
figplot




%% Strategy #3: based on Strategy#2, but using mask instead of histogram
mdwi_b0 = squeeze(mean(dwi(:,:,:,(bvals0<50)),4));
mdwi_b1k = squeeze(mean(dwi(:,:,:,(bvals0>50)&(bvals0<1500)),4));
mdwi_b2k = squeeze(mean(dwi(:,:,:,(bvals0>1500)),4));
% load FA
% Dir0 = '/home/naxos2-raid1/maxiao/Projects/DiffusionAnalysis/diffusion-analysis/BrainSimu_2Shell_ConstCSM/T1w/';
Dir0 = '/home/naxos2-raid1/maxiao/Projects/DiffusionAnalysis/diffusion-analysis/BrainSimu_2Shell_LinearCSM/T1w/';
dwiName={ 'im_2shelll_ground_truth'};
dir_data = [Dir0,dwiName{1},filesep,'dti/'];
mycmd1=['fslchfiletype NIFTI ',dir_data,'dti_FA.nii.gz ',...
    dir_data,'dti_FA_ch.nii'];
system(mycmd1)
FA = load_nii([dir_data,'dti_FA_ch.nii']);
FA = FA.img;
FA = permute(flip(FA,2),[2 1 3]);

mask_wm = FA>0.25;

dwi = single(dwi);
Sigma0 = single(Sigma0);
SNR_map_b0 = repmat(mdwi_b0,[1 1 1 size(IM_R,5)])./Sigma0;
SNR_map_b1k = repmat(mdwi_b1k,[1 1 1 size(IM_R,5)])./Sigma0;
SNR_map_b2k = repmat(mdwi_b2k,[1 1 1 size(IM_R,5)])./Sigma0;

% display SNR maps
% % b0 [0 25]
% figure;myimagesc(SNR_map_All(:,:,45,1,5));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(:,40,:,1,5))));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(60,:,:,1,5))));colorbar;imcontrast
% %b1k [0 7]
% figure;myimagesc(SNR_map_All(:,:,45,3,5));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(:,40,:,3,5))));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(60,:,:,3,5))));colorbar;imcontrast
% %b2k [0 5]
% figure;myimagesc(SNR_map_All(:,:,45,36,5));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(:,40,:,36,5))));colorbar;imcontrast
% figure;myimagesc(rot90(squeeze(SNR_map_All(60,:,:,36,5))));colorbar;imcontrast

%% average SNR across whole brain
for idx = 1:numel(levels)
    SNR_ij = SNR_map_b0(:,:,:,idx);
    SNR_wb_b0(idx) = mean(mean(SNR_ij(mask)));
    SNR_ij = SNR_map_b1k(:,:,:,idx);
    SNR_wb_b1k(idx) = mean(mean(SNR_ij(mask)));
    SNR_ij = SNR_map_b2k(:,:,:,idx);
    SNR_wb_b2k(idx) = mean(mean(SNR_ij(mask)));
end

clear opt
opt.Markers={'v','+','o'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
opt.YLabel='SNR Whole Brain';
opt.FileName='SNR_wholeBrain';
Y{1} = SNR_wb_b0;
Y{2} = SNR_wb_b1k;
Y{3} = SNR_wb_b2k;
opt.Colors=[0,0,0;0,0,1;1,0,0];
opt.Legend= {'b0','b1k','b2k'};
opt.LegendLoc= 'NorthEast';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=5;
figplot

%% average SNR across white matter
for idx = 1:numel(levels)
    SNR_ij = SNR_map_b0(:,:,:,idx);
    SNR_wm_b0(idx) = mean(mean(SNR_ij(mask_wm)));
    SNR_ij = SNR_map_b1k(:,:,:,idx);
    SNR_wm_b1k(idx) = mean(mean(SNR_ij(mask_wm)));
    SNR_ij = SNR_map_b2k(:,:,:,idx);
    SNR_wm_b2k(idx) = mean(mean(SNR_ij(mask_wm)));
end

clear opt
opt.Markers={'v','+','o'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
opt.YLabel='SNR White Matter';
opt.FileName='SNR_WhiteMatter';
Y{1} = SNR_wm_b0;
Y{2} = SNR_wm_b1k;
Y{3} = SNR_wm_b2k;
opt.Colors=[0,0,0;0,0,1;1,0,0];
opt.Legend= {'b0','b1k','b2k'};
opt.LegendLoc= 'NorthEast';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=5;
figplot




%% compare three strategies

clear opt
opt.Markers={'v','+','o','v','+','o','v','+','o'};
opt.LineStyle={'--','--','--',':',':',':','-','-','-'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
X{4} = levels;
X{5} = levels;
X{6} = levels;
X{7} = levels;
X{8} = levels;
X{9} = levels;
opt.YLabel='SNR';
opt.FileName='SNR_wholeBrain_AllStrategies';
Y{1} = snrave_b0;
Y{2} = snrave_b1k;
Y{3} = snrave_b2k;
Y{4} = snrave_b0_Stg2;
Y{5} = snrave_b1k_Stg2;
Y{6} = snrave_b2k_Stg2;
Y{7} = SNR_wb_b0;
Y{8} = SNR_wb_b1k;
Y{9} = SNR_wb_b2k;
opt.Colors=[0,0,0;0,0,1;1,0,0;0,0,0;0,0,1;1,0,0;0,0,0;0,0,1;1,0,0];
opt.Legend= {'b0-Stg1','b1k-Stg1','b2k-Stg1','b0-Stg2','b1k-Stg2','b2k-Stg2'...
            'b0-Mask','b1k-Mask','b2k-Mask'};
opt.LegendLoc= 'NorthEast';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=7;
figplot





%% Strategy #1+ (Correct the signal by M=(M0^2-N^2))
%% calculate the mean image for each b value
for idx=1:size(IM_R,5)
    imgave_b0{idx}= nanmean(IM_R(:,:,:,bvals0<500,idx),4);
    imgave_b1k{idx}= nanmean(IM_R(:,:,:,bvals0>500& bvals0<1500,idx),4);
    imgave_b2k{idx}= nanmean(IM_R(:,:,:,bvals0>1500,idx),4);
end
sigmaVST=cell(1,size(Sigma_VST2_b1k,4));
for idx=1:size(Sigma_VST2_b1k,4)
    sigmaVST{idx}.sigma_vst = Sigma_VST2_b1k(:,:,:,idx);
end

% correction
for idx=1:size(IM_R,5)
    imgave_b0{idx}= sqrt(imgave_b0{idx}.^2 - (sigmaVST{idx}.sigma_vst).^2);
    imgave_b1k{idx}= sqrt(imgave_b1k{idx}.^2 - (sigmaVST{idx}.sigma_vst).^2);
    imgave_b2k{idx}= sqrt(imgave_b2k{idx}.^2 - (sigmaVST{idx}.sigma_vst).^2);
end

figure, position_plots(imgave_b0{1}(:,:,3:6:69),[3,4],[0 1])
figure, position_plots(imgave_b1k{1}(:,:,3:6:69),[3,4],[0 0.5])
figure, position_plots(imgave_b2k{1}(:,:,3:6:69),[3,4],[0 0.3])
figure, position_plots(sigmaVST{1}.sigma_vst(:,:,3:6:69),[3,4])

x= imgave_b1k{1}(:);
x= imgave_b0{1}(:);
x= imgave_b2k{1}(:);
x= sigmaVST{1}.sigma_vst(:);
figure, histogram(x(x>prctile(x,0))) % even for b2k, the mean value can be viewed as being large enough (relative to the mean sigma) that the mean image (ie, expectation of Rician data) can be used as good approximate for the signal mean (ie, mean of underlying Gaussian data). 

%% calculate snr
for idx=1:length(imgave_b0)
    snr_b0{idx}= imgave_b0{idx}./sigmaVST{idx}.sigma_vst;
    snr_b1k{idx}= imgave_b1k{idx}./sigmaVST{idx}.sigma_vst;
    snr_b2k{idx}= imgave_b2k{idx}./sigmaVST{idx}.sigma_vst;
end
figure, position_plots(snr_b0{1}(:,:,3:6:69),[3,4])
figure, position_plots(snr_b1k{1}(:,:,3:6:69),[3,4])
figure, position_plots(snr_b2k{1}(:,:,3:6:69),[3,4])

% hist
x= snr_b0{1}(:);
x= snr_b1k{1}(:);
x= snr_b2k{1}(:);
figure, histogram(x(x>prctile(x,0))) % check histogram to make sure there are two peaks

%% derive average snr for each b value by locating the second peak in the histogram of snr
%%% 
nbins=100;
mpp=1e-4;
for idx=1:length(snr_b0)
    
    % b0
    x= snr_b0{idx}(:);
    [p,edge]=histcounts(x(x>prctile(x,2.5)&x<prctile(x,97.5)),nbins,'Normalization','probability');
    xx= edge(1:end-1)+ 0.5*diff(edge);
    %figure, plot(xx,p)
    %[pks,locs]=findpeaks(p,xx,'MinPeakHeight',1e-3,'MinPeakWidth',1);
    [pks,locs]=findpeaks(p,xx,'MinPeakProminence',mpp,...
        'SortStr','descend','NPeaks',2);
    if numel(locs)>1
        snrave_b0_correct(idx)=locs(2);
    else
        snrave_b0_correct(idx)=locs;
    end
    
    % b1k
    x= snr_b1k{idx}(:);
    [p,edge]=histcounts(x(x>prctile(x,2.5)&x<prctile(x,97.5)),nbins,'Normalization','probability');
    xx= edge(1:end-1)+ 0.5*diff(edge);
    %figure, plot(xx,p)
    %[pks,locs]=findpeaks(p,xx,'MinPeakHeight',1e-3,'MinPeakWidth',1);
    [pks,locs]=findpeaks(p,xx,'MinPeakProminence',mpp,...
        'SortStr','descend','NPeaks',2);
    if numel(locs)>1
        snrave_b1k_correct(idx)=locs(2);
    else
        snrave_b1k_correct(idx)=locs;
    end
    
    % b2k
    x= snr_b2k{idx}(:);
    [p,edge]=histcounts(x(x>prctile(x,2.5)&x<prctile(x,97.5)),nbins,'Normalization','probability');
    xx= edge(1:end-1)+ 0.5*diff(edge);
    %figure, plot(xx,p)
    %[pks,locs]=findpeaks(p,xx,'MinPeakHeight',1e-3,'MinPeakWidth',1);
    [pks,locs]=findpeaks(p,xx,'MinPeakProminence',mpp,...
        'SortStr','descend','NPeaks',2);
    if numel(locs)>1
        snrave_b2k_correct(idx)=locs(2);
    else
        snrave_b2k_correct(idx)=locs;
    end
end
% mean(snrave_b0) % 6.7
% mean(snrave_b1k) % 3.9
% mean(snrave_b2k) % 2.5
%% display

clear opt
opt.Markers={'v','+','o'};
opt.LineStyle={'-','-','-'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
opt.YLabel='SNR Whole Brain';
opt.FileName='SNR_wholeBrain_Strategy1';
Y{1} = snrave_b0_correct;
Y{2} = snrave_b1k_correct;
Y{3} = snrave_b2k_correct;
opt.Colors=[0,0,0;0,0,1;1,0,0];
opt.Legend= {'b0','b1k','b2k'};
opt.LegendLoc= 'NorthEast';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=5;
figplot





%% compare four strategies

clear opt
opt.Markers={'v','+','o','v','+','o','v','+','o','v','+','o'};
opt.LineStyle={'--','--','--',':',':',':','-.','-.','-.','-','-','-'};
opt.XLabel='Noise level (%)';
opt.XLim=[0 11];
opt.YLim=[0 20];
clear X Y
X{1} = levels;
X{2} = levels;
X{3} = levels;
X{4} = levels;
X{5} = levels;
X{6} = levels;
X{7} = levels;
X{8} = levels;
X{9} = levels;
X{10} = levels;
X{11} = levels;
X{12} = levels;
opt.YLabel='SNR';
opt.FileName='SNR_wholeBrain_AllStrategies';
Y{1} = snrave_b0;
Y{2} = snrave_b1k;
Y{3} = snrave_b2k;
Y{4} = snrave_b0_Stg2;
Y{5} = snrave_b1k_Stg2;
Y{6} = snrave_b2k_Stg2;
Y{7} = snrave_b0_correct;
Y{8} = snrave_b1k_correct;
Y{9} = snrave_b2k_correct;
Y{10} = SNR_wb_b0;
Y{11} = SNR_wb_b1k;
Y{12} = SNR_wb_b2k;
opt.Colors=[0,0,0;0,0,1;1,0,0;0,0,0;0,0,1;1,0,0;0,0,0;0,0,1;1,0,0;0,0,0;0,0,1;1,0,0];
opt.Legend= {'b0-Stg1','b1k-Stg1','b2k-Stg1','b0-Stg2','b1k-Stg2','b2k-Stg2',...
            'b0-Stg1-correct','b1k-Stg1-correct','b2k-Stg1-correct',...
            'b0-Mask','b1k-Mask','b2k-Mask'};
opt.LegendLoc= 'NorthEast';
opt.FileName=[opt.FileName,'.png'];
maxBoxDim=6;
figplot

%% save SNR values
save SNR_noisy_ConstCSM  snrave_b* SNR_wb_b*

%% show the images for visual check
%axial
figure, position_plots(squeeze(IM_R(:,:,45,1,[3 4 5])),[1,3],[],[],[],[],[],'gray')
figure, position_plots(squeeze(IM_R(:,:,45,3,[3 4 5])),[1,3],[],[],[],[],[],'gray')
figure, position_plots(squeeze(IM_R(:,:,45,36,[3 4 5])),[1,3],[],[],[],[],[],'gray')
%sagital
figure, position_plots(flip(flip(permute(squeeze(IM_R(:,40,:,1,[3 4 5])),[2 1 3]),1),2),[1,3],[],[],[],[],[],'gray')
figure, position_plots(flip(flip(permute(squeeze(IM_R(:,40,:,3,[3 4 5])),[2 1 3]),1),2),[1,3],[],[],[],[],[],'gray')
figure, position_plots(flip(flip(permute(squeeze(IM_R(:,40,:,36,[3 4 5])),[2 1 3]),1),2),[1,3],[],[],[],[],[],'gray')
%coronal



