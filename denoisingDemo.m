%%% a script demonstrating how to use the proposed method to denoise
%%% magnitude diffusion images. 
%% create noisy data with 2 shell b1000 and b2000
load data_2shell dwi0 bvals0 Mask0 gs0
nz= 11;
if numel(size(dwi0))<4
    dwi1= repmat(reshape(dwi0,size(dwi0,1),size(dwi0,2),1,size(dwi0,3)),1,1,nz,1);
    Mask1= repmat(Mask0,1,1,nz);
end
numDWI= size(dwi1,4);

ksize=7;%5;
[i1,i2,i3]= ind2sub(size(Mask1),find(Mask1));
dwi= dwi1(min(i1):max(i1)+ksize,min(i2)-ksize:max(i2)+ksize,:,:); % full fov but with reduced background.
mask= Mask1(min(i1):max(i1)+ksize,min(i2)-ksize:max(i2)+ksize,round(nz/2));
figure, myimagesc(dwi(:,:,1,1),mask)

s= rng;
%tmp=repmat(Mask,[1 1 1 numDWI]);
dwi00= squeeze(dwi(:,:,1,:));
% estimate spatially varying nois
% spatial modulation
% - fast variation (gaussian + sine wave modulation)
%clear Sigma_MPPCA* Sigma_VST*

sm1= customgauss([size(dwi,1),size(dwi,2)], round(0.5*size(dwi,2)), round(0.5*size(dwi,2)), 0, 0.2, 1, [1 1]);
sm2= sin(repmat(linspace(-5*pi,5*pi,size(dwi,2)), size(dwi,1),1));
%figure, myimagesc(0.1*sm2+sm1)

sm= sm1+ 0.1*sm2;
sm= sm./max(sm(:));
figure, myimagesc(sm,mask)

%save data_2shell_trimmed dwi dwi00 sm mask nz ksize
%
%
levels=10;%1:1:10;% percent

%parfor idx=1:length(levels)
idx=1;
    level= levels(idx);
    % im_r is the simulated noisy data with varying noise level
    rng(s);
    
    noisemap= level*randn(size(dwi))/100 .* repmat(sm, [1 1 size(dwi,3) size(dwi,4)]);    
    noisemap1= level*randn(size(dwi))/100 .* repmat(sm, [1 1 size(dwi,3) size(dwi,4)]);
    
    Sigma0(:,:,idx)= 0.01* level* sm;
    Sigma1(:,:,idx)= 0.5*(std(noisemap(:,:,:),0,3)+std(noisemap1(:,:,:),0,3) );

    im_r0=sqrt((dwi+noisemap).^2+(noisemap1).^2);
    IM_R(:,:,:,:,idx)= im_r0; % save this for denoising

%end
disp('->done..')
%save -v7.3 data_2shell_noisy_lvl10 dwi dwi00 sm mask nz ksize IM_R Sigma0 Sigma1 levels
%%
if isempty(gcp)
    mypool= parpool(8);
end
%% estimate noise
ks=5; % kernel size
VST_ABC='B';

im_r0= IM_R(:,:,:,:,idx);
    
%     im_r= im_r0;
%     sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; % 
%     Sigma_VST2_all(:,:,idx)= sigma_vst(:,:,round(nz/2));
    
    im_r= im_r0(:,:,:,bvals0>500&bvals0<1500); %b1000
    sigma_vst= estimate_noise_vst3(im_r,ks,VST_ABC) ; 
    % Total elapsed time = 46.579587 min (using 20 workers on atlas13)
    Sigma_VST2_b1k(:,:,idx)= sigma_vst(:,:,round(nz/2));
  
    
  figure, myimagesc(Sigma_VST2_b1k), caxis([0 .1])
  
%     im_r= im_r0(:,:,:,bvals0>1500);
%     sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; % 
%     Sigma_VST2_b2k(:,:,idx)= sigma_vst(:,:,round(nz/2));
%     
%     im_r= im_r0(:,:,:,bvals0>500);
%     sigma_vst= estimate_noise_vst2(im_r,ks,VST_ABC) ; %
%     Sigma_VST2_b1k2k(:,:,idx)= sigma_vst(:,:,round(nz/2));

%% denoise
ws=5;
Sigma_VST= Sigma_VST2_b1k;

Sigma_VST= Sigma0; % for a quick check w/o noise estimation.

% IMVST= IM_R;
%     IMVST(:)=0;
    %parfor idx=1:size(IM_R,5)
        
        im_r= IM_R(:,:,:,:,idx);
        sig= repmat(Sigma_VST(:,:,idx),[1 1 size(im_r,3)]);
        
        % VST
        rimavst= perform_riceVST3(im_r,sig,ws,VST_ABC) ; 
        % Total elapsed time = 4.029004 min 
%        IMVST(:,:,:,:,idx)= rimavst;

        
        % estimate noise
        [im_denoised,sig_mppca]= denoise_mppca3(rimavst,ws);
        % Total elapsed time = 6.345065 min
        IMVSTd_mppca(:,:,:,:,idx)= im_denoised;
        %Sigma_MPPCA{idx}= sig_mppca;
        
       
%    end

% denoise using a standard method

        sig_med= sig_mppca;
        
        % optimal shrinkage
        stepsize= 2;
        [im_denoised,rankmap]= denoise_svs3(rimavst,ksize,stepsize,sig_med,'shrink');
        % Total elapsed time = 2.798461 min
        IMVSTd_shrink(:,:,:,:,idx)= im_denoised;
        
        
%         % optimal hard threshold
%         im_denoised= denoise_svs(rimavst,ksize,1,sig_med,'hard');
%         IMVSTd_hard(:,:,:,:,idx)= im_denoised;
%         
% 

%% EUI VST
sig= repmat(Sigma_VST(:,:,idx),[1 1 nz]);
        
%         % mppca+
%         im_denoised= IMVSTd_mppca(:,:,:,:,idx);
%         im_denoised= perform_riceVST_EUI3(im_denoised,sig,ws,VST_ABC);
%         %Total elapsed time = 3.839819 min
%         IMVSTd_mppca_EUIVST(:,:,:,idx)= squeeze(im_denoised(:,:,round(nz/2),:));
        
        % optimal shrinkage
        im_denoised= IMVSTd_shrink(:,:,:,:,idx);
        im_denoised= perform_riceVST_EUI3(im_denoised,sig,ws,VST_ABC);
        % Total elapsed time = 3.635731 min
        IMVSTd_shrink_EUIVST(:,:,:,idx)= squeeze(im_denoised(:,:,round(nz/2),:));
        
%% mppca
im_r= IM_R(:,:,:,:,idx);
        [im_denoised,sigma_mppca]= MPdenoising(im_r,[],ksize,'full');
        IMd_mppca(:,:,:,idx)= squeeze(im_denoised(:,:,round(nz/2),:));
        Sigma_mppca(:,:,idx)= sigma_mppca(:,:,round(nz/2));

%% display
IMs_r= squeeze(IM_R(:,:,1,:));
mystr={'Ground truth','Noisy','MPPCA','Proposed'};
% b0
ind=1;
ims{1}= dwi00(:,:,ind);
ims{2}= IMs_r(:,:,ind);% highest noise
ims{3}= IMd_mppca(:,:,ind); % mppca
ims{4}= IMVSTd_shrink_EUIVST(:,:,ind); % shrink

figure, position_plots(ims,[1 4],[0 1],[],mask,mystr,'y','gray',1)

% b1000
ind=2;
ims{1}= dwi00(:,:,ind);
ims{2}= IMs_r(:,:,ind);% highest noise
ims{3}= IMd_mppca(:,:,ind); % mppca
ims{4}= IMVSTd_shrink_EUIVST(:,:,ind); % shrink

figure, position_plots(ims,[1 4],[0 .5],[],mask,mystr,'y','gray',1)

% b2000
ind=3;
ims{1}= dwi00(:,:,ind);
ims{2}= IMs_r(:,:,ind);% highest noise
ims{3}= IMd_mppca(:,:,ind); % mppca
ims{4}= IMVSTd_shrink_EUIVST(:,:,ind); % shrink

figure, position_plots(ims,[1 4],[0 .3],[],mask,mystr,'y','gray',1)
