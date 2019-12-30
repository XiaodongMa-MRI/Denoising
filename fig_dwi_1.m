% create dwi and difference images
% !!!!!Note: the highest noise level (1st index) is shown

%ind20=2;
ims{1}= dwi00(:,:,ind);
ims{2}= IM_R(:,:,nzToShow_idx,ind,1);% highest noise
ims{3}= IMd_mppca(:,:,ind,1); % mppca
ims{4}= IMVSTd_mppca_EUIVST(:,:,ind,1); % mppca+
ims{5}= IMVSTd_shrink_EUIVST(:,:,ind,1); % shrink
ims{6}= IMVSTd_hard_EUIVST(:,:,ind,1); % hard
ims{7}= IMVSTd_soft_EUIVST(:,:,ind,1); % soft
ims{8}= IMVSTd_tsvd_EUIVST(:,:,ind,1); % tsvd

%figure, position_plots(ims,[2 .5*length(ims)],[0 .5],[],[],[],[],'gray')

% diff
%sf= 20;
ims1{1}= dwi00(:,:,ind);
ims1{2}= IM_R(:,:,nzToShow_idx,ind,1);% highest noise
ims1{3}= sf*abs(IMd_mppca(:,:,ind,1)-dwi00(:,:,ind)); % mppca
ims1{4}= sf*abs(IMVSTd_mppca_EUIVST(:,:,ind,1)-dwi00(:,:,ind)); % mppca+
ims1{5}= sf*abs(IMVSTd_shrink_EUIVST(:,:,ind,1)-dwi00(:,:,ind)); % shrink
ims1{6}= sf*abs(IMVSTd_hard_EUIVST(:,:,ind,1)-dwi00(:,:,ind)); % hard
ims1{7}= sf*abs(IMVSTd_soft_EUIVST(:,:,ind,1)-dwi00(:,:,ind)); % soft
ims1{8}= sf*abs(IMVSTd_tsvd_EUIVST(:,:,ind,1)-dwi00(:,:,ind)); % tsvd
%figure, position_plots(ims1(3:end),[2 3],[0 .5],[],[],[],[],'gray')

%% combine
[r,c]=size(ims{1});
c0= floor(c/2);

n=1;
ims2{n}= dwi00(:,:,ind);
n=n+1;
% ims2{n}= IMs_r(:,:,ind,end);% highest noise
ims2{n}= IM_R(:,:,nzToShow_idx,ind,1);% highest noise
n=n+1;
ims2{n}= [ims{n}(:,1:c0) ims1{n}(:,c0+1:end)]; % mppca
n=n+1;
ims2{n}= [ims{n}(:,1:c0) ims1{n}(:,c0+1:end)]; % mppca+
n=n+1;
ims2{n}= [ims{n}(:,1:c0) ims1{n}(:,c0+1:end)]; % shrink
n=n+1;
ims2{n}= [ims{n}(:,1:c0) ims1{n}(:,c0+1:end)]; % hard
n=n+1;
ims2{n}= [ims{n}(:,1:c0) ims1{n}(:,c0+1:end)]; % soft
n=n+1;
ims2{n}= [ims{n}(:,1:c0) ims1{n}(:,c0+1:end)]; % tsvd

mystr={'ground truth','noisy','mppca','mppca+','shrink','hard','soft','tsvd'};
%figure, position_plots(ims2,[2 .5*length(ims2)],[0 .5],[],mask,mystr,'y','gray',1)
