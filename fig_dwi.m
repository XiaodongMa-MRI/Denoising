% create dwi and difference images

%ind=2;
ims{1}= dwi00(:,:,ind);
ims{2}= IMs_r(:,:,ind,end);% highest noise
ims{3}= mppca.IMs_denoised2(:,:,ind,end); % mppca
ims{4}= IMs_denoised1(:,:,ind,end); % mppca+
ims{5}= IMs_denoised22(:,:,ind,end); % shrink
ims{6}= IMs_denoised33(:,:,ind,end); % hard
ims{7}= IMs_denoised44(:,:,ind,end); % soft
ims{8}= IMs_denoised55(:,:,ind,end); % tsvd

%figure, position_plots(ims,[2 .5*length(ims)],[0 .5],[],[],[],[],'gray')

% diff
%sf= 20;
ims1{1}= dwi00(:,:,ind);
ims1{2}= IMs_r(:,:,ind,end);% highest noise
ims1{3}= sf*abs(mppca.IMs_denoised2(:,:,ind,end)-dwi00(:,:,ind)); % mppca
ims1{4}= sf*abs(IMs_denoised1(:,:,ind,end)-dwi00(:,:,ind)); % mppca+
ims1{5}= sf*abs(IMs_denoised22(:,:,ind,end)-dwi00(:,:,ind)); % shrink
ims1{6}= sf*abs(IMs_denoised33(:,:,ind,end)-dwi00(:,:,ind)); % hard
ims1{7}= sf*abs(IMs_denoised44(:,:,ind,end)-dwi00(:,:,ind)); % soft
ims1{8}= sf*abs(IMs_denoised55(:,:,ind,end)-dwi00(:,:,ind)); % tsvd
%figure, position_plots(ims1(3:end),[2 3],[0 .5],[],[],[],[],'gray')

%% combine
[r,c]=size(ims{1});
c0= floor(c/2);

n=1;
ims2{n}= dwi00(:,:,ind);
n=n+1;
ims2{n}= IMs_r(:,:,ind,end);% highest noise
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
