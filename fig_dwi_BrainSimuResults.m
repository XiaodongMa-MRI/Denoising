
% ind=2;
ims{1}= dwi00(:,:,ind);
ims{2}= IMs_r(:,:,ind,end);% highest noise
ims{3}= IMs_denoised2(:,:,ind,end); % mppca
ims{4}= IMs_denoised1(:,:,ind,end); % shrink

%figure, position_plots(ims,[2 .5*length(ims)],[0 .5],[],[],[],[],'gray')

% diff
%sf= 20;
ims1{1}= dwi00(:,:,ind);
ims1{2}= sf*abs(IMs_r(:,:,ind,end)-dwi00(:,:,ind));% highest noise
ims1{3}= sf*abs(IMs_denoised2(:,:,ind,end)-dwi00(:,:,ind)); % mppca
ims1{4}= sf*abs(IMs_denoised1(:,:,ind,end)-dwi00(:,:,ind)); % shrink
%figure, position_plots(ims1(3:end),[2 3],[0 .5],[],[],[],[],'gray')

% combine

[r,c]=size(ims{1});
c0= floor(c/2);

n=1;
ims2{n}= dwi00(:,:,ind);
n=n+1;
% ims2{n}= IMs_r(:,:,ind,end);% highest noise
ims2{n}= [ims{n}(:,1:c0) ims1{n}(:,c0+1:end)];% highest noise
n=n+1;
ims2{n}= [ims{n}(:,1:c0) ims1{n}(:,c0+1:end)]; % mppca
n=n+1;
ims2{n}= [ims{n}(:,1:c0) ims1{n}(:,c0+1:end)]; % shrink

% mystr={'ground truth','noisy','mppca','mppca+','shrink','hard','soft','tsvd'};
%
