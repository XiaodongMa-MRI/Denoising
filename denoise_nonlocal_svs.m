
function denoise_image_all=denoise_nonlocal_svs(KSP2,nordic,kernel)



mask_all = zeros(size(KSP2));
denoise_image_all = zeros(size(KSP2));

for slice=floor(kernel/2)+1:1:size(KSP2,3)-floor(kernel/2)
im_r= squeeze(KSP2(:,:,slice-floor(kernel/2):slice+floor(kernel/2),:));
%
%    im_r(:,:,1:34)= real(squeeze(dwi0_noisy(:,:,slice,1:34)));
% im_r(:,:,35:68) =imag(squeeze(dwi0_noisy(:,:,slice,1:34)));
%im_r2=zeros(250,270,32); im_r2(:,:,:)=im_r(11:260,6:275,:);
%im_r2=im_r;
%% denoise with proposed framework
% VST
rimavst= im_r;
x_image=rimavst;
% denoise using optimal shrinkage
for iter=1:1
% tic
y_image=x_image+0.2*(rimavst-x_image);
noisy_image=y_image;
step=1;
dir=size(im_r,4);
numm=0;
for i=1:step:size(noisy_image,1)-kernel+1
for j=1:step:size(noisy_image,2)-kernel+1
numm=numm+1;
end
end
noisy_patch2=zeros(kernel,kernel,kernel,dir,numm);
noisy_patch3=zeros(kernel,kernel,kernel,dir,numm);
numm=0;
for i=1:step:size(noisy_image,1)-kernel+1
for j=1:step:size(noisy_image,2)-kernel+1
numm=numm+1;
noisy_patch2(:,:,:,:,numm)=squeeze(noisy_image(i:i+kernel-1,j:j+kernel-1,:,:));
end
end
noisy_patch2=reshape(noisy_patch2,[kernel*kernel*kernel*dir,numm]);
numm=0;
for i=1:step:size(noisy_image,1)-kernel+1
for j=1:step:size(noisy_image,2)-kernel+1
numm=numm+1;
noisy_patch3(:,:,:,:,numm)=squeeze(nordic(i:i+kernel-1,j:j+kernel-1,slice-floor(kernel/2):slice+floor(kernel/2),:));
end
end
noisy_patch3=reshape(noisy_patch3,[kernel*kernel*kernel*dir,numm]);
D=pdist2(noisy_patch3',noisy_patch3');
k=140;
denoise_patch=[];
denoise_image=zeros(size(noisy_image));
mask=zeros(size(noisy_image));
numm2=0;
for i=1:step:size(noisy_image,1)-kernel+1
for j=1:step:size(noisy_image,2)-kernel+1
numm2=numm2+1;

[disall,numberall] = sort(D(:,numm2));

first_dis = disall(2);
k = min(sum(disall<first_dis*1.2),140);

[dis,number] = sort(D(:,numm2));
sorted=number(1:1+k);
sorted_patches=squeeze(noisy_patch2(:,sorted));
sorted_patches=reshape(sorted_patches,[kernel,kernel,kernel,dir,k+1]);
sorted_patches=permute(sorted_patches,[1 2 4 3 5]);
sorted_patches=reshape(sorted_patches,[kernel*kernel*dir,kernel*(k+1)]);
tempsorted_patches=sorted_patches;
sigg=1;
M= min(size(tempsorted_patches));% assuming M<=N
N= max(size(tempsorted_patches));% assuming M<=N
flag=0;
if size(tempsorted_patches,1)> size(tempsorted_patches,2)
flag=1;
tempsorted_patches=tempsorted_patches';
end
tempsorted_patches= tempsorted_patches./sqrt(N)./sigg;
[u,sigma,v]=svd(tempsorted_patches, 'econ');
sigmas= diag(sigma);
sigmas= optshrink_impl(sigmas,M/N,'fro');
R= length(find(sigmas));


temp= u(:,:) * diag(sigmas) * v(:,:)';
temp= sqrt(N).* sigg.* temp;
if flag==1
temp=temp';
end

denoise_patch(:,1:kernel)=temp(:,1:kernel);
tempp=reshape(denoise_patch(:,1:kernel),[kernel,kernel,dir,kernel]);
tempp=permute(tempp,[1 2 4 3]);
denoise_image(i:i+kernel-1,j:j+kernel-1,:,:)=  denoise_image(i:i+kernel-1,j:j+kernel-1,:,:)+tempp(1:kernel,1:kernel,:,:)*1;
mask(i:i+kernel-1,j:j+kernel-1,:,:)=mask(i:i+kernel-1,j:j+kernel-1,:,:)+1;
end
end
denoise_image=denoise_image./(eps+mask);

%clear D
end
denoise_image_all(:,:,slice-floor(kernel/2):slice+floor(kernel/2),:) = denoise_image_all(:,:,slice-floor(kernel/2):slice+floor(kernel/2),:) +denoise_image;
mask_all(:,:,slice-floor(kernel/2):slice+floor(kernel/2),:) =mask_all(:,:,slice-floor(kernel/2):slice+floor(kernel/2),:) +1;

end
denoise_image_all=denoise_image_all./(eps+mask_all);

end