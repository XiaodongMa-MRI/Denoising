%%% Generate noisy data of one specific noise level for brain simulation
%%% using data from fiberfox
%%% Note: 3D noise maps are applied, with gaussian modulation in three
%%% dimensions and sinusoid modulation along 2nd dimension.

%% create noisy data with 2 shell b1000 and b2000
s = rng;

% simulate 3D spatially variant random noise to mimic parallel imaging with
% the characteristic of fast variations in x, and slow in y and z
% dimensions.
sm1 = customgauss([size(dwi0,1),size(dwi0,2)], round(0.5*size(dwi0,2)), round(0.5*size(dwi0,2)), 0, 0.2, 1, [1 1]);
sm1 = repmat(sm1,[1 1 size(dwi0,3)]);
sm1_z = customgauss([size(dwi0,1),size(dwi0,3)], round(0.7*size(dwi0,1)), round(0.7*size(dwi0,1)), 0, 0.1, 1, [1 1]);
sm1_z = sm1_z(floor(size(sm1_z,1)/2),:);
sm1 = sm1.*repmat(reshape(sm1_z,[1 1 size(dwi0,3)]),[size(dwi0,1),size(dwi0,2)]);
% add additional sinusoidal variation along x dimension
sm2 = sin(repmat(linspace(-5*pi,5*pi,size(dwi0,2)), size(dwi0,1),1));
sm = sm1 + 0.1*repmat(sm2,[1 1 size(dwi0,3)]);

% normalize the std to 1
sm = sm./max(sm(:));

rng(s);

noisemap  = 0.5*level*randn(size(dwi0))/100 .* repmat(sm, [1 1 1 size(dwi0,4)]);    
noisemap1 = 0.5*level*randn(size(dwi0))/100 .* repmat(sm, [1 1 1 size(dwi0,4)]);

for slice=1:size(dwi0,3)
[phi pt] = genphi(size(dwi0,1),size(dwi0,2),size(dwi0,4));
for pp=1:size(dwi0,4)
phaseall(:,:,slice,pp)=phi(:,:,pp);   %0.5 1 2
end



end

% ground-truth noise std
Sigma0 = 0.01* level* sm*0.5;

% sampled noise std
Sigma1 = std(noisemap,0,4);%Sigma1 = 0.5*(std(noisemap,0,4)+std(noisemap1,0,4) );

% simulated noisy data for denoising
dwi02 =  dwi0.*exp(-1i*phaseall);

dwi0_noisy = dwi02+noisemap + 1i*noisemap1;%dwi0_noisy = sqrt((dwi0+noisemap).^2+(noisemap1).^2);

clear sm sm1 sm1_z sm2 noisemap noisemap1
disp('->done with generating simulated noisy data..')