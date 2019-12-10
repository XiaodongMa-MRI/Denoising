function [im_out, SigmaMPPCA] = denoise_mppca3 (nim, b, step)
% 
% DENOISE_MPPCA3 to denoise using the extended MPPCA where multiple estimates
% derived from neighboring patches are aggregated for a voxel. The code is
% basically a wrapper of the MPPCA approach (MPdenoising.m) introduced by
% Veraart et al MRM/NI 2016, and is created based on glhosvd.m provided by
% Xinyuan Zhang (Zhang et al. NI 2017). This is basically a modification of
% denoise_mppca.m to enable parfor.
%
% Usage: [im_out, SigmaMPPCA] = denoise_mppca3 (nim, b, step)
%
% Returns
% -------
% im_out: denoised image [x y z M]
% 
% SigmaMPPCA: estimated noise [x y z]
%
% Expects
% -------
% nim: input noisy image [x y z M]
% 
% b:   (optional)  window size, defaults to 5 ie 5x5x5 kernel
% 
% step: step length between neighboring patches. defaults to 1.
%
%
% See also: estimate_noise denoise_mppca.m
%
%
% Copyright (C) 2019 CMRR at UMN
% Author: Xiaoping Wu <xpwu@cmrr.umn.edu> 
% Created: Tue Sep  3 14:40:55 2019
%


time0         =   clock;
if nargin<2
    b             =   5; % block size
end
if nargin<3
    step          =   1; % step length
end

fprintf('--------start denoising--------\n');
%%%The local mppca denoising stage
Ys            =   zeros( size(nim) );
W             =   zeros( size(nim) );
SIGs= W(:,:,:,1);
W1= SIGs;

len_i= length([1:step:size(nim,1)-b size(nim,1)-b+1]);
len_j= length([1:step:size(nim,2)-b size(nim,2)-b+1]);

% segment the data for parfor
disp('-> segment data...')
[~,sy,sz,M] = size(nim);
data0= zeros(b,sy,sz,M,len_i);
for i  =  [1:step:size(nim,1)-b size(nim,1)-b+1]
    data0(:,:,:,:,i)= nim(i:i+b-1, :, :, :);
end

% denoise
disp('-> denoise...')
Ys0= zeros(b,sy,sz,M,len_i);
W0= Ys0;
SIGs0= zeros(b,sy,sz,len_i);
W10= SIGs0;

parfor  i  =  [1:step:size(nim,1)-b size(nim,1)-b+1]
    
    iB1= data0(:, :, :, :,i);
    iYs = zeros(b,sy,sz,M);
    iW= iYs;
    iSIGs= zeros(b,sy,sz);
    iW1= iSIGs;
    
    for j = [1:step:size(nim,2)-b size(nim,2)-b+1]
        
        fprintf('--- denoising: i=%i (%i total), j=%i (%i total) --- \n',i, len_i, j, len_j)
        
        for k = [1:step:size(nim,3)-b size(nim,3)-b+1]
            
            %B1=nim(i:i+b-1,j:j+b-1,k:k+b-1,:);
            B1=iB1(:, j:j+b-1, k:k+b-1, :);
            
            [Ysp, Wp, sig]   =   Low_rank_SSC(double(B1));
            
            %                 Ys(i:i+b-1,j:j+b-1,k:k+b-1,:)=Ys(i:i+b-1,j:j+b-1,k:k+b-1,:)+Ysp;
            %                 W(i:i+b-1,j:j+b-1,k:k+b-1,:)=W(i:i+b-1,j:j+b-1,k:k+b-1,:)+Wp;
            
            iYs(:,j:j+b-1,k:k+b-1,:)=iYs(:,j:j+b-1,k:k+b-1,:)+ Ysp;
            iW(:,j:j+b-1,k:k+b-1,:)=iW(:,j:j+b-1,k:k+b-1,:)+ Wp;
            
            iSIGs(:,j:j+b-1,k:k+b-1)=iSIGs(:,j:j+b-1,k:k+b-1)+sig;
            iW1(:,j:j+b-1,k:k+b-1)=iW1(:,j:j+b-1,k:k+b-1)+1;
            
        end
    end
    
    Ys0(:,:,:,:,i)= iYs;
    W0(:,:,:,:,i)= iW;
    SIGs0(:,:,:,i)= iSIGs;
    W10(:,:,:,i)= iW1;
    
end

% aggregate data
disp('-> aggregate segmented results...')
for i  =  [1:step:size(nim,1)-b size(nim,1)-b+1]
    
    Ys(i:i+b-1, :, :, :)= Ys(i:i+b-1, :, :, :)+ Ys0(:,:,:,:,i);
    W(i:i+b-1, :, :, :)= W(i:i+b-1, :, :, :)+ W0(:,:,:,:,i);
    
    SIGs(i:i+b-1, :, :)= SIGs(i:i+b-1, :, :)+ SIGs0(:,:,:,i);
    W1(i:i+b-1, :, :)= W1(i:i+b-1, :, :)+ W10(:,:,:,i);
end

%
im_out  =  Ys./W;
SigmaMPPCA = SIGs./W1;
fprintf('Total elapsed time = %f min\n\n', (etime(clock,time0)/60) );

end

function  [X, W, sig]   =   Low_rank_SSC( Y1)

siz=size(Y1);
[X, Sig, R]= MPdenoising(Y1,[],siz(1:3),'fast');
r= R(R>0);
sig= Sig(Sig>0);
%   [Sigma2 U1] = hosvd2(full(Y1),full(Y2));
%
%   Sigma2(abs(Sigma2) < sigmah*sqrt( 2*log(length(Y1(:))) ) )=0;
%   r   =   sum( abs(Sigma2(:))>0 );
%   X   =   tprod(Sigma2, U1);
if isempty(r)
    r=0;
end
wei =   1/(1+r);
W   =   wei*ones( size(X) );
X   =   X*wei;
end

