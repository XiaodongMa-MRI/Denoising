%----------------------------------------------------
% This code is modified based on the code of SAIST algorithm (Nonlocal image restoration with bilateral variance estimation: a low-rank approach)
% Data: May 6th, 2017
% Author: Xinyuan Zhang (519573769@qq.com)
% Article: Denoise diffusion-weighted images using higher-order singular value decomposition
% parfor supported
%----------------------------------------------------
function  [im_out, rank_out] =denoise_TSVD3(nim,b,step,Sig)
% to denoise using the truncated SVD method.
% Additionally multiple estimates are aggregated for a voxel.
% The code is modified based on glhosvd.m provided
% by Xinyuan Zhang and the matlab codes by Gavish and Donoho from Stanford.
%
% [im_out] =denoise_TSVD(nim,b,step,Sig)
%
%
% See also: denoise_mppca denoise_optim_SVShrinkage estimate_noise

time0         =   clock;
if nargin<2
    b             =   5; % block size
end
if nargin<3
    step          =   1; % step length
end

fprintf('--------start denoising--------\n');
%%%The local mppca denoising stage
[sx,sy,sz,M] = size(nim);
Ys            =   zeros( size(nim) );
W             =   zeros( size(nim) );
R = zeros(sx,sy,sz); % rank

% 
indices_x= [1:step:sx-b sx-b+1];
indices_y= [1:step:sy-b sy-b+1];
indices_z= [1:step:sz-b sz-b+1];

% segment the data for parfor
disp('-> segment data...')
data0= zeros(b,sy,sz,M,length(indices_x));
sigma0= zeros(b,sy,sz,length(indices_x));
for i  =  1:length(indices_x) %[1:step:size(nim,1)-b size(nim,1)-b+1]
    data0(:,:,:,:,i)= nim(indices_x(i): indices_x(i)+b-1, :, :, :);
    sigma0(:,:,:,i)= Sig(indices_x(i):indices_x(i)+b-1, :, :);
end

% denoise
disp('-> denoise...')
Ys0= zeros(b,sy,sz,M,length(indices_x));
W0= Ys0;
R0= zeros(sy,sz,length(indices_x));
        
parfor  i  =  1:length(indices_x) %[1:step:size(nim,1)-b size(nim,1)-b+1]
    
    iB1= data0(:, :, :, :,i);
    iSigma= sigma0(:,:,:,i);
    iYs = zeros(b,sy,sz,M);
    iW= iYs;
    iR= zeros(sy,sz);
        
    for j = indices_y % [1:step:size(nim,2)-b size(nim,2)-b+1]
        
        fprintf('--- denoising: i=%i (%i total), j=%i (%i total) --- \n',i, length(indices_x), j, length(indices_y))
            
        for k = indices_z % [1:step:size(nim,3)-b size(nim,3)-b+1]
            
            B1=iB1(:, j:j+b-1, k:k+b-1, :);            
            sig1= iSigma(:, j:j+b-1, k:k+b-1);
            
            [Ysp, Wp, rp]   =   Low_rank_SSC(double(B1), mean(sig1(:)));
            
            iYs(:,j:j+b-1,k:k+b-1,:)=iYs(:,j:j+b-1,k:k+b-1,:)+ Ysp;
            iW(:,j:j+b-1,k:k+b-1,:)=iW(:,j:j+b-1,k:k+b-1,:)+ Wp;
            iR(j,k)= rp;
            
        end
    end
    
    Ys0(:,:,:,:,i)= iYs;
    W0(:,:,:,:,i)= iW;
    R0(:,:,i)= iR;
    
end

% aggregate data
disp('-> aggregate segmented results...')
for i  =  1:length(indices_x) %[1:step:size(nim,1)-b size(nim,1)-b+1]
    
    Ys(indices_x(i):indices_x(i)+b-1, :, :, :)= Ys(indices_x(i):indices_x(i)+b-1, :, :, :)+ Ys0(:,:,:,:,i);
    W(indices_x(i):indices_x(i)+b-1, :, :, :)= W(indices_x(i):indices_x(i)+b-1, :, :, :)+ W0(:,:,:,:,i);
    R(indices_x(i),:,:)= R0(:,:,i);
    
end

%
im_out  =  Ys./W;
rank_out = R;

%Sigma = SIGs./W1;
fprintf('Total elapsed time = %f min\n\n', (etime(clock,time0)/60) );

end

function  [X, W, r]   =   Low_rank_SSC( Y1, sig1)
% sig1: used to determine the optimal hard threshold
[X, r]= denoise(Y1,sig1);
wei =   1/(1+r);
W   =   wei*ones( size(X) );
X   =   X*wei;
end

function [X, R]= denoise(Y,sig)

[sx,sy,sz,M]= size(Y);
N= sx*sy*sz;% assuming M<=N
Y = reshape(Y, N, M); Y = Y.'; % MxN

Y= Y./sqrt(N)./sig;

[u, vals, v] = svd(Y, 'econ');
y= diag(vals);
y(y<(1+ sqrt(M/N)))= 0;
X= u* diag(y)* v';

X= sqrt(N).*sig.* X;

X= X.';
X= reshape(X,sx,sy,sz,M);

R= length(find(y));

end


