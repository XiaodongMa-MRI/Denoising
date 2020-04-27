function [im_out, rank_out] = denoise_optim_SVShrinkage3 (nim, b, step, Sig)
%
% DENOISE_OPTIM_SVSHRINKAGE3 to denoise using the optimal singular value
% shrinkage. Additionally multiple estimates derived from neighboring patches
% are aggregated for a voxel. The code is modified based on glhosvd.m provided
% by Xinyuan Zhang and the matlab codes by Gavish and Donoho from Stanford.
%
% Usage: [im_out, rank_out] = denoise_optim_SVShrinkage3 (nim, b, step, Sig)
%
% Returns
% -------
% im_out: [x y z M]
%
% rank_out: [x y z], ranks for individual patches after singular value
% shrinkage. 
%
% Expects
% -------
% nim: input image [x y z M]
% 
% b:   (optional)  window size, defaults to 5 ie 5x5x5 kernel
% 
% step: step length between neighboring patches. defaults to 1.
% 
% Sig: noise level [x y z]
%
%
% See also: denoise_mppca denoise_optim_SVHT estimate_noise
%
%
% Copyright (C) 2019 CMRR at UMN
% Author: Xiaoping Wu <xpwu@cmrr.umn.edu> 
% Created: Tue Sep  3 15:02:51 2019
%


time0         =   clock;
if nargin<2
    b             =   5; % block size
end
if nargin<3
    step          =   1; % step length
end

fprintf('--------start denoising--------\n');
%%%The local denoising stage
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

function  [X, W, r]   =   Low_rank_SSC( Y, sig)
% sig: used to determine the optimal shrinkage
[sx,sy,sz,M]= size(Y);
N= sx*sy*sz;
Y = reshape(Y, N, M); 

if M<N
    Y = Y.'; % MxN
end

[X, r]= denoise(Y,sig);

if M<N
    X= X.';
end

X= reshape(X,sx,sy,sz,M);

wei =   1/(1+r);
W   =   wei*ones( size(X) );
X   =   X*wei;
end

function [X, R]= denoise(Y,sig)
% denoising
[M, N]= size(Y);% assuming M<=N
Y= Y./sqrt(N)./sig;

[u, vals, v] = svd(Y, 'econ');
y= diag(vals);
y= optshrink_impl(y,M/N,'fro');
X= u* diag(y)* v';

X= sqrt(N).* sig.* X;
R= length(find(y));

end

%
function singvals = optshrink_impl(singvals,beta,loss)

%y = @(x)( (1+sqrt(beta)).*(x<=beta^0.25) + sqrt((x+1./x) ...
%     .* (x+beta./x)).*(x>(beta^0.25)) );
%     assert(sigma>0)
%     assert(prod(size(sigma))==1)

x = @(y)( sqrt(0.5*((y.^2-beta-1 )+sqrt((y.^2-beta-1).^2 - 4*beta) ))...
    .* (y>=1+sqrt(beta)));

% this is found to be not exactly right for all y's < 1+sqrt(beta).
%opt_fro_shrink = @(y)( sqrt(max(((y.^2-beta-1).^2 - 4*beta),0) ) ./ y);
opt_fro_shrink = @(y)( sqrt((y.^2-beta-1).^2 - 4*beta) ./ y);

opt_op_shrink = @(y)(max(x(y),0));
opt_nuc_shrink = @(y)(max(0, (x(y).^4 - sqrt(beta)*x(y).*y - beta)) ...
    ./((x(y).^2) .* y));

switch loss
    case 'fro'
        %singvals = sigma * opt_fro_shrink(singvals/sigma);
        % to fix the bug:
        singvals1 = opt_fro_shrink(singvals);
        singvals1(singvals<1+sqrt(beta))=0;
        singvals= singvals1;
    case 'nuc'
        %         y = singvals/sigma;
        %         singvals = sigma * opt_nuc_shrink(y);
        singvals = opt_nuc_shrink(y);
        singvals((x(y).^4 - sqrt(beta)*x(y).*y - beta)<=0)=0;
    case 'op'
        %         singvals = sigma * opt_op_shrink(singvals/sigma);
        singvals = opt_op_shrink(singvals);
    otherwise
        error('loss unknown')
end

end

% function lambda_star= calcLambda(beta)
% w = (8 * beta) ./ (beta + 1 + sqrt(beta.^2 + 14 * beta +1));
% lambda_star = sqrt(2 * (beta + 1) + w);
% end

% function omega= calcOmega(beta)
% omega= calcLambda(beta)./ sqrt(MedianMarcenkoPastur(beta));
% end
%
% function omega= approxOmega(beta)
% omega= 0.56*beta^3- 0.95*beta^2+ 1.82*beta+ 1.43;
% end
