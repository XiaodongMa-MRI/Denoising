function im_out = denoise_optim_SVShrinkage3 (nim, b, step, Sig)
%
% DENOISE_OPTIM_SVSHRINKAGE3 to denoise using the optimal singular value
% shrinkage. Additionally multiple estimates derived from neighboring patches
% are aggregated for a voxel. The code is modified based on glhosvd.m provided
% by Xinyuan Zhang and the matlab codes by Gavish and Donoho from Stanford.
%
% Usage: im_out = denoise_optim_SVShrinkage3 (nim, b, step, Sig)
%
% Returns
% -------
% im_out: [x y z M]
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
%%%The local mppca denoising stage
Ys            =   zeros( size(nim) );
W             =   zeros( size(nim) );

len_i= length([1:step:size(nim,1)-b size(nim,1)-b+1]);
len_j= length([1:step:size(nim,2)-b size(nim,2)-b+1]);

% segment the data for parfor
disp('-> segment data...')
[~,sy,sz,M] = size(nim);
data0= zeros(b,sy,sz,M,len_i);
sigma0= zeros(b,sy,sz,len_i);
for i  =  [1:step:size(nim,1)-b size(nim,1)-b+1]
    data0(:,:,:,:,i)= nim(i:i+b-1, :, :, :);
    sigma0(:,:,:,i)= Sig(i:i+b-1, :, :);
end

% denoise
disp('-> denoise...')
Ys0= zeros(b,sy,sz,M,len_i);
W0= Ys0;

parfor  i  =  [1:step:size(nim,1)-b size(nim,1)-b+1]
    
    iB1= data0(:, :, :, :,i);
    iSigma= sigma0(:,:,:,i);
    iYs = zeros(b,sy,sz,M);
    iW= iYs;
    
    for j = [1:step:size(nim,2)-b size(nim,2)-b+1]
        
        fprintf('--- denoising: i=%i (%i total), j=%i (%i total) --- \n',i, len_i, j, len_j)
        
        for k = [1:step:size(nim,3)-b size(nim,3)-b+1]
            
            %B1=nim(i:i+b-1,j:j+b-1,k:k+b-1,:);
            B1=iB1(:, j:j+b-1, k:k+b-1, :);
            
            %sig1= Sig(i:i+b-1,j:j+b-1,k:k+b-1);
            sig1= iSigma(:, j:j+b-1, k:k+b-1);
            
            [Ysp, Wp]   =   Low_rank_SSC(double(B1), mean(sig1(:)));
            
            %             Ys(i:i+b-1,j:j+b-1,k:k+b-1,:)=Ys(i:i+b-1,j:j+b-1,k:k+b-1,:)+Ysp;
            %             W(i:i+b-1,j:j+b-1,k:k+b-1,:)=W(i:i+b-1,j:j+b-1,k:k+b-1,:)+Wp;
            
            iYs(:,j:j+b-1,k:k+b-1,:)=iYs(:,j:j+b-1,k:k+b-1,:)+ Ysp;
            iW(:,j:j+b-1,k:k+b-1,:)=iW(:,j:j+b-1,k:k+b-1,:)+ Wp;
            
        end
    end
    
    Ys0(:,:,:,:,i)= iYs;
    W0(:,:,:,:,i)= iW;
    
end

% aggregate data
disp('-> aggregate segmented results...')
for i  =  [1:step:size(nim,1)-b size(nim,1)-b+1]
    
    Ys(i:i+b-1, :, :, :)= Ys(i:i+b-1, :, :, :)+ Ys0(:,:,:,:,i);
    W(i:i+b-1, :, :, :)= W(i:i+b-1, :, :, :)+ W0(:,:,:,:,i);
    
end

%
im_out  =  Ys./W;
%Sigma = SIGs./W1;
fprintf('Total elapsed time = %f min\n\n', (etime(clock,time0)/60) );

end

function  [X, W]   =   Low_rank_SSC( Y1, sig1)
% sig1: used to determine the optimal shrinkage
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
y= optshrink_impl(y,M/N,'fro');
X= u* diag(y)* v';

X= sqrt(N).* sig.* X;

X= X.';
X= reshape(X,sx,sy,sz,M);

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


