function [im_out, rank_out] = denoise_svs3 (nim, b, step, Sig, whichmethod)
% 
% DENOISE_SVS3 to denoise using the optimal singular value shrinkage methods
% including truncated SVD, optimal soft and hard thresholding, optimal
% shrinkage. Additionally multiple estimates derived from neighboring patches
% are aggregated for a voxel. The code is modified based on glhosvd.m provided
% by Xinyuan Zhang and the matlab codes by Gavish and Donoho from Stanford.
%
% This is basically a modification of denoise_svs.m to enable parfor.
%
% Usage: [im_out, rank_out] = denoise_svs3 (nim, b, step, Sig, whichmethod)
%
% Returns
% -------
% im_out: denoised image [x y z M]
%
% rank_out: [x y z], ranks for individual patches after singular value
% manipulation. 
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
% whichmethod: which method for shrinkage.
% could be one of the following:
% 'tsvd': conventional truncated svd
% 'soft': optimal soft thresholding
% 'hard': optimal hard thresholding
%  'shrink': optimal shrinkage.
%
%
% See also: denoise_mppca3 estimate_noise denoise_optim*3
%
%
% Copyright (C) 2019 CMRR at UMN
% Author: Xiaoping Wu <xpwu@cmrr.umn.edu> 
% Created: Tue Sep  3 14:56:50 2019
%


% time0         =   clock;
if nargin<2
    b             =   5; % block size
end
if nargin<3
    step          =   1; % step length
end

rank_out= zeros(size(Sig));

switch whichmethod
    case 'tsvd'
        im_out =denoise_TSVD3(nim,b,step,Sig);
    case 'soft'
        im_out =denoise_optim_SVST3(nim,b,step,Sig);
    case 'hard'
        im_out =denoise_optim_SVHT3(nim,b,step,Sig);
    case 'shrink'
        [im_out, rank_out] =denoise_optim_SVShrinkage3(nim,b,step,Sig);
    otherwise
        
        error('-> Denoising method specified is not supported...')
        
end

end


