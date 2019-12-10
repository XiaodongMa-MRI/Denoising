function Sigma = estimate_noise_vst3 (data, b, VST_ABC)
% ESTIMATE_NOISE_VST3 estimate noise map using riceVST method (ie, variance stabilization
% transform). This is a modification of estimate_noise_vst2.m to enable parfor.
%
% Usage: Sigma = estimate_noise_vst3 (data, b, VST_ABC)
%
% Returns
% -------
% Sigma: [x, y, z] noise map
%
% Expects
% -------
% data: [x, y, z, M] data matrix
%
% b: (optional)  window size, defaults to 5 ie 5x5x5 kernel
%
% VST_ABC: name or filename of variance-stabilizing transform to be used (default='B')
%
%
% See also: estimate_noise
%
%
% Copyright (C) 2019 CMRR at UMN
% Author: Xiaoping Wu <xpwu@cmrr.umn.edu> 
% Created: Tue Sep  3 14:24:25 2019
%

if isa(data,'integer')
    data = single(data);
end
[sx,sy,sz,M] = size(data);

if ~exist('VST_ABC', 'var') || isempty(VST_ABC)
    VST_ABC= 'B';
end

time0         =   clock;
if nargin<2
    b             =   5; % block size
end
%if nargin<3
step          =   1; % step length
%end

fprintf('--------start noise estimation --------\n');
%%%The local mppca denoising stage
Ys            =   zeros( sx,sy,sz );
W             =   Ys;
N= b^3;

len_i= length([1:step:size(data,1)-b size(data,1)-b+1]);
len_j= length([1:step:size(data,2)-b size(data,2)-b+1]);

% segment the data for parfor
disp('-> segment data...')
data0= zeros(b,sy,sz,M,len_i);
for i  =  [1:step:size(data,1)-b size(data,1)-b+1]
    data0(:,:,:,:,i)= data(i:i+b-1, :, :, :);
end

% estimate noise
disp('-> estimate noise...')
Ys0= zeros(b,sy,sz,len_i);
W0= Ys0;
parfor  i  =  [1:step:size(data,1)-b size(data,1)-b+1]
    
    iB1= data0(:, :, :, :,i);
    iYs = zeros(b,sy,sz);
    iW= iYs;
    
    for j = [1:step:size(data,2)-b size(data,2)-b+1]
        
        fprintf('--- i=%i (%i total), j=%i (%i total) --- \n',i, len_i, j, len_j)
        
        for k = [1:step:size(data,3)-b size(data,3)-b+1]
            
            %B1=data(i:i+b-1, j:j+b-1, k:k+b-1, :);
            B1=iB1(:, j:j+b-1, k:k+b-1, :);
            
            sig= riceVST_sigmaEst2(reshape(B1,N,M),0,1,VST_ABC ) ;
            
            %             Ys(i:i+b-1,j:j+b-1,k:k+b-1)=Ys(i:i+b-1,j:j+b-1,k:k+b-1)+ sig;
            %             W(i:i+b-1,j:j+b-1,k:k+b-1)=W(i:i+b-1,j:j+b-1,k:k+b-1)+ 1;
            
            iYs(:,j:j+b-1,k:k+b-1)=iYs(:,j:j+b-1,k:k+b-1)+ sig;
            iW(:,j:j+b-1,k:k+b-1)=iW(:,j:j+b-1,k:k+b-1)+ 1;
            
        end
    end
    
    
    Ys0(:,:,:,i)= iYs;
    W0(:,:,:,i)= iW;
    
end

% aggregate data
disp('-> aggregate segmented noise estimations...')
for i  =  [1:step:size(data,1)-b size(data,1)-b+1]
    
    Ys(i:i+b-1, :, :)= Ys(i:i+b-1, :, :)+ Ys0(:,:,:,i);
    W(i:i+b-1, :, :)= W(i:i+b-1, :, :)+ W0(:,:,:,i);
    
end

%
Sigma  =  Ys./W;
fprintf('Total elapsed time = %f min\n\n', (etime(clock,time0)/60) );

end


