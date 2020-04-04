function myimagesc (img, roi, mycolor, lw)

% MYIMAGESC display an image with some configs. If roi is specified, the boundaries of
% roi are shown over the image.
%
% Usage: myimagesc (img) or myimagesc (img, roi, mycolor, lw)
%
%
% Expects
% -------
% img: image to display 
% roi: contains one or more than one region of interest. defaults to empty
% 
% mycolor: a char, or a RGB vec for color of the roi boundary, it defaults to 'w' for white.
%
% lw: linewidth for the roi. defaults to 4.
% 
% See also myimagesc2
% 

if nargin < 3 || isempty(mycolor)
    mycolor = 'w';
end
if nargin < 4
    lw=4;
end

% if ~ischar(mycolor)
%     error('color should be specified in char.')
% end

if nargin < 2
  roi = [];
end


  if ndims(img)~=2
      error('The number of image dimensions should be 2!')
  end
  
  imagesc(img)
  axis image
  axis off
  colormap jet
  
  if ~isempty(roi)
    B = bwboundaries(roi);                % the boundaries of the roi.
  
    hold on;
    for k = 1:length(B),
      boundary = B{k};
      plot(boundary(:,2),boundary(:,1),...
          'color',mycolor,'linestyle','-','Linewidth',lw);
    end
    hold off
  end

  
