function position_plots (plots, siz, clrLim, aspect, mask, mystr,mycolor,mycolormap,lw)

% POSITION_PLOTS Position a number of plots using matlab subplot. this is useful
% especially for the case where the images have different dimensions.
%
% Usage: position_plots (plots, siz, clrLim, aspect, mask, mystr, mycolor,mycolormap,lw)
%
%
% Expects
% -------
% plots: 1 x nPlots cell or regular array.
% siz: [nRows nCols]
%
% clrLim: [cmin cmax] or a cell having a color scale for each subplot. if its empty, the min and max of all plots will be
% used for caxis.
%
% aspect: data aspect ratio. Specify the aspect ratio as three relative values
% representing the ratio of the x-, y-, and z-axis scaling (e.g., [1 1 3] means
% one unit in x is equal in length to one unit in y and three units in z).
% when not specified or empty, defaults to [1 1 1]
%
% mask: a roi. 1 x nPlots cell or regular array.
%
% mystr: a cell containing strings that will be used to text individual plots.
% defaults to [].
%
% mycolor: a string, or a RGB 3-element vector, specifying color for the mask. defaults to 'w' for white.
%
% mycolormap: colormap used for plotting. defaults to 'jet'
%
% lw: line width for masking. defaults to 2.
%
% See also: subplot_image myimagesc myMontagemn plot_loop plot_distrib
%
%
% Copyright (C) 2007 by Xiaoping Wu@UMN, Tue Nov 27 13:05:31 2007
%
% Feb 18, 2008
% now caxis limits can be specified.

if nargin < 2
    error('invalid inputs!')
end

if nargin < 3
    clrLim = [];
end

if nargin < 4 || isempty(aspect)
    aspect = [1 1 1];
end

if nargin < 5
    mask = [];
end


if iscell(plots)
    nPlots = length(plots);
else
    nPlots = size(plots,3);
    if isempty(mask)
        mask=false(size(plots));
    elseif size(mask,3)~=nPlots
        mask= repmat(mask,[1 1 nPlots]);
    end
end

if nargin < 6|| isempty(mystr)
    mystr= cell(1,nPlots);
    for ind=1:nPlots,
        mystr{ind}='';
    end
end

if nargin< 7|| isempty(mycolor)
    mycolor='w';
end

if nargin< 8|| isempty(mycolormap)
    mycolormap='jet';
end
if nargin< 9
    lw=2;
end

w = 1/siz(2);
h = 1/siz(1);
Ind = 1:nPlots;
%[I,J] = ind2sub(siz,Ind);
[J,I] = ind2sub([siz(2) siz(1)],Ind);

hdl = Ind;

if iscell(plots) && ~iscell(mask)
    for idx = 1:length(Ind),
        i = I(idx);
        j = J(idx);
        hdl(idx) = subplot('Position',[(j-1)*w (1-i*h) w h]);
        myimagesc(abs(plots{idx}),mask,mycolor,lw); colormap(mycolormap)
        daspect(hdl(idx),aspect);
        text(2,6,mystr{idx},'fontsize',20,'color','w')
        
        if isempty(clrLim)
            caxis auto
            clrLim = [clrLim caxis]; %#ok<AGROW>
        end
    end
elseif iscell(plots) && iscell(mask)
    
    for idx = 1:length(Ind),
        i = I(idx);
        j = J(idx);
        hdl(idx) = subplot('Position',[(j-1)*w (1-i*h) w h]);
        myimagesc(abs(plots{idx}),mask{idx},mycolor,lw);colormap(mycolormap)
        daspect(hdl(idx),aspect);
        text(2,6,mystr{idx},'fontsize',20,'color','w')
        
        if isempty(clrLim)
            caxis auto
            clrLim = [clrLim caxis]; %#ok<AGROW>
        end
    end
    
else
    for idx = 1:length(Ind),
        i = I(idx);
        j = J(idx);
        hdl(idx) = subplot('Position',[(j-1)*w (1-i*h) w h]);
        myimagesc(abs(plots(:,:,idx)),mask(:,:,idx),mycolor,lw);colormap(mycolormap)
        daspect(hdl(idx),aspect);
        text(2,6,mystr{idx},'fontsize',20,'color','w')
        
        if isempty(clrLim)
            caxis auto
            clrLim = [clrLim caxis]; %#ok<AGROW>
        end
    end
end

for idx = 1:length(Ind),
    if iscell(clrLim)
        caxis(hdl(idx),[min(clrLim{idx}) max(clrLim{idx})])
    else
        caxis(hdl(idx),[min(clrLim) max(clrLim)])
    end
end


