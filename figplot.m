%% create plot using plotPub.m

%opt.Markers={'o','x','^'};
opt.LineWidth=[2,2,2];

if ~isfield(opt,'Colors')
opt.Colors=[0,1,0;1,0,0;0,0,0;0,0,1;...
    0.5,0.5,0.5;1,0.5,0.5;0.5,1,0.5;0.5,0.5,1;...
    0.25,0.25,0.25;1,0.25,0.25;0.25,1,0.25;0.25,0.25,1;...
    0.75,0.75,0.75;1,0.75,0.75;0.75,1,0.75;0.75,0.75,1];end
%opt.XLabel='Time (ms)';
%opt.YLabel='Ampl. (a.u.)';
%opt.YLim=[-81 81];

%maxBoxDim=5;

%% Start point
%load ptxsol_example

%max_x= ceil(max(x));

%t= (0:length(g)-1)*dt*1000;

opt.BoxDim=maxBoxDim.*[1 1];
if ~isfield(opt,'XLim')
opt.XLim=[0 ceil(max(X{1}))];end

% X{1}=t;
% Y{1}=g;

% opt.FileName='ptx_example.eps';
% opt.YLabel='Gradient (mT/m)';
plotPub(X,Y,length(X),opt)
