function [sys,x0,str,ts] = cessnac550_3d_sfnc(t,x,u,flag)
%3D-Display S-function for making aircraft animation.
%

%
% Joao Oliveira, 2005
%

% Plots every major integration step, but has no states of its own
switch flag,

  %%%%%%%%%%%%%%%%%%
  % Initialization %
  %%%%%%%%%%%%%%%%%%
  case 0,
    [sys,x0,str,ts] = mdlInitializeSizes;

  %%%%%%%%%%
  % Update %
  %%%%%%%%%%
  case 2,
    sys = mdlUpdate(t,x,u);

  %%%%%%%%%%
  % Output %
  %%%%%%%%%%
  case 3
    sys = mdlOutputs(t,x,u); 

  %%%%%%%%%%%%%%%%
  % Unused flags %
  %%%%%%%%%%%%%%%%
  case { 1, 4, 9 },
    sys = [];
    
  %%%%%%%%%%%%%%%
  % DeleteBlock %
  %%%%%%%%%%%%%%%
  case 'DeleteBlock',
    sys = LocalDeleteBlock;
    
  %%%%%%%%%%%%%%%
  % DeleteFigure %
  %%%%%%%%%%%%%%%
  case 'DeleteFigure',
    sys = LocalDeleteFigure;
  
  %%%%%%%%%
  % Close %
  %%%%%%%%%
  case 'Close',
    sys = LocalClose;
     
  %%%%%%%%%%%%%%%%%%%%
  % Unexpected flags %
  %%%%%%%%%%%%%%%%%%%%
  otherwise
    error(['Unhandled flag = ',num2str(flag)]);
end

% end cessnac550_3d_sfnc

%
%=============================================================================
% mdlInitializeSizes
% Return the sizes, initial conditions, and sample times for the S-function.
%=============================================================================
%
function [sys,x0,str,ts]=mdlInitializeSizes

%
% call simsizes for a sizes structure, fill it in and convert it to a
% sizes array.
%
sizes = simsizes;

sizes.NumContStates  = 0;
sizes.NumDiscStates  = 0;
sizes.NumOutputs     = 1;
sizes.NumInputs      = 6;
sizes.DirFeedthrough = 1;
sizes.NumSampleTimes = 1;

sys = simsizes(sizes);

%
% initialize the initial conditions
%
x0  = [];

%
% str is always an empty matrix
%
str = [];

%
% initialize the array of sample times, for the aircraft demo,
% the animation is updated every 0.1 seconds
%
ts  = [0 0];

%
% create the figure, if necessary
%
LocalAircraftInit;

% end mdlInitializeSizes

%
%=============================================================================
% mdlUpdate
% Update the aircraft animation.
%=============================================================================
%
function sys = mdlUpdate(t,x,u)

persistent count

par = get_param(gcbh,'UserData');
Fig = par(1);
if ishandle(Fig)&~mod(count,10),
   
   roll  = u(1);
   pitch = u(2);
   yaw   = u(3);
   x     = u(4);
   y     = u(5);
   z     = u(6);
   Z   = [  0,  0,  0 ];
   P   = [  x;  y;  z ];
   Up  = [  0;  0;  1 ];
   Rob = [ 100;  0;  0 ];
   Cob = get_Cnb(roll,pitch,yaw);
   Cbo = Cob';
   CameraPosition = Cbo*Rob;
   CameraUpVector = Cbo*Up;
   CameraTarget   = Cbo*(P);
   set(0,'CurrentFigure',Fig);
   %figure(Fig);
   set(gca,'CameraTarget',CameraTarget);
   set(gca,'CameraPosition',CameraPosition);
   set(gca,'CameraUpVector',CameraUpVector);
   %view(Tob)
   
   count = count + 1;
else
    count = 0;
end

sys = [];
% end mdlUpdate

%
%=============================================================================
% mdlOutputs
% Return the output vector for the S-function
%=============================================================================
%
function sys = mdlOutputs(t,x,u)

par = get_param(gcbh,'UserData');
sys = par(2);


% end mdlOutputs


%
%=============================================================================
% LocalDeleteBlock
% The animation block is being deleted, delete the associated figure.
%=============================================================================
%
function sys = LocalDeleteBlock

par = get_param(gcbh,'UserData');
fig = par(1);
if ishandle(fig),
  delete(fig);
  set_param(gcbh,'UserData',[0,-1])
end

sys = [];

% end LocalDeleteBlock

%
%=============================================================================
% LocalDeleteFigure
% The animation figure is being deleted, set the S-function UserData to -1.
%=============================================================================
%
function sys = LocalDeleteFigure

ud = get(gcbf,'UserData');
set_param(ud.Block,'UserData',[0,-1]);
  
sys = [];

% end LocalDeleteFigure

%
%=============================================================================
% LocalClose
% The callback function for the animation window close button.  Delete
% the animation figure window.
%=============================================================================
%
function sys = LocalClose

if ~isempty(gcbf)
    ud = get(gcbf,'UserData');
    set_param(ud.Block,'UserData',-1);
    delete(gcbf);
else
    delete(gcf);
end
sys = [];

% end LocalClose


%
%=============================================================================
% LocalAircraftSets
%=============================================================================
%
function LocalAircraftSets(time,ud,u)

disp('LocalAircraftSets')

% end LocalAircraftSets

%
%=============================================================================
% LocalAircraftInit
% Local function to initialize the aircraft animation.  If the animation
% window already exists, it is brought to the front.  Otherwise, a new
% figure window is created.
%=============================================================================
%
function LocalAircraftInit

%
% The name of the reference is derived from the name of the
% subsystem block that owns the aircraft animation S-function block.
% This subsystem is the current system and is assumed to be the same
% layer at which the reference block resides.
%

%
% The animation figure handle is stored in the 3D display block's UserData.
% If it exists, initialize the reference states roll/pitch/yaw.
%
par = get_param(gcbh,'UserData');
if isempty(par)
   Fig = 0;
else
   Fig = par(1);
end
if ishandle(Fig) & Fig,
  FigUD = get(Fig,'UserData');
      
  %
  % bring it to the front
  %
  figure(Fig);
  return
end

%
% the animation figure doesn't exist, create a new one and store its
% handle in the animation block's UserData
%
%=========================
xplane = [];
yplane = [];
zplane = [];
R = [ acos([0;0.4;0.8;1]) ];
[x,y,z] = cylinder(R,10);
z(end,:) = [3.*z(end,:) + 4.*z(end-1,:)]./7;
aux=x;x=-z;z=aux;
x = x./max(max(abs(x)));
y = y./max(max(abs(y)));
z = z./max(max(abs(z)));
x = flipud(x);
y = flipud(y);
z = flipud(z);
xnose = x.*2.50 + 3.00;
ynose = y.*0.80 + 0.00;
znose = z.*0.85 + 3.15;
xplane = [ xplane; xnose ];
yplane = [ yplane; ynose ];
zplane = [ zplane; znose ];
xfus1 = x(end,:).*0.00 + 8.5;
yfus1 = ynose(end,:);
zfus1 = znose(end,:);
xplane = [ xplane; xfus1 ];
yplane = [ yplane; yfus1 ];
zplane = [ zplane; zfus1 ];
xfus2 = x(end,:).*0.00 + 12.5;
yfus2 = y(end,:).*0.30;
zfus2 = z(end,:).*0.55 + 3.7;
xplane = [ xplane; xfus2 ];
yplane = [ yplane; yfus2 ];
zplane = [ zplane; zfus2 ];
xtail = [13.8; 14.1; 14.8]*ones(size(x(end,:)));
ytail = [y(end,:).*0.10 + 0.00; y(end,:).*0.10 + 0.00; y(end,:).*0.00 + 0.00]; 
ztail = [z(end,:).*1.45 + 4.85; z(end,:).*1.40 + 4.90; z(end,:).*0.00 + 6.30];
xplane = [ xplane; xtail ];
yplane = [ yplane; ytail ];
zplane = [ zplane; ztail ];
xwing = [ ...
      7.0   7.25 8.0  7.25 7.0; ...
      6.0   7.0  9.5 7.0  6.0; ...
      7.0   7.25 8.0  7.25 7.0; ...
   ];
ywing = [ ...
      -8.0 -8.0 -8.0 -8.0 -8.0; ...
      0.0   0.0  0.0  0.0  0.0; ...
      8.0   8.0  8.0  8.0  8.0];
zwing = 0.2 + [ ...
      3.0   3.0  3.0  3.0  3.0; ...
      2.6   2.55 2.4  2.45 2.6; ...
      3.0   3.0  3.0  3.0  3.0; ...
      ];
xstab = (xwing - 7.0).*(1.5/3.5) + 13;
ystab = (ywing - 0.0).*(3.0/8.0) +  0;
zstab = [ ...
      4.5   4.5  4.5  4.5  4.5; ...
      4.1   4.1  4.1  4.1  4.1; ...
      4.5   4.5  4.5  4.5  4.5; ...
   ];

x_cg   = 7;
xstab  = xstab  - x_cg;
xwing  = xwing  - x_cg;
xplane = xplane - x_cg;
y_cg   = 0;
ystab  = ystab  - y_cg;
ywing  = ywing  - y_cg;
yplane = yplane - y_cg;
z_cg   = 3.2;
zstab  = zstab  - z_cg;
zwing  = zwing  - z_cg;
zplane = zplane - z_cg;

%=========================
FigureName = 'Cessna Citation II Visualization';
Fig = double(figure(...
  'Units',           'pixel',...
  'Position',        [780 0 500 500],...
  'Toolbar',         'none',...
  'Name',            FigureName,...
  'NumberTitle',     'off',...
  'IntegerHandle',   'off',...
  'HandleVisibility','on', ... 'callback',...
  'Resize',          'off',...
  'DeleteFcn',       'cessnac550_3d_sfnc([],[],[],''DeleteFigure'')',...
  'CloseRequestFcn', 'cessnac550_3d_sfnc([],[],[],''Close'');'));
%=========================

surf(xstab,ystab,zstab,ones(size(xwing)))
hold on
surf(xwing,ywing,zwing,ones(size(xwing)))
surf(xplane,yplane,zplane,ones(size(xplane)))
light
axis image
axis off
set(gcf,'DoubleBuffer','on')
set(gca,'DrawMode','fast')
%set(gca,'NextPlot','replace')
set(gca,'CameraTargetMode','manual')
set(gca,'CameraViewAngleMode','manual')
set(gca,'YDir','reverse')
map = [ ...
      0    0    1; ...
      0    0    1; ...
      1    1    0; ...
      ];
colormap(map)
hidden
shading flat
%=========================
% uicontrol(...
%   'Parent',  Fig,...
%   'Style',   'pushbutton',...
%   'Position',[415 15 70 20],...
%   'String',  'Close', ...
%   'Callback','cessnac550_3d_sfnc([],[],[],''Close'');');

%
% all the HG objects are created, store them into the Figure's UserData
%
FigUD.Block = get_param(gcbh,'Handle');
set(Fig,'UserData',FigUD);

drawnow

%
% store the figure handle and output state 0 in the animation block's UserData
%
x = 0;
set_param(gcbh,'UserData',[Fig,x]);

% end LocalAircraftInit
