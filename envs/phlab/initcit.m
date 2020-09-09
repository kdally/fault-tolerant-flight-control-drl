% INITCITATION.M
%
%
%-----------------------------------------------------------------------------
%
% Author : Clark Borst
%
% September 2004
%
% Control and Simulation Division
% Faculty of Aerospace Engineering
% Delft University of Technology
%
%-----------------------------------------------------------------------------
%
% Version : EXPORT (for Analysis), no gear, no wind.

% Load general and Citation data
load ac_genrl;
load citdata;
load jt15data;

% Define variables in models

% logarithmic wind model
Vw915 = 0;
winddir = 45*(pi/180);

% Define Runway Altitude hrunway
%
% Used by 
% * Aerodynamic Ground Effect Model
% * Landing Gear Model

%hrunway=2.5;

trimdatafile = 'citast.tri';

if not(isempty(trimdatafile))
load(trimdatafile,'-mat');
disp(' ');
disp('**********************************');
disp('*    Citation Library            *');
disp('*                                *');
disp('* Delft University of Technology *');
disp('* Clark Borst, 2004              *');
disp('**********************************');
disp(' ');
disp(['Trim condition  : ' trimdatafile]);
disp(['Aircraft weight : ' num2str(massinit(1)) ' kg']);
disp(['Altitude        : ' num2str(x0(10)) ' m']);
disp(['Speed Vtas      : ' num2str(x0(4)) ' m/s']);
disp(['Gamma           : ' num2str(round((x0(8)-x0(5))/pi*180)) ' deg']);
disp(['Power Lever     : ' num2str(ut0(1)) ]);
disp(' ');
end

gear_params;

