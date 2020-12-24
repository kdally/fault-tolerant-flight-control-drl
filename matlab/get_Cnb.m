function Cnb = Get_Cnb(phi,theta,psi)
% Build the Ref Frame Rotation Matrix from Body to NED: NED (n) <= Body (b) : (Cnb)
%
% function Cnb = Get_Cnb(phi,theta,psi)
%
% Cnb = Cbn'
%

%-------------------------------------------------------------------------------------
% Set trigonometric quantities
cos_phi    = cos(phi);
cos_theta  = cos(theta);
cos_psi    = cos(psi);
sin_phi    = sin(phi);
sin_theta  = sin(theta);
sin_psi    = sin(psi);

%-------------------------------------------------------------------------------------
% Cnb
%
% Build the Ref Frame Rotation Matrix from Body to NED: NED (n) <= Body (b) : (Cnb)
Cnb = [ ...
      cos_theta*cos_psi, sin_phi*sin_theta*cos_psi-cos_phi*sin_psi, cos_phi*sin_theta*cos_psi+sin_phi*sin_psi; ...
      cos_theta*sin_psi, sin_phi*sin_theta*sin_psi+cos_phi*cos_psi, cos_phi*sin_theta*sin_psi-sin_phi*cos_psi; ...
      -sin_theta,        sin_phi*cos_theta,                         cos_phi*cos_theta];

return