% gear_params.m
%
% Cessna Citation 500
% Landing Gear Parameters
%
% Based on Fortran Routine flag6m.f NLR
%
% (c) A.R. Veldhuijzen September 2001

% Definitions
% 1	Left Main Landing Gear
% 2 	Right Main Landing gear
% 3	Nose Gear
% 4	Tail Strike Area

% Correction for CG

xcg	= massinit(2);
ycg	= massinit(3);
zcg   = massinit(4);


% Damping coefficients

cv1	= 20000;
cv2	= cv1;
cv3	= 4000;
cv4	= 20000;

% Strut Preloads
P1		= 2000;
P2		= P1;
P3		= 2550;
P4		= 40000;

% Maximum Strut Deflection
deltagmax1	=	0.25;
deltagmax2	=	deltagmax1;
deltagmax3	=	0.24;
deltagmax4  =  0.02;

% Maximum Tire Deflection;
% (Lecture Notes Simulation Model}
%deltatp1		= 0.01;
%deltatp2		= deltatp1;
%deltatp3		= deltatp1;

% Tire Parameters
% (NLR Model)

kt1	=525020;
kt2	=kt1;
kt3	=457050;


% Landing Gear posititions

xl1	= xcg-6.858;
yl1	= 1.918;
zl1	= zcg-1.701;

xl2	= xcg-6.858;
yl2	= -1.918;
zl2	= zcg-1.701;

xl3	= xcg-2.0752;
yl3	= 0;
zl3	= zcg-1.7488;

xl4	= xcg-12.742;
yl4	= 0;
zl4	= zcg-2.78;

% Brake and Roll coefficients

murol	= 0.03*10; %0.0213;
