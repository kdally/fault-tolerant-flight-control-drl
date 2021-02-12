%% Example script to visualize the aircraft simulation data
% Add the path of the aircraft_3d_animation function
model_info_file = '3d_models/citation.mat';
% Load the simulation data
fc = '3attitude_step_GT0PLE_normal';
load(strcat('mat_files/',fc,'.mat'))
state_history = state_history';
tout = 0.01:0.01:(size(state_history,1)/100);
tout = tout';
% define the reproduction speed factor
speedx = 1; 
% Do you want to save the animation in a mp4 file? (0.No, 1.Yes)
isave_movie = 1;

% -------------------------------------------------------------------------
% The frame sample time shall be higher than 0.02 seconds to be able to 
% update the figure (CPU/GPU constraints)
frame_sample_time = max(0.02, tout(2)-tout(1));
% Resample the time vector to modify the reproduction speed
t_new   = tout(1):frame_sample_time*(speedx):tout(end);
% Resample the recorded data
y_new   = interp1(tout, state_history, t_new','linear');
% We have to be careful with angles with ranges
% y_new(:, 9)  = atan2(interp1(tout, sin(state_history(:, 9)), t_new','linear'), interp1(tout, cos(state_history(:, 9)), t_new','linear'));
% y_new(:, 8/)  = atan2(interp1(tout, sin(state_history(:, 8)), t_new','linear'), interp1(tout, cos(state_history(:, 8)), t_new','linear'));
% y_new(:, 7)  = atan2(interp1(tout, sin(state_history(:, 7)), t_new','linear'), interp1(tout, cos(state_history(:, 7)), t_new','linear'));
% Assign the data
heading_deg           =  y_new(:, 9);
pitch_deg             =  y_new(:, 8);
bank_deg              =  y_new(:, 7);
angle_of_attack_deg   =  y_new(:, 5);
angle_of_sideslip_deg =  y_new(:, 6);
fligh_path_angle_deg  =  y_new(:, 8)-y_new(:, 6);
altitude_ft           =  y_new(:, 10);

%% Run aircraft_3d_animation function
% ------------------------------------------------------------------------
aircraft_3d_animation(model_info_file,...
    heading_deg, ...            Heading angle [deg]
    pitch_deg, ...              Pitch angle [deg]
    bank_deg, ...               Roll angle [deg]
    angle_of_attack_deg, ...    AoA [deg]
    angle_of_sideslip_deg, ...  AoS [deg]
    fligh_path_angle_deg, ...   Flight path angle [deg]
    altitude_ft, ...            Altitude [ft]
    frame_sample_time, ...      Sample time [sec]
    speedx, ...                 Reproduction speed
    isave_movie, ...            Save the movie? 0-1
    fc);           % Movie file name