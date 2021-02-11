

load 'mat_files/3attitude_step_GT0PLE_normal.mat';
time = 0:0.01:(120-0.01);


% load 'mat_files/altitude_2attitude_P7V00G_normal.mat';
% load 'mat_files/altitude_2pitch_PZ5QGW_5K6QFG_cg.mat';
% load 'mat_files/altitude_2pitch_PZ5QGW_9MUWUB_ice.mat';
% load 'mat_files/altitude_2pitch_PZ5QGW_GT0PLE_normal_noise.mat';
% load 'mat_files/altitude_2pitch_PZ5QGW_GT0PLE_normal.mat';
% load 'mat_files/altitude_2pitch_PZ5QGW_HNAKCC_dr.mat';
% load 'mat_files/altitude_2pitch_PZ5QGW_R0EV0U_ht.mat';
% time = 0:0.01:(120-0.01);

ts = timeseries(state_history',time);

ts.TimeInfo.Increment = 0.01;

options = simset('SrcWorkspace','current');

simOut = sim('sim_3D.slx',[],options);


