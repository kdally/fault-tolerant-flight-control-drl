
function run_3d_sim(state)

options = simset('SrcWorkspace','current');

simOut = sim('sim_3D.slx',[],options);

end