%% Python File Configuration
%   Configures the python environment and file to run as part of the
%   Simulink simulation. Assumes to use of Anaconda or MiniConda on
%   Windows. In Linux the environment needs to be reachable using: 'source
%   activate <ENV_NAME>'. 

%% Python Environment Name
env_name = 'ms-thesis';

%% Python File
% Python file to run in the background from the Simulink simulation. Can be
% an absolute path or a relative path with respect to this config file. 
python_file = '../../code/DHP/dhp_phlab.py';  

%% Config Script
% Constuct path to the python file
python_file = strip(strip(python_file, 'left', '/'), 'left', '\');
python_file_path = join([fileparts(mfilename('fullpath')), '/', python_file]);

% Find the python environment
if ispc
    if exist(join([getenv('HOMEDRIVE'), getenv('HOMEPATH'), '/Anaconda3/envs/', env_name ]), 'dir')
        env_cmd_str = join([getenv('HOMEDRIVE'), getenv('HOMEPATH'), '/Anaconda3/envs/', env_name, '/python ']);
    elseif exist(join([getenv('HOMEDRIVE'), getenv('HOMEPATH'), '/MiniConda3/envs/', env_name ]), 'dir')
        env_cmd_str = join([getenv('HOMEDRIVE'), getenv('HOMEPATH'), '/MiniConda3/envs/', env_name, '/python ']);
    elseif exist(join([getenv('HOMEDRIVE'), getenv('HOMEPATH'), '/Anaconda/envs/', env_name ]), 'dir')
        env_cmd_str = join([getenv('HOMEDRIVE'), getenv('HOMEPATH'), '/Anaconda/envs/', env_name, '/python ']);
    elseif exist(join([getenv('HOMEDRIVE'), getenv('HOMEPATH'), '/MiniConda/envs/', env_name ]), 'dir')
        env_cmd_str = join([getenv('HOMEDRIVE'), getenv('HOMEPATH'), '/MiniConda/envs/', env_name, '/python ']);
    else
        getenv('HOMEPATH')
        error('Python (Anaconda or MiniConda) or environment not found.')
    end 
elseif isunix
    env_cmd_str = join(['source activate', env_name, '&& python ']);
elseif ismac                % Run if on MacOs
    error('MacOS is not yet supported by the configuration script. Please update the python_config.m with a correct command string or buy another computer =).')
end

% Construct system command
cmd_str = join([env_cmd_str, python_file_path, ' &']);

%% Call command in bash or cmd window
system(cmd_str);
