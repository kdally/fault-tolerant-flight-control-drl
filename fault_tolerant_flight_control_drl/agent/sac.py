import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import zipfile
from abc import ABC
from collections import OrderedDict
from typing import Optional
import multiprocessing
import numpy as np
import tensorflow as tf
tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

from fault_tolerant_flight_control_drl.agent import ReplayBuffer
from fault_tolerant_flight_control_drl.tools.save_util import data_to_json, json_to_data, params_to_bytes, bytes_to_params # todo: check saving tools
from fault_tolerant_flight_control_drl.tools.math_util import unscale_action, scale_action, set_global_seeds


class SAC(ABC):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code the from original implementation (https://github.com/haarnoja/sac),
    from OpenAI Spinning Up (https://github.com/openai/spinningup),
    from the Softlearning repo (https://github.com/rail-berkeley/softlearning/) and
    from the StableBaselines repo (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290

    :param policy: The LnMlpPolicy policy model
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for SAC normally but can help exploring.
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, batch_size=64, tau=0.005, action_noise=None,
                 random_exploration=0.0, _init_setup_model=True, policy_kwargs=None,
                 seed=None):

        self.ID = None
        self.policy = policy
        self.env = env
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.num_timesteps = 0
        self.sess = None
        self.params = None
        self.seed = seed
        self._param_load_ops = None
        self.episode_reward = None

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = 1
        self.batch_size = batch_size
        self.tau = tau
        self.ent_coef = 'auto'
        self.target_update_interval = 1
        self.gradient_steps = 1
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration

        self.value_fn = None
        self.graph = None
        self.replay_buffer = None
        self.params = None
        self.policy_tf = None
        self.target_entropy = 'auto'

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.set_random_seed(self.seed)

            num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
            tf_config = tf.ConfigProto(
                inter_op_parallelism_threads=num_cpu,
                intra_op_parallelism_threads=num_cpu)
            self.sess = tf.Session(config=tf_config, graph=self.graph)

            self.replay_buffer = ReplayBuffer(self.buffer_size)

            with tf.variable_scope("input", reuse=False):
                # Create policy and target TF objects
                self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                             **self.policy_kwargs)
                self.target_policy = self.policy(self.sess, self.observation_space, self.action_space,
                                                 **self.policy_kwargs)

                # Initialize Placeholders
                self.observations_ph = self.policy_tf.obs_ph
                # Normalized observation for pixels
                self.processed_obs_ph = self.policy_tf.processed_obs
                self.next_observations_ph = self.target_policy.obs_ph
                self.processed_next_obs_ph = self.target_policy.processed_obs
                self.action_target = self.target_policy.action_ph
                self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                 name='actions')
                self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

            with tf.variable_scope("model", reuse=False):
                # Create the policy
                # first return value corresponds to deterministic actions
                # policy_out corresponds to stochastic actions, used for training
                # logp_pi is the log probability of actions taken by the policy
                self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                # Monitor the entropy of the policy,
                # this is not used for training
                self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                #  Use two Q-functions to improve performance by reducing overestimation bias.
                qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                 create_qf=True, create_vf=True)
                qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                policy_out, create_qf=True, create_vf=False,
                                                                reuse=True)

                self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)

                # The entropy coefficient is learned automatically
                # see Automating Entropy Adjustment for Maximum Entropy RL section
                # of https://arxiv.org/abs/1812.05905
                # Default initial value of ent_coef when learned
                init_value = 1.0

                self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                    initializer=np.log(init_value).astype(np.float32))
                self.ent_coef = tf.exp(self.log_ent_coef)

            with tf.variable_scope("target", reuse=False):
                # Create the value network
                _, _, self.value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                     create_qf=False, create_vf=True)

            with tf.variable_scope("loss", reuse=False):
                # Take the min of the two Q-Values (Double-Q Learning)
                min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                # Target for Q value regression
                q_backup = tf.stop_gradient(
                    self.rewards_ph +
                    (1 - self.terminals_ph) * self.gamma * self.value_target
                )

                # Compute Q-Function loss
                qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                # Compute the entropy temperature loss
                ent_coef_loss = -tf.reduce_mean(
                    self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                # Compute the policy loss
                policy_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                # Target for value fn regression
                # We update the vf towards the min of two Q-functions in order to
                # reduce overestimation bias from function approximation error.
                v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                values_losses = qf1_loss + qf2_loss + value_loss

                # Policy train op (separate from value train operation, because min_qf_pi appears in policy_loss)
                policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                policy_train_op = policy_optimizer.minimize(policy_loss,
                                                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/pi'))

                # Value train operation
                value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                values_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/values_fn')

                source_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model/values_fn")
                target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target/values_fn")

                # Polyak averaging for target variables
                self.target_update_op = [
                    tf.assign(target, (1 - self.tau) * target + self.tau * source)
                    for target, source in zip(target_params, source_params)
                ]
                # Initializing target to match source variables
                target_init_op = [
                    tf.assign(target, source)
                    for target, source in zip(target_params, source_params)
                ]

                # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                # and we first need to compute the policy action before computing q values losses
                with tf.control_dependencies([policy_train_op]):
                    train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                    self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                    # All ops to call during one training step
                    self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                     value_loss, qf1, qf2, value_fn, logp_pi,
                                     self.entropy, policy_train_op, train_values_op]

                    # Add entropy coefficient optimization operation
                    with tf.control_dependencies([train_values_op]):
                        ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                        self.infos_names += ['ent_coef_loss', 'ent_coef']
                        self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

            # Retrieve parameters that must be saved
            self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")
            self.target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target/values_fn")

            # Initialize Variables and target network
            with self.sess.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.sess.run(target_init_op)

    def _train_step(self, learning_rate):
        # Sample a batch from the replay buffer
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = self.replay_buffer.sample(self.batch_size)

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        # Do one gradient step
        self.sess.run(self.step_ops, feed_dict)
        return

    def learn(self, total_timesteps, callback=None, online=False, initial_state=None):

        callback.init_callback(self)

        self._setup_learn()

        episode_rewards = [0.0]
        episode_successes = []
        if self.action_noise is not None:
            self.action_noise.reset()

        if online:
            obs = initial_state
        else:
            obs = self.env.reset()
        n_updates = 0

        for step in range(total_timesteps):
            # Before training starts, randomly sample actions
            # from a uniform distribution for better exploration.
            # Afterwards, use the learned policy
            # if random_exploration is set to 0 (normal setting)
            if (self.num_timesteps < self.learning_starts or np.random.rand() < self.random_exploration or \
                    (len(episode_successes) > 1 and not bool(episode_successes[-1]))) and not online:
                # actions sampled from action space are from range specific to the environment
                # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                unscaled_action = self.env.action_space.sample()
                action = scale_action(self.action_space, unscaled_action)

            else:
                action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                # Add noise to the action (improve exploration,
                # not needed in general)
                if self.action_noise is not None:
                    action = np.clip(action + self.action_noise(), -1, 1)
                # inferred actions need to be transformed to environment action_space before stepping
                unscaled_action = unscale_action(self.action_space, action)

            assert action.shape == self.env.action_space.shape

            new_obs, reward, done, info = self.env.step(unscaled_action)

            self.num_timesteps += 1

            # Only stop training if return value is False
            if callback.on_step() is False:
                break

            # Avoid changing the original ones
            obs_, new_obs_, reward_ = obs, new_obs, reward

            # Store transition in the replay buffer.
            self.replay_buffer_add(obs_, action, reward_, new_obs_, done, info)
            obs = new_obs

            if self.num_timesteps % self.train_freq == 0:

                # Update policy, critics and target networks
                for grad_step in range(self.gradient_steps):
                    # Break if the warmup phase is not over
                    # or if there are not enough samples in the replay buffer
                    if not self.replay_buffer.can_sample(self.batch_size) \
                            or self.num_timesteps < self.learning_starts:
                        break

                    n_updates += 1
                    # Compute current learning_rate
                    frac = 1.0 - step / total_timesteps
                    current_lr = self.learning_rate(frac)
                    # Update policy and critics (q functions)
                    self._train_step(current_lr)
                    # Update target network
                    if (step + grad_step) % self.target_update_interval == 0:
                        # Update target network
                        self.sess.run(self.target_update_op)

            episode_rewards[-1] += reward_
            if done:
                if self.action_noise is not None:
                    self.action_noise.reset()

                obs = self.env.reset()
                episode_rewards.append(0.0)

                maybe_is_success = info.get('is_success')
                if maybe_is_success is not None:
                    episode_successes.append(float(maybe_is_success))

        return self

    def predict(self, observation, deterministic=True):
        observation = np.array(observation)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions)  # scale the output for the prediction

        actions = actions[0]

        return actions, None

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def get_parameters(self):
        """
        Get current model parameters as dictionary of variable name -> ndarray.
        :return: (OrderedDict) Dictionary of variable name -> ndarray of model's parameters.
        """
        parameters = self.get_parameter_list()
        parameter_values = self.sess.run(parameters)
        return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_values))
        return return_dictionary

    def _setup_load_operations(self):
        """
        Create tensorflow operations for loading model parameters
        """
        # Assume tensorflow graphs are static -> check
        # that we only call this function once
        if self._param_load_ops is not None:
            raise RuntimeError("Parameter load operations have already been created")
        # For each loadable parameter, create appropiate
        # placeholder and an assign op, and store them to
        # self.load_param_ops as dict of variable.name -> (placeholder, assign)
        loadable_parameters = self.get_parameter_list()
        # Use OrderedDict to store order for backwards compatibility with
        # list-based params
        self._param_load_ops = OrderedDict()
        with self.graph.as_default():
            for param in loadable_parameters:
                placeholder = tf.placeholder(dtype=param.dtype, shape=param.shape)
                # param.name is unique (tensorflow variables have unique names)
                self._param_load_ops[param.name] = (placeholder, param.assign(placeholder))

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        :param env: (Gym Environment) The environment for learning a policy
        """
        if env is None and self.env is None:
            return
        elif env is None:
            raise ValueError("Error: trying to replace the current environment with None")

        # sanity checking the environment
        assert self.observation_space == env.observation_space, \
            f"Error: the environment ({env.observation_space}) must have the same observation space that " \
            f"the model was trained on ({self.observation_space})."
        assert self.action_space == env.action_space, \
            f"Error: the environment ({env.action_space}) must have the same action space that the model" \
            f" was trained on ({self.action_space})."

        self.env = env
        self.episode_reward = None

    def replay_buffer_add(self, obs_t, action, reward, obs_tp1, done, info):
        """
        Add a new transition to the replay buffer
        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        :param info: (dict) extra values used to compute the reward when using HER
        """
        # Pass info dict when using HER, as it can be used to compute the reward
        kwargs = {}
        self.replay_buffer.add(obs_t, action, reward, obs_tp1, float(done), **kwargs)

    def set_random_seed(self, seed: Optional[int]) -> None:
        """
        :param seed: (Optional[int]) Seed for the pseudo-random generators. If None,
            do not change the seeds.
        """
        # Ignore if the seed is None
        if seed is None:
            return
        # Seed python, numpy and tf random generator
        set_global_seeds(seed)
        if self.env is not None:
            self.env.seed(seed)
            # Seed the action space
            # useful when selecting random actions
            self.env.action_space.seed(seed)
        self.action_space.seed(seed)

    def _setup_learn(self):
        """
        Check the environment.
        """
        if self.env is None:
            raise ValueError(
                "Error: cannot train the model without a valid environment, please set an environment with"
                "set_env(self, env) method.")
        if self.episode_reward is None:
            self.episode_reward = np.zeros((1,))

    def load_parameters(self, params, exact_match=True):
        """
        Load model parameters from a file or a dictionary
        Dictionary keys should be tensorflow variable names, which can be obtained
        with ``get_parameters`` function. If ``exact_match`` is True, dictionary
        should contain keys for all model's parameters, otherwise RunTimeError
        is raised. If False, only variables included in the dictionary will be updated.
        This does not load agent's hyper-parameters.
        .. warning::
            This function does not update trainer/optimizer variables (e.g. momentum).
            As such training after using this function may lead to less-than-optimal results.
        :param load_path_or_dict: (str or file-like or dict) Save parameter location
            or dict of parameters as variable.name -> ndarrays to be loaded.
        :param exact_match: (bool) If True, expects load dictionary to contain keys for
            all variables in the model. If False, loads parameters only for variables
            mentioned in the dictionary. Defaults to True.
        """
        # Make sure we have assign ops
        if self._param_load_ops is None:
            self._setup_load_operations()

        feed_dict = {}
        param_update_ops = []
        # Keep track of not-updated variables
        not_updated_variables = set(self._param_load_ops.keys())
        for param_name, param_value in params.items():
            placeholder, assign_op = self._param_load_ops[param_name]
            feed_dict[placeholder] = param_value
            # Create list of tf.assign operations for sess.run
            param_update_ops.append(assign_op)
            # Keep track which variables are updated
            not_updated_variables.remove(param_name)

        # Check that we updated all parameters if exact_match=True
        if exact_match and len(not_updated_variables) > 0:
            raise RuntimeError("Load dictionary did not contain all variables. " +
                               "Missing variables: {}".format(", ".join(not_updated_variables)))

        self.sess.run(param_update_ops, feed_dict=feed_dict)

    @classmethod
    def load(cls, load_path, env, custom_objects=None, **kwargs):
        """
        Load the model from file
        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            data['policy_kwargs'] = kwargs['policy_kwargs']
            # raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
            #                  "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
            #                                                                   kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=env, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()

        model.load_parameters(params)

        return model

    def save(self, save_path):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": 'auto',
            "target_entropy": self.target_entropy,
            # "replay_buffer": self.replay_buffer
            "gamma": self.gamma,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "policy_kwargs": self.policy_kwargs
        }

        params = self.get_parameters()

        # Check postfix if save_path is a string
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".zip"

        # Create a zip-archive and write our objects
        # there. This works when save_path
        # is either str or a file-like
        with zipfile.ZipFile(save_path, "w") as file_:
            # Do not try to save "None" elements
            if data is not None:
                serialized_data = data_to_json(data)
                file_.writestr("data", serialized_data)
            if params is not None:
                serialized_params = params_to_bytes(params)
                # We also have to store list of the parameters
                # to store the ordering for OrderedDict.
                serialized_param_list = json.dumps(
                    list(params.keys()),
                    indent=4
                )
                file_.writestr("parameters", serialized_params)
                file_.writestr("parameter_list", serialized_param_list)

    @staticmethod
    def _load_from_file(load_path, load_data=True, custom_objects=None):
        """Load model data from a .zip archive
        :param load_path: (str or file-like) Where to load model from
        :param load_data: (bool) Whether we should load and return data
            (class parameters). Mainly used by `load_parameters` to
            only load model parameters (weights).
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :return: (dict, OrderedDict) Class parameters and model parameters
        """
        # Check if file exists if load_path is
        # a string
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".zip"):
                    load_path += ".zip"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

        # Open the zip archive and load data.
        try:
            # print('here1',load_path)
            with zipfile.ZipFile(load_path, "r") as file_:
                namelist = file_.namelist()
                # print(namelist)
                data = None
                params = None
                if "data" in namelist and load_data:
                    # print('here2')
                    # Load class parameters and convert to string
                    # (Required for json library in Python 3.5)
                    json_data = file_.read("data").decode()
                    data = json_to_data(json_data, custom_objects=custom_objects)

                if "parameters" in namelist:
                    # print('here3')
                    # Load parameter list and and parameters
                    parameter_list_json = file_.read("parameter_list").decode()
                    parameter_list = json.loads(parameter_list_json)
                    serialized_params = file_.read("parameters")
                    params = bytes_to_params(
                        serialized_params, parameter_list
                    )
        except zipfile.BadZipFile:
            warnings.warn("It appears you are loading from a file with wrong format. ",
                          DeprecationWarning)

        return data, params