import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from abc import ABC
import numpy as np
import tensorflow as tf
from gym.spaces import Box
tf.get_logger().warning('test')
# WARNING:tensorflow:test
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def apply_squashing_func(mu_, pi_, logp_pi):
    """
    Squash the output of the Gaussian distribution
    and account for that in the log probability
    The squashed mean is also returned for using
    deterministic actions.

    :param mu_: (tf.Tensor) Mean of the gaussian
    :param pi_: (tf.Tensor) Output of the policy before squashing
    :param logp_pi: (tf.Tensor) Log probability before squashing
    :return: ([tf.Tensor])
    """
    # Squash the output
    deterministic_policy = tf.tanh(mu_)
    policy = tf.tanh(pi_)
    # Squash correction (from original implementation)
    logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS), axis=1)
    return deterministic_policy, policy, logp_pi


def mlp(input_tensor, layers, activ_fn=tf.nn.relu, layer_norm=True):
    """
    Create a multi-layer fully connected neural network.

    :param input_tensor: (tf.placeholder)
    :param layers: ([int]) Network architecture
    :param activ_fn: (tf.function) Activation function
    :param layer_norm: (bool) Whether to apply layer normalization or not
    :return: (tf.Tensor)
    """
    output = input_tensor
    for i, layer_size in enumerate(layers):
        output = tf.layers.dense(output, layer_size, name=f'fc{i}')
        if layer_norm:
            output = tf.contrib.layers.layer_norm(output, center=True, scale=True)
        output = activ_fn(output)
    return output


class LnMlpPolicy(ABC):
    """
    Class that implements a Layer-Normalization Gaussian policy with ReLu activation.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_steps: (int) The number of steps to run for each environment
    :param reuse: (bool) If the policy is reusable or not
    :param layers: (list) size (int) of the hidden layers
    """

    recurrent = False

    def __init__(self, sess, ob_space, ac_space, n_steps=1, reuse=False,
                 layers=None):

        assert isinstance(ac_space, Box), "Error: the action space must be of type gym.spaces.Box"

        self.n_steps = n_steps
        with tf.variable_scope("input", reuse=False):
            self.obs_ph = tf.placeholder(shape=(None,) + ob_space.shape, dtype=ob_space.dtype, name='Ob')
            self.processed_obs = tf.cast(self.obs_ph, tf.float32)
            self.action_ph = None

        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

        self.qf1 = None
        self.qf2 = None
        self.value_fn = None
        self.policy = None
        self.deterministic_policy = None
        self.act_mu = None
        self.std = None
        self.layer_norm = True
        if layers is None:
            layers = [64, 64]
        self.layers = layers
        self.entropy = None
        self.activ_fn = tf.nn.relu

    def make_actor(self, obs=None, reuse=False, scope="pi"):

        with tf.variable_scope(scope, reuse=reuse):
            pi_h = tf.layers.flatten(obs)
            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            self.act_mu = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            # With SAC, the std depends on the state
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)

        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        # Gaussian likelihood
        logp_pi = tf.reduce_sum(-0.5 * (((pi_ - mu_) / (tf.exp(log_std) + EPS)) ** 2 +
                                        2 * log_std + np.log(2 * np.pi)), axis=1)
        # Gaussian entropy
        self.entropy = tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
        # Apply squashing and account for it in the probability
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn",
                     create_vf=True, create_qf=True):

        with tf.variable_scope(scope, reuse=reuse):
            critics_input = tf.layers.flatten(obs)

            if create_vf:
                # Value function
                with tf.variable_scope('vf', reuse=reuse):
                    vf_h = mlp(critics_input, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    value_fn = tf.layers.dense(vf_h, 1, name="vf")
                    # outputs one value for a given observation
                self.value_fn = value_fn

            if create_qf:
                # Concatenate preprocessed state and action
                qf_h = tf.concat([critics_input, action], axis=-1)  # maps the state + action spaces to a value

                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf1 = tf.layers.dense(qf1_h, 1, name="qf1")  # no activation

                with tf.variable_scope('qf2', reuse=reuse):
                    qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf2 = tf.layers.dense(qf2_h, 1, name="qf2")  # no activation

                self.qf1 = qf1
                self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def step(self, obs, deterministic=False):
        """
        function to run on every policy iteration step
        """
        if deterministic:
            return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        return self.sess.run(self.policy, {self.obs_ph: obs})
