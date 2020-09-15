import csv
import os
import time
from abc import ABC
from typing import Union

import gym
import numpy as np


class SaveOnBestReturn(ABC):
    """
    Callback for evaluating an agent.

    :param eval_env: (gym.Env) The environment used for initialization
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: (str) Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: (bool) Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: (bool) Whether to render or not the environment during evaluation
    :param verbose: (int)
    """
    # todo: option to average out the training plot
    def __init__(self, eval_env: gym.Env,
                 log_path: str,
                 eval_freq: int = 10000,
                 best_model_save_path: str = None,
                 deterministic: bool = True,
                 render: bool = False,
                 verbose: int = 1):
        super(SaveOnBestReturn, self).__init__()

        # The RL model
        self.model = None
        # An alias for self.model.get_env(), the environment used for training
        self.training_env = None  # type: Union[gym.Env, None]
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int
        self.verbose = verbose

        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path

        self.log_path = log_path
        filename = os.path.join(log_path, "monitor.csv")
        self.file_handler = open(filename, "wt")
        self.logger = csv.DictWriter(self.file_handler, fieldnames=('r', 'l', 't'))
        self.logger.writeheader()
        self.file_handler.flush()
        self.file_handler.flush()
        self.t_start = time.time()

    def init_callback(self, model) -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self.training_env = model.get_env()

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            obs = self.eval_env.reset()
            done, state = False, None
            episode_reward = 0.0
            episode_length = 0
            while not done:
                action, state = self.model.predict(obs, state=state, deterministic=self.deterministic)
                obs, reward, done, _info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1

            self.last_mean_reward = episode_reward
            ep_info = {"r": round(episode_reward, 6), "l": self.num_timesteps, "t": round(time.time() - self.t_start, 6)}
            self.logger.writerow(ep_info)
            self.file_handler.flush()

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"current return={episode_reward:.2f}, best return={self.best_mean_reward:.2f}")

            if episode_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_mean_reward = episode_reward

        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()