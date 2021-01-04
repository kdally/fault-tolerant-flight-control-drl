import csv
import os
import time
from abc import ABC
from typing import Union

import gym
import numpy as np


class SaveOnBestReturn(ABC):
    """
    Callback when evaluating the controller during training.

    :param eval_env: (gym.Env) The environment used for evaluation.
    :param eval_freq: (int) Evaluate the controller every eval_freq call of the callback.
        Best controller so-far will be saved.
    :param best_model_save_path: (str) Path to a folder where the best model will be saved.
    :param verbose: (int)
    """

    def __init__(self, eval_env,
                 log_path: str,
                 eval_freq: int = 2000,
                 best_model_save_path: str = None,
                 verbose: int = 1):
        super(SaveOnBestReturn, self).__init__()

        self.model = None
        self.training_env = None  # type: Union[gym.Env, None]
        self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int
        self.verbose = verbose

        self.eval_freq = eval_freq
        self.best_reward = -np.inf
        self.weights_sample = None

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path

        self.log_path = log_path
        filename = os.path.join(log_path, "monitor.csv")
        self.file_handler = open(filename, "wt")
        self.logger = csv.DictWriter(self.file_handler, fieldnames=('r', 'l', 't')
                                                                   + tuple([f'w{i}' for i in range(10)]))
        self.logger.writeheader()
        self.file_handler.flush()
        self.file_handler.flush()
        self.t_start = time.time()

    def init_callback(self, model):
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self.training_env = model.env
        return self

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:

            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _info = self.eval_env.step(action)
                episode_reward += reward

            self.weights_sample = self.model.get_parameter_list()[4][5, :10].eval(session=self.model.sess)

            ep_info = {"r": round(episode_reward, 6), "l": self.num_timesteps,
                       "t": round(time.time() - self.t_start, 6)}
            ep_info = {**ep_info, **{f'w{i}': self.weights_sample[i] for i in range(10)}}
            self.logger.writerow(ep_info)
            self.file_handler.flush()

            if episode_reward > self.best_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                self.model.save(os.path.join(self.best_model_save_path, 'best_model'))
                self.best_reward = episode_reward
                self.eval_env.render(ext_agent=self.model)

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"current return={episode_reward:.2f}, best return={self.best_reward:.2f}")
                # print('Entropy : ', self.model.ent_coef.eval(session=self.model.sess))

        return True
