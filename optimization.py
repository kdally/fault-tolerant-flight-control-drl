import os
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from agent.sac import SAC
from agent.policy import LnMlpPolicy
from agent.callback import SaveOnBestReturn

from envs.citation_rates import Citation
from tools.schedule import schedule


def sample_sac_params(trial):
    """
    Sampler for SAC hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    net_arch = trial.suggest_categorical('net_arch', ["small", "medium", "big"])
    batch_size = trial.suggest_categorical('batch_size', [256, 512])

    net_arch = {
        'small': [64, 64],
        'medium': [256, 256],
        'big': [400, 300],
    }[net_arch]

    return {
        'learning_rate': schedule(learning_rate),
        'batch_size': batch_size,
        'policy_kwargs': dict(layers=net_arch)
    }


def hyperparam_optimization(n_trials=30, n_timesteps=int(1e6),
                            n_jobs=1):
    """
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param n_jobs: (int) number of parallel jobs
    :return: (pd.Dataframe) detailed result of the optimization
    """

    n_startup_trials = 5

    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=3)

    pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=10)

    study = optuna.create_study(sampler=sampler, pruner=pruner)
    algo_sampler = sample_sac_params

    def objective(trial):

        kwargs = {}
        trial.model_class = None

        kwargs.update(algo_sampler(trial))

        eval_env = Citation()
        model = create_model(eval_env, **kwargs)

        eval_callback = SaveOnBestReturn(eval_env=eval_env, eval_freq=2000, log_path='optimization_logs/tmp/',
                                         best_model_save_path='optimization_logs/tmp/')

        try:
            model.learn(n_timesteps, callback=eval_callback)
            # Free memory
            model.env.close()
            eval_env.close()
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            raise optuna.exceptions.TrialPruned()
        cost = -1 * eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        return cost

    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except KeyboardInterrupt:
        pass

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('Value: ', trial.value)

    print('Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()


def create_model(env1, **kwargs):
    """
    Helper to create a model with different hyperparameters
    """
    return SAC(LnMlpPolicy, env=env1, verbose=0, ent_coef='auto', **kwargs)


data_frame = hyperparam_optimization()

report_name = "report_100-trials_{}.csv".format(int(time.time()))

log_path = os.path.join('optimization_logs', report_name)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
data_frame.to_csv(log_path)
