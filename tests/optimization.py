import os
import time

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import fault_tolerant_flight_control_drl as ft


def sample_sac_params(trial):
    """
    Sampler for SAC hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    learning_rate = trial.suggest_loguniform('lr', 1e-4, 2e-3)
    net_arch = trial.suggest_categorical('net_arch', ["small", "medium", "big"])
    train_freq = trial.suggest_categorical('train_freq', [1, 50, 100])
    batch_size = trial.suggest_categorical('batch_size', [256, 512])
    buffer_size = trial.suggest_categorical('buffer_size', [int(5e4), int(1e6)])

    net_arch = {
        'small': [16, 16],
        'medium': [32, 32],
        'big': [64, 64],
    }[net_arch]

    return {
        'learning_rate': ft.tools.constant(learning_rate),
        'train_freq': train_freq,
        'buffer_size':buffer_size,
        'batch_size': batch_size,
        'policy_kwargs': dict(layers=net_arch)
    }


def hyperparam_optimization(n_trials=60, n_timesteps=int(1e5),
                            n_jobs=1):
    """
    :param n_trials: (int) maximum number of trials for finding the best hyperparams
    :param n_timesteps: (int) maximum number of timesteps per trial
    :param n_jobs: (int) number of parallel jobs
    :return: (pd.Dataframe) detailed result of the optimization
    """

    n_startup_trials = 5

    sampler = TPESampler(n_startup_trials=n_startup_trials, seed=3)

    pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=15)

    study = optuna.create_study(sampler=sampler, pruner=pruner)
    algo_sampler = sample_sac_params

    def objective(trial):

        kwargs = {}
        trial.model_class = None

        kwargs.update(algo_sampler(trial))

        eval_env = ft.envs.CitationNormal()
        env_train = ft.envs.CitationNormal()
        model = create_model(env_train, **kwargs)

        eval_callback = ft.agent.SaveOnBestReturn(eval_env=eval_env, eval_freq=2000, log_path='optimization_logs/tmp/',
                                         best_model_save_path='optimization_logs/tmp/', verbose=0)

        try:
            model.learn(n_timesteps, callback=eval_callback)
            # Free memory
            model.env.close()
            eval_env.close()
            env_train.close()
        except AssertionError:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            env_train.close()
            raise optuna.exceptions.TrialPruned()
        except IndexError:
            model.env.close()
            eval_env.close()
            env_train.close()
            raise optuna.exceptions.TrialPruned()
        cost = -1 * eval_callback.best_reward

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
    return ft.agent.SAC(ft.agent.LnMlpPolicy, env=env1, verbose=0, ent_coef='auto', **kwargs)


if not os.path.exists('optimization_logs'):
    os.makedirs('optimization_logs')
data_frame = hyperparam_optimization()

report_name = "report_100-trials_{}.csv".format(int(time.time()))

log_path = os.path.join('optimization_logs', report_name)
os.makedirs(os.path.dirname(log_path), exist_ok=True)
data_frame.to_csv(log_path)
