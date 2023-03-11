import ray
from ray import air, tune
from ray.tune.logger import TBXLoggerCallback

from stocktradingv2.agent.mysac.my_sac import MySAC, MySACConfig


def main_test():
    param_space = MySACConfig().to_dict()
    param_space.update(
        {
            "framework": "torch",
            "env": "Pendulum-v1",
            "policy_model_config": {
                "lstm_dim": 64,
                "net_arch": [256, 256],
            },
            "q_model_config": {
                "type": tune.grid_search(["dqn", "cqn", "qrdqn", "iqn"]),
                "lstm_dim": 64,
                "num_atoms": 50,
                "net_arch": [256, 256],
                "num_critics": 1,
                # cqn
                "vmin": -80.0,
                "vmax": 0,
                # iqn
                "risk_distortion_measure": None,
                "cos_embedding_dim": 64,
            },
            "tau": 0.01,
            "target_entropy": "auto",
            "n_step": 1,
            "train_batch_size": 256,
            "target_network_update_freq": 1,
            "grad_clip": 40,
            "min_sample_timesteps_per_iteration": 200,
            "num_steps_sampled_before_learning_starts": 256,
            "metrics_num_episodes_for_smoothing": 5,
            "num_workers": 0,
            "num_envs_per_worker": 2,
            "num_cpus_per_worker": 1,
            "num_steps_sampled_before_learning_starts": 256,
            "train_batch_size": 256,
            "target_network_update_freq": 1,
        }
    )

    tuner = tune.Tuner(
        MySAC,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=air.RunConfig(
            stop={
                "episode_reward_mean": -250,
                "timesteps_total": 3_000_000,
            },
            callbacks=[TBXLoggerCallback()],
        ),
    )
    results = tuner.fit()
    # best_result = results.get_best_result()  # Get best result object
    # best_config = best_result.config  # Get best trial's hyperparameters
    # best_logdir = best_result.log_dir  # Get best trial's logdir
    # best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint
    # best_metrics = best_result.metrics  # Get best trial's last results
    # best_result_df = best_result.metrics_dataframe  # Get best result as pandas dataframe

    # Get a dataframe with the last results for each trial
    df_results = results.get_dataframe()
    print(df_results.head())


if __name__ == "__main__":
    ray.init()
    main_test()
