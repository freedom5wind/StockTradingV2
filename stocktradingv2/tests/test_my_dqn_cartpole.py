from typing import Dict

import numpy as np
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.logger import TBXLoggerCallback

from stocktradingv2.agent.mydqn.my_dqn import MyDQNConfig, MyDQN


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length <= 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        episode.user_data["pole_angles"] = []
        episode.hist_data["pole_angles"] = []

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        for agent_id in episode.get_agents():
            pole_angle = episode._agent_collectors[agent_id].buffers[SampleBatch.OBS][
                0
            ][-1][2]
        episode.user_data["pole_angles"].append(pole_angle)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: EpisodeV2,
        env_index: int,
        **kwargs
    ):
        pole_angle = np.mean(episode.user_data["pole_angles"])
        print(
            "episode {} (env-idx={}) ended with length {} and pole "
            "angles {}".format(
                episode.episode_id, env_index, episode.length, pole_angle
            )
        )
        episode.custom_metrics["pole_angle"] = pole_angle
        episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        print(
            "Algorithm.train() result: {} -> {} episodes".format(
                algorithm, result["episodes_this_iter"]
            )
        )
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True


def main_test():
    param_space = MyDQNConfig().to_dict()
    param_space.update(
        {
            "framework": "torch",
            "env": "CartPole-v1",
            "model": {
                "custom_model_config": {
                    "type": tune.grid_search(["iqn", "cqn", "dqn", "qrdqn"]),
                    "vmin": 0,
                    "vmax": 100,
                    "net_arch": [64],
                }
            },
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.02,
                "epsilon_timesteps": 10000,
            },
        }
    )

    tuner = tune.Tuner(
        MyDQN,
        param_space=dict(
            param_space,
            **{
                "num_workers": 5,
                "num_envs_per_worker": 4,
                "num_cpus_per_worker": 1,
                "callbacks": MyCallbacks,
            }
        ),
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=air.RunConfig(
            stop={
                "episode_reward_mean": 100,
                "timesteps_total": 100000,
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
    main_test()
