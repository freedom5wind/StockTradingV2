import ray
from ray.tune.logger import pretty_print

from stocktradingv2.agent.mysac import MySACConfig


def main_test():
    for dqn_type in ["iqn", "dqn", "cqn", "qrdqn"]:
        config = MySACConfig()
        config.environment(env="Pendulum-v1")
        config.q_model_config.update(
            {
                "type": dqn_type,
            }
        )
        config.num_steps_sampled_before_learning_starts = 10
        config.min_sample_timesteps_per_iteration = 100
        config.train_batch_size = 32
        algo = config.build()

        for _ in range(2):
            result = algo.train()
            print(pretty_print(result))

        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == "__main__":
    ray.init(address="auto")
    main_test()
