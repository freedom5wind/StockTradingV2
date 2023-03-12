from ray.tune.logger import pretty_print

from stocktradingv2.agent.mydqn import MyDQNConfig


def main_test():
    for dqn_type in ["dqn", "cqn", "qrdqn", "iqn"]:
        config = MyDQNConfig()
        config.framework(framework="torch")
        config.environment(env="CartPole-v1")
        config.model["custom_model_config"].update(
            {
                "type": dqn_type,
                "num_atoms": 51,
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
    main_test()
