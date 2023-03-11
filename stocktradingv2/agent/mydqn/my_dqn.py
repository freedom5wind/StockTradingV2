from typing import Optional, Type

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQN, DQNConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE

from stocktradingv2.agent.mydqn.my_dqn_policy import MyDQNPolicy


class MyDQNConfig(DQNConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class or MyDQN)
        self.framework_str = "torch"
        self.gamma = 0.99
        self.n_step = 1
        self.replay_buffer_config = {
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": 50000,
            "prioritized_replay": DEPRECATED_VALUE,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
            "replay_sequence_length": 1,
        }
        self.twin_q = False
        self.policy_model_config = {
            "lstm_dim": 64,
            "net_arch": [256, 256],
        }
        self.model_config = {
            "custom_model_config": {
                "type": "dqn",
                "lstm_dim": 64,
                "num_atoms": 50,
                "net_arch": [256, 256],
                "num_critics": 3,
                # cqn
                "vmin": -10.0,
                "v_max": 10.0,
                # iqn
                "risk_distortion_measure": None,
                "cos_embedding_dim": 64,
            }
        }
        self.exploration_config = {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000,
        }
        self.zero_init_states = True

        self.num_steps_sampled_before_learning_starts = 100
        self.train_batch_size = 480
        self.target_network_update_freq = 480
        self.tau = 0.3
        # "train_batch_size": 480,
        # "target_network_update_freq": 480,
        # "tau": 0.3,
        # "zero_init_states": True,

        self.evaluation_interval = 3
        self.evaluation_duration = 1
        self.evaluation_duration_unit = "episodes"
        self.evaluation_config = {"explore": False}


class MyDQN(DQN):
    @classmethod
    @override(DQN)
    def get_default_config(cls) -> AlgorithmConfig:
        return MyDQNConfig()

    @classmethod
    @override(DQN)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return MyDQNPolicy
        else:
            raise NotImplementedError
