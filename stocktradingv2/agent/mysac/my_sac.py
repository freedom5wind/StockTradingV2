from typing import Type, Optional

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.sac.rnnsac import RNNSAC, RNNSACConfig
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import DEPRECATED_VALUE

from stocktradingv2.agent.mysac.my_sac_policy import MySACPolicy


class MySACConfig(RNNSACConfig):
    """Defines a configuration class from which an MySAC can be built.

    Example:
        >>> config = MySACConfig().training(gamma=0.9, lr=0.01)\
        ...     .resources(num_gpus=0)\
        ...     .rollouts(num_rollout_workers=4)
        >>> print(config.to_dict())  # doctest: +SKIP
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> algo = config.build(env="CartPole-v1")
        >>> algo.train()  # doctest: +SKIP
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or MySAC)
        # self.batch_mode = "complete_episodes"
        self.zero_init_states = True

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
        self.model.update(
            {
                "max_seq_len": 1,
            }
        )
        self.policy_model_config = {
            "lstm_dim": 64,
            "net_arch": [256, 256],
        }
        self.q_model_config = {
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

        # self.optimization = {
        #     "actor_learning_rate":,
        #     "critic_learning_rate":,
        #     "entropy_learning_rate":,
        # }
        # self.initial_alpha
        # self.target_entropy

    @override(RNNSACConfig)
    def training(
        self,
        *,
        zero_init_states: Optional[bool] = NotProvided,
        **kwargs,
    ) -> "MySACConfig":
        """Sets the training related configuration.

        Args:
            zero_init_states: If True, assume a zero-initialized state input (no matter
                where in the episode the sequence is located).
                If False, store the initial states along with each SampleBatch, use
                it (as initial state when running through the network for training),
                and update that initial state during training (from the internal
                state outputs of the immediately preceding sequence).

        Returns:
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)
        if zero_init_states is not NotProvided:
            self.zero_init_states = zero_init_states

        return self

    @override(RNNSACConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super(SACConfig, self).validate()

        if self.framework_str != "torch":
            raise ValueError("Only `framework=torch` supported so far for MySAC!")


class MySAC(RNNSAC):
    def __init__(self, *args, **kwargs):
        self._allow_unknown_configs = True
        super().__init__(*args, **kwargs)

    @classmethod
    @override(RNNSAC)
    def get_default_config(cls) -> AlgorithmConfig:
        return MySACConfig()

    @classmethod
    @override(RNNSAC)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        return MySACPolicy
