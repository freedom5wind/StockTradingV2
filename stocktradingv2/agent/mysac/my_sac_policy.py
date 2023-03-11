from typing import Dict, Tuple, Type, Union, List

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.algorithms.sac.rnnsac_torch_policy import RNNSACTorchPolicy
from ray.rllib.algorithms.sac.sac_torch_policy import _get_dist_class
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_action_dist import (
    TorchDiagGaussian,
    TorchDistributionWrapper,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import TargetNetworkMixin
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import global_norm, apply_grad_clipping
from ray.rllib.utils.typing import TensorType, AlgorithmConfigDict
import tree
import torch

import stocktradingv2
from stocktradingv2.agent.mysac.my_sac_model import MySACModel


class MyTorchDiagGaussian(TorchDiagGaussian):
    @override(ActionDistribution)
    def sample(self) -> TensorType:
        self.last_sample = self.dist.rsample()
        return self.last_sample


def build_my_sac_model_and_action_dist(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    """Constructs the necessary ModelV2 and action dist class for the Policy.

    Args:
        policy: The TFPolicy that will use the models.
        obs_space (gym.spaces.Space): The observation space.
        action_space (gym.spaces.Space): The action space.
        config: The SAC's config dict.

    Returns:
        ModelV2: The ModelV2 to be used by the Policy. Note: An additional
            target model will be created in this function and assigned to
            `policy.target_model`.
    """
    # With separate state-preprocessor (before obs+action concat).
    num_outputs = int(np.product(obs_space.shape))

    # Force-ignore any additionally provided hidden layer sizes.
    # Everything should be configured using SAC's `q_model_config` and
    # `policy_model_config` config settings.
    policy_model_config = MODEL_DEFAULTS.copy()
    policy_model_config.update(config["policy_model_config"])
    q_model_config = MODEL_DEFAULTS.copy()
    q_model_config.update(config["q_model_config"])

    # Override custom_loss function of RNNSACTorchModel
    # to customize loss.
    default_model_cls = MySACModel

    model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        default_model=default_model_cls,
        name="sac_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
    )

    assert isinstance(model, default_model_cls)

    # Create an exact copy of the model and store it in `policy.target_model`.
    # This will be used for tau-synched Q-target models that run behind the
    # actual Q-networks and are used for target q-value calculations in the
    # loss terms.
    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        default_model=default_model_cls,
        name="target_sac_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        initial_alpha=config["initial_alpha"],
        target_entropy=config["target_entropy"],
    )
    policy.target_model.eval()

    assert isinstance(policy.target_model, default_model_cls)
    assert (
        model.get_initial_state() != []
    ), "MySAC requires its model to be a recurrent one!"

    action_dist_class = _get_dist_class(policy, config, action_space)
    # action_dist_class = MyTorchDiagGaussian

    # Reference used in custom loss function.
    model.get_target_model = lambda: policy.target_model
    model.action_dist_class = action_dist_class

    return model, action_dist_class


def dummy_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: Type[TorchDistributionWrapper],
    train_batch: SampleBatch,
) -> Union[TensorType, List[TensorType]]:
    """Add dummy model towerstat"""
    train_batch[SampleBatch.ACTIONS]
    train_batch[SampleBatch.SEQ_LENS]
    train_batch[SampleBatch.NEXT_OBS]
    model.tower_stats["q_t"] = torch.tensor(0.0)
    model.tower_stats["actor_loss"] = torch.tensor(0.0)
    model.tower_stats["critic_loss"] = torch.tensor(0.0)
    model.tower_stats["alpha_loss"] = torch.tensor(0.0)
    model.tower_stats["policy_t"] = torch.tensor(0.0)
    model.tower_stats["log_pis_t"] = torch.tensor(0.0)
    model.tower_stats["reward_t"] = torch.tensor(0.0)


class ComputeTDErrorMixin:
    """Mixin class calculating TD-error (part of critic loss) per batch item.

    - Adds `policy.compute_td_error()` method for TD-error calculation from a
      batch of observations/actions/rewards/etc..
    """

    def __init__(self):
        def compute_td_error(input_dict):
            # Do forward pass on loss to update td errors attribute
            # (one TD-error value per item in batch to update PR weights).
            self.model.custom_loss(None, input_dict)

            # `self.model.td_error` is set within actor_critic_loss call.
            # Return its updated value here.
            return self.model.tower_stats["td_error"]

        # Assign the method to policy (self) for later usage.
        self.compute_td_error = compute_td_error


def postprocess_nstep_and_prio(
    policy: Policy, batch: SampleBatch, other_agent=None, episode=None
) -> SampleBatch:
    # N-step Q adjustments.
    if policy.config["n_step"] > 1:
        adjust_nstep(policy.config["n_step"], policy.config["gamma"], batch)

    # Create dummy prio-weights (1.0) in case we don't have any in
    # the batch.
    if PRIO_WEIGHTS not in batch:
        batch[PRIO_WEIGHTS] = np.ones_like(batch[SampleBatch.REWARDS])

    # Prioritize on the worker side.
    if batch.count > 0 and policy.config["replay_buffer_config"].get(
        "worker_side_prioritization", False
    ):
        td_errors = policy.compute_td_error(batch)
        # Retain compatibility with old-style Replay args
        epsilon = policy.config.get("replay_buffer_config", {}).get(
            "prioritized_replay_eps"
        ) or policy.config.get("prioritized_replay_eps")
        if epsilon is None:
            raise ValueError("prioritized_replay_eps not defined in config.")

        new_priorities = np.abs(convert_to_numpy(td_errors)) + epsilon
        batch[PRIO_WEIGHTS] = new_priorities

    return batch


def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Stats function for SAC. Returns a dict with important loss stats.

    Args:
        policy: The Policy to generate stats for.
        train_batch: The SampleBatch (already) used for training.

    Returns:
        Dict[str, TensorType]: The stats dict.
    """
    q_t = torch.stack(policy.get_tower_stats("q_t"))

    return {
        "actor_loss": torch.mean(torch.stack(policy.get_tower_stats("actor_loss"))),
        "critic_loss": torch.mean(
            torch.stack(tree.flatten(policy.get_tower_stats("critic_loss")))
        ),
        "alpha_loss": torch.mean(torch.stack(policy.get_tower_stats("alpha_loss"))),
        "alpha_value": torch.exp(policy.model.log_alpha),
        "log_alpha_value": policy.model.log_alpha,
        "target_entropy": policy.model.target_entropy,
        "policy_t": torch.mean(torch.stack(policy.get_tower_stats("policy_t"))),
        "log_pis_t": torch.mean(torch.stack(policy.get_tower_stats("log_pis_t"))),
        "mean_q": torch.mean(q_t),
        "max_q": torch.max(q_t),
        "min_q": torch.min(q_t),
        "policy_var_norm": global_norm(policy.model.action_model.trainable_variables()),
        "q_net_var_norm": global_norm(policy.model.q_net.trainable_variables()),
        "reward_t": torch.mean(torch.stack(policy.get_tower_stats("reward_t"))),
    }


MySACPolicy = RNNSACTorchPolicy.with_updates(
    name="MySACPolicy",
    get_default_config=lambda: stocktradingv2.agent.mysac.my_sac.MySACConfig().to_dict(),
    make_model_and_action_dist=build_my_sac_model_and_action_dist,
    loss_fn=dummy_loss,
    postprocess_fn=postprocess_nstep_and_prio,
    mixins=[TargetNetworkMixin, ComputeTDErrorMixin],
    stats_fn=stats,
    extra_grad_process_fn=apply_grad_clipping,
)
