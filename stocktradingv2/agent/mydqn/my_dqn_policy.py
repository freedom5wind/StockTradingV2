from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import gym.spaces
import gymnasium
import numpy as np
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchDistributionWrapper,
    TorchCategorical,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.torch_mixins import TargetNetworkMixin, LearningRateSchedule
from ray.rllib.utils.annotations import (
    override,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    concat_multi_gpu_td_errors,
    convert_to_torch_tensor,
    FLOAT_MIN,
    huber_loss,
    l2_loss,
    softmax_cross_entropy_with_logits,
)
from ray.rllib.utils.typing import TensorStructType, TensorType
import torch
import torch.nn.functional as F

import stocktradingv2
from stocktradingv2.agent.mydqn.my_dqn_model import IQNModel, MyDQNModel


Q_SCOPE = "q_func"
Q_TARGET_SCOPE = "target_q_func"
PRIO_WEIGHTS = "weights"


# model.tower_stats: key: "q_loss" -> "loss"


class ComputeTDErrorMixin:
    """Assign the `compute_td_error` method to the DQNTorchPolicy

    This allows us to prioritize on the worker side.
    """

    def __init__(self):
        def compute_td_error(input_dict: SampleBatch):
            # Do forward pass on loss to update td error attribute
            self.loss(self.model, None, input_dict)

            return self.model.tower_stats["td_error"]

        self.compute_td_error = compute_td_error


class MyDQNPolicy(
    LearningRateSchedule,
    TargetNetworkMixin,
    TorchPolicyV2,
):
    def __init__(self, observation_space, action_space, config):

        assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
            action_space, gymnasium.spaces.discrete.Discrete
        ), "Unsurported action space type: {}.".format(type(action_space))

        config = dict(
            stocktradingv2.agent.mydqn.my_dqn.MyDQNConfig().to_dict(), **config
        )
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        TargetNetworkMixin.__init__(self)
        ComputeTDErrorMixin.__init__(self)

        # Shortcut
        self.action_mask_fn = self.model.action_mask_fn

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()

    @override(TorchPolicyV2)
    def make_model_and_action_dist(
        self,
    ) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
        """Create model and action distribution function.

        Returns:
            ModelV2 model.
            ActionDistribution class.
        """
        # Shortcut
        model_config = self.config["model"]["custom_model_config"]
        model_cls = model_config.get("type")
        if model_cls is None or model_cls != "iqn":
            model_cls = MyDQNModel
        else:
            model_cls = IQNModel

        model = model_cls(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=None,
            model_config=model_config,
            name=Q_SCOPE,
        )

        self.target_model = model_cls(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=None,
            model_config=model_config,
            name=Q_TARGET_SCOPE,
        )
        self.target_model.eval()

        return model, TorchCategorical

    @override(TorchPolicyV2)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List["Episode"]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:

        with torch.no_grad():
            seq_lens = torch.ones(len(obs_batch), dtype=torch.int32)
            input_dict = self._lazy_tensor_dict(
                {
                    SampleBatch.CUR_OBS: obs_batch,
                    "is_training": False,
                }
            )
            state_batches = [
                convert_to_torch_tensor(s, self.device) for s in (state_batches or [])
            ]

            output, state_out = self.model(
                input_dict=input_dict, state=state_batches, seq_lens=seq_lens
            )
            if self.model.dqn_type != "cqn":
                output = output.mean(dim=2)
            else:
                num_atoms = self.model.n_atoms
                v_min = self.model.vmin
                v_max = self.model.vmax
                z = torch.arange(0.0, num_atoms, dtype=torch.float32).to(output.device)
                z = v_min + z * (v_max - v_min) / float(num_atoms - 1)
                output = torch.sum(z * F.softmax(output, dim=2), dim=2)

            if self.action_mask_fn:
                output += self.action_mask_fn(obs_batch)

            actions = output.argmax(dim=1)

        extra_fetches = {}
        if self.model.dqn_type == "iqn":
            extra_fetches["taus"] = self.model.tower_stats["taus"]

        return convert_to_numpy((actions, state_out, extra_fetches))

    @override(TorchPolicyV2)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Constructs the loss

        Args:
            model (ModelV2): The Model to calculate the loss for.
            train_batch: The training data.

        Returns:
            TensorType: A single loss tensor.
        """
        seq_lens = train_batch[SampleBatch.SEQ_LENS]
        input_dict = self._lazy_tensor_dict(
            {
                SampleBatch.OBS: train_batch[SampleBatch.OBS],
                "is_training": False,
            }
        )
        state_batches = [
            train_batch[k] for k in train_batch.keys() if "state_in" in k[:8]
        ]
        state_batches = [
            convert_to_torch_tensor(s, self.device) for s in (state_batches or [])
        ]

        # Q-network evaluation.
        q_t, state_out = self.model(
            input_dict=input_dict, state=state_batches, seq_lens=seq_lens
        )

        input_dict = self._lazy_tensor_dict(
            {
                SampleBatch.OBS: train_batch[SampleBatch.NEXT_OBS],
                "is_training": False,
            }
        )

        # Target Q-network evaluation.
        q_tp1, state_out = self.target_model(
            input_dict=input_dict, state=state_out, seq_lens=seq_lens
        )

        # Q scores for actions which we know were selected in the given state.
        one_hot_selection = F.one_hot(
            train_batch[SampleBatch.ACTIONS].long(), self.action_space.n
        ).unsqueeze(-1)
        q_t_selected = torch.sum(
            torch.where(q_t > FLOAT_MIN, q_t, torch.tensor(0.0, device=q_t.device))
            * one_hot_selection,
            1,
        )

        # Shortcut
        rewards = train_batch[SampleBatch.REWARDS]
        done_mask = train_batch[SampleBatch.TERMINATEDS].float()
        importance_weights = train_batch[PRIO_WEIGHTS]
        gamma = self.config["gamma"]
        n_step = self.config["n_step"]
        num_atoms = self.model.n_atoms

        # 1) Compute greedy action at t+1 step.
        # 2) Compute TD-error.
        # 3) compute loss.
        if self.model.dqn_type == "cqn":

            # cf. ray.rllib.algorithms.dqn.dqn_torch_policy.QLoss
            v_min = self.model.vmin
            v_max = self.model.vmax
            z = torch.arange(0.0, num_atoms, dtype=torch.float32).to(rewards.device)
            z = v_min + z * (v_max - v_min) / float(num_atoms - 1)

            # Best action at t+1.
            q_logits_tp1 = q_tp1
            q_expect_tp1 = torch.sum(z * F.softmax(q_logits_tp1, dim=2), dim=2)
            if self.action_mask_fn:
                q_expect_tp1 += self.action_mask_fn(train_batch[SampleBatch.NEXT_OBS])

            q_tp1_best_one_hot_selection = F.one_hot(
                torch.argmax(q_expect_tp1, 1), self.action_space.n
            ).unsqueeze(-1)
            q_logits_tp1_best = torch.sum(
                torch.where(
                    q_logits_tp1 > FLOAT_MIN,
                    q_logits_tp1,
                    torch.tensor(0.0, device=q_expect_tp1.device),
                )
                * q_tp1_best_one_hot_selection,
                1,
            )

            q_probs_tp1_best = F.softmax(q_logits_tp1_best, dim=1)

            # (batch_size, 1) * (1, num_atoms) -> (batch_size, num_atoms)
            r_tau = torch.unsqueeze(rewards, -1) + gamma**n_step * torch.unsqueeze(
                1.0 - done_mask, -1
            ) * torch.unsqueeze(z, 0)
            r_tau = torch.clamp(r_tau, v_min, v_max)
            b = (r_tau - v_min) / ((v_max - v_min) / float(num_atoms - 1))
            lb = torch.floor(b)
            ub = torch.ceil(b)

            # Indispensable judgement which is missed in most implementations
            # when b happens to be an integer, lb == ub, so pr_j(s', a*) will
            # be discarded because (ub-b) == (b-lb) == 0.
            floor_equal_ceil = ((ub - lb) < 0.5).float()

            # (batch_size, num_atoms, num_atoms)
            l_project = F.one_hot(lb.long(), num_atoms)
            # (batch_size, num_atoms, num_atoms)
            u_project = F.one_hot(ub.long(), num_atoms)
            ml_delta = q_probs_tp1_best * (ub - b + floor_equal_ceil)
            mu_delta = q_probs_tp1_best * (b - lb)
            ml_delta = torch.sum(l_project * torch.unsqueeze(ml_delta, -1), dim=1)
            mu_delta = torch.sum(u_project * torch.unsqueeze(mu_delta, -1), dim=1)
            m = ml_delta + mu_delta

            # Rainbow paper claims that using this cross entropy loss for
            # priority is robust and insensitive to `prioritized_replay_alpha`
            td_error = softmax_cross_entropy_with_logits(
                logits=q_t_selected, labels=m.detach()
            )
            loss = torch.mean(td_error * importance_weights)

            # Store values for stats function in model (tower), such that for
            # multi-GPU, we do not override them during the parallel loss phase.
            # TD-error tensor in final stats
            # will be concatenated and retrieved for each individual batch item.
            self.model.tower_stats["td_error"] = td_error
            self.model.tower_stats["loss"] = loss
            self.model.tower_stats["stat"] = {
                # TODO: better Q stats for dist dqn
            }

            return loss

        else:
            q_expect_tp1 = torch.mean(q_tp1, dim=2)
            if self.action_mask_fn:
                q_expect_tp1 += self.action_mask_fn(train_batch[SampleBatch.OBS])

            q_tp1_best_one_hot_selection = F.one_hot(
                torch.argmax(q_expect_tp1, 1), self.action_space.n
            ).unsqueeze(-1)
            q_tp1_best = torch.sum(
                torch.where(
                    q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
                )
                * q_tp1_best_one_hot_selection,
                1,
            )
            q_tp1_best_masked = (1.0 - done_mask).unsqueeze(-1) * q_tp1_best
            # compute RHS of bellman equation
            q_t_selected_target = (
                rewards.unsqueeze(-1) + gamma**n_step * q_tp1_best_masked
            )

            if self.model.dqn_type == "dqn":
                td_error = torch.squeeze(
                    q_t_selected_target.detach() - q_t_selected, dim=-1
                )
                loss_fn = (
                    huber_loss
                    if self.config["td_error_loss_fn"] == "huber"
                    else l2_loss
                )
                loss = torch.mean(importance_weights.float() * loss_fn(td_error))

                self.model.tower_stats["td_error"] = td_error
                self.model.tower_stats["loss"] = loss

                return loss

            else:
                if self.model.dqn_type == "qrdqn":
                    taus = (
                        torch.arange(
                            num_atoms, device=q_t_selected.device, dtype=torch.float
                        )
                        + 0.5
                    ) / num_atoms
                elif self.model.dqn_type == "iqn":
                    taus = self.model.tower_stats["taus"]

                pairwise_delta = q_t_selected_target.detach().unsqueeze(
                    -2
                ) - q_t_selected.unsqueeze(-1)
                abs_pairwise_delta = torch.abs(pairwise_delta)
                td_error = abs_pairwise_delta.mean()

                quantile_loss = torch.where(
                    abs_pairwise_delta > 1,
                    abs_pairwise_delta - 0.5,
                    pairwise_delta**2 * 0.5,
                )
                quantile_loss = (
                    torch.abs(
                        taus.unsqueeze(-2) - (pairwise_delta.detach() < 0).float()
                    )
                    * quantile_loss
                ).sum(-2)
                loss = torch.mean(importance_weights.float() * quantile_loss.mean(-1))

                self.model.tower_stats["td_error"] = td_error.reshape(-1)
                self.model.tower_stats["loss"] = loss.reshape(-1)
                self.model.tower_stats["stat"] = {
                    # TODO: better Q stats for dist dqn
                }

                return loss

    @override(TorchPolicyV2)
    def extra_compute_grad_fetches(self) -> Dict[str, Any]:
        # Call super to auto-add empty learner stats dict if needed.
        fetches = super().extra_compute_grad_fetches()
        fetches.update(convert_to_numpy(concat_multi_gpu_td_errors(self)))
        return fetches

    @override(TorchPolicyV2)
    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return convert_to_numpy(
            {
                "cur_lr": self.cur_lr,
                "td_error": torch.mean(torch.stack(self.get_tower_stats("td_error"))),
                "loss": torch.mean(torch.stack(self.get_tower_stats("loss"))),
                "mean_r_t": torch.mean(train_batch[SampleBatch.REWARDS]),
            }
        )

    @override(TorchPolicyV2)
    def optimizer(
        self,
    ) -> "torch.optim.Optimizer":

        # By this time, the models have been moved to the GPU - if any - and we
        # can define our optimizers using the correct CUDA variables.
        if not hasattr(self, "q_func_vars"):
            self.q_func_vars = self.model.variables()

        return torch.optim.Adam(
            self.q_func_vars, lr=self.config["lr"], eps=self.config["adam_epsilon"]
        )

    @override(TorchPolicyV2)
    def extra_grad_process(
        self, optimizer: torch.optim.Optimizer, loss: TensorType
    ) -> Dict[str, TensorType]:
        super().extra_grad_process(optimizer, loss)
        # Clip grads if configured.
        return apply_grad_clipping(self, optimizer, loss)

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[Dict[Any, SampleBatch]] = None,
        episode: Optional["Episode"] = None,
    ) -> SampleBatch:
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )
        # N-step Q adjustments.
        if self.config["n_step"] > 1:
            adjust_nstep(self.config["n_step"], self.config["gamma"], sample_batch)

        # Create dummy prio-weights (1.0) in case we don't have any in
        # the batch.
        if PRIO_WEIGHTS not in sample_batch:
            sample_batch[PRIO_WEIGHTS] = np.ones_like(sample_batch[SampleBatch.REWARDS])

        # Prioritize on the worker side.
        if sample_batch.count > 0 and self.config["replay_buffer_config"].get(
            "worker_side_prioritization", False
        ):
            td_errors = self.compute_td_error(sample_batch)
            # Retain compatibility with old-style Replay args
            epsilon = self.config.get("replay_buffer_config", {}).get(
                "prioritized_replay_eps"
            ) or self.config.get("prioritized_replay_eps")
            if epsilon is None:
                raise ValueError("prioritized_replay_eps not defined in config.")

            new_priorities = np.abs(convert_to_numpy(td_errors)) + epsilon
            sample_batch[PRIO_WEIGHTS] = new_priorities

        return sample_batch

    @override(TorchPolicyV2)
    def action_distribution_fn(
        self,
        model: ModelV2,
        *,
        obs_batch: TensorType,
        state_batches: TensorType,
        **kwargs,
    ) -> Tuple[TensorType, type, List[TensorType]]:
        """Action distribution function for this Policy.

        Args:
            model: Underlying model.
            obs_batch: Observation tensor batch.
            state_batches: Action sampling state batch.

        Returns:
            Distribution input.
            ActionDistribution class.
            State outs.
        """
        dist_class = self.dist_class
        with torch.no_grad():
            dist_inputs, state_out = model(obs_batch, state_batches, kwargs["seq_lens"])
        if self.model.dqn_type != "cqn":
            dist_inputs = torch.mean(dist_inputs, dim=-1)
        else:
            num_atoms = self.model.n_atoms
            v_min = self.model.vmin
            v_max = self.model.vmax
            z = torch.arange(0.0, num_atoms, dtype=torch.float32).to(dist_inputs.device)
            z = v_min + z * (v_max - v_min) / float(num_atoms - 1)
            dist_inputs = torch.sum(z * F.softmax(dist_inputs, dim=2), dim=2)

        if self.action_mask_fn:
            dist_inputs += self.action_mask_fn(obs_batch[SampleBatch.OBS])

        return dist_inputs, dist_class, state_out
