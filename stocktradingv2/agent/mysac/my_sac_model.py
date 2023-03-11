from typing import List, Dict, Union, Tuple

import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.sac.rnnsac_torch_model import RNNSACTorchModel
from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import override, force_list
from ray.rllib.utils.torch_utils import huber_loss, softmax_cross_entropy_with_logits
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import tree
import torch
import torch.nn as nn
import torch.nn.functional as F

from stocktradingv2.agent.mydqn.my_dqn_model import CosineEmbeddingNetwork
from stocktradingv2.agent.mydqn.my_dqn_policy import PRIO_WEIGHTS
from stocktradingv2.utils.utils import create_mlp


class MyLSTMModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.input_dim = int(np.product(obs_space.shape))
        self.lstm_dim = self.model_config.get("lstm_dim", 64)
        self.lstm_layers = self.model_config.get("lstm_layers", 1)
        self.net_arch = self.model_config.get("net_arch", [256, 256])
        self.activation_fn = self.model_config.get("activation_fn", nn.ReLU)
        self.num_atoms = self.num_outputs = num_outputs

        self.dqn_type = model_config.get("type", "dqn")
        if self.dqn_type == "iqn":
            self.num_outputs = 1
            self.risk_distortion_measure = self.model_config.get(
                "risk_distortion_measure", None
            )
            self.cos_embedding_dim = self.model_config.get("cos_embedding_dim", 64)
            # Build cosine embedding here.
            self.cos_embedding = CosineEmbeddingNetwork(
                self.lstm_dim, self.cos_embedding_dim
            )

        self._build()

    @override(ModelV2)
    def get_initial_state(self):
        return [
            torch.zeros(self.lstm_layers, self.lstm_dim),
            torch.zeros(self.lstm_layers, self.lstm_dim),
        ]

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        flat_inputs = input_dict["obs_flat"].float()
        # Note that max_seq_len != input_dict.max_seq_len != seq_lens.max()
        # as input_dict may have extra zero-padding beyond seq_lens.max().
        # Use add_time_dimension to handle this
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            seq_lens=seq_lens,
            framework="torch",
            time_major=self.time_major,
        )

        # Forward lstm.
        state = tree.map_structure(lambda x: x.transpose(0, 1), state)
        output, state_out = self.lstm(inputs, state)
        state_out = tree.map_structure(lambda x: x.transpose(0, 1), state_out)
        state_out = list(state_out)
        output = output.reshape([-1, self.lstm_dim])

        if self.dqn_type == "iqn":
            B = output.shape[0]
            N = self.num_atoms
            # sample taus
            taus = torch.rand(B, N, dtype=output.dtype, device=output.device)
            if self.risk_distortion_measure is not None:
                self.risk_distortion_measure(taus)
            self.tower_stats["taus"] = taus
            # shape: (B, N, self.lstm_dim)
            tau_embedding = self.cos_embedding(taus)
            output = output.view(B, 1, self.lstm_dim) * tau_embedding
            output = output.view(B * N, self.lstm_dim)
            output = self.mlp(output).view(B, N)
        else:
            # Forward mlp
            output = self.mlp(output).reshape(-1, self.num_outputs)

        return output, state_out

    def _build(self):
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
        )
        mlp = create_mlp(
            input_dim=self.lstm_dim,
            output_dim=self.num_outputs if self.dqn_type != "iqn" else 1,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )
        self.mlp = nn.Sequential(*mlp)


class EnsembleModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.num_critics = model_config.get("num_critics", 3)
        self.dqn_type = model_config.get("type", "dqn")
        if self.dqn_type == "dqn":
            self.num_atoms = 1
        else:
            self.num_atoms = model_config.get("num_atoms")
        if self.dqn_type == "cqn":
            self.vmin = self.model_config.get("vmin", -10)
            self.vmax = self.model_config.get("vmax", 10)
            self.atoms = torch.arange(0.0, self.num_atoms, dtype=torch.float32)
        self.models = []
        for i in range(self.num_critics):
            model = MyLSTMModel(
                obs_space, action_space, self.num_atoms, model_config, f"{name}_{i}"
            )
            self.add_module(f"critic_{i}", model)
            self.models.append(model)

        # To control forward output
        self._raw = False

    @override(ModelV2)
    def get_initial_state(self):
        states = []
        for model in self.models:
            states.extend(model.get_initial_state())
        return states

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        outputs = []
        states_out = []
        for i in range(self.num_critics):
            output, state_out = self.models[i](
                input_dict, state[i * 2 : i * 2 + 2], seq_lens
            )
            outputs.append(output)
            states_out.extend(state_out)
        outputs = torch.stack(outputs, dim=-2)

        # Raw output for loss calculation.
        if self._raw:
            if self.dqn_type == "iqn":
                self.tower_stats["taus"] = torch.stack(
                    [model.tower_stats["taus"].unsqueeze(-2) for model in self.models],
                    dim=-2,
                )
            return outputs, states_out

        if self.dqn_type != "cqn":
            outputs = outputs.mean(dim=-1)
        else:
            outputs = torch.sum(
                self.atoms.to(outputs.device) * F.softmax(outputs, dim=-1), dim=-1
            )

        return outputs.mean(dim=1), states_out

    def _get_last_taus(self):
        if self.dqn_type != "iqn":
            return None
        taus = []
        for model in self.models:
            taus.append(model.tower_stats["taus"])
        return torch.concat(taus, dim=1)


class MySACModel(RNNSACTorchModel):
    @override(RNNSACTorchModel)
    def _get_q_value(
        self,
        model_out: TensorType,
        actions,
        net,
        state_in: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        # Continuous case -> concat actions to model_out.
        if actions is None:
            actions = torch.zeros(
                model_out[SampleBatch.OBS].shape[:1] + self.action_space.shape,
                dtype=model_out[SampleBatch.OBS].dtype,
                device=model_out[SampleBatch.OBS].device,
            )

        # Make sure that, if we call this method twice with the same
        # input, we don't concatenate twice
        model_out["obs_and_action_concatenated"] = True

        if self.concat_obs_and_actions:
            model_out[SampleBatch.OBS] = torch.cat(
                [model_out[SampleBatch.OBS], actions], dim=-1
            )
        else:
            model_out[SampleBatch.OBS] = force_list(model_out[SampleBatch.OBS]) + [
                actions
            ]

        # Switch on training mode (when getting Q-values, we are usually in
        # training).
        model_out["is_training"] = True

        # q_net_output shape: [Batch_size, num_critics, num_atoms]
        q_net_output, states_out = net(model_out, state_in, seq_lens)

        return q_net_output, states_out

    @override(SACTorchModel)
    def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
        return MyLSTMModel(
            obs_space, self.action_space, num_outputs, policy_model_config, name
        )

    @override(SACTorchModel)
    def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
        self.concat_obs_and_actions = False
        if self.discrete:
            input_space = obs_space
        else:
            orig_space = getattr(obs_space, "original_space", obs_space)
            if isinstance(orig_space, gym.spaces.Box) and len(orig_space.shape) == 1:
                input_space = gym.spaces.Box(
                    float("-inf"),
                    float("inf"),
                    shape=(orig_space.shape[0] + action_space.shape[0],),
                )
                self.concat_obs_and_actions = True
            else:
                input_space = gym.spaces.Tuple([orig_space, action_space])

        return EnsembleModel(
            input_space, action_space, num_outputs, q_model_config, name
        )

    @override(ModelV2)
    def custom_loss(
        self, policy_loss: TensorType, loss_inputs: Dict[str, TensorType]
    ) -> Union[List[TensorType], TensorType]:
        """Constructs the loss for the Soft Actor Critic.

        Args:
            policy: The Policy to calculate the loss for.
            model (ModelV2): The Model to calculate the loss for.
            dist_class (Type[TorchDistributionWrapper]: The action distr. class.
            train_batch: The training data.

        Returns:
            Union[TensorType, List[TensorType]]: A single loss tensor or a list
                of loss tensors.
        """
        target_model = self.get_target_model()
        self.q_net._raw = target_model.q_net._raw = True
        if self.twin_q_net is not None:
            self.twin_q_net._raw = target_model.twin_q_net._raw = True

        i = 0
        state_batches = []
        while "state_in_{}".format(i) in loss_inputs:
            state_batches.append(loss_inputs["state_in_{}".format(i)])
            i += 1
        assert state_batches
        seq_lens = loss_inputs.get(SampleBatch.SEQ_LENS)

        model_out_t, state_in_t = self(
            SampleBatch(
                obs=loss_inputs[SampleBatch.CUR_OBS],
                prev_actions=loss_inputs[SampleBatch.PREV_ACTIONS]
                if SampleBatch.PREV_ACTIONS in loss_inputs
                else None,
                prev_rewards=loss_inputs[SampleBatch.PREV_REWARDS]
                if SampleBatch.PREV_REWARDS in loss_inputs
                else None,
                _is_training=True,
            ),
            state_batches,
            seq_lens,
        )
        states_in_t = self.select_state(state_in_t, ["policy", "q", "twin_q"])

        alpha = torch.exp(self.log_alpha)

        # Discrete case.
        if self.discrete:
            assert NotImplementedError, "Discrete action space is not suppported."

        # Continuous actions case.
        # Sample single actions from distribution.
        action_dist_class = self.action_dist_class
        states_out_t = []
        action_dist_inputs_t, state_out_t = self.get_action_model_outputs(
            model_out_t, states_in_t["policy"], seq_lens
        )
        states_out_t.extend(state_out_t)
        action_dist_t = action_dist_class(
            action_dist_inputs_t,
            self,
        )
        policy_t = action_dist_t.sample()

        log_pis_t = action_dist_t.logp(policy_t)
        # Q-values for the actually selected actions.
        q_t, state_out_t = self.get_q_values(
            SampleBatch(
                obs=loss_inputs[SampleBatch.CUR_OBS],
                prev_actions=loss_inputs[SampleBatch.PREV_ACTIONS]
                if SampleBatch.PREV_ACTIONS in loss_inputs
                else None,
                prev_rewards=loss_inputs[SampleBatch.PREV_REWARDS]
                if SampleBatch.PREV_REWARDS in loss_inputs
                else None,
                _is_training=True,
            ),
            states_in_t["q"],
            seq_lens,
            loss_inputs[SampleBatch.ACTIONS],
        )
        states_out_t.extend(state_out_t)

        # Q-values for current policy in given current state.
        q_t_det_policy, _ = self.get_q_values(
            SampleBatch(
                obs=loss_inputs[SampleBatch.CUR_OBS],
                prev_actions=loss_inputs[SampleBatch.PREV_ACTIONS]
                if SampleBatch.PREV_ACTIONS in loss_inputs
                else None,
                prev_rewards=loss_inputs[SampleBatch.PREV_REWARDS]
                if SampleBatch.PREV_REWARDS in loss_inputs
                else None,
                _is_training=True,
            ),
            states_in_t["q"],
            seq_lens,
            policy_t,
        )
        taus = self.q_net._get_last_taus()

        model_out_tp1, state_in_tp1 = self(
            SampleBatch(
                obs=loss_inputs[SampleBatch.NEXT_OBS],
                prev_actions=loss_inputs[SampleBatch.ACTIONS],
                prev_rewards=loss_inputs[SampleBatch.REWARDS],
                _is_training=True,
            ),
            state_batches,
            seq_lens,
        )

        # Use states from previous forward.
        state_in_tp1 = states_out_t
        states_in_tp1 = self.select_state(state_in_tp1, ["policy", "q", "twin_q"])

        target_model_out_tp1, target_state_in_tp1 = target_model(
            SampleBatch(
                obs=loss_inputs[SampleBatch.NEXT_OBS],
                prev_actions=loss_inputs[SampleBatch.ACTIONS],
                prev_rewards=loss_inputs[SampleBatch.REWARDS],
                _is_training=True,
            ),
            state_batches,
            seq_lens,
        )
        target_states_in_tp1 = target_model.select_state(
            state_in_tp1, ["policy", "q", "twin_q"]
        )

        action_dist_inputs_t, _ = self.get_action_model_outputs(
            model_out_tp1, states_in_tp1["policy"], seq_lens
        )
        action_dist_tp1 = action_dist_class(
            action_dist_inputs_t,
            self,
        )
        policy_tp1 = action_dist_tp1.sample()

        log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1).unsqueeze(
            -1
        )

        # Target q network evaluation.
        q_tp1, _ = target_model.get_q_values(
            target_model_out_tp1, target_states_in_tp1["q"], seq_lens, policy_tp1
        )

        # q_tp1 -= alpha * log_pis_tp1

        q_tp1_masked = (
            1.0
            - loss_inputs[SampleBatch.TERMINATEDS].unsqueeze(-1).unsqueeze(-1).float()
        ) * q_tp1

        # compute RHS of bellman equation
        gamma = self.model_config.get("gamma", 0.99)
        n_step = self.model_config.get("n_step", 1)
        dqn_type = self.q_net.model_config.get("type", "dqn")
        tqc = self.q_net.model_config.get("tqc", False)
        num_atoms = self.q_net.model_config.get("num_atoms")
        importance_weights = loss_inputs[PRIO_WEIGHTS]
        reward = loss_inputs[SampleBatch.REWARDS]
        batch_size = len(reward)

        if dqn_type == "dqn":
            # Q value shape: [batch_size, num_critic, 1]
            q_t = q_t.squeeze(dim=2).mean(dim=1)
            q_tp1_masked = q_tp1_masked.squeeze(dim=2).mean(dim=1)

            q_t_selected_target = (reward + (gamma**n_step) * q_tp1_masked).detach()
            td_error = torch.abs(q_t_selected_target - q_t)
            critic_loss = torch.mean(importance_weights * huber_loss(td_error))
            q_t_det_policy = q_t_det_policy.view(-1)

        elif dqn_type == "cqn":
            v_min = self.q_net.model_config.get("vmin")
            v_max = self.q_net.model_config.get("vmax")
            z = torch.arange(0.0, num_atoms, dtype=torch.float32).to(q_t.device)
            z = v_min + z * (v_max - v_min) / float(num_atoms - 1)
            r_tau = torch.unsqueeze(reward, -1) + gamma**n_step * torch.unsqueeze(
                1.0 - loss_inputs[SampleBatch.TERMINATEDS].float(), -1
            ) * torch.unsqueeze(z, 0)
            r_tau = torch.clamp(r_tau, v_min, v_max)
            b = (r_tau - v_min) / ((v_max - v_min) / float(num_atoms - 1))
            lb = torch.floor(b)
            ub = torch.ceil(b)
            floor_equal_ceil = ((ub - lb) < 0.5).float()

            # Average atoms at the same position in different critics.
            # Shape: [batch_size, num_critics, num_atoms] -> [...]
            q_t = q_t.mean(dim=1)
            q_tp1_masked = q_tp1_masked.mean(dim=1)
            q_probs_tp1 = F.softmax(q_tp1_masked, dim=1)

            # (batch_size, num_atoms, num_atoms)
            l_project = F.one_hot(lb.long(), num_atoms)
            u_project = F.one_hot(ub.long(), num_atoms)
            ml_delta = q_probs_tp1 * (ub - b + floor_equal_ceil)
            mu_delta = q_probs_tp1 * (b - lb)
            ml_delta = torch.sum(l_project * torch.unsqueeze(ml_delta, -1), dim=1)
            mu_delta = torch.sum(u_project * torch.unsqueeze(mu_delta, -1), dim=1)
            m = ml_delta + mu_delta
            td_error = softmax_cross_entropy_with_logits(logits=q_t, labels=m.detach())
            critic_loss = torch.mean(td_error * importance_weights)
            q_t_det_policy = torch.sum(
                F.softmax(q_t_det_policy.mean(dim=1), dim=1) * z, dim=1
            )

        elif dqn_type == "iqn":
            # Gather samples from critics
            q_t = q_t.reshape(batch_size, -1)
            q_tp1_masked = q_tp1_masked.reshape(batch_size, -1)

            q_t_selected_target = (
                reward.unsqueeze(1) + (gamma**n_step) * q_tp1_masked
            ).detach()
            pairwise_delta = q_t_selected_target.unsqueeze(-2) - q_t.unsqueeze(-1)
            abs_pairwise_delta = torch.abs(pairwise_delta)
            q_quantile_loss = torch.where(
                abs_pairwise_delta > 1,
                abs_pairwise_delta - 0.5,
                pairwise_delta**2 * 0.5,
            )
            quantile_loss = (
                torch.abs(taus.unsqueeze(-1) - (pairwise_delta.detach() < 0).float())
                * q_quantile_loss
            ).sum(-2)
            critic_loss = torch.mean(
                importance_weights.float() * quantile_loss.mean(dim=-1).sum(dim=-1)
            )
            td_error = abs_pairwise_delta.detach().mean(dim=(1, 2))
            q_t_det_policy = q_t_det_policy.mean(dim=(1, 2))

        elif dqn_type == "qrdqn":
            # tqc: number of truncated atoms
            tqc = self.model_config.get("tqc", None)
            q_tp1_masked = q_tp1_masked.reshape(batch_size, -1)
            if tqc and tqc > 0:
                q_tp1_masked = torch.sort(q_tp1_masked, descending=True)
                q_tp1_masked = q_tp1_masked[:, tqc:]
            q_t_selected_target = (
                reward.unsqueeze(1) + (gamma**n_step) * q_tp1_masked
            ).detach()
            taus = (
                torch.arange(
                    q_t.shape[-1],
                    device=q_t.device,
                    dtype=q_t.dtype,
                )
                + 0.5
            ) / q_t.shape[-1]

            pairwise_delta = q_t_selected_target.view(
                *q_t_selected_target.shape, 1, 1
            ) - q_t.unsqueeze(-1)
            abs_pairwise_delta = torch.abs(pairwise_delta)
            q_quantile_loss = torch.where(
                abs_pairwise_delta > 1,
                abs_pairwise_delta - 0.5,
                pairwise_delta**2 * 0.5,
            )
            quantile_loss = (
                (
                    torch.abs(
                        taus.view(1, 1, -1, 1) - (pairwise_delta.detach() < 0).float()
                    )
                    * q_quantile_loss
                )
                .mean(-1)
                .sum(-1)
                .mean(-1)
            )  # Aggregate n_target_quantiles, current_quantiles, n_critics in turn.
            critic_loss = torch.mean(importance_weights.float() * quantile_loss)
            td_error = abs_pairwise_delta.detach().mean(dim=(1, 2, 3))
            q_t_det_policy = q_t_det_policy.mean(dim=(1, 2))

        # Alpha- and actor losses.
        # Note: In the papers, alpha is used directly, here we take the log.
        # Discrete case: Multiply the action probs as weights with the original
        # loss terms (no expectations needed).
        alpha_loss = -torch.mean(
            self.log_alpha * (log_pis_t + self.target_entropy).detach()
        )
        # Note: Do not detach q_t_det_policy here b/c is depends partly
        # on the policy vars (policy sample pushed through Q-net).
        # However, we must make sure `actor_loss` is not used to update
        # the Q-net(s)' variables.
        actor_loss = torch.mean(alpha.detach() * log_pis_t - q_t_det_policy)

        self.q_net._raw = target_model.q_net._raw = False
        if self.twin_q_net is not None:
            self.twin_q_net._raw = target_model.twin_q_net._raw = False

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        self.tower_stats["q_t"] = q_t_det_policy
        self.tower_stats["policy_t"] = policy_t
        self.tower_stats["log_pis_t"] = log_pis_t
        self.tower_stats["actor_loss"] = actor_loss
        self.tower_stats["critic_loss"] = critic_loss
        self.tower_stats["alpha_loss"] = alpha_loss
        # Store per time chunk (b/c we need only one mean
        # prioritized replay weight per stored sequence).
        self.tower_stats["td_error"] = td_error
        self.tower_stats["reward_t"] = loss_inputs[SampleBatch.REWARDS]

        # Return all loss terms corresponding to our optimizers.
        return tuple([actor_loss, critic_loss, alpha_loss])
