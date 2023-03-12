import tree
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import (
    override,
)
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import torch
import torch.nn as nn

from stocktradingv2.utils.utils import create_mlp


# TODO: Unify two different models and derive directly from TorchModeV2


class MyDQNModel(RecurrentNetwork, nn.Module):
    """DQN + CQN + QR-DQN"""

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
        self.dqn_type = self.model_config.get("type", "dqn")

        if self.dqn_type == "cqn":
            self.num_atoms = self.model_config.get("num_atoms", 50)
            self.vmin = self.model_config.get("vmin", -10)
            self.vmax = self.model_config.get("vmax", 10)
            self.output_dim = action_space.n * self.num_atoms
        elif self.dqn_type == "qrdqn":
            self.num_atoms = self.model_config.get("num_atoms", 20)
            self.output_dim = action_space.n * self.num_atoms
        else:
            self.num_atoms = 1
            self.output_dim = action_space.n * self.num_atoms
        self.num_outputs = self.action_space.n * self.num_atoms

        self.action_mask_fn = self.model_config.get("action_mask_fn", None)

        self._build()

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [
            torch.zeros(self.lstm_layers, self.lstm_dim),
            torch.zeros(self.lstm_layers, self.lstm_dim),
        ]

    @override(RecurrentNetwork)
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
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.action_space.n, self.num_atoms])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(
        self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        states = tree.map_structure(lambda x: x.transpose(0, 1), state)
        output, state_out = self.lstm(inputs, states)
        state_out = tree.map_structure(lambda x: x.transpose(0, 1), state_out)
        output = output.reshape([-1, self.lstm_dim])
        output = self.mlp(output).view(-1, self.action_space.n, self.num_atoms)
        return output, list(state_out)

    def _build(self):
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
        )
        mlp = create_mlp(
            input_dim=self.lstm_dim,
            output_dim=self.output_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )
        self.mlp = nn.Sequential(*mlp)


class CosineEmbeddingNetwork(nn.Module):
    """Borrow from https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch/fqf_iqn_qrdqn/network.py"""

    def __init__(self, feature_dim, cosine_embedding=64):

        super().__init__()

        self.feature_dim = feature_dim
        self.cosine_embedding = cosine_embedding

        self.net = nn.Sequential(nn.Linear(cosine_embedding, feature_dim), nn.ReLU())

    def forward(self, taus):
        # B: batch size, N: sample num
        B = taus.shape[0]
        N = taus.shape[1]

        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=0, end=self.cosine_embedding, dtype=taus.dtype, device=taus.device
        ).view(1, 1, self.cosine_embedding)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(taus.view(B, N, 1) * i_pi).view(
            B * N, self.cosine_embedding
        )

        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(B, N, self.feature_dim)

        return tau_embeddings


class IQNModel(RecurrentNetwork, nn.Module):
    """IQN"""

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
        self.risk_distortion_measure = self.model_config.get("risk_distortion_measure")
        self.cos_embedding_dim = self.model_config.get("cos_embedding_dim", 64)
        self.net_arch = self.model_config.get("net_arch", [256, 256])
        self.activation_fn = self.model_config.get("activation_fn", nn.ReLU)
        self.dqn_type = "iqn"
        self.num_atoms = self.model_config.get("num_atoms", 64)

        self.action_mask_fn = self.model_config.get("action_mask_fn", None)
        self.num_outputs = self.action_space.n

        self._build()

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [
            torch.zeros(self.lstm_layers, self.lstm_dim),
            torch.zeros(self.lstm_layers, self.lstm_dim),
        ]

    @override(RecurrentNetwork)
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
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.action_space.n, self.num_atoms])
        return output, new_state

    @override(RecurrentNetwork)
    def forward_rnn(
        self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        states = tree.map_structure(lambda x: x.transpose(0, 1), state)
        output, state_out = self.lstm(inputs, states)
        state_out = tree.map_structure(lambda x: x.transpose(0, 1), state_out)

        # B: batch size, N: sample num
        output = output.reshape([-1, self.lstm_dim])
        B = output.shape[0]
        N = self.num_atoms

        # sample taus
        taus = torch.rand(B, N, dtype=output.dtype, device=output.device)
        if self.risk_distortion_measure is not None:
            self.risk_distortion_measure(taus)
        self.tower_stats["taus"] = taus

        # shape: (B, N, self.lstm_dim)
        tau_embedding = self.cos_embedding(taus)

        embedding = output.view(B, 1, self.lstm_dim) * tau_embedding
        embedding = embedding.view(B * N, self.lstm_dim)

        output = self.mlp(embedding).view(B, N, self.action_space.n).swapaxes(1, 2)

        return output, list(state_out)

    def _build(self):

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_dim,
            num_layers=self.lstm_layers,
            batch_first=True,
        )

        self.cos_embedding = CosineEmbeddingNetwork(
            self.lstm_dim, self.cos_embedding_dim
        )

        mlp = create_mlp(
            input_dim=self.lstm_dim,
            output_dim=self.action_space.n,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )
        self.mlp = nn.Sequential(*mlp)
