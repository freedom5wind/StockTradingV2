from typing import Callable, List, Type

import torch
import torch.nn as nn


TRAIN_START_DAY = "2008-01-01"
TRAIN_END_DAY = "2016-12-31"
TEST_START_DAY = "2017-01-01"
TEST_END_DAY = "2019-12-31"
TRADE_START_DAY = "2020-01-01"
TRADE_END_DAY = "2022-12-31"


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
) -> List[nn.Module]:
    """
    Borrow from stable_baselines3/common/torch_layers.
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param with_bias: If set to False, the layers will not learn an additive bias
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0], bias=with_bias)]
        if activation_fn is not None:
            modules.append(activation_fn())
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        if activation_fn is not None:
            modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


def quantile_huber_loss(
    q_dist_t: torch.Tensor,
    q_dist_tp1: torch.Tensor,
    taus: torch.Tensor,
    sum_over_quantiles: bool = True,
) -> torch.Tensor:
    """
    The quantile-regression loss.
    Code borrowed from sb3_contrib.common.utils.quantile_huber_loss

    :param q_dist_t: distribution at t. Shape: (batch_size, num_atoms).
    :param q_dist_tp1: distribution at t + 1. Shape: (batch_size, num_atoms).
    :param taus: taus correspond to quantiles of q_dist_t. Shape: (batch_size, num_atoms).
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """
    if q_dist_t.ndim != q_dist_tp1.ndim:
        raise ValueError(
            f"Error: The dimension of distribution at t ({q_dist_t.ndim}) needs to match "
            f"the dimension of distributions at t+1 ({q_dist_tp1.ndim})."
        )
    if q_dist_t.shape[0] != q_dist_tp1.shape[0]:
        raise ValueError(
            f"Error: The batch size of distribution at t ({q_dist_t.shape[0]}) needs to match "
            f"the batch size of distributions at t+1 ({q_dist_tp1.shape[0]})."
        )

    # target_dist: (batch_size, num_atoms) -> (batch_size, 1, num_atoms)
    # current_dist: (batch_size, num_atoms) -> (batch_size, num_atoms, 1)
    # pairwise_delta: (batch_size, num_atoms, num_atoms)
    pairwise_delta = q_dist_tp1.unsqueeze(-2) - q_dist_t.unsqueeze(-1)
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(
        abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5
    )
    loss = (
        torch.abs(taus.unsqueeze(-2) - (pairwise_delta.detach() < 0).float())
        * huber_loss
    )
    if sum_over_quantiles:
        # (batch_size, num_atoms)
        loss = loss.sum(dim=-1)
    else:
        # (batch_size, num_atoms, num_atoms)
        loss = loss
    return loss


def CPW_factory(eta: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def CPW(taus: torch.Tensor) -> torch.Tensor:
        taus = (taus**eta) / ((taus**eta + (1 - taus) ** eta) ** (1 / eta))
        return taus

    return CPW


def Wang_factory(eta: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def Wang(taus: torch.Tensor) -> torch.Tensor:
        n = torch.distributions.normal.Normal(
            torch.zeros_like(taus), torch.ones_like(taus)
        )
        finfo = torch.finfo(taus.dtype)
        # clamp to prevent +-inf
        taus = n.cdf(torch.clamp(n.icdf(taus) + eta, min=finfo.min, max=finfo.max))
        return taus

    return Wang


def CVaR_factory(eta: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def CVaR(taus: torch.Tensor) -> torch.Tensor:
        taus = taus * eta
        return taus

    return CVaR


def Norm_factory(eta: int) -> Callable[[torch.Tensor], torch.Tensor]:
    def Norm(taus: torch.Tensor) -> torch.Tensor:
        taus = (
            taus.unsqueeze(-1)
            .repeat_interleave(repeats=eta, dim=-1)
            .uniform_(0, 1)
            .mean(axis=-1)
        )
        return taus

    return Norm


def Pow_factory(eta: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def Pow(taus: torch.Tensor) -> torch.Tensor:
        if eta >= 0:
            taus = taus ** (1 / (1 + eta))
        else:
            taus = 1 - (1 - taus) ** (1 / (1 - eta))
        return taus

    return Pow


DEFAULT_RISK_DISTORTION_MEASURES = {
    "CPW_0.71": CPW_factory(0.71),
    "Wang_-0.75": Wang_factory(-0.75),
    "Wang_0.75": Wang_factory(0.75),
    "CVaR_0.25": CVaR_factory(0.25),
    "CVaR_0.4": CVaR_factory(0.4),
    "Norm_3": Norm_factory(3),
    "Pow_-2": Pow_factory(-2),
}
