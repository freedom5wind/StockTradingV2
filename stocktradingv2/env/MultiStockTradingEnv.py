import logging
import warnings
from typing import Dict, Tuple

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
from ray.rllib.utils.spaces.simplex import Simplex
from ray.tune.registry import register_env
import scipy


class MultiStockTradingEnv(gym.Env):
    """Multiple stock trading environment (also known as
    portfolio management) with continuous action space.

    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config) -> None:

        assert (
            "df" in env_config and len(env_config["df"]) > 0
        ), "Input data frame can't be empty."

        self.df = env_config.get("df")

        assert (
            isinstance(self.df.index, pd.MultiIndex)
            and "date" in self.df.index.names
            and "tic" in self.df.index.names
        ), "Data frame should have multi-index with date and tic."
        assert "change" in self.df.columns, "Data frame should have column 'change'."
        assert self.df.isna().sum().sum() == 0, "Nan in input date frame."

        self.cost_pct = env_config.get("cost_pct", 0.001)
        self.stack_frame = max(int(env_config.get("stack_frame", 1)), 1)
        self.verbose = env_config.get("verbose", False)

        self.df.reset_index(inplace=True)
        self.num_tickers = len(self.df["tic"].unique())
        self.feature_dims = len(self.df.columns) - 2  # exclude date and tic
        self.date = pd.Series(self.df.date.unique())
        self.num_day = len(self.date)
        self.df.set_index(["date", "tic"], inplace=True)

        self.change_matrix = self.df["change"]

        self.feature_matrix = self.df[[col for col in self.df.columns]]
        self.change_matrix = (
            self.change_matrix.to_numpy().reshape(-1, self.num_tickers) / 100.0
        )  # Percentage to decimal.
        self.feature_matrix = self.feature_matrix.to_numpy().reshape(
            -1, self.num_tickers, self.feature_dims
        )

        # Observation space includes n features for each day and previous action.
        self.observation_shape = (
            self.stack_frame * self.feature_dims * self.num_tickers
            + (self.num_tickers + 1),
        )
        # Action space includes cash(index 0) and n tickers.
        # self.action_space = Box(low=-1e5, high=1e5, shape=(self.num_tickers + 1,))
        self.action_memory = Simplex(shape=(self.num_tickers + 1,))
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape
        )

        self.logger = logging.getLogger(__name__)

        self.reset()

    def reset(self, *, seed=None, options=None) -> Tuple[np.array, Dict]:
        self.terminal = False
        # skip n days for frame stacking
        self.day = self.stack_frame - 1
        self.asset = 1.0
        self.portfolio = np.zeros((self.num_tickers + 1,))
        self.portfolio[0] = 1.0
        self.asset_memory = [self.asset]
        self.portfolio_memory = [self.portfolio]
        self.action_memory = []
        self.reward_memory = []

        return self.state, {}

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, Dict]:
        assert len(action) == self.num_tickers + 1, f"Invalid action {action}."
        # action = scipy.special.softmax(action)
        action.flags.writeable = True
        action = np.clip(action, 0.0, 1.0)
        action /= action.sum()
        if np.abs(action.sum() - 1.0) > 1e-8:
            action[0] += 1 - action.sum()
            warnings.warn(f"{action.sum()} != 0. Adjust cash position to {action[0]}.")

        self.terminal = self.day >= self.date.shape[0] - 2

        # Exclude cash to compute cost
        cost = np.abs(self.portfolio - action)[1:].sum() * self.cost_pct
        self.asset *= 1 - cost
        self.portfolio = action

        # state: s -> s+1
        self.day += 1

        changes = np.insert(self.change_matrix[self.day], 0, 0)
        self.asset *= 1 + np.dot(changes, self.portfolio)
        self.asset_memory.append(self.asset)

        assert self.asset_memory[-2] > 0, f"Non positive asset in: {self.asset_memory}"
        reward = np.log2(self.asset_memory[-1] / self.asset_memory[-2])

        self.action_memory.append(action)
        self.reward_memory.append(reward)

        if self.terminal:
            msg = "-" * 30 + "\n"
            msg += f"Reward memory: {self.reward_memory}\n \
                        Action memory: {self.action_memory}\n \
                        Asset memory: {self.asset_memory}"
            if self.verbose:
                print(msg)
            else:
                self.logger.log(logging.INFO, msg=msg)

        return self.state, reward, self.terminal, False, {}

    @property
    def state(self) -> None:
        return np.concatenate(
            [self.feature_matrix[self.day, :, :].flatten(), self.portfolio], axis=0
        )


def _env_creator(env_config):
    return MultiStockTradingEnv(env_config)


register_env("MultiStockTrading", _env_creator)
