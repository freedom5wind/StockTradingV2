from collections import deque
import logging
from typing import Any, Dict, List, Tuple

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import pandas as pd
from ray.tune.registry import register_env
import torch as th


class SingleStockTradingEnv(gym.Env):
    """A single stock trading environment with discrete action space"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_config: Dict[str, Any]):
        """
        :param df: Data frame of a ticker. First column should be "date" and
            should have column "close". Use a copy of "close" if you want a
            "close" feature.
        :param initial_cash: initial amount of cash
        :param buy_cost_pct: cost for buying shares
        :param sell_cost_pct: cost for selling shares
        :param stack_frame: how many days for data present in observation space
        :param reward_bias: whether to adjust reward based on ticker's average daily return
        :param ignore_close: whether to return 'close' column to state
        """

        assert (
            "df" in env_config and len(env_config["df"]) > 0
        ), "Input data frame can't be empty."

        self._dfs = env_config.get("df")
        if not isinstance(self._dfs, List):
            self._dfs = [self._dfs]
        self.num_df = env_config.get("num_df", len(self._dfs))
        self._dfs = deque(self._dfs[: self.num_df])
        self.df = self._dfs[0]

        self.seed = env_config.get("seed")
        self.shuffle = env_config.get("shuffle", False)
        self.np_random = np.random.RandomState(self.seed)

        self.initial_cash = env_config.get("initial_cash", 1000_000)
        self.cost_pct = env_config.get("cost_pct", 0.001)
        self.stack_frame = max(int(env_config.get("stack_frame", 1)), 1)
        self.verbose = env_config.get("verbose", False)

        self.date = self.df["date"]
        self.prices = self.df["close"].to_numpy()
        self.data = self.df[
            [col for col in self.df.columns if col != "date" and col != "close"]
        ].to_numpy()
        self.num_day, self.feature_dims = self.data.shape
        self.action_space = Discrete(3)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                (self.stack_frame) * (self.feature_dims) + 1,
            ),  # 1 for position state.
        )

        for i in self._dfs:
            assert "close" in i.columns, "Data frame doesn't have column 'close'."
            assert (
                i.columns[0] == "date"
            ), "First column of the data frame is not 'date'"
            assert i.isna().sum().sum() == 0, "Nan in input date frame."
            assert (
                i.shape[0] == self.data.shape[0]
                and i.shape[1] - 2 == self.data.shape[1]
            ), f"{i.shape}, {self.data.shape} not match."

        # TODO: figure out how to log in rollout worker process.
        self.logger = logging.getLogger(__name__)

        self.success_trade = 0
        self.total_trading = 0
        self.account = self.initial_cash
        self.asset_memory = [self.account]

        self.reset()

    def reset(self, *, seed=None, options=None) -> Tuple[np.array, Dict]:
        # if self.shuffle:
        #     np.random.shuffle(self._dfs)
        # else:
        #     self._dfs.rotate(-1)
        # self.df = self._dfs[0]
        # Use callback 'on_episode_create' to set df instead.

        self.date = self.df["date"]
        self.prices = self.df["close"].to_numpy()
        self.data = self.df[
            [col for col in self.df.columns if col != "date" and col != "close"]
        ].to_numpy()

        self.terminal = False
        # skip n days for frame stacking
        self.day = self.stack_frame - 1
        self.share_num = 0
        self.account = self.initial_cash

        self._last_total_trading = self.total_trading
        self._last_success_trade = self.success_trade
        self._last_asset_memory = self.asset_memory

        self.asset_memory = [self.account]
        self.action_memory = []
        self.buying_price = 0
        self.success_trade = 0
        self.total_trading = 0

        return self.state, {}

    def step(self, action: int) -> Tuple[np.array, float, bool, bool, Dict]:
        assert action in [0, 1, 2], f"Action {action} out of bounds."
        assert self.action_masks()[
            action
        ], f"Invalid action {action} with {self.share_num} shares of stocks."

        self.terminal = self.day >= self.num_day - 2

        if action == 0:
            self._sell(self.share_num)
            self.total_trading += 1
            self.success_trade += 1 if self.price > self.buying_price else 0
            self.action_memory.append(0)
        elif action == 2:
            num_share_to_buy = np.floor(
                self.account / (self.price * (1 + self.cost_pct))
            )
            self._buy(num_share_to_buy)
            self.buying_price = self.price
            self.action_memory.append(2)
        else:
            self.action_memory.append(0)

        # state: s -> s+1
        self.day += 1
        self.asset_memory.append(self.asset)

        assert self.asset_memory[-2] > 0, f"Non positive asset in: {self.asset_memory}"
        reward = np.log2(self.asset_memory[-1] / self.asset_memory[-2])

        if self.terminal:
            msg = "-" * 30 + "\n"
            msg += f"Total trading times: {self.total_trading}\n \
                        Success times: {self.success_trade}\n \
                        Action memory: {self.action_memory}\n \
                        Asset memory: {self.asset_memory}"
            if self.verbose:
                print(msg)
            else:
                self.logger.log(logging.INFO, msg=msg)

        return self.state, reward, self.terminal, False, {}

    def render(self, mode="human", close=False) -> pd.DataFrame:
        return self.state

    def action_masks(self) -> List[bool]:
        if self.share_num > 0:
            return [True, True, False]
        else:
            return [False, True, True]

    @property
    def state(self) -> np.array:
        assert (
            self.day < self.num_day
        ), f"Day no. {self.day} > total days {self.num_day}"
        feat = self.data[self.day - (self.stack_frame - 1) : self.day + 1].flatten()
        state = np.concatenate([feat.flatten(), self.position], axis=0)

        return state

    @property
    def price(self) -> float:
        return self.prices[self.day]

    @property
    def asset(self) -> float:
        return self.account + self.price * self.share_num

    @property
    def position(self) -> np.array:
        if self.share_num > 0:
            return np.array([1])
        else:
            return np.array([0])

    def _sell(self, share_num: int) -> None:
        self.share_num -= share_num
        self.account += self.price * share_num * (1 - self.cost_pct)

    def _buy(self, share_num: int) -> None:
        self.share_num += share_num
        self.account -= share_num * self.price * (1 + self.cost_pct)

    @staticmethod
    def action_mask_fn(obs: th.Tensor) -> th.Tensor:
        # obs: (batch_size, action_space_n)
        pos = obs[:, -1:]
        no_buying = th.Tensor([0, 0, -th.inf]).to(obs.device)
        no_selling = th.Tensor([-th.inf, 0, 0]).to(obs.device)
        action_masks = (
            pos.mul(no_buying).nan_to_num() + (1 - pos).mul(no_selling).nan_to_num()
        )
        return action_masks


def _env_creator(env_config):
    return SingleStockTradingEnv(env_config)


register_env("SingleStockTrading", _env_creator)
