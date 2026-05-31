from __future__ import annotations

import numpy as np
import pandas as pd

try:  # optional dependency for lightweight installation
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None


class NiftyTradingEnv(gym.Env if gym else object):
    """Cross-sectional swing-trading environment with risk-adjusted rewards.

    Actions: 0 hold, 1 long, 2 short, 3 increase, 4 reduce, 5 cash.
    The agent learns when confidence is sufficient by receiving turnover, volatility, and drawdown penalties.
    """

    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame, feature_columns: list[str], cost_bps: float = 8, drawdown_penalty: float = 2.0):
        if gym is None:
            raise RuntimeError("Install gymnasium to use NiftyTradingEnv")
        self.features = features.sort_values(["date", "symbol"]).reset_index(drop=True)
        self.feature_columns = feature_columns
        self.dates = list(self.features["date"].drop_duplicates())
        self.cost = cost_bps / 10_000
        self.drawdown_penalty = drawdown_penalty
        width = len(feature_columns)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(width,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.step_id = 0
        self.position = 0.0
        self.equity = 1.0
        self.peak = 1.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.step_id = 0
        self.position = 0.0
        self.equity = 1.0
        self.peak = 1.0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        day = self.features[self.features["date"] == self.dates[self.step_id]]
        return day[self.feature_columns].median(numeric_only=True).fillna(0).to_numpy(dtype=np.float32)

    def step(self, action: int):
        old_position = self.position
        if action == 1:
            self.position = 0.5
        elif action == 2:
            self.position = -0.5
        elif action == 3:
            self.position = min(0.75, self.position + 0.25)
        elif action == 4:
            self.position *= 0.5
        elif action == 5:
            self.position = 0.0
        day = self.features[self.features["date"] == self.dates[self.step_id]]
        ret = float(day.get("target_return_1d", pd.Series([0])).mean())
        turnover_cost = abs(self.position - old_position) * self.cost
        self.equity *= 1 + self.position * ret - turnover_cost
        self.peak = max(self.peak, self.equity)
        drawdown = self.equity / self.peak - 1
        reward = self.position * ret - turnover_cost + self.drawdown_penalty * drawdown
        self.step_id += 1
        terminated = self.step_id >= len(self.dates) - 1
        return self._obs(), float(reward), terminated, False, {"equity": self.equity, "drawdown": drawdown}
