from __future__ import annotations

from typing import Any

from ai_trading_system.rl.environment import NiftyTradingEnv


def train_rl_agent(env: NiftyTradingEnv, algorithm: str = "PPO", total_timesteps: int = 10_000) -> Any:
    try:
        from stable_baselines3 import A2C, DQN, PPO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install stable-baselines3 to train RL agents.") from exc
    algos = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
    if algorithm not in algos:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")
    model = algos[algorithm]("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model
