# Research Playbook

## Objective

Predict next-day and next-5-day returns for Nifty 50 constituents, rank stocks, allocate dynamically across long, short, and cash, and select the architecture that maximizes conservative risk-adjusted performance.

## Leakage Controls

- All labels are forward returns generated after features.
- Feature columns exclude OHLCV levels, identifiers, and all target/ranking columns.
- Walk-forward splits advance chronologically.
- Exogenous data should be timestamped by publication date before joining.
- Options-chain features should be used only for prediction dates after the chain snapshot timestamp.

## Model Comparison

Compare each model family across regression, classification, and ranking outputs:

| Family | Regression | Classification | Ranking |
| --- | --- | --- | --- |
| Random Forest | supported | supported | via expected-return rank |
| XGBoost | optional | optional | via expected-return rank |
| LightGBM | optional | optional | via expected-return rank |
| CatBoost | optional | optional | via expected-return rank |
| PPO/A2C/DQN | policy value | policy action | realized portfolio score |

## Promotion Gates

A candidate model should be promoted only if it passes:

1. Positive out-of-sample CAGR.
2. Sharpe and Sortino above benchmark.
3. Max drawdown within conservative mandate.
4. Stable turnover after costs.
5. No performance concentration in one short regime.
6. Paper-trading acceptance period.
