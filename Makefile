.PHONY: install install-all test train backtest

install:
	pip install -e '.[dev]'

install-all:
	pip install -e '.[all]' yfinance kiteconnect pyarrow

test:
	pytest

train:
	ai-trade-train --config config/config.yaml

backtest:
	ai-trade-backtest
