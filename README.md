# Stock Buyer & Backtesting System

A Python-based trading strategy framework using **LightGBM** machine learning models and technical indicators to predict stock/ETF price movements and backtest trading strategies. This system supports multiple tickers, rolling training, cash management, trailing stops, and performance metrics.

---

## Features

* **Data Acquisition**: Downloads historical stock/ETF data via Yahoo Finance.
* **Feature Engineering**:

  * Moving averages: SMA20, SMA50
  * Momentum, Rate of Change (ROC)
  * RSI, MACD, Bollinger Bands
  * Volatility indicators
* **Machine Learning**:

  * Rolling LightGBM classifier for predicting next-day price movement
  * Probability-based dynamic position sizing
* **Trading Strategy**:

  * Buy/Sell signals based on ML predictions
  * Trailing stop-loss
  * Trend-aware exit/entry logic
  * Memory-based signal filtering to avoid whipsaws
  * Cash reserve handling for risk management
* **Backtesting**:

  * Strategy vs Buy & Hold comparison
  * Performance metrics: Sharpe ratio, CAGR, Max Drawdown, Calmar ratio
* **Visualization**:

  * Portfolio vs Buy & Hold plots
  * Prediction signals over time

---

## Installation

Requires Python 3.8+ and the following packages:

```
pip install yfinance ta xgboost plotly dash jupyter-dash backtrader lightgbm matplotlib scikit-learn
```

---

## Usage

1. **Download Historical Data**:

```python
import yfinance as yf

tickers = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
start = '2017-01-01'
end = '2024-12-31'

for ticker in tickers:
    df = yf.download(ticker, start=start, end=end, progress=False)
```

2. **Compute Features**:

```python
from indicators import compute_features

df = compute_features(df[['Close']])
```

3. **Train & Predict**:

```python
from model import rolling_train_predict

df = rolling_train_predict(df, retrain_period=180)
```

4. **Backtest Strategy**:

```python
from backtest import backtest

df, trades = backtest(df)
```

5. **Evaluate Performance**:

```python
from metrics import compute_sharpe, compute_cagr, compute_max_drawdown, compute_calmar

sharpe = compute_sharpe(df)
cagr = compute_cagr(df)
max_dd = compute_max_drawdown(df)
calmar = compute_calmar(df)
```

6. **Plot Results**:

```python
from plot import plot_results

plot_results(df, trades, ticker)
```

---

## Configuration Options

* `initial_cash`: Starting capital for backtesting (default: 100,000)
* `transaction_cost`: Trading fees per trade (default: 0.0005)
* `base_frac`: Base fraction of cash to deploy per trade (default: 0.3)
* `trail_stop_pct`: Trailing stop percentage for exits (default: 0.08)
* `retrain_period`: Rolling ML training window in days (default: 180)
* `memory_window`: Number of previous signals to consider for whipsaw avoidance (default: 5)
* `cash_reserve`: Maximum fraction of cash to deploy in any trade (default: 0.95)

---

## Example Metrics Output

```
SPY | Sharpe: 1.45 | CAGR: 12.5% | MaxDD: 8.3% | Calmar: 1.50 | Trades: 120
QQQ | Sharpe: 1.62 | CAGR: 14.8% | MaxDD: 10.1% | Calmar: 1.47 | Trades: 110
```

---

## License

MIT License.
Feel free to modify and use for personal or research purposes.
