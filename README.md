# XAUUSD AI Signal System

Local Python service for XAUUSD signal generation. It uses Deriv WebSocket candles for live inference, Twelve Data for historical training export, computes deterministic features, loads a calibrated XGBoost model, asks Groq for advisory JSON review, applies hard risk gates, and sends accepted signals to Discord.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
cp data/news_blackouts.csv.example data/news_blackouts.csv
```

Deriv market data is the default live feed and does not require a trading account. For development, `DERIV_APP_ID=1089` is enough; create your own Deriv app ID before production use. For historical training export, add `TWELVE_DATA_API_KEY`. Fill `.env` with Groq and Discord credentials when those integrations are enabled. Place the trained model at `models/xgb_xauusd_v1.pkl`.

## Data Strategy

- Historical training data: Twelve Data `XAU/USD` 15-minute candles.
- Live inference data: Deriv WebSocket `frxXAUUSD` candles.
- Execution: manual for v1.
- Yahoo/yfinance is not the primary 5-year M15 source because intraday history is usually limited to recent windows.

## Run

```bash
python main.py
```

The scheduler fires at `:01`, `:16`, `:31`, and `:46` UTC.

## Validate Deriv Data

```bash
python scripts/validate_deriv_market_data.py
```

This checks the configured Deriv symbol and prints the latest M15 candle without running the model, LLM, risk pipeline, or Discord.

## Dry Signal Cycle

```bash
python scripts/dry_signal_cycle.py
```

This fetches real market candles, stores them in SQLite, computes the latest feature snapshot, and stops before model inference, Groq, risk delivery, or Discord.

## Train

First export historical data:

```bash
python scripts/export_training_bundle.py --years 5 --output-dir data/training
```

The exporter is resumable: rerun the same command if it is interrupted.

Validate the exported CSV before training:

```bash
python scripts/validate_training_data.py --csv data/training/xauusd_m15.csv
```

```bash
pip install -e ".[train]"
python scripts/train_xgboost.py \
  --m15 data/training/xauusd_m15.csv \
  --h1 data/training/xauusd_h1.csv \
  --h4 data/training/xauusd_h4.csv \
  --dxy data/training/eurusd_m15.csv \
  --output models/xgb_xauusd_v1.pkl \
  --trials 50
```

Run training in Colab or another training environment, not on the local signal runner unless you explicitly want to. The training path uses chronological train/validation/test splits and excludes HOLD rows from the binary classifier.

## Safety Defaults

- No automated trade execution.
- Missing model suppresses signals.
- Missing economic calendar suppresses signals.
- Missing Groq key falls back to deterministic ML-only rationale for local testing.
- Discord receives only signals that pass all risk filters.
