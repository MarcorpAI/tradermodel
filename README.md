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

### Local Training (for testing)

First export historical data (requires TWELVE_DATA_API_KEY):

```bash
python scripts/export_training_bundle.py --years 5 --output-dir data/training
python scripts/export_fred_series.py --output data/training/us10y_daily.csv
```

The exporter is resumable: rerun the same command if it is interrupted.

Validate and sanitize:

```bash
python scripts/validate_training_data.py --csv data/training/xauusd_m15.csv
python scripts/sanitize_training_data.py data/training/xauusd_m15.csv
```

Train with macro features (recommended):

```bash
pip install -e ".[train]"
python scripts/train_xgboost.py \
  --m15 data/training/xauusd_m15.csv \
  --h1 data/training/xauusd_h1.csv \
  --h4 data/training/xauusd_h4.csv \
  --dxy data/training/eurusd_m15.csv \
  --us10y data/training/us10y_daily.csv \
  --real-dxy data/training/dxy_m15.csv \
  --output models/xgb_xauusd_v1.pkl \
  --trials 50 \
  --binary
```

Without macro features (backward compatible 3-class):

```bash
python scripts/train_xgboost.py \
  --m15 data/training/xauusd_m15.csv \
  --h1 data/training/xauusd_h1.csv \
  --h4 data/training/xauusd_h4.csv \
  --dxy data/training/eurusd_m15.csv \
  --output models/xgb_xauusd_v1.pkl \
  --trials 50
```

### Colab Training (for serious retraining)

Use `notebooks/train_xgboost.ipynb` — it supports two paths:

**Path A — Auto-export**: Sets up env, exports data via Twelve Data + CalcFi, validates, sanitizes, trains with macro features, and downloads the artifact. Requires a Twelve Data API key (stored in Colab Secrets).

**Path B — Upload bundle**: Upload a zip of `data/training/` exported from your local machine.

New CLI flags:

| Flag | Description |
|------|-------------|
| `--us10y` | Path to US10Y daily CSV (CalcFi/FRED export) |
| `--real-dxy` | Path to real DXY M15 CSV (not EUR/USD proxy) |
| `--binary` | Train binary classifier (produces overlap_macro_trend artifact) |
| `--target-mode` | Labeling strategy: `close_return`, `first_touch`, `three_class`, `binary` |

When `--us10y` and `--real-dxy` are provided, the model is trained on 42 features (22 base + 12 regime + 8 macro). Without them, training uses 22 base features.

### Deployment After Training

```bash
# Move the downloaded/retrained model into place
mv models/xgb_xauusd_v1.pkl models/overlap_macro_trend_xgb.pkl

# Verify
python -c "import joblib; a=joblib.load('models/overlap_macro_trend_xgb.pkl'); print('OK, type=' + a.get('artifact_type','raw'))"

# Dry run
python scripts/dry_model_signal.py --ignore-calendar
```

## Paper Signal Gate

The system includes a research-only paper gate (`paper_signal_gate` in `config.yaml`) that restricts signals to BUY trend_continuation candidates with favorable macro conditions (DXY falling fast or US10Y yields falling). It requires 50 paper trades before demo/live is considered.

Track collection progress:

```bash
python scripts/paper_signal_stats.py
```

## Paper Signal Tracking

```bash
# Quick status
python scripts/paper_signal_stats.py

# Watch mode (refreshes every 60s)
python scripts/paper_signal_stats.py --watch
```

## Safety Defaults

- No automated trade execution.
- Paper signal gate in research-only mode (min 50 paper trades before demo/live).
- Missing model suppresses signals.
- Missing economic calendar suppresses signals.
- Missing Groq key falls back to deterministic ML-only rationale for local testing.
- Discord receives only signals that pass all risk filters.
