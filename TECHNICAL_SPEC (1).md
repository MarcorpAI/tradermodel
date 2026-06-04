# XAUUSD AI Signal System — Technical Specification

## 1. System Architecture

### Runtime Environment
- **Local machine**: Python 3.11+ scheduler, inference, Discord delivery
- **Google Colab**: Model training only (one-time + periodic retraining)
- **External APIs**: OANDA (data), Claude API (LLM), Discord Webhook (delivery), Forex Factory or Tradays API (economic calendar)

### Process Flow

```
OANDA API (M15 candle close)
    ↓
data_ingest.py          — fetch latest candles, validate, store to local SQLite
    ↓
feature_engine.py       — compute all indicators and context features
    ↓
model_inference.py      — load trained XGBoost model, run prediction
    ↓
llm_layer.py            — build structured context, call Claude API
    ↓
risk_filter.py          — evaluate confidence, check news calendar, apply guards
    ↓
discord_notify.py       — format and fire webhook if threshold passed
    ↓
logger.py               — write full signal record to SQLite regardless of outcome
```

### Directory Structure

```
xauusd-signal/
├── .env                        # API keys — never committed
├── config.yaml                 # Tunable parameters
├── main.py                     # Scheduler entrypoint
├── src/
│   ├── data_ingest.py
│   ├── feature_engine.py
│   ├── model_inference.py
│   ├── llm_layer.py
│   ├── risk_filter.py
│   ├── discord_notify.py
│   └── logger.py
├── models/
│   └── xgb_xauusd_v1.pkl       # Trained model artifact
├── data/
│   └── xauusd.db               # SQLite: candles + signal log
├── notebooks/
│   └── train_xgboost.ipynb     # Colab training notebook
└── logs/
    └── system.log
```

---

## 2. Data Layer

### Source
OANDA v20 REST API. Requires a free practice account. Provides XAUUSD (instrument: `XAU_USD`) historical and live candles.

### Granularities Fetched
| Granularity | Purpose |
|---|---|
| M15 | Primary signal timeframe, model inference |
| H1 | Higher timeframe trend context |
| H4 | Macro trend filter |

### Historical Data Pull (Training)
- **Period**: 5 years (adjustable in config)
- **Instrument**: XAU_USD
- **Granularity**: M15
- **Estimated rows**: ~120,000 after weekend/gap removal
- **Storage**: CSV for Colab training, SQLite for live system
- OANDA limits responses to 5,000 candles per request. The data fetch script paginates automatically using the `from` parameter, walking backwards in time.

### Live Data Poll
- Runs on M15 candle close (scheduler fires at :01 past every 15 minutes to allow candle to fully close)
- Fetches last 200 candles to ensure all indicators have sufficient lookback
- Validates that the latest candle timestamp is within expected range before proceeding — if OANDA returns stale data, the cycle is skipped and logged

### Schema (SQLite: `candles` table)
```sql
CREATE TABLE candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL UNIQUE,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    granularity TEXT,
    complete INTEGER   -- 1 if candle is closed, 0 if still forming
);
```

---

## 3. Feature Engineering

All features are computed fresh on each inference cycle from the latest candle window. No features are stored — they are recomputed from raw candles every run.

### Price-Based Indicators

| Feature | Parameters | Library |
|---|---|---|
| RSI | Period 14 | pandas-ta |
| MACD | 12/26/9 | pandas-ta |
| MACD histogram | derived | pandas-ta |
| Bollinger Bands | 20/2 | pandas-ta |
| BB %B (position within bands) | derived | pandas-ta |
| ATR | Period 14 | pandas-ta |
| EMA 20 | Period 20 | pandas-ta |
| EMA 50 | Period 50 | pandas-ta |
| EMA 200 | Period 200 | pandas-ta |
| Price vs EMA 200 | binary: above/below | derived |
| Candlestick body ratio | (close-open) / (high-low) | derived |

### Higher Timeframe Context Features
- H1 trend direction: derived from H1 EMA20 vs EMA50 slope (1 = up, -1 = down, 0 = flat)
- H4 trend direction: same method on H4 candles
- These are added as static features to every M15 row during training and inference

### Session Features
XAUUSD behaves differently across sessions. These are binary flag features:

| Feature | Definition |
|---|---|
| `session_london` | 1 if UTC hour is 07–16 |
| `session_newyork` | 1 if UTC hour is 13–21 |
| `session_overlap` | 1 if UTC hour is 13–16 (London/NY overlap) |
| `session_asian` | 1 if UTC hour is 00–07 |
| `day_of_week` | Integer 0–4 (Monday–Friday) |

### DXY Proxy
True DXY feed requires a paid data source. Use USDIndex (OANDA instrument: `USB02Y_USD` or EURUSD inverse as a proxy). Fetch alongside XAUUSD and compute:
- DXY proxy 14-period RSI
- DXY proxy direction (above/below EMA20): binary

### Sentiment Score
- Source: RSS feeds from Reuters, FXStreet, ForexLive filtered for "gold" keyword
- Simple keyword scoring: positive words (rally, surge, demand, safe-haven) = +1, negative words (drop, pressure, selloff) = -1
- Score is summed over last 6 hours of headlines and normalized to range [-1, 1]
- This is a rough but useful signal — it does not require an NLP model

### Target Variable (Training Only)
```
target = 1 if close[t+4] > close[t] + (0.5 * ATR[t])   # BUY signal
target = 0 if close[t+4] < close[t] - (0.5 * ATR[t])   # SELL signal
```
Rows where neither condition is met are labeled as HOLD and excluded from the binary classifier. A separate HOLD filter is applied at the risk layer, not in the model.

This means the model answers a specific question: *given that a move is happening, which direction?* The confidence score determines whether a move is likely enough to act on.

---

## 4. ML Model

### Algorithm: XGBoost Classifier

**Why XGBoost over LSTM or other options:**
- XAUUSD features are tabular — XGBoost is purpose-built for tabular data
- Trains in minutes on Colab CPU, no GPU needed
- Outputs calibrated probability scores, not just class labels
- Resistant to overfitting with proper regularization
- Interpretable: feature importance plots show what the model is using
- Does not require feature scaling or normalization

### Training Pipeline (Google Colab Notebook)

**Step 1 — Data preparation**
- Load 5 years of M15 candles from CSV
- Compute all features using pandas-ta
- Drop rows with NaN (first ~200 rows due to indicator lookback)
- Generate target variable as described above
- Remove HOLD rows from training set

**Step 2 — Train/test split**
- Use a time-based split, NOT random shuffle
- Train: first 80% of data chronologically
- Validation: next 10%
- Test: final 10% (most recent — never touched during training)
- Random shuffle is wrong for time series. It leaks future data into training.

**Step 3 — Hyperparameter tuning**
Use Optuna for automated hyperparameter search. Key parameters to tune:
```python
params = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 0.5],    # L1 regularization
    "reg_lambda": [1, 1.5, 2.0]    # L2 regularization
}
```
Run 50 Optuna trials. Optimize for F1 score on validation set, not accuracy (class imbalance is expected).

**Step 4 — Probability calibration**
Raw XGBoost probability outputs are not always well-calibrated. Apply `CalibratedClassifierCV` with `method='isotonic'` on the validation set after training. This ensures that a 70% confidence output actually means 70% historical accuracy at that threshold.

**Step 5 — Evaluation metrics**
Report on test set:
- Precision, Recall, F1 for both BUY and SELL classes
- ROC-AUC score
- Confusion matrix
- Profit simulation: backtest with fixed lot size, ATR-based SL, 1.5 R:R TP — track net pips

**Step 6 — Export**
```python
import joblib
joblib.dump(model, "xgb_xauusd_v1.pkl")
```
Download from Colab and place in `models/` directory on local machine.

### Retraining Schedule
Retrain every 8 weeks minimum. XAUUSD market structure evolves — a model trained only on 2022 data will degrade in 2025. Set a calendar reminder. Each retrain uses a rolling window (most recent 5 years), not cumulative data.

---

## 5. LLM Signal Layer

### Model
Claude API — `claude-sonnet-4-20250514`

### Role of the LLM
The LLM does not make the primary direction decision — that is the model's job. The LLM:
- Reviews the ML probability alongside all indicators
- Checks for contradictions (e.g. model says BUY but price is deep in overbought RSI)
- Factors in news sentiment
- Produces a final adjusted confidence score
- Writes the human-readable rationale for the Discord card

### Context Object Sent to Claude
The context is a structured Python dict serialized to a clean text block. It is not freeform. Every inference cycle sends the same template — only the values change.

```
XAUUSD Signal Analysis Request
================================
Timestamp: 2025-06-03 14:15 UTC
Session: London/NY Overlap

--- Price Context ---
Current Price: 2345.80
M15 Candle: O:2342.10 H:2347.50 L:2341.80 C:2345.80
Body Ratio: 0.61 (bullish)
ATR(14): 18.4 pips

--- Indicators ---
RSI(14): 58.2 — neutral, room to run
MACD: +0.42 histogram, line above signal
BB %B: 0.71 — upper half, not overbought
EMA20: 2341.2 | EMA50: 2335.8 | EMA200: 2298.4
Price vs EMA200: ABOVE (bullish structure)

--- Higher Timeframe ---
H1 Trend: UP (EMA20 > EMA50, positive slope)
H4 Trend: UP (strong)

--- DXY Proxy ---
DXY RSI: 44.1 — weakening dollar (supportive for gold)
DXY Direction: BELOW EMA20

--- Sentiment ---
Sentiment Score: +0.62 (positive for gold, last 6hrs)
Recent headlines: "Gold climbs on Fed uncertainty", "Safe-haven demand rises"

--- ML Model Output ---
Direction Probability: BUY 73%, SELL 27%
Model Confidence: 73%

--- Task ---
Review all signals above. Identify any contradictions or confluence.
Output a JSON object with exactly these keys:
{
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": <integer 0-100>,
  "entry_zone": "<low>-<high>",
  "stop_loss": <float>,
  "take_profit": <float>,
  "rr_ratio": <float>,
  "rationale": "<2-3 sentences explaining the signal>"
}
Return only the JSON. No preamble.
```

### SL/TP Calculation
These are computed before the LLM call and passed in the context so Claude can validate and confirm them:
- **Stop Loss**: entry - (1.5 × ATR) for BUY, entry + (1.5 × ATR) for SELL
- **Take Profit**: entry + (2.25 × ATR) for BUY, entry - (2.25 × ATR) for SELL
- **R:R Ratio**: always targeting 1.5 minimum. If ATR conditions don't support 1.5 R:R, the risk filter rejects the signal before the LLM call.

### Parsing Claude's Response
```python
import json, re

def parse_llm_response(raw: str) -> dict:
    # Strip any accidental markdown
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)
```

---

## 6. Risk Filter

The risk filter is the last gate before Discord delivery. It is pure Python logic — no ML, no LLM. Hard rules only.

### Rules (applied in order)

**Rule 1 — Minimum confidence threshold**
```python
if signal["confidence"] < config["min_confidence"]:
    reject("confidence below threshold")
```
Default threshold: 65. Configurable in `config.yaml`.

**Rule 2 — Minimum R:R ratio**
```python
if signal["rr_ratio"] < 1.5:
    reject("insufficient risk-reward ratio")
```

**Rule 3 — High-impact news blackout**
Call economic calendar API (Tradays free tier or Forex Factory RSS).
```python
if high_impact_event_within_hours(2):
    reject("high-impact news event within 2 hours")
```
Events flagged: NFP, FOMC rate decision, CPI, PPI, US GDP, Fed Chair speech.
This rule alone will prevent most catastrophic losses from news spikes.

**Rule 4 — Session filter**
Signals generated during Asian session (00:00–07:00 UTC) are suppressed unless H4 trend is strongly aligned. XAUUSD liquidity is lowest in this window and spreads widen.
```python
if session == "asian" and h4_trend_strength < 0.8:
    reject("low liquidity session, insufficient H4 confluence")
```

**Rule 5 — Consecutive signal cooldown**
If the same direction signal was sent in the last 2 M15 cycles (30 minutes), suppress.
```python
if last_signal_direction == signal["signal"] and minutes_since_last < 30:
    reject("cooldown period active")
```

**Rule 6 — Daily drawdown guard**
Track simulated daily P&L from signals sent. If simulated drawdown exceeds 3 R (3 times the risk unit), suppress all signals for remainder of the day.
```python
if daily_simulated_drawdown >= 3:
    reject("daily drawdown limit reached")
```

---

## 7. Discord Notification

### Delivery Method
Discord Incoming Webhook. No bot required. One webhook URL stored in `.env`.

### Message Format
```
🟢 XAUUSD BUY SIGNAL
━━━━━━━━━━━━━━━━━━━
Confidence     74%
Entry Zone     2344.50 – 2346.00
Stop Loss      2316.90
Take Profit    2387.40
R:R Ratio      1.58

📊 Rationale
Gold is trading above all major EMAs with strong H1 and H4 uptrend alignment.
RSI at 58 leaves room before overbought territory. DXY weakening supports
continued upside momentum. ML model conviction at 73%.

⏰ 2025-06-03 14:17 WAT | M15 | London/NY Overlap
```

For SELL signals replace 🟢 with 🔴. For suppressed signals, nothing is sent to Discord — only logged locally.

### Webhook Payload
```python
import requests

def send_discord_signal(signal: dict, webhook_url: str):
    content = format_signal_card(signal)
    requests.post(webhook_url, json={"content": content})
```

---

## 8. Scheduler

### Library: APScheduler (local, no external dependency)

```python
from apscheduler.schedulers.blocking import BlockingScheduler

scheduler = BlockingScheduler(timezone="UTC")
scheduler.add_job(run_signal_cycle, "cron", minute="1,16,31,46")
scheduler.start()
```

Fires at :01, :16, :31, :46 past each hour — one minute after each M15 candle close. This gives OANDA time to finalize the candle.

---

## 9. Local Storage (SQLite)

### Tables

**`candles`** — raw price data as described in section 2

**`signals`**
```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    direction TEXT,
    confidence INTEGER,
    entry_zone TEXT,
    stop_loss REAL,
    take_profit REAL,
    rr_ratio REAL,
    rationale TEXT,
    sent_to_discord INTEGER,  -- 1 or 0
    reject_reason TEXT,       -- null if sent, reason string if suppressed
    ml_probability REAL,
    session TEXT
);
```

This table is your feedback loop. After trades, you can manually annotate outcome (win/loss) and use it to evaluate signal quality over time.

---

## 10. Configuration File

```yaml
# config.yaml

oanda:
  account_type: practice       # practice or live
  instrument: XAU_USD
  primary_granularity: M15
  candles_lookback: 200

model:
  path: models/xgb_xauusd_v1.pkl
  retrain_interval_weeks: 8

risk:
  min_confidence: 65
  min_rr_ratio: 1.5
  atr_sl_multiplier: 1.5
  atr_tp_multiplier: 2.25
  news_blackout_hours: 2
  session_cooldown_minutes: 30
  daily_drawdown_limit_r: 3
  suppress_asian_session: true

sentiment:
  lookback_hours: 6
  feeds:
    - https://feeds.reuters.com/reuters/businessNews
    - https://www.fxstreet.com/rss/news

discord:
  min_confidence_to_notify: 65

logging:
  level: INFO
  file: logs/system.log
```

---

## 11. Risks and Mitigations

### Risk 1 — Model Overfitting
**Description**: XGBoost memorizes training data patterns that don't generalize to live market conditions.

**Mitigation**:
- Strict time-based train/test split (no shuffle)
- Regularization parameters (L1 + L2) tuned via Optuna
- Walk-forward validation: test the model on each successive 3-month block after training, not just the final test set
- Monitor live signal accuracy in the signals table — if win rate drops below 45% over 30 signals, trigger retraining

---

### Risk 2 — News Spike Losses
**Description**: High-impact macroeconomic events (NFP, FOMC, CPI) cause XAUUSD to spike 200–400 pips in seconds. No model predicts these.

**Mitigation**:
- Hard news blackout rule in risk filter: no signals 2 hours before and 30 minutes after tier-1 events
- Economic calendar checked on every cycle via API
- If calendar API is unavailable, system defaults to suppressing all signals (fail-safe, not fail-open)

---

### Risk 3 — Data Leakage During Training
**Description**: Using future data to compute features or labels, causing the model to appear accurate in testing but fail live.

**Mitigation**:
- All indicators computed strictly from past candles only using `.shift(1)` on the close price before feature computation
- Target variable computed from `close[t+4]` only during training — never available at inference time
- Train/test split is chronological, never random
- Final test set (most recent 10%) is never touched during tuning

---

### Risk 4 — LLM Hallucinated Signal
**Description**: Claude returns a malformed JSON, invents numbers, or returns a signal that contradicts its own rationale.

**Mitigation**:
- Response is parsed with strict JSON validation — any parse failure suppresses the signal and logs an error
- All numeric fields (SL, TP, confidence) are validated against expected ranges before use
- SL and TP are independently calculated in Python before the LLM call. If Claude's returned values deviate by more than 20% from the Python calculation, the Python values override
- Retry logic: if Claude returns invalid JSON, retry once with a stricter prompt before rejecting

---

### Risk 5 — OANDA API Downtime or Rate Limiting
**Description**: OANDA API becomes unavailable mid-cycle, causing stale data to be processed.

**Mitigation**:
- Candle timestamp is validated before every inference cycle. If the latest candle is more than 20 minutes old, cycle is skipped
- Exponential backoff on API failures: retry after 30s, 60s, 120s before giving up
- All failures logged with full error detail

---

### Risk 6 — Model Staleness
**Description**: Market regime changes (new macro cycle, Fed pivot, geopolitical shift) make the trained model's patterns obsolete.

**Mitigation**:
- Scheduled retraining every 8 weeks (calendar reminder)
- Signal win/rate tracking in SQLite — automated alert if rolling 30-signal accuracy drops below 48%
- Model versioned by filename (e.g. `xgb_xauusd_v2.pkl`) so rollback is trivial

---

### Risk 7 — XAUUSD Volatility and Spread Widening
**Description**: During low-liquidity periods (Asian session, major holidays) spreads widen significantly, making nominal SL/TP levels unreliable.

**Mitigation**:
- Asian session signals suppressed by default unless H4 trend is strongly aligned
- ATR-based SL already adapts to current volatility regime — wider ATR = wider SL automatically
- R:R minimum of 1.5 enforced even after spread is accounted for

---

### Risk 8 — Sentiment Feed Failures or Noise
**Description**: RSS feeds go down, or headlines are irrelevant/misleading, corrupting the sentiment score.

**Mitigation**:
- If RSS fetch fails, sentiment score defaults to 0 (neutral) — system continues without it
- Sentiment is a supplementary feature only. The model and signal can function without it
- Keyword scoring is simple by design — a complex NLP model here would be brittle and harder to debug

---

### Risk 9 — Overtrading from Too Many Signals
**Description**: System fires too many signals in a short window, leading to correlated positions and compounded losses.

**Mitigation**:
- 30-minute cooldown between same-direction signals
- Daily drawdown guard: 3R loss in a day shuts off signals for the rest of the day
- Minimum confidence threshold of 65% filters the majority of marginal signals

---

### Risk 10 — Psychological Override
**Description**: Trader ignores HOLD signals, takes signals below confidence threshold, or revenge trades after losses. The system cannot prevent this — it is a human risk.

**Mitigation** (process-based, not technical):
- Commit to a rule: only act on signals with confidence ≥ 65% that were delivered to Discord
- Log every trade taken manually alongside the signal that prompted it
- Review weekly: compare trades taken vs signals delivered, identify deviations

---

## 12. Tech Stack Summary

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Data source | OANDA v20 REST API |
| Indicators | pandas-ta |
| ML model | XGBoost + scikit-learn (CalibratedClassifierCV) |
| Hyperparameter tuning | Optuna |
| Training environment | Google Colab (free CPU) |
| LLM | Claude API (claude-sonnet-4-20250514) |
| Scheduler | APScheduler |
| Local storage | SQLite (via Python sqlite3) |
| Notifications | Discord Incoming Webhook |
| Economic calendar | Tradays API (free tier) |
| Config | PyYAML |
| Environment variables | python-dotenv |
| Logging | Python logging module |

---

## 13. Python Dependencies

```
oandapyV20
pandas
pandas-ta
xgboost
scikit-learn
optuna
anthropic
apscheduler
requests
feedparser
pyyaml
python-dotenv
joblib
sqlite3  # stdlib
```

---

## 14. What is NOT in v1

These are deliberately excluded from the first version to keep scope manageable:

- Backtesting engine (use the Colab notebook profit simulation instead)
- Web dashboard (Discord is sufficient)
- Multiple currency pairs
- Automated trade execution
- Portfolio-level risk management
- Fine-tuned LLM (Claude base model is sufficient)
- Deep learning models (LSTM, Transformer) — revisit in v2 if XGBoost proves insufficient
