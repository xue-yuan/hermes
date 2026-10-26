# hermes

Taiwan stock analysis tool: strategy signals, daily screening, backtesting, institutional and margin trading data, and machine learning. Works on TWSE-listed stocks, with data from 2020-01-01 onward.

## Requirements

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Node.js and pnpm (only for the web UI)

## Install

```sh
uv sync
```

## Data

Data is downloaded from [FinMind](https://finmindtrade.com/) and stored in `~/.hermes` (override with `HERMES_HOME`). No token is needed; set `FINMIND_TOKEN` for a higher rate limit.

Only `fetch`, `update` and `watchlist add` download and write data. All other commands read local data only.

```sh
uv run hermes fetch 2330 2317        # download everything for these stocks
uv run hermes watchlist add 2330     # add to the watchlist
uv run hermes watchlist list
uv run hermes update                 # incremental update of the watchlist, for a daily job
uv run hermes verify                 # compare local data against official TWSE data
```

## Commands

| Command      | Description                                                         |
| ------------ | ------------------------------------------------------------------- |
| `show`       | Daily price data                                                    |
| `tech`       | Indicator values for a day                                          |
| `chips`      | Institutional net buying, foreign ownership, margin trading         |
| `strategies` | List strategies                                                     |
| `indicators` | List indicator types, parameters and outputs for strategy files     |
| `signals`    | Historical buy and sell signals for a strategy                      |
| `screen`     | Daily screen: new signals, signals still active, setups forming     |
| `backtest`   | Backtest one strategy                                               |
| `compare`    | Compare several strategies side by side                             |
| `ml`         | Machine learning: labeling, cross-validation, walk-forward backtest |

Run `uv run hermes <command> --help` for all options.

```sh
uv run hermes signals 2330 -s connors_rsi2
uv run hermes screen
uv run hermes backtest 2330 -s turtle_system2 --fee-discount 0.28
uv run hermes compare 2330 -s connors_rsi2 -s siegel_ma200 --ml
uv run hermes ml 2330
```

Signals use actual traded prices by default; add `--adjust` to use prices adjusted for dividends and splits. Backtest returns always use adjusted prices (dividends included), and fills are at actual traded prices.

## Configuration

| Path           | Contents                                     | Override             |
| -------------- | -------------------------------------------- | -------------------- |
| `strategies/`  | Strategy files (YAML)                        | `HERMES_STRATEGIES`  |
| `indicators/`  | Indicator sets for `tech` (YAML)             | `HERMES_INDICATORS`  |
| `reports/`     | Charts and reports                           | `HERMES_REPORTS`     |

## Web

A read-only viewer with Google sign-in. It cannot download or change data.

1. In Google Cloud Console, create an OAuth client ID of type "Web application" and add `http://localhost:8000` to its authorized JavaScript origins.
2. Copy `.env.example` to `.env` and fill it in.
3. Build the frontend:
   ```sh
   cd web && pnpm install && pnpm build
   ```
4. Start the server and open http://localhost:8000:
   ```sh
   uv run hermes-web            # --host, --port
   ```

## Tests

```sh
uv run pytest
```
