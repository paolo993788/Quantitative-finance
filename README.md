# Quantitative Finance

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=cplusplus&logoColor=white)
![pybind11](https://img.shields.io/badge/bindings-pybind11-5C6BC0)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![Data](https://img.shields.io/badge/data-ECB%20%7C%20FRED-2E7D32)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Derivatives pricing, hedging and market-risk analytics with a C++17 engine driven from Python notebooks, validated against published benchmarks and applied to official European Central Bank data.**

The repository answers questions that a derivatives desk, a market-risk function or a corporate treasury actually face: how to price and hedge an option sold to a client, how much a hedged position can lose, whether a VaR model passes regulatory backtesting, and how much capital a trading book requires under FRTB. Numerically intensive parts (Monte Carlo, finite differences, Fourier integration, hundreds of maximum-likelihood fits) run in C++; analysis, validation and reporting stay in Python.

## Highlights

| Area | What is done | Key result |
| --- | --- | --- |
| FX options desk (case study) | Price and delta-hedge a six-month EUR/USD option sold to an Italian exporter, with dynamics estimated on ECB data | A constant-volatility model understates the 97.5% ES of the hedged position by a factor of about 4 compared with filtered historical simulation; the quote is set by a cost-of-capital rule on the simulated tail |
| Market risk and capital | One-day 99% VaR and 97.5% ES of a currency book on ECB reference rates, 1999-2025, with regulatory backtests and FRTB capital | Filtered historical simulation passes every test (1.12% exceptions, Z2 about 0); historical simulation fails (red zone in 2008); FRTB charge about 9% of the book versus 22% under Basel 2.5, stressed period April 2008 - March 2009 |
| Option pricing engines | Black-Scholes, Crank-Nicolson with PSOR for American options, Heston by Fourier inversion and QE Monte Carlo, calibration | Heston price matches the Fang-Oosterlee (2008) benchmark to 2e-8; second-order convergence of Crank-Nicolson; C++ Monte Carlo about 7x faster than vectorised NumPy |
| Engineering | pybind11 extension, parallel and thread-independent random streams, NumPy reference implementations | 60 automated tests with justified tolerances; results identical for any number of threads |

## Catalogue

| Item | Business question | Data |
| --- | --- | --- |
| [`scripts/quant_engine`](scripts/quant_engine/README.md) (library) | Reusable C++/Python engines for pricing, hedging simulation, GARCH risk models, backtests and regulatory capital | ECB, FRED loaders |
| [FX option desk: pricing and hedging](notebooks/case_studies/fx_option_desk_hedging.ipynb) | Which volatility should the desk quote for a EUR call / USD put sold to an exporter, how should it hedge, and how large is the model risk? | ECB EUR/USD, ECB AAA curve, US Treasury bill (FRED) |
| [FX market risk and FRTB capital](notebooks/risk_measurement/fx_garch_var_backtesting.ipynb) | Does a GARCH-based VaR/ES model pass regulatory backtests, and how much capital does the book need under FRTB and Basel 2.5? | ECB reference rates for USD, GBP, JPY, CHF |
| [Heston pricing and calibration](notebooks/derivatives_pricing/heston_pricing_and_calibration.ipynb) | Are the pricing engines accurate and fast enough for production-style calibration, and what drives the smile? | ECB AAA yield curve |

Each notebook states its assumptions, fixes its random seeds, validates the numbers it relies on and ends with conclusions and limitations. Figures and tables are written to `outputs/`.

## Architecture

```text
notebooks/  ──►  quant_engine (Python)                  ──►  quant_engine._core (C++17, pybind11)
                 data loaders (ECB, FRED)                     Black-Scholes, implied volatility
                 NumPy/SciPy reference implementations        Crank-Nicolson + PSOR (American options)
                 calibration, backtests, capital              Heston Fourier + QE Monte Carlo
                 reporting                                    GJR-GARCH MLE, rolling VaR/ES (parallel)
                                                              discrete delta-hedging simulator
```

## Getting started

Requirements: Python 3.10 or later and a C++17 compiler (Visual Studio Build Tools on Windows, Xcode Command Line Tools on macOS, GCC or Clang on Linux). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/quant_engine/requirements.txt
python -m pip install -e scripts/quant_engine
python -m pytest tests/quant_engine
```

Open a notebook in Visual Studio Code (extensions *Python*, *Jupyter* and *C/C++*) and select the `.venv` environment as kernel. Official data are downloaded and cached on first use; set `QUANT_ENGINE_DATA_MODE=synthetic` to work offline. Details, methods and the full validation table are in the [project README](scripts/quant_engine/README.md).

## Repository layout

```text
.
├── scripts/quant_engine/   C++ engine (cpp/), Python package, build and dependency files
├── notebooks/              case_studies/, derivatives_pricing/, risk_measurement/
├── tests/quant_engine/     validation suite (pytest)
├── docs/                   project README template and publishing workflow
├── data/                   download cache (ignored by Git) and small examples
└── outputs/                generated figures and tables (ignored by Git)
```

## Roadmap

SABR and local volatility calibration to FX and equity smiles; jump-diffusion and Lévy models; LSTM volatility forecasts benchmarked against GARCH; deep hedging with transaction costs; CVA for the FX option case study.

## Conventions

- Each project documents its purpose, inputs, outputs and exact run command, following the [project README template](docs/script-template.md).
- Model assumptions, parameter values, calibration data, conventions and sources are stated explicitly.
- Numerical methods are validated against closed-form solutions or published benchmarks, with justified tolerances.
- Simulations use a fixed, documented random seed and report Monte Carlo standard errors.
- Market data is committed only when its license allows redistribution; otherwise, the repository provides the code to download it.
- Paths are relative to the repository root, and no credentials or confidential data are ever committed.

## Development workflow

Changes follow the [publishing workflow](docs/publishing.md). The [`CLAUDE.md`](CLAUDE.md) file provides project instructions for [Claude Code](https://claude.com/claude-code), so that AI-assisted contributions meet the same standards.

## Disclaimer

Research and educational code. Results depend on the stated assumptions and are not investment advice or a validated production model.

## License

Released under the [MIT License](LICENSE).
