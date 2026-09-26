# Quantitative Finance

Implementations and research notes on quantitative finance, covering derivatives pricing, stochastic modeling, volatility forecasting and risk measurement.

## Scope

The repository is organized around the following topics:

- **Derivatives pricing**: Black–Scholes, Heston and SABR models, with closed-form, Monte Carlo and finite-difference methods, and model calibration.
- **Stochastic processes**: Brownian motion, jump-diffusion and Lévy processes for modeling asset returns.
- **Volatility modeling**: GARCH-family models and deep-learning approaches, such as LSTM networks, for volatility forecasting.
- **Risk measurement**: Value at Risk and Expected Shortfall, together with the backtesting of risk models.
- **Machine learning in finance**: data-driven methods such as deep hedging.

## Repository layout

```text
.
├── scripts/     One folder per project, each with its own README
├── notebooks/   Jupyter notebooks, grouped by topic
├── docs/        Project README template and publishing workflow
├── data/        Small synthetic or publicly redistributable datasets (data/examples/)
├── outputs/     Generated results, not tracked by Git
└── tests/       Automated checks
```

Folders are created when their first content is added.

## Catalogue

| Item | Type | Description |
| --- | --- | --- |
| [`scripts/quant_engine`](scripts/quant_engine/README.md) | Python + C++ library | C++17 engines exposed with pybind11: Black-Scholes, Crank-Nicolson (European and American), Heston Fourier and QE Monte Carlo, GARCH/GJR-GARCH estimation and rolling VaR/ES; NumPy reference implementations, backtests and ECB data loaders. |
| [`notebooks/derivatives_pricing/heston_pricing_and_calibration.ipynb`](notebooks/derivatives_pricing/heston_pricing_and_calibration.ipynb) | Notebook | Validation of the pricing engines against closed forms and published benchmarks, Heston smiles and calibration, American puts, discounting with the ECB AAA yield curve. |
| [`notebooks/risk_measurement/fx_garch_var_backtesting.ipynb`](notebooks/risk_measurement/fx_garch_var_backtesting.ipynb) | Notebook | One-day 99% VaR and 97.5% ES of a euro investor's currency portfolio on ECB reference rates: historical simulation, GARCH-N, GJR-t and filtered historical simulation with Kupiec, Christoffersen, traffic-light and Acerbi-Szekely backtests. |

## Getting started

The notebooks run in Visual Studio Code (with the *Python*, *Jupyter* and *C/C++* extensions) or in Jupyter. The numerical engines are written in C++ and compiled into a Python extension, so a C++17 compiler is required (Visual Studio Build Tools on Windows, Xcode Command Line Tools on macOS, GCC or Clang on Linux). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/quant_engine/requirements.txt
python -m pip install -e scripts/quant_engine
python -m pytest tests/quant_engine
```

Then open a notebook and select the `.venv` environment as kernel. The notebooks download official data from the European Central Bank on first use; see the [project README](scripts/quant_engine/README.md) for details and for the offline mode.

## Conventions

- Each project documents its purpose, inputs, outputs and exact run command, following the [project README template](docs/script-template.md).
- Model assumptions, parameter values, calibration data, conventions and sources are stated explicitly.
- Numerical methods are validated against closed-form solutions or published benchmarks, with justified tolerances.
- Simulations use a fixed, documented random seed.
- Market data is committed only when its license allows redistribution; otherwise, the repository provides the code to download it.
- Paths are relative to the repository root, and no credentials or confidential data are ever committed.

## Development workflow

Changes follow the [publishing workflow](docs/publishing.md). The [`CLAUDE.md`](CLAUDE.md) file provides project instructions for [Claude Code](https://claude.com/claude-code), so that AI-assisted contributions meet the same standards.

## License

Released under the [MIT License](LICENSE).
