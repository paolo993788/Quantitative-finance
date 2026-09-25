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

No projects have been published yet. Each new script or notebook will be listed here with a short description.

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
