# Quant engine: C++ pricing and risk engines for Python

A C++17 library exposed to Python with pybind11, used by the notebooks of this repository. It prices European and American options (Black-Scholes closed form, Crank-Nicolson finite differences, Heston Fourier inversion and Quadratic-Exponential Monte Carlo), simulates the profit and loss of discretely delta-hedged option positions under GBM, Heston and GARCH filtered-historical-simulation dynamics, and estimates GARCH-family models for rolling Value at Risk and Expected Shortfall. The Python package adds NumPy/SciPy reference implementations for validation, Heston calibration, VaR/ES backtests, FRTB and Basel 2.5 capital calculations and loaders for official data from the European Central Bank and FRED.

## Requirements

- Python 3.10 or later (developed and tested with Python 3.11).
- A C++17 compiler:
  - Windows: Visual Studio 2019 or later, or the free *Build Tools for Visual Studio* with the "Desktop development with C++" workload;
  - macOS: Xcode Command Line Tools (`xcode-select --install`);
  - Linux: GCC 9 or later, or Clang 10 or later.
- Python dependencies: [`requirements.txt`](requirements.txt) (NumPy, SciPy, pandas, Matplotlib, pybind11, pytest, ipykernel). No other C++ libraries are needed.

## Usage

### Set-up (from the repository root)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/quant_engine/requirements.txt
python -m pip install -e scripts/quant_engine
```

The last command compiles `cpp/bindings.cpp` into the extension module `quant_engine._core` and installs the package in editable mode. After editing any C++ file, run it again to rebuild.

### Visual Studio Code

1. Install the *Python*, *Jupyter* and *C/C++* extensions.
2. Open the repository folder, then select the `.venv` interpreter (*Python: Select Interpreter*).
3. Open a notebook in `notebooks/` and choose the same environment as kernel (*Select Kernel*).
4. For C++ IntelliSense, add the pybind11 and Python include folders to the C/C++ configuration; the command `python -m pybind11 --includes` prints them.

### Tests and data

```bash
python -m pytest tests/quant_engine                           # validation suite
python -m quant_engine.data --fx --yield-curve --fred DTB3    # optional: pre-download the official data
```

### Notebooks

| Notebook | Content |
| --- | --- |
| [`notebooks/derivatives_pricing/heston_pricing_and_calibration.ipynb`](../../notebooks/derivatives_pricing/heston_pricing_and_calibration.ipynb) | ECB discount curve, validation of the finite-difference, Fourier and Monte Carlo engines, Heston smiles, calibration and American puts. |
| [`notebooks/risk_measurement/fx_garch_var_backtesting.ipynb`](../../notebooks/risk_measurement/fx_garch_var_backtesting.ipynb) | GJR-GARCH-t and filtered historical simulation VaR/ES for a currency portfolio on ECB reference rates, regulatory backtests and FRTB versus Basel 2.5 capital. |
| [`notebooks/case_studies/fx_option_desk_hedging.ipynb`](../../notebooks/case_studies/fx_option_desk_hedging.ipynb) | Case study: pricing, delta hedging, model risk, hedging frequency and a risk-based quote for a EUR/USD option sold to an exporter. |

Both notebooks download official data by default. Set the environment variable `QUANT_ENGINE_DATA_MODE=synthetic` before starting Jupyter to run them offline on synthetic data.

## Inputs

| Name | Format | Description |
| --- | --- | --- |
| ECB euro foreign exchange reference rates | ZIP with CSV, downloaded from `https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip` | Daily fixings since 4 January 1999, units of foreign currency per euro. |
| ECB euro area yield curve (AAA) | CSV from the ECB Data Portal, series `YC.B.U2.EUR.4F.G_N_A.SV_C_YM.{BETA0..BETA3,TAU1,TAU2}` | Daily Svensson parameters: betas in percent, taus in years. |
| US 3-month Treasury bill rate | CSV from FRED, series `DTB3` (Federal Reserve H.15) | Percent; converted to continuous compounding with $\ln(1 + y)$ as an approximation. |

Downloads are cached in `data/raw/ecb/` and `data/raw/fred/`, which Git ignores. The environment variable `QUANT_ENGINE_DATA_DIR` changes the base cache folder. The ECB allows reuse of its statistics free of charge provided the source is acknowledged and Federal Reserve data are in the public domain; the data are downloaded rather than committed, so that the source and version remain explicit. If an interest rate cannot be downloaded, the case-study notebook prints a warning and uses the documented assumption.

## Outputs

| File | Description |
| --- | --- |
| `outputs/heston_pricing/*.png` | Figures of the pricing notebook (curve, convergence, QE bias, smiles, calibration, American put). |
| `outputs/fx_var_backtesting/*.png`, `backtest_summary.csv` | Figures, backtest table and capital chart of the risk notebook. |
| `outputs/fx_option_desk/*.png`, `management_summary.txt` | P&L distributions, hedging frontier, pricing chart and the management summary of the case study. |
| `docs/figures/*-light.png`, `*-dark.png` | README charts drawn by `python -m quant_engine.readme_figures` (official data; `--synthetic` offline); the only generated files committed. |

## Method

**Conventions.** Rates and dividend yields are continuously compounded; maturities are year fractions. The ECB Svensson curve gives continuously compounded zero rates of AAA-rated euro area central government bonds, used as a proxy for the risk-free rate. Returns in the risk module are daily log-returns in percent; VaR and ES are positive losses.

**Black-Scholes-Merton** (`cpp/black_scholes.hpp`): closed-form prices and Greeks; implied volatility by Newton iterations safeguarded with bisection, NaN outside the no-arbitrage bounds.

**Finite differences** (`cpp/finite_difference.hpp`): Crank-Nicolson in log-price on a uniform grid centred on $S_0$ (half-width 5 standard deviations plus the log-moneyness), Dirichlet boundaries, Rannacher start (the first two steps replaced by four implicit Euler half steps), Thomas algorithm for European options and projected SOR (relaxation 1.2, tolerance $10^{-12} K$) for American options. The early-exercise boundary is extracted at each time step.

**Heston** (`cpp/heston.hpp`): characteristic function in the Albrecher et al. (2007) form; call prices by

$$C = \tfrac12\left(S e^{-qT} - K e^{-rT}\right) + \frac{e^{-rT}}{\pi}\int_0^\infty \mathrm{Re}\left[\frac{e^{iuk}\left(S\varphi(u-i) - K\varphi(u)\right)}{iu}\right]du,\qquad k=\ln(S/K),$$

integrated with composite 16-point Gauss-Legendre panels until four consecutive panels contribute less than $10^{-14}\max(S,K)$; puts by put-call parity. Monte Carlo uses the QE scheme of Andersen (2008) with $\psi_c = 1.5$ and central discretisation ($\gamma_1=\gamma_2=\tfrac12$), without martingale correction. Calibration (`quant_engine/heston.py`) minimises vega-weighted price errors with `scipy.optimize.least_squares`.

**GARCH** (`cpp/garch.hpp`): GARCH(1,1) and GJR-GARCH(1,1) with Gaussian or unit-variance Student-t innovations, recursion initialised with the sample variance, maximum likelihood with a Nelder-Mead simplex on an unconstrained reparametrisation ($\omega,\alpha,\beta,\alpha+\gamma>0$, $\nu>2$) and a penalty for $\alpha+\gamma/2+\beta\ge 1$, with restarts. Rolling forecasts refit the model every `refit_every` days on a moving window; fits run in parallel. Filtered historical simulation uses the empirical quantile (linear interpolation) of the standardised residuals of the window.

**Backtests and capital** (`quant_engine/risk.py`): Kupiec proportion of failures, Christoffersen independence and conditional coverage, Basel traffic light, Acerbi-Szekely $Z_2$ for ES. FRTB internal-models charge from the 10-day 97.5% ES on overlapping returns calibrated to the worst 12-month window, $\max(\text{IMCC}_{t-1}, m_c\,\overline{\text{IMCC}}_{60})$ with $m_c$ from 1.5 (green zone) to 2.0 (red zone); Basel 2.5 charge from 99% VaR and stressed VaR with square-root-of-time scaling and multiplier 3 plus the traffic-light add-on.

**Delta-hedging simulator** (`cpp/hedging.hpp`, `quant_engine/hedging.py`): the dealer sells a European option priced at one volatility and holds the Black-Scholes/Garman-Kohlhagen delta computed at another, rebalanced every $k$ steps; cash accrues at the domestic rate, the underlying position at the foreign rate (or dividend yield), each trade pays a proportional cost, and the payoff is settled at maturity. The underlying follows GBM, Heston (QE scheme) or GJR-GARCH with resampled standardised residuals (filtered historical simulation). Paths are shared across hedging policies (common random numbers) and blocks of 1,024 paths have their own random streams.

**Random numbers and parallelism** (`cpp/random.hpp`, `cpp/parallel.hpp`): xoshiro256** seeded through SplitMix64 and Box-Muller normals, instead of the implementation-defined `std::normal_distribution`, so a seed gives the same numbers with any compiler. Monte Carlo paths are simulated in blocks of 4,096 with one random stream per block and block sums are combined in a fixed order: results are identical for any number of threads. Every notebook fixes its seed (`SEED`) and reports standard errors.

## Verification

Run `python -m pytest tests/quant_engine` from the repository root (61 tests, about 15 seconds). The main checks and their tolerances:

| Check | Tolerance and justification |
| --- | --- |
| C++ Black-Scholes prices and Greeks vs NumPy/SciPy | relative $10^{-12}$ for prices, $10^{-10}$ for Greeks: same formulas, rounding only |
| Greeks vs central finite differences | relative $10^{-6}$ ($10^{-4}$ for gamma): truncation error of the differences |
| Implied volatility round trip | $10^{-9}$ in volatility |
| Heston vs Fang and Oosterlee (2008) reference 5.785155450 | $2\times10^{-8}$: the published value has ten significant digits and was itself computed numerically |
| Heston vs SciPy adaptive quadrature of an independent NumPy characteristic function | $10^{-9}$ |
| Heston with vanishing vol of vol vs Black-Scholes with the integrated variance | $10^{-5}$ (error of order $\sigma^2$) |
| QE Monte Carlo vs Fourier | 4 standard errors plus 0.01 for the discretisation bias at 50 steps per year |
| Monte Carlo reproducibility with 1 and 4 threads | bitwise equality |
| Crank-Nicolson European prices vs closed form (1,600 x 1,600) | $10^{-4}$; observed convergence order 2 (error ratio between 3.5 and 4.5 when the grid is doubled) |
| American put vs 20,000-step binomial tree | $5\times10^{-4}$ (tree error of order $10^{-4}$) |
| Calibration round trip on noise-free data | implied-volatility RMSE below $10^{-6}$ |
| GARCH likelihood C++ vs pure Python | relative $10^{-11}$ |
| Nelder-Mead vs SciPy SLSQP maximum | log-likelihood not lower by more than $10^{-4}$ |
| Parameter recovery on 20,000 simulated observations | within 4 asymptotic standard errors |
| Rolling forecasts | no look-ahead: shocking a future return leaves earlier forecasts unchanged |
| VaR/ES formulas | known normal quantiles, Student-t ES by numerical integration, $Z_2 \approx 0$ under the true model |
| Capital helpers | FRTB and Basel 2.5 multiplier tables; stressed window located on data with a known stressed year; capital charge takes the larger of the two terms |
| Unhedged option P&L equals premium minus discounted payoff, path by path | relative $10^{-12}$ |
| Daily hedge at the true volatility: unbiased, standard deviation vs Kamal and Derman (1999) $\sqrt{\pi/4}\,\text{vega}\,\sigma/\sqrt{n}$ | mean within 4 standard errors; 10% (asymptotic formula) |
| Hedging error scales with the square root of the rebalancing interval | ratio within 10% of 2 when rebalancing every 4 days instead of daily |
| Selling above the true volatility and hedging at the true volatility locks in the premium difference | 4 standard errors plus $2\times10^{-5}$ |
| Transaction costs: P&L difference equals the reported costs path by path | relative $10^{-9}$ |
| GARCH-FHS with constant variance reproduces the variance of log returns; reproducibility across threads | 3%; bitwise equality |

The notebooks repeat the main validations on the data used (for example the convergence study of Crank-Nicolson and the bias study of the QE scheme).

## References

- Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*, January, 83-92.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational Finance*, 11(3), 1-42.
- Acerbi, C. and Szekely, B. (2014). Backtesting expected shortfall. *Risk*, December.
- Barone-Adesi, G., Giannopoulos, K. and Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5), 583-602.
- Blackman, D. and Vigna, S. (2021). Scrambled linear pseudorandom number generators. *ACM Transactions on Mathematical Software*, 47(4). Reference code in the public domain: https://prng.di.unimi.it/
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841-862.
- Fang, F. and Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM Journal on Scientific Computing*, 31(2), 826-848.
- Basel Committee on Banking Supervision (2019). Minimum capital requirements for market risk, MAR33 and MAR99.
- Garman, M. B. and Kohlhagen, S. W. (1983). Foreign currency option values. *Journal of International Money and Finance*, 2(3), 231-237.
- Glosten, L. R., Jagannathan, R. and Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779-1801.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6(2), 327-343.
- Kamal, M. and Derman, E. (1999). When you cannot hedge continuously: the corrections of Black-Scholes. *Risk*, 12(1), 82-85.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73-84.
- Lagarias, J. C., Reeds, J. A., Wright, M. H. and Wright, P. E. (1998). Convergence properties of the Nelder-Mead simplex method in low dimensions. *SIAM Journal on Optimization*, 9(1), 112-147.
- Marsaglia, G. and Tsang, W. W. (2000). A simple method for generating gamma variables. *ACM Transactions on Mathematical Software*, 26(3), 363-372.
- McNeil, A. J., Frey, R. and Embrechts, P. (2015). *Quantitative Risk Management*, revised edition. Princeton University Press.
- Rannacher, R. (1984). Finite element solution of diffusion problems with irregular data. *Numerische Mathematik*, 43, 309-327.
- Svensson, L. E. O. (1994). Estimating and interpreting forward interest rates: Sweden 1992-1994. NBER Working Paper 4871.
- European Central Bank: euro foreign exchange reference rates and euro area yield curves, https://www.ecb.europa.eu/stats/
