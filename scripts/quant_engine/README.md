# Quant engine: C++ pricing and risk engines for Python

A C++17 library exposed to Python with pybind11, used by the notebooks of this repository. It prices European and American options (Black-Scholes closed form, Crank-Nicolson finite differences, Heston Fourier inversion and Quadratic-Exponential Monte Carlo), simulates the profit and loss of discretely delta-hedged option positions under GBM, Heston and GARCH filtered-historical-simulation dynamics, estimates GARCH-family models for rolling Value at Risk and Expected Shortfall, and runs the Kalman filter of the dynamic Nelson-Siegel yield-curve model. The Python package adds NumPy/SciPy reference implementations for validation, Heston calibration, VaR/ES backtests, FRTB and Basel 2.5 capital calculations, yield-curve factor models and forecast evaluation, interest-rate risk in the banking book (IRRBB) with swap-hedge optimisation, and loaders for official data from the European Central Bank and FRED.

## Requirements

- Python 3.10 or later (developed and tested with Python 3.11).
- A C++17 compiler:
  - Windows: Visual Studio 2019 or later, or the free *Build Tools for Visual Studio* with the "Desktop development with C++" workload;
  - macOS: Xcode Command Line Tools (`xcode-select --install`);
  - Linux: GCC 9 or later, or Clang 10 or later.
- Python dependencies: [`requirements.txt`](requirements.txt) (NumPy, SciPy, pandas, Matplotlib, pybind11, pytest, ipykernel). No other C++ libraries are needed.
- Development tools: [`requirements-dev.txt`](requirements-dev.txt) (ruff, nbconvert).
- Exact versions: [`requirements-lock.txt`](requirements-lock.txt) pins the two files above to versions that pass the test suite and execute every notebook (Python 3.11, Linux). CI installs it on Python 3.11 and the unpinned files on 3.10 and 3.12.
- For the standalone C++ tests: CMake 3.16 or later (Ninja optional).

## Usage

### Set-up (from the repository root)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r scripts/quant_engine/requirements.txt
python -m pip install -e scripts/quant_engine
```

For the exact environment used by CI, install `requirements-lock.txt` instead of `requirements.txt`. The last command compiles `cpp/bindings.cpp` into the extension module `quant_engine._core` and installs the package in editable mode. After editing any C++ file, run it again to rebuild.

### Visual Studio Code

1. Install the *Python*, *Jupyter* and *C/C++* extensions.
2. Open the repository folder, then select the `.venv` interpreter (*Python: Select Interpreter*).
3. Open a notebook in `notebooks/` and choose the same environment as kernel (*Select Kernel*).
4. For C++ IntelliSense, add the pybind11 and Python include folders to the C/C++ configuration; the command `python -m pybind11 --includes` prints them.

### Tests and data

```bash
python -m pytest tests/quant_engine                           # validation suite
ruff check --select F scripts tests                           # lint (pyflakes rules)
python -m quant_engine.data --fx --yield-curve --fred DTB3    # optional: pre-download the official data

# Standalone C++ tests (strict warnings; add -DQE_SANITIZE=address,undefined or =thread)
cmake -S scripts/quant_engine/cpp -B build/cpp -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/cpp && ctest --test-dir build/cpp --output-on-failure
```

### Notebooks

| Notebook | Content |
| --- | --- |
| [`notebooks/derivatives_pricing/heston_pricing_and_calibration.ipynb`](../../notebooks/derivatives_pricing/heston_pricing_and_calibration.ipynb) | ECB discount curve, validation of the finite-difference, Fourier and Monte Carlo engines, Heston smiles, calibration and American puts. |
| [`notebooks/risk_measurement/fx_garch_var_backtesting.ipynb`](../../notebooks/risk_measurement/fx_garch_var_backtesting.ipynb) | GJR-GARCH-t and filtered historical simulation VaR/ES for a currency portfolio on ECB reference rates, regulatory backtests and FRTB versus Basel 2.5 capital. |
| [`notebooks/case_studies/fx_option_desk_hedging.ipynb`](../../notebooks/case_studies/fx_option_desk_hedging.ipynb) | Case study: pricing, delta hedging, model risk, hedging frequency and a risk-based quote for a EUR/USD option sold to an exporter. |
| [`notebooks/risk_measurement/yield_curve_dynamics_and_irrbb.ipynb`](../../notebooks/risk_measurement/yield_curve_dynamics_and_irrbb.ipynb) | Case study for a bank's ALCO: PCA and dynamic Nelson-Siegel models of the ECB curve (two-step, Kalman maximum likelihood and the arbitrage-free AFNS model), out-of-sample forecasts with Diebold-Mariano tests, EVE and NII under the BCBS scenarios, EBA outlier tests and minimax swap hedges. |

The notebooks download official data by default. Set the environment variable `QUANT_ENGINE_DATA_MODE=synthetic` before starting Jupyter to run them offline on synthetic data.

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
| `outputs/yield_curve_irrbb/*.png` | Zero curves, PCA loadings, Nelson-Siegel factors, forecast accuracy by regime, IRRBB scenarios, historical EVE changes and hedge comparison. |
| `docs/figures/*-light.png`, `*-dark.png` | README charts drawn by `python -m quant_engine.readme_figures` (official data; `--synthetic` offline); the only generated files committed. |

## Method

**Conventions.** Rates and dividend yields are continuously compounded; maturities are year fractions. The ECB Svensson curve gives continuously compounded zero rates of AAA-rated euro area central government bonds, used as a proxy for the risk-free rate. Returns in the risk module are daily log-returns in percent; VaR and ES are positive losses.

**Black-Scholes-Merton** (`cpp/black_scholes.hpp`): closed-form prices and Greeks; implied volatility by Newton iterations safeguarded with bisection on $[10^{-9}, 10]$, NaN outside the no-arbitrage bounds and whenever the iteration ends without repricing the input to $10^{-12}$ (prices that would need a volatility below the bracket).

**Finite differences** (`cpp/finite_difference.hpp`): Crank-Nicolson in log-price on a uniform grid centred on $S_0$ (half-width 5 standard deviations plus the log-moneyness), Dirichlet boundaries, Rannacher start (the first two steps replaced by four implicit Euler half steps), Thomas algorithm for European options and projected SOR (relaxation 1.2, tolerance $10^{-12} K$) for American options. The early-exercise boundary is extracted at each time step from the interior nodes where the value equals the payoff (NaN when early exercise is never optimal, as for a call without dividends).

**Heston** (`cpp/heston.hpp`): characteristic function in the Albrecher et al. (2007) form; call prices by

$$C = \tfrac12\left(S e^{-qT} - K e^{-rT}\right) + \frac{e^{-rT}}{\pi}\int_0^\infty \mathrm{Re}\left[\frac{e^{iuk}\left(S\varphi(u-i) - K\varphi(u)\right)}{iu}\right]du,\qquad k=\ln(S/K),$$

integrated with composite 16-point Gauss-Legendre panels whose width is the smaller of $0.5/\sqrt{\max(v_0,\theta)T}$ (the decay scale of the characteristic function) and a quarter period $\pi/(2|k|)$ of the oscillation $e^{iuk}$ (which limits the resolution at short maturities away from the money), continued at least to $u = 4/\sqrt{\max(v_0,\theta)T}$ and until four consecutive panels contribute less than $10^{-14}\max(S,K)$; puts by put-call parity. Monte Carlo uses the QE scheme of Andersen (2008) with $\psi_c = 1.5$ and central discretisation ($\gamma_1=\gamma_2=\tfrac12$), without martingale correction. Calibration (`quant_engine/heston.py`) minimises vega-weighted price errors with `scipy.optimize.least_squares`.

**GARCH** (`cpp/garch.hpp`): GARCH(1,1) and GJR-GARCH(1,1) with Gaussian or unit-variance Student-t innovations, recursion initialised with the sample variance, maximum likelihood with a Nelder-Mead simplex on an unconstrained reparametrisation ($\omega,\alpha,\beta,\alpha+\gamma>0$, $\nu>2$) and a penalty for $\alpha+\gamma/2+\beta\ge 1$, with restarts. Rolling forecasts refit the model every `refit_every` days on a moving window; fits run in parallel. Filtered historical simulation uses the empirical quantile (linear interpolation) of the standardised residuals of the window.

**Backtests and capital** (`quant_engine/risk.py`): Kupiec proportion of failures, Christoffersen independence and conditional coverage, Basel traffic light, Acerbi-Szekely $Z_2$ for ES. FRTB internal-models charge from the 10-day 97.5% ES on overlapping returns calibrated to the worst 12-month window, $\max(\text{IMCC}_{t-1}, m_c\,\overline{\text{IMCC}}_{60})$ with $m_c$ from 1.5 (green zone) to 2.0 (red zone); Basel 2.5 charge from 99% VaR and stressed VaR with square-root-of-time scaling and multiplier 3 plus the traffic-light add-on.

**Delta-hedging simulator** (`cpp/hedging.hpp`, `quant_engine/hedging.py`): the dealer sells a European option priced at one volatility and holds the Black-Scholes/Garman-Kohlhagen delta computed at another, reviewed every $k$ steps and traded either at every review or only when the delta has moved by more than a band (move-based hedging, Whalley and Wilmott, 1997); cash accrues at the domestic rate, the underlying position at the foreign rate (or dividend yield), each trade pays a proportional cost, and the payoff is settled at maturity. The underlying follows GBM, Heston (QE scheme) or GJR-GARCH with resampled standardised residuals (filtered historical simulation). One simulation step is $T/n$ years; the GARCH recursion is daily, so with the GARCH-FHS dynamics $n$ must equal $\mathrm{round}(252\,T)$ trading days (the simulator rejects any other step, since the recursion does not scale with it). Paths are shared across hedging policies (common random numbers) and blocks of 1,024 paths have their own random streams.

**Yield-curve factor models** (`quant_engine/term_structure.py`, `cpp/term_structure.hpp`): month-end zero rates from the ECB Svensson parameters at 11 maturities (3 months to 30 years). Principal components of the covariance of monthly changes, with sign conventions for level, slope and curvature. Dynamic Nelson-Siegel (Diebold and Li, 2006): two-step estimation with the decay fixed at 0.0609 per month and AR(1) or VAR(1) factors; one-step state-space estimation (Diebold, Rudebusch and Aruoba, 2006) with $y_t = \Lambda(\lambda) f_t + e_t$, $f_t - \mu = A(f_{t-1}-\mu) + u_t$, the state started from its unconditional distribution (the estimator admits only transition matrices whose eigenvalues have modulus below `STATIONARITY_BOUND` = 0.9999, the same bound at which the filter switches to a diffuse prior for user-supplied models, so the likelihood is continuous inside the admissible region), $Q$ parameterised by its Cholesky factor and measurement standard deviations either free by maturity or common. Both filters use the covariance form with the Joseph update of the state covariance, $(I-K\Lambda)P(I-K\Lambda)' + KHK'$, which stays symmetric and positive semi-definite in floating point: the NumPy reference factorises the innovation covariance $S = \Lambda P \Lambda' + H$ of the maturities observed on each date by Cholesky, and the C++ filter, since $H$ is diagonal, assimilates the maturities one at a time with scalar innovation variances $\lambda_i' P \lambda_i + h_i^2$ (sequential processing, the same posterior and likelihood with 3 x 3 operations only). This keeps full precision when some measurement standard deviations approach the 0.01 bp floor or the prior is diffuse (an information-form update lost the prior to rounding there, with log-likelihood errors of $10^{-3}$ and gradients of the wrong sign); missing values are skipped. The likelihood is maximised by BFGS restarted from its own solution until it stops improving, first with forward-difference and then with central-difference gradients, and polished by Newton steps with the numerical Hessian while it is positive definite (the polish brings the gradient from rounding noise of order $10^{-2}$ down to about $10^{-5}$, so that the convergence check no longer depends on the last digits of the inputs), convergence is checked with a central-difference gradient, and standard errors come from the inverse numerical Hessian. With free measurement variances on the ECB curve the likelihood is degenerate (some maturities become observed without error), so the notebook uses the common-variance model.

**Arbitrage-free Nelson-Siegel** (`quant_engine/afns.py`): the independent-factor AFNS model of Christensen, Diebold and Rudebusch (2011): the Nelson-Siegel loadings are those of a Gaussian affine model with short rate $X_1 + X_2$ and risk-neutral mean reversion $K^Q = \begin{psmallmatrix}0&0&0\\0&\lambda&-\lambda\\0&0&\lambda\end{psmallmatrix}$, so absence of arbitrage adds the yield adjustment $-C(\tau)/\tau = -\operatorname{Var}(\int_0^\tau r_s\,ds)/(2\tau)$, in closed form for diagonal volatilities; physical dynamics $dX = K^P(\theta - X)\,dt + \Sigma\,dW$ with diagonal $K^P$ and $\Sigma$, discretised exactly at the monthly interval ($A = e^{-\kappa\,\Delta t}$, $Q_{ii} = \sigma_i^2(1-e^{-2\kappa_i\Delta t})/(2\kappa_i)$); the likelihood is the Kalman filter of the DNS model applied to the adjusted yields, maximised as for the DNS model (mean reversions of at least 0.01 per year). With `arbitrage_free=False` the same model without the adjustment gives the like-for-like comparison of the authors.

**Forecast evaluation**: recursive expanding-window forecasts at 1, 6 and 12 months (random walk; two-step DNS with direct AR(1) or iterated VAR(1); state-space DNS re-estimated every 12 months), each using data up to the origin only; Diebold-Mariano test on squared errors with $h-1$ autocovariances and the Harvey, Leybourne and Newbold (1997) correction.

**IRRBB** (`quant_engine/term_structure.py`, `quant_engine/banking_book.py`): the six standardised scenarios of BCBS (2016) with euro shock sizes (parallel 200 bp, short 250 bp, long 100 bp), the post-shock floor of EBA/GL/2022/14 (-150 bp rising by 3 bp a year to 0% at 50 years, or the observed rate when lower), EVE as the present value of all interest-rate cash flows at continuously compounded zero rates, NII over one year with a constant balance sheet by the repricing-gap method, and the EBA supervisory outlier tests (EVE decline above 15% of Tier 1, NII decline above 5%). The stylised bank's behavioural assumptions are stated in the module: mortgage prepayment at a constant rate scaled by the BCBS (2016) Table 3 multipliers (0.8 in the parallel-up, steepener and short-rates-up scenarios, 1.2 in the parallel-down, flattener and short-rates-down scenarios), non-maturity deposits split into an overnight part and a core part slotted evenly over a fixed horizon, quarterly repricing of floating items. Swaps are valued as a floating leg worth par at the next reset against a fixed bullet leg at the par rate. The hedge that maximises the worst EVE change over a scenario set (the six BCBS scenarios, optionally with every historical 12-month curve move) is a linear programme solved with HiGHS, with optional receive-fixed swaps, a gross-notional penalty and an NII constraint.

**Random numbers and parallelism** (`cpp/random.hpp`, `cpp/parallel.hpp`): xoshiro256** seeded through SplitMix64 and Box-Muller normals, instead of the implementation-defined `std::normal_distribution`, so a seed gives the same numbers with any compiler. Monte Carlo paths are simulated in blocks of 4,096 with one random stream per block and block sums are combined in a fixed order: results are identical for any number of threads. Every notebook fixes its seed (`SEED`) and reports standard errors.

## Verification

Run `python -m pytest tests/quant_engine` from the repository root (141 tests, about two minutes). The main checks and their tolerances:

| Check | Tolerance and justification |
| --- | --- |
| C++ Black-Scholes prices and Greeks vs NumPy/SciPy | relative $10^{-12}$ for prices, $10^{-10}$ for Greeks: same formulas, rounding only |
| Greeks vs central finite differences | relative $10^{-6}$ ($10^{-4}$ for gamma): truncation error of the differences |
| Implied volatility round trip; NaN for prices that need a volatility below the bracket; every finite result reprices its input | $10^{-9}$ in volatility; $10^{-12}$ in price |
| Heston vs Fang and Oosterlee (2008) reference 5.785155450 | $2\times10^{-8}$: the published value has ten significant digits and was itself computed numerically |
| Heston vs SciPy adaptive quadrature of an independent NumPy characteristic function, including 36 short-dated cases ($T$ from 0.001 to 1/52, variance from $10^{-4}$ to 0.04, strikes from 70% to 135% of spot) with the no-arbitrage bounds | $10^{-9}$ (absolute, per unit of $\max(S,K)$) |
| Heston with vanishing vol of vol vs Black-Scholes with the integrated variance | $10^{-5}$ (error of order $\sigma^2$) |
| QE Monte Carlo vs Fourier | 4 standard errors plus 0.01 for the discretisation bias at 50 steps per year |
| Monte Carlo reproducibility with 1 and 4 threads | bitwise equality |
| Crank-Nicolson European prices vs closed form (1,600 x 1,600) | $10^{-4}$; observed convergence order 2 (error ratio between 3.5 and 4.5 when the grid is doubled) |
| American put vs 20,000-step binomial tree; American calls with dividend yields and a put with a negative rate vs the same tree; American delta and gamma vs bumped prices; no exercise boundary (NaN) when early exercise is never optimal | $5\times10^{-4}$ (tree error of order $10^{-4}$); $10^{-4}$ and $10^{-3}$ (grid error of the bumped prices); exact |
| Calibration round trip on noise-free data (and the optimiser reports success) | implied-volatility RMSE below $10^{-6}$ |
| GARCH likelihood C++ vs pure Python | relative $10^{-11}$ |
| Nelder-Mead vs SciPy SLSQP maximum | log-likelihood not lower by more than $10^{-4}$ |
| Parameter recovery on 20,000 simulated observations; every fit reports convergence; half-life and news-impact curve vs the GJR recursion | within 4 asymptotic standard errors; exact; relative $10^{-12}$ |
| Rolling forecasts | no look-ahead: shocking a future return leaves earlier forecasts unchanged |
| VaR/ES formulas | known normal quantiles, Student-t ES by numerical integration, $Z_2 \approx 0$ under the true model |
| Backtest table: reproduces Kupiec, Christoffersen, conditional coverage (LR sum, $\chi^2_2$) and $Z_2$ computed directly; forecasts that start later or contain NaN are aligned by date; zero and all exceptions | exact |
| Capital helpers | FRTB and Basel 2.5 multiplier tables; stressed window located on data with a known stressed year; capital charge takes the larger of the two terms |
| Unhedged option P&L equals premium minus discounted payoff, path by path (call and put) | relative $10^{-12}$ |
| Hedged call and put P&L coincide path by path (put-call parity of the deltas) | $10^{-10}$ |
| Delta band: a zero band reproduces the time-based hedge bitwise; a band wider than any move keeps the initial hedge (costs equal the initial trade plus the discounted liquidation, path by path); a moderate band lies between the two in costs and dispersion and hedges better than weekly rebalancing at similar cost | exact; relative $10^{-12}$; order |
| Daily hedge at the true volatility: unbiased, standard deviation vs Kamal and Derman (1999) $\sqrt{\pi/4}\,\text{vega}\,\sigma/\sqrt{n}$ | mean within 4 standard errors; 10% (asymptotic formula) |
| Hedging error scales with the square root of the rebalancing interval | ratio within 10% of 2 when rebalancing every 4 days instead of daily |
| Selling above the true volatility and hedging at the true volatility locks in the premium difference | 4 standard errors plus $2\times10^{-5}$ |
| Transaction costs: P&L difference equals the reported costs path by path | relative $10^{-9}$ |
| GARCH-FHS with constant variance reproduces the variance of log returns; reproducibility across threads; a step other than one trading day is rejected | 3%; bitwise equality; `ValueError` |
| Nelson-Siegel loadings: limits at zero and infinite maturity, short end to second order ($1 - x/2$, computed with `expm1`), curvature peak at 30 months with the Diebold-Li decay | $10^{-15}$ and $10^{-3}$; 0.1 month |
| Cross-sectional fit recovers exact factors from noise-free curves | $10^{-10}$ |
| Kalman log-likelihood (NumPy and C++) vs the joint Gaussian density of all observations | relative $10^{-10}$: the prediction-error decomposition is exact |
| Same with two maturities at the 0.01 bp floor (200 months, 2,200 observations); central-difference gradients of both engines | relative $10^{-9}$; $10^{-3}$ (rounding of $10^{-12}$ relative divided by the step $10^{-5}$) |
| Diffuse prior vs a 50-digit filter (mpmath, offline) | $10^{-5}$ on 655 (relative $2\times10^{-8}$) |
| Likelihood continuous across the former prior switch at 0.999; the prior and the estimator use the same stationarity bound | $10^{-2}$ for a $2\times10^{-6}$ change in the eigenvalue; exact |
| C++ vs NumPy filter with missing values: log-likelihood, filtered and predicted states | relative $10^{-11}$, absolute $10^{-10}$ |
| Maximum likelihood on 500 simulated months: decay, autoregressive coefficients, measurement error (free and common); convergence diagnostic and estimates unchanged when the data are perturbed by $10^{-12}$ | 5%; within 4 standard errors and 0.05; 20% (10% with a common variance); $10^{-4}$ |
| AFNS yield adjustment: closed form vs numerical integration of the convexity for three parameter sets; vs Monte Carlo of $-\log E[e^{-\int r}]$ under the risk-neutral dynamics (20,000 paths); negative, decreasing in maturity, zero at zero maturity or volatility | relative $10^{-10}$; 4 standard errors plus 0.2 bp of Euler bias |
| AFNS exact discretisation ($Q$ by numerical integration), parameter round trip, stationary prior $\sigma^2/(2\kappa)$; likelihood of both engines vs the joint Gaussian density with the adjustment in the mean | relative $10^{-12}$; relative $10^{-10}$ |
| AFNS maximum likelihood on 400 simulated months: convergence, decay, measurement error, volatilities, likelihood at least that of the true parameters; forecasts tend to the long-run curve | 5%; 5%; 20%; exact; $10^{-8}$ |
| Recursive forecasts (two-step, state space and AFNS) | no look-ahead: shifting the yields from a date leaves earlier errors unchanged and moves the errors of forecasts made before the date for targets after it by exactly the shift (a forecast using any later data would move by less) |
| Diebold-Mariano statistic at horizon 1 vs the direct formula | relative $10^{-12}$ |
| BCBS scenarios and EBA floor at known maturities; zero-coupon EVE change with and without the floor | exact values; relative $10^{-12}$ |
| Mortgage pool: no prepayment equals the annuity; any prepayment rate repays the notional and is worth par at the loan rate; prepayment multipliers equal BCBS (2016) Table 3 and front-load the cash flows when above 1 | relative $10^{-12}$; exact |
| Par swap worth zero on the base curve; payer swap gains when rates rise; NII by repricing gap vs hand calculation | $10^{-10}$; sign; exact |
| Bank book: balance sheet balances, floating items at par, EVE by position adds up, swap effects linear in notional | relative $10^{-9}$ to $10^{-12}$ |
| Hedge optimisation: minimax value equals the worst scenario, larger feasible sets never do worse, NII limit respected | $10^{-6}$ |
| Data parsers (ECB FX, ECB yield curve, FRED with `.` and blank missing values and unsorted dates) | exact |
| README figures: a failed rate download warns and is named in the figure's source line | exact |

The notebooks repeat the main validations on the data used and now assert them (Crank-Nicolson error and order of convergence, Fourier price against the published benchmark, QE bias within the documented bound, calibration at the noise level, C++ against NumPy Kalman likelihood, convergence of every fit, exception recounts, capital formulas, EVE adding up by position), so the offline CI run of the notebooks fails when a validation drifts; see `notebooks/README.md` for what that run does not cover.

## C++ unit tests and sanitizers

**Tests.** The engines are header-only, so [`cpp/CMakeLists.txt`](cpp/CMakeLists.txt) also compiles them without Python. It uses `-Wall -Wextra -Wpedantic -Wshadow -Wold-style-cast -Wnull-dereference -Wdouble-promotion -Werror` (`/W4 /WX` with MSVC) and bounds-checked standard containers outside Release builds. [`cpp/tests/test_core.cpp`](cpp/tests/test_core.cpp) runs 1,207 checks without any test framework. The Python suite compares the engines with independent NumPy and SciPy implementations; these tests add published values, exact identities, thread invariance and argument validation on the C++ code alone:

| Check | Tolerance |
| --- | --- |
| SplitMix64 and xoshiro256** against the published reference outputs | exact |
| Uniform, normal and gamma variates: range, mean, variance, 5% two-sided tail frequency | 4 standard errors |
| `parallel_for`: same result for 1, 3 and 8 threads; an exception in a task is rethrown | exact |
| Black-Scholes against Hull (10.4506; Example 15.6: 4.76 and 0.81) | half a unit in the last published digit |
| Put-call parity, parity of the Greeks, Greeks as central differences of the price, no-arbitrage bounds, zero volatility and zero time | $10^{-12}S$; $10^{-6}$ (truncation $O(h^2)$) |
| Implied volatility: repricing on 90 cases, volatility where the vega exceeds $10^{-3}$, NaN outside the no-arbitrage bounds and for prices below the bracket, every finite result reprices its input | $10^{-11}$; $10^{-7}$; exact; $10^{-12}$ |
| Crank-Nicolson (400 x 400) price, delta and gamma vs closed form; error cut at least 16 times from 100 to 400 nodes | $10^{-4}$, $5\times10^{-5}$, $10^{-6}$ |
| American put vs a Cox-Ross-Rubinstein tree (mean of 4,000 and 4,001 steps); value above the payoff; exercise boundary below the strike and monotone in maturity; American call without dividends and put with a negative rate equal the European with no exercise boundary; call with a yield above the rate has a boundary above the strike | $2\times10^{-4}$; exact; $10^{-6}$; exact |
| Heston characteristic function: $\varphi(0)=1$, martingale $\varphi(-i)=e^{(r-q)T}$, conjugate symmetry; 16-point Gauss-Legendre exact to degree 31 | $10^{-13}$; $10^{-14}$ |
| Heston vs Fang and Oosterlee (2008) 5.785155450; vanishing vol of vol vs Black-Scholes; bounds; parameter validation | $2\times10^{-8}$; $10^{-5}$; exact |
| QE Monte Carlo vs Fourier (calls, puts, forward); put-call parity of the estimates; 1 vs 5 threads | 4 standard errors plus 0.01; $10^{-9}$; bitwise |
| $\ln\Gamma$ vs `std::lgamma`; Nelder-Mead on the Rosenbrock function | $10^{-12}$; $10^{-5}$ |
| GARCH likelihood with constant variance and with the GJR recursion written out; Student-t tending to the normal; reparametrisation round trip | $10^{-9}$; $10^{-3}$; $10^{-14}$ |
| GJR-GARCH-t fit on 4,000 simulated returns: likelihood at least as high as at the true parameters; convergence reported; persistence, $\beta$, $\gamma$, $\nu$ | exact; 0.02, 0.06, 0.06, 2.1 (about three asymptotic standard errors) |
| Rolling GARCH forecasts: 1 vs 4 threads; expected shortfall above VaR; VaR falls with the tail probability; argument validation | bitwise; exact |
| Delta hedging: unhedged mean P&L zero at the true volatility (call and put, the put identity path by path); daily hedge unbiased (a martingale, no slack) and cuts the standard deviation below 15%; call and put hedged P&L coincide path by path; weekly vs daily ratio near $\sqrt{5}$; P&L falls by exactly the reported costs, path by path; Heston and GARCH-FHS dynamics identical for 1 and 3 threads; GARCH-FHS with a step other than a trading day rejected | 4 standard errors; $10^{-12}$; $10^{-9}$; ratio in (1.9, 2.6); $10^{-10}$; bitwise; exact |
| Nelson-Siegel loadings at one month and 1,000 years | $10^{-13}$ |
| Kalman filter vs an independent dense matrix implementation with missing values and an empty date, also with two measurement standard deviations at the 0.01 bp floor: log-likelihood and filtered states; a non-positive-definite prior gives $-\infty$; argument validation | relative $10^{-9}$; $10^{-10}$ ($10^{-8}$ at the floor); exact |

**Mutation check.** Seven deliberate bugs were injected into the engines, and each makes the suite fail:

- the sign of the interest term in the call theta;
- the original Heston characteristic function instead of the "little trap" form;
- projected SOR without the early-exercise constraint;
- transaction costs not charged to the hedger's cash;
- the Kalman covariance update skipped;
- the GJR leverage term applied after positive shocks (in the likelihood and, separately, in the simulator);
- the expected shortfall of filtered historical simulation replaced by the VaR.

The leverage bug was missed until the written-out GJR recursion and the check on the fitted leverage coefficient were added. A control mutation that seeds the Monte Carlo streams differently, but still independently of the thread count, correctly passes.

**Sanitizers.** The same tests pass under two sanitizer builds:

- AddressSanitizer with UndefinedBehaviorSanitizer (`-DQE_SANITIZE=address,undefined`): out-of-bounds access, use after free, signed overflow and similar;
- ThreadSanitizer (`-DQE_SANITIZE=thread`): data races in the parallel Monte Carlo, hedging and rolling-GARCH loops.

CI runs both, plus GCC and Clang builds with warnings as errors.

## Continuous integration

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) runs on every pull request and on pushes to `main`. It needs no credentials and downloads no market data. Its jobs:

1. **Python:** lint (ruff, pyflakes rules) and the test suite, on Python 3.11 with the locked versions and on 3.10 and 3.12 with the newest allowed versions.
2. **C++:** the standalone tests with GCC and Clang (warnings as errors), and with GCC under ASan/UBSan and under TSan.
3. **Notebooks:** every Python notebook executed offline on synthetic data. The job checks that the data cache stays empty, i.e. that synthetic mode does not touch the network.

## References

- Albrecher, H., Mayer, P., Schoutens, W. and Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*, January, 83-92.
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational Finance*, 11(3), 1-42.
- Acerbi, C. and Szekely, B. (2014). Backtesting expected shortfall. *Risk*, December.
- Barone-Adesi, G., Giannopoulos, K. and Vosper, L. (1999). VaR without correlations for portfolios of derivative securities. *Journal of Futures Markets*, 19(5), 583-602.
- Blackman, D. and Vigna, S. (2021). Scrambled linear pseudorandom number generators. *ACM Transactions on Mathematical Software*, 47(4). Reference code in the public domain: https://prng.di.unimi.it/
- Basel Committee on Banking Supervision (2016). Interest rate risk in the banking book. Standards, April 2016.
- Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841-862.
- Diebold, F. X. and Li, C. (2006). Forecasting the term structure of government bond yields. *Journal of Econometrics*, 130(2), 337-364.
- Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics*, 13(3), 253-263.
- Christensen, J. H. E., Diebold, F. X. and Rudebusch, G. D. (2011). The affine arbitrage-free class of Nelson-Siegel term structure models. *Journal of Econometrics*, 164(1), 4-20.
- Diebold, F. X., Rudebusch, G. D. and Aruoba, S. B. (2006). The macroeconomy and the yield curve: a dynamic latent factor approach. *Journal of Econometrics*, 131(1-2), 309-338.
- European Banking Authority (2022). Guidelines on IRRBB and CSRBB (EBA/GL/2022/14) and regulatory technical standards on supervisory outlier tests (EBA/RTS/2022/10).
- Fang, F. and Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. *SIAM Journal on Scientific Computing*, 31(2), 826-848.
- Basel Committee on Banking Supervision (2019). Minimum capital requirements for market risk, MAR33 and MAR99.
- Garman, M. B. and Kohlhagen, S. W. (1983). Foreign currency option values. *Journal of International Money and Finance*, 2(3), 231-237.
- Glosten, L. R., Jagannathan, R. and Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *Journal of Finance*, 48(5), 1779-1801.
- Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281-291.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6(2), 327-343.
- Kamal, M. and Derman, E. (1999). When you cannot hedge continuously: the corrections of Black-Scholes. *Risk*, 12(1), 82-85.
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*, 3(2), 73-84.
- Litterman, R. and Scheinkman, J. (1991). Common factors affecting bond returns. *Journal of Fixed Income*, 1(1), 54-61.
- Lagarias, J. C., Reeds, J. A., Wright, M. H. and Wright, P. E. (1998). Convergence properties of the Nelder-Mead simplex method in low dimensions. *SIAM Journal on Optimization*, 9(1), 112-147.
- Marsaglia, G. and Tsang, W. W. (2000). A simple method for generating gamma variables. *ACM Transactions on Mathematical Software*, 26(3), 363-372.
- McNeil, A. J., Frey, R. and Embrechts, P. (2015). *Quantitative Risk Management*, revised edition. Princeton University Press.
- Rannacher, R. (1984). Finite element solution of diffusion problems with irregular data. *Numerische Mathematik*, 43, 309-327.
- Nelson, C. R. and Siegel, A. F. (1987). Parsimonious modeling of yield curves. *Journal of Business*, 60(4), 473-489.
- Svensson, L. E. O. (1994). Estimating and interpreting forward interest rates: Sweden 1992-1994. NBER Working Paper 4871.
- European Central Bank: euro foreign exchange reference rates and euro area yield curves, https://www.ecb.europa.eu/stats/
