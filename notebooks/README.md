# Notebooks

Store notebooks here, grouped by topic. Each notebook states its language and version, its dependencies, its data sources and, where relevant, the order in which the notebooks should be run.

Before publishing a notebook, restart the kernel and run all cells in order. Remove outputs that contain confidential data or credentials, as well as unnecessarily large results, and save generated files to `outputs/`.

Add every published notebook to the catalogue in the [main README](../README.md).

Notebooks are published with their outputs, produced by a full run on official data, so that tables and charts can be read on GitHub without running them. `r_crosschecks/` contains R notebooks that verify the main results with independent R packages.

## What the CI run of the notebooks covers

CI executes every Python notebook offline with `QUANT_ENGINE_DATA_MODE=synthetic`. The validation cells of the notebooks are
`assert` statements, so the run fails when a check drifts: Crank-Nicolson error and order of convergence, the Heston Fourier
price against the published benchmark and the SciPy quadrature, the QE Monte Carlo bias, the calibration error, the agreement
of the C++ and NumPy Kalman filters, the convergence of every GARCH and DNS fit, the exception recounts, the capital formulas
and the decomposition of EVE by position. A deliberately wrong benchmark value makes the CI step fail (checked when the asserts
were added).

The synthetic run does not exercise: the `DATA_MODE == "official"` branches (ECB and FRED downloads, `rate_or_assumption` in
the FX desk notebook), the negative-rate headline of the yield-curve notebook (synthetic rates stay positive), the checks that
describe the official data (Brexit and March 2020 in the stress table, the dates of the stressed GARCH variance), and the R
notebook. Those are covered by the published runs on official data, whose outputs are committed with the notebooks.
