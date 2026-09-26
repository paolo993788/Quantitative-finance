# R cross-checks

Notebooks written in R that recompute the key results of the Python notebooks with independent R packages, starting again from the
official data, and compare the two implementations number by number. They also add analyses that are standard in R.

| Notebook | Question | Data |
| --- | --- | --- |
| [`fx_garch_var_r.ipynb`](fx_garch_var_r.ipynb) | Do an independent R estimation (fGarch) and independent backtest statistics reproduce the C++ risk engine? | ECB reference rates |

## Requirements

- R 4.2 or later (developed with R 4.3.3) and the packages listed in [`install_packages.R`](install_packages.R): `IRkernel`, `jsonlite`, `ggplot2`, `fGarch`, `tseries`.
- For the comparison with the Python results, the Python environment of the repository (see the main README). The notebook runs the
  interpreter in the `PYTHON` environment variable, or `.venv` in the repository root, or `python3` on the `PATH`; if none has the
  package installed, the comparison cells are skipped and the R results are still shown.

## Run

```bash
Rscript notebooks/r_crosschecks/install_packages.R      # once: packages and the Jupyter kernel "R"
```

Then open the notebook in Visual Studio Code (extension *Jupyter*) or Jupyter and select the kernel **R**. Downloads are cached in
`data/raw/r/` (ignored by Git). The notebook is published with the outputs of the September 2026 run on official data.
