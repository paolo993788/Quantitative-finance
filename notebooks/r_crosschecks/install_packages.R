# R packages used by the notebooks in this folder (R 4.2 or later).
# Run once from the repository root:  Rscript notebooks/r_crosschecks/install_packages.R
packages <- c("IRkernel", "jsonlite", "ggplot2", "fGarch", "tseries")
missing <- setdiff(packages, rownames(installed.packages()))
if (length(missing) > 0) install.packages(missing, repos = "https://cloud.r-project.org")
# Register the R kernel for Jupyter and Visual Studio Code (requires Jupyter on the PATH, e.g. the project's .venv activated).
IRkernel::installspec(name = "ir", displayname = "R")
