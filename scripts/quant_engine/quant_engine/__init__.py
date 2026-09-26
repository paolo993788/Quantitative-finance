"""C++-accelerated option pricing and market risk tools.

The compiled extension ``quant_engine._core`` contains the numerical engines
(Black-Scholes, Heston Fourier and Monte Carlo pricing, Crank-Nicolson finite
differences and GARCH estimation). The Python modules provide reference
implementations used for validation, data loaders for official sources and
statistical backtests.
"""

__version__ = "0.1.0"

try:
    from . import _core  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on the local build
    _core = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

HAS_CPP = _core is not None


def require_cpp():
    """Return the compiled extension or raise an informative error."""
    if _core is None:
        raise ImportError(
            "The C++ extension quant_engine._core is not built. From the repository root run "
            "`python -m pip install -e scripts/quant_engine` (a C++17 compiler is required)."
        ) from _IMPORT_ERROR
    return _core
