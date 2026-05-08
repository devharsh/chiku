"""chiku: polynomial function approximation library.

Approximators are accessed via their submodules:

    from chiku import taylor, pade, chebyshev, fourier, remez, sk_lr, tf_ann
    poly = taylor.taylor(f, degree=5, frange=(-1, 1))

Arbitrary-precision variants live in the ``chiku.mpmath`` subpackage:

    from chiku.mpmath import mptaylor, mppade, mpchebyshev, mpfourier, mpremez
"""

__version__ = "2.0.0"

from . import taylor      # noqa: F401
from . import pade        # noqa: F401
from . import chebyshev   # noqa: F401
from . import fourier     # noqa: F401
from . import remez       # noqa: F401
from . import sk_lr       # noqa: F401
from . import sk_lr_cheb  # noqa: F401


def __getattr__(name):
    """Lazy import of tf_ann so the package works without TensorFlow installed."""
    if name == "tf_ann":
        import importlib
        _tf_ann = importlib.import_module(".tf_ann", __name__)
        globals()["tf_ann"] = _tf_ann
        return _tf_ann
    raise AttributeError("module 'chiku' has no attribute {!r}".format(name))


__all__ = [
    "__version__",
    "taylor",
    "pade",
    "chebyshev",
    "fourier",
    "remez",
    "sk_lr",
    "sk_lr_cheb",
    "tf_ann",
]
