"""mpChebyshev: Chebyshev series approximation in arbitrary precision via mpmath.

Computes the truncated Chebyshev series of a function on ``[a, b]`` of length
``degree`` using collocation at Chebyshev nodes of the second kind followed
by a discrete cosine projection. All arithmetic is performed at the current
``mp.dps`` precision.

Coefficient convention: ``f(x) ~ c0/2 + sum_{k>=1} c_k T_k(t(x))`` where
``t = (2x - a - b) / (b - a)``. The first coefficient is stored unhalved
(Numerical Recipes convention) and halved internally during evaluation.
"""

import mpmath as mp


class mpChebyshev:
    """Chebyshev series approximation in arbitrary precision.

    Parameters
    ----------
    func : callable
        Target function. Must accept and return mpmath types.
    domain : tuple of (mpf or float), optional
        Interval ``[a, b]``. Default ``(-1, 1)``.
    degree : int, optional
        Number of Chebyshev coefficients to compute. Default 5.

    Attributes
    ----------
    coeffs : list of mpf
        Chebyshev coefficients ``[c_0, c_1, ..., c_{degree-1}]``.
    """

    def __init__(self, func=mp.cos, domain=(-1, 1), degree=5):
        self.func = func
        self.a = mp.mpf(domain[0])
        self.b = mp.mpf(domain[1])
        self.n = int(degree)
        if self.b <= self.a:
            raise ValueError("require domain[1] > domain[0]")
        if self.n < 1:
            raise ValueError("degree must be >= 1")

        bma = (self.b - self.a) / 2
        bpa = (self.b + self.a) / 2
        pi = mp.pi
        n = self.n

        # f sampled at the n Chebyshev nodes mapped onto [a, b]
        f = []
        for k in range(n):
            t = mp.cos(pi * (k + mp.mpf(1) / 2) / n)
            f.append(self.func(t * bma + bpa))

        fac = mp.mpf(2) / n
        self.coeffs = []
        for j in range(n):
            s = mp.mpf(0)
            for k in range(n):
                s += f[k] * mp.cos(pi * j * (k + mp.mpf(1) / 2) / n)
            self.coeffs.append(fac * s)

    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __setitem__(self, idx, val):
        self.coeffs[idx] = val

    def predict(self, x):
        """Evaluate the truncated Chebyshev series at x using Clenshaw recurrence."""
        x = mp.mpf(x) if not isinstance(x, mp.mpc) else x
        # Map x in [a, b] to y in [-1, 1]
        y = (2 * x - self.a - self.b) / (self.b - self.a)
        y2 = 2 * y
        # Clenshaw recurrence
        d = self.coeffs[-1]
        dd = mp.mpf(0)
        for cj in self.coeffs[-2:0:-1]:
            d, dd = y2 * d - dd + cj, d
        return y * d - dd + self.coeffs[0] / 2