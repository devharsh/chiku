"""mpTaylor: Taylor series approximation in arbitrary precision via mpmath.

Computes the truncated Taylor series of an mpmath-typed function around an
expansion point by numerical differentiation in mp precision.

The expected callable signature accepts and returns mpmath types
(``mpf``/``mpc``) so all arithmetic stays at the working precision
``mp.dps``.
"""

import mpmath as mp


class mpTaylor:
    """Truncated Taylor series of an mpmath callable.

    Parameters
    ----------
    func : callable
        Target function. Must accept and return mpmath types.
    fpoint : mpf or float, optional
        Expansion point. Default 0.
    degree : int, optional
        Highest power retained. Default 5.

    Attributes
    ----------
    coeffs : list of mpf
        Taylor coefficients ``[c0, c1, ..., c_degree]`` such that
        ``f(x) ~ sum_k c_k * (x - fpoint)**k``.
    """

    def __init__(self, func=mp.sin, fpoint=0, degree=5):
        self.func = func
        self.fpoint = mp.mpf(fpoint)
        self.degree = int(degree)

        # Compute coefficients c_k = f^(k)(fpoint) / k!
        # mpmath.diff(f, x, k) gives the k-th derivative at x.
        self.coeffs = []
        for k in range(self.degree + 1):
            if k == 0:
                deriv = self.func(self.fpoint)
            else:
                deriv = mp.diff(self.func, self.fpoint, k)
            self.coeffs.append(deriv / mp.factorial(k))

    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __setitem__(self, idx, val):
        self.coeffs[idx] = val

    def predict(self, x):
        """Evaluate the truncated Taylor series at x."""
        x = mp.mpf(x) if not isinstance(x, mp.mpc) else x
        u = x - self.fpoint
        result = mp.mpf(0)
        u_pow = mp.mpf(1)
        for c in self.coeffs:
            result += c * u_pow
            u_pow *= u
        return result