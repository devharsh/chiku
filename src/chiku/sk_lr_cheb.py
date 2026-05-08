"""Polynomial regression fitted in the Chebyshev basis.

Numerically stable alternative to ``sk_lr``: fits a Chebyshev series on
``frange`` via ``numpy.polynomial.Chebyshev.fit`` and converts back to the
monomial basis so ``predict(x) = sum_i coeffs[i] * x**i`` matches the
other approximators.

Note that ``degree`` is a list of powers like in ``sk_lr``, but the fit
uses every power from 0 to ``max(degree)`` (Chebyshev fitting is
inherently dense). Powers absent from the list are still represented in
the returned coefficients.
"""

import numpy as np


class sk_lr_cheb:

    def __init__(self, f, degree=(1, 2, 3, 4, 5), frange=(-1.0, 1.0),
                 points=2 ** 11):
        self.f = f
        self.degree = list(degree)
        self.max_deg = max(self.degree)
        self.a, self.b = float(frange[0]), float(frange[1])
        if self.b <= self.a:
            raise ValueError("require frange[1] > frange[0]")
        self.points = int(points)

        xs = np.linspace(self.a, self.b, self.points)
        ys = np.array([float(f(x)) for x in xs], dtype=np.float64)

        cheb = np.polynomial.Chebyshev.fit(
            xs, ys, deg=self.max_deg, domain=[self.a, self.b]
        )
        # Convert to standard power basis.
        poly = cheb.convert(kind=np.polynomial.Polynomial)
        coeffs = list(poly.coef)
        while len(coeffs) < self.max_deg + 1:
            coeffs.append(0.0)
        self.coeffs = np.array(coeffs, dtype=float)

    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __setitem__(self, idx, val):
        self.coeffs[idx] = val

    def get_coeffs(self):
        return np.array(self.coeffs, dtype=float)

    def print_coeffs(self):
        print(self.coeffs)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        powers = np.array([x ** i for i in range(len(self.coeffs))])
        return np.tensordot(self.coeffs, powers, axes=1)