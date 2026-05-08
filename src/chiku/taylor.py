"""Taylor series approximation in numpy.

Computes the truncated Taylor series of ``f`` around the midpoint of
``frange``. Numerical differentiation is performed with numdifftools.
"""

import math

import numpy as np
import numdifftools as nd


class taylor:

    def __init__(self, f, degree=5, frange=(-1.0, 1.0)):
        self.f = f
        self.degree = int(degree)
        self.a, self.b = float(frange[0]), float(frange[1])
        if self.b <= self.a:
            raise ValueError("require frange[1] > frange[0]")
        self.fpoint = 0.5 * (self.a + self.b)

        coeffs = []
        for i in range(self.degree + 1):
            ider = nd.Derivative(f, n=i)
            coeffs.append(float(ider(self.fpoint)) / math.factorial(i))
        self.coeffs = np.array(coeffs, dtype=float)

    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __setitem__(self, idx, val):
        self.coeffs[idx] = val

    def get_coeffs(self):
        """Coefficients ``[c0, c1, ..., c_degree]`` of ``(x - fpoint)**k``."""
        return np.array(self.coeffs, dtype=float)

    def print_coeffs(self):
        print(self.coeffs)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        u = x - self.fpoint
        powers = np.array([u ** i for i in range(len(self.coeffs))])
        return np.tensordot(self.coeffs, powers, axes=1)