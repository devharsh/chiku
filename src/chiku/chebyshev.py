"""Chebyshev series approximation in numpy.

Computes the truncated Chebyshev series of ``f`` on ``[a, b]`` of length
``degree`` using collocation at Chebyshev nodes of the second kind followed
by a discrete cosine projection.

Coefficient convention: ``f(x) ~ c0/2 + sum_{k>=1} c_k T_k(t(x))`` where
``t = (2x - a - b) / (b - a)``. The first coefficient is stored unhalved
(Numerical Recipes convention) and halved internally during Clenshaw
evaluation.
"""

import numpy as np


class chebyshev:

    def __init__(self, f, degree=5, frange=(-1.0, 1.0)):
        self.f = f
        self.n = int(degree)
        self.a, self.b = float(frange[0]), float(frange[1])
        if self.b <= self.a:
            raise ValueError("require frange[1] > frange[0]")
        if self.n < 1:
            raise ValueError("degree must be >= 1")

        bma = 0.5 * (self.b - self.a)
        bpa = 0.5 * (self.b + self.a)

        k = np.arange(self.n)
        nodes_t = np.cos(np.pi * (k + 0.5) / self.n)        # t in [-1, 1]
        nodes_x = nodes_t * bma + bpa                        # mapped to [a, b]
        fk = np.array([float(f(x)) for x in nodes_x], dtype=float)

        # discrete cosine projection: c_j = (2/n) sum_k f_k * cos(pi j (k+1/2)/n)
        j = k.reshape(-1, 1)
        kk = k.reshape(1, -1)
        M = np.cos(np.pi * j * (kk + 0.5) / self.n)
        self.c = (2.0 / self.n) * (M @ fk)

    def __len__(self):
        return len(self.c)

    def __getitem__(self, idx):
        return self.c[idx]

    def __setitem__(self, idx, val):
        self.c[idx] = val

    def get_coeffs(self):
        """Chebyshev coefficients (first coefficient unhalved)."""
        return np.array(self.c, dtype=float)

    def print_coeffs(self):
        print(self.c)

    def predict(self, x):
        """Evaluate the Chebyshev series at x using Clenshaw recurrence."""
        x = np.asarray(x, dtype=float)
        y = (2.0 * x - self.a - self.b) / (self.b - self.a)
        y2 = 2.0 * y
        d = np.full_like(y, self.c[-1])
        dd = np.zeros_like(y)
        for cj in self.c[-2:0:-1]:
            d, dd = y2 * d - dd + cj, d
        return y * d - dd + 0.5 * self.c[0]