"""Truncated Fourier series approximation on a finite interval.

Computes coefficients ``a_0, {a_k}, {b_k}`` for k = 1..K such that

    f(x) ~ a_0 + sum_k [ a_k cos(k pi x / L) + b_k sin(k pi x / L) ]

where ``L = b - a``. Coefficients are obtained by ``scipy.integrate.quad``.
"""

import numpy as np
from scipy.integrate import quad


class fourier:

    def __init__(self, f, degree=4, frange=(-1.0, 1.5)):
        self.f = f
        self.degree = int(degree)
        self.a, self.b = float(frange[0]), float(frange[1])
        if self.b <= self.a:
            raise ValueError("require frange[1] > frange[0]")
        self.L = self.b - self.a

        val, _ = quad(f, self.a, self.b)
        self.a0 = val / (2.0 * self.L)

        self.ak = np.zeros(self.degree, dtype=float)
        self.bk = np.zeros(self.degree, dtype=float)
        for i in range(self.degree):
            n = i + 1
            ck = (np.pi * n) / self.L
            val, _ = quad(lambda x, ck=ck: f(x) * np.cos(ck * x), self.a, self.b)
            self.ak[i] = val / self.L
            val, _ = quad(lambda x, ck=ck: f(x) * np.sin(ck * x), self.a, self.b)
            self.bk[i] = val / self.L

    def __len__(self):
        return 1 + 2 * self.degree

    def __getitem__(self, idx):
        if idx == 0:
            return self.a0
        if idx <= self.degree:
            return self.ak[idx - 1]
        return self.bk[idx - 1 - self.degree]

    def __setitem__(self, idx, val):
        if idx == 0:
            self.a0 = val
        elif idx <= self.degree:
            self.ak[idx - 1] = val
        else:
            self.bk[idx - 1 - self.degree] = val

    def get_coeffs(self):
        """Return ``(a0, ak, bk)`` as numpy arrays."""
        return float(self.a0), np.array(self.ak), np.array(self.bk)

    def print_coeffs(self):
        print("a0:", self.a0)
        print("ak:", self.ak)
        print("bk:", self.bk)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        n = np.arange(1, self.degree + 1)
        ck = np.pi * n / self.L
        cos_part = np.cos(np.multiply.outer(x, ck))
        sin_part = np.sin(np.multiply.outer(x, ck))
        return self.a0 + cos_part @ self.ak + sin_part @ self.bk