"""Polynomial regression in the monomial basis using sklearn.

Fits ``f`` on a uniform grid over ``frange`` with sklearn's
``LinearRegression``. Features are the requested powers of ``x``,
standardized to keep the design matrix well-conditioned for high degrees
on wide intervals; the resulting coefficients are unscaled back to the
original power basis so ``predict(x) = sum_i coeffs[i] * x**i``.
"""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class sk_lr:

    def __init__(self, f, degree=(1, 2, 3, 4, 5), frange=(-1.0, 1.0),
                 points=2 ** 11):
        self.f = f
        self.degree = list(degree)
        self.a, self.b = float(frange[0]), float(frange[1])
        if self.b <= self.a:
            raise ValueError("require frange[1] > frange[0]")
        self.points = int(points)

        # Standardization is essential. For x in [-100, 100] and d=7 the
        # raw design matrix has condition number ~1e24, well past float64
        # precision. Centering and scaling each feature column drops the
        # condition number to O(1) and lets LinearRegression recover the
        # true minimax-like coefficients.
        self.model = make_pipeline(StandardScaler(), LinearRegression())

        xs = np.linspace(self.a, self.b, self.points)
        X_raw = self._features(xs)
        y_raw = np.array([float(f(x)) for x in xs], dtype=np.float64)

        self.model.fit(X_raw, y_raw)

        # Unscale: y_hat = sum_j W_j * (x^{d_j} - mu_j)/sigma_j + B
        scaler = self.model.named_steps["standardscaler"]
        lr = self.model.named_steps["linearregression"]
        mu, sigma = scaler.mean_, scaler.scale_
        W, B = lr.coef_, lr.intercept_

        feat_coeffs = W / sigma
        const_term = float(B - np.sum(feat_coeffs * mu))

        max_deg = max(self.degree)
        coeffs = np.zeros(max_deg + 1, dtype=float)
        coeffs[0] = const_term
        for i, d in enumerate(self.degree):
            coeffs[d] += float(feat_coeffs[i])  # += handles d=0 in self.degree
        self.coeffs = coeffs

    def _features(self, xs):
        xs = np.asarray(xs, dtype=np.float64)
        return np.stack([xs ** d for d in self.degree], axis=1)

    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __setitem__(self, idx, val):
        self.coeffs[idx] = val

    def get_coeffs(self):
        """Coefficients ``[c0, c1, ..., c_max_deg]`` for ``predict``."""
        return np.array(self.coeffs, dtype=float)

    def print_coeffs(self):
        print(self.coeffs)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        powers = np.array([x ** i for i in range(len(self.coeffs))])
        return np.tensordot(self.coeffs, powers, axes=1)