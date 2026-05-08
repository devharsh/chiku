"""Pade rational approximation in numpy.

Constructs the [pd / qd] Pade approximant of ``f`` around the midpoint of
``frange``. Internally:

1. Computes ``pd + qd + 1`` Taylor coefficients of ``f`` by numerical
   differentiation at the midpoint.
2. Solves the linear Pade equations directly with ``numpy.linalg.solve``
   (with an ``lstsq`` fallback for singular systems).
3. Stores ``p`` (numerator) and ``q`` (denominator), with ``q[0] = 1``.

The resulting rational function is evaluated as ``P(u) / Q(u)`` where
``u = x - fpoint``.
"""

import math

import numpy as np
import numdifftools as nd


class pade:

    def __init__(self, f, pd=3, qd=3, frange=(-1.0, 1.0)):
        self.f = f
        self.pd = int(pd)
        self.qd = int(qd)
        self.a, self.b = float(frange[0]), float(frange[1])
        if self.b <= self.a:
            raise ValueError("require frange[1] > frange[0]")
        self.fpoint = 0.5 * (self.a + self.b)

        # 1. Taylor coefficients a_0 .. a_{pd+qd} around fpoint
        N = self.pd + self.qd + 1
        a = np.empty(N, dtype=float)
        for k in range(N):
            ider = nd.Derivative(f, n=k)
            a[k] = float(ider(self.fpoint)) / math.factorial(k)

        # 2. Solve  P(u) = Q(u) * sum_k a_k u^k  mod u^N, with Q(0) = 1.
        #    Unknowns are [p_0..p_pd, q_1..q_qd].
        #    Equation k (k = 0..N-1):
        #        sum_{i=0..min(k,qd)} a_{k-i} q_i  =  p_k         (with q_0 = 1)
        #    rearranged:
        #        p_k - sum_{i=1..min(k,qd)} a_{k-i} q_i  =  a_k   for k <= pd
        #            - sum_{i=1..min(k,qd)} a_{k-i} q_i  =  a_k   for k >  pd
        A = np.zeros((N, N), dtype=float)
        b_vec = np.zeros(N, dtype=float)
        for k in range(N):
            if k <= self.pd:
                A[k, k] = 1.0
            for i in range(1, min(k, self.qd) + 1):
                A[k, self.pd + i] = -a[k - i]
            b_vec[k] = a[k]

        try:
            sol = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            sol, *_ = np.linalg.lstsq(A, b_vec, rcond=None)

        self.p = np.array(sol[: self.pd + 1], dtype=float)
        self.q = np.concatenate(([1.0], sol[self.pd + 1 :])).astype(float)

    def __len__(self):
        # combined view: numerator then denominator (q_1..q_qd; q_0 fixed)
        return len(self.p) + len(self.q) - 1

    def __getitem__(self, idx):
        if idx < len(self.p):
            return self.p[idx]
        return self.q[idx - len(self.p) + 1]

    def __setitem__(self, idx, val):
        if idx < len(self.p):
            self.p[idx] = val
        else:
            self.q[idx - len(self.p) + 1] = val

    def get_coeffs(self):
        """Return ``(p, q)`` as numpy arrays, ascending power order."""
        return np.array(self.p, dtype=float), np.array(self.q, dtype=float)

    def print_coeffs(self):
        print("p:", self.p)
        print("q:", self.q)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        u = x - self.fpoint
        # numpy.polyval expects descending coefficients
        nom = np.polyval(self.p[::-1], u)
        den = np.polyval(self.q[::-1], u)
        den = np.where(den == 0, 1.0, den)
        return nom / den