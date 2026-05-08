"""mpRemez: minimax (Remez exchange) polynomial approximation in arbitrary
precision via mpmath.

Iteratively constructs the polynomial of degree ``n`` that minimizes the
uniform error against ``func`` on ``[a, b]``. Internally works with a
Chebyshev-series representation in the standard variable ``t in [-1, 1]``;
``predict`` accepts the original variable ``x``.

Coefficient layout (returned by ``get_coeffs``): ascending power-series
coefficients in the original variable ``x``.
"""

import mpmath as mp


class mpRemez:
    """Iterative Remez minimax approximation in arbitrary precision.

    Parameters
    ----------
    func : callable
        Target function. Must accept and return mpmath types.
    degree : int
        Degree of the approximating polynomial.
    domain : tuple of (mpf or float), optional
        Interval ``(a, b)``. Default ``(-1, 1)``.
    max_iter : int, optional
        Maximum Remez exchange iterations. Default 50.

    Attributes
    ----------
    coeffs : list of mpf
        Chebyshev-series coefficients in the standard variable ``t``.
    E : mpf
        Signed equioscillation error from the final linear solve.
    """

    def __init__(self, func, degree, domain=(-1, 1), max_iter=50):
        self.func = func
        self.n = int(degree)
        self.a, self.b = mp.mpf(domain[0]), mp.mpf(domain[1])
        if self.b <= self.a:
            raise ValueError("require domain[1] > domain[0]")
        self.max_iter = int(max_iter)

        self.map_to_std = lambda x: (2 * x - (self.a + self.b)) / (self.b - self.a)
        self.map_from_std = lambda t: mp.mpf(0.5) * (self.b - self.a) * t + mp.mpf(0.5) * (self.a + self.b)
        self.f_std = lambda t: self.func(self.map_from_std(t))

        # initial reference: Chebyshev nodes of T_{n+1} (n+2 points in [-1, 1])
        self.x = sorted(
            mp.cos(mp.pi * k / mp.mpf(self.n + 1))
            for k in range(self.n + 2)
        )
        self.coeffs = None
        self.E = None
        self._solve()

    # ---------- helpers ----------
    def _chebT(self, t, n):
        if n == 0:
            return mp.mpf(1)
        if n == 1:
            return t
        T0, T1 = mp.mpf(1), t
        for _ in range(2, n + 1):
            T0, T1 = T1, 2 * t * T1 - T0
        return T1

    def _cheb_series(self, t, coeffs):
        return sum(c * self._chebT(t, i) for i, c in enumerate(coeffs))

    def _find_all_extrema(self, r, n_grid=8000):
        grid = [mp.mpf(-1) + 2 * i / mp.mpf(n_grid) for i in range(n_grid + 1)]
        vals = [r(t) for t in grid]
        pts, rvs = [grid[0]], [vals[0]]
        for i in range(1, n_grid):
            dl = vals[i] - vals[i - 1]
            dr = vals[i + 1] - vals[i]
            if dl * dr < 0:
                lo, hi = grid[i - 1], grid[i + 1]
                fine = [lo + (hi - lo) * j / mp.mpf(200) for j in range(201)]
                fine_v = [r(t) for t in fine]
                best = max(range(201), key=lambda j: abs(fine_v[j]))
                pts.append(fine[best])
                rvs.append(fine_v[best])
        pts.append(grid[-1])
        rvs.append(vals[-1])
        return pts, rvs

    def _select_alternating(self, pts, rvs, n_needed):
        t_alt, v_alt = [pts[0]], [rvs[0]]
        for t, v in zip(pts[1:], rvs[1:]):
            if mp.sign(v) == mp.sign(v_alt[-1]):
                if abs(v) > abs(v_alt[-1]):
                    t_alt[-1], v_alt[-1] = t, v
            else:
                t_alt.append(t)
                v_alt.append(v)
        if len(t_alt) <= n_needed:
            return t_alt
        best_ref, best_score = None, mp.mpf(-1)
        for s in range(len(t_alt) - n_needed + 1):
            score = min(abs(v) for v in v_alt[s : s + n_needed])
            if score > best_score:
                best_score = score
                best_ref = t_alt[s : s + n_needed]
        return best_ref

    def _cheb_to_power(self, coeffs):
        n = len(coeffs)
        poly = [mp.mpf(0)] * n
        for k, c in enumerate(coeffs):
            if k == 0:
                poly[0] += c
            elif k == 1:
                poly[1] += c
            else:
                T0 = [mp.mpf(1)]
                T1 = [mp.mpf(0), mp.mpf(1)]
                for _ in range(2, k + 1):
                    Tn = [mp.mpf(0)] * (len(T1) + 1)
                    for i in range(len(T1)):
                        Tn[i + 1] += 2 * T1[i]
                    for i in range(len(T0)):
                        Tn[i] -= T0[i]
                    T0, T1 = T1, Tn
                for i in range(len(T1)):
                    poly[i] += c * T1[i]
        return poly

    # ---------- main loop ----------
    def _solve(self):
        N = self.n + 2
        coeffs = E = None
        for _ in range(self.max_iter):
            A = mp.matrix(N, N)
            b_vec = mp.matrix(N, 1)
            for i, t in enumerate(self.x):
                for j in range(self.n + 1):
                    A[i, j] = self._chebT(t, j)
                A[i, self.n + 1] = (-1) ** i
                b_vec[i] = self.f_std(t)
            sol = mp.lu_solve(A, b_vec)
            coeffs = [sol[i] for i in range(self.n + 1)]
            E = sol[self.n + 1]

            def r(t, coeffs=coeffs):
                return self.f_std(t) - self._cheb_series(t, coeffs)

            pts, rvs = self._find_all_extrema(r)
            new_x = self._select_alternating(pts, rvs, N)
            if len(new_x) < N:
                break

            r_abs = [abs(r(t)) for t in new_x]
            max_e, min_e = max(r_abs), min(r_abs)
            if max_e > 0 and (max_e - min_e) / max_e < mp.mpf("1e-10"):
                self.x = new_x
                break
            self.x = new_x

        self.coeffs = coeffs
        self.E = E

    # ---------- container API ----------
    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __setitem__(self, idx, val):
        self.coeffs[idx] = val

    # ---------- public API ----------
    def predict(self, x):
        """Evaluate the minimax polynomial at x."""
        x = mp.mpf(x) if not isinstance(x, mp.mpc) else x
        t = self.map_to_std(x)
        return self._cheb_series(t, self.coeffs)

    def get_coeffs(self):
        """Power-series coefficients in the original variable x.

        Returns ``[c0, c1, ..., cn]`` such that
        ``p(x) = c0 + c1*x + ... + cn*x**n``.
        """
        power_t = self._cheb_to_power(self.coeffs)  # p(t) = sum c_k t^k

        # t = alpha*x + beta
        alpha = mp.mpf(2) / (self.b - self.a)
        beta = -(self.a + self.b) / (self.b - self.a)

        n = len(power_t)
        power_x = [mp.mpf(0)] * n
        for k in range(n):
            if power_t[k] == 0:
                continue
            # c_k * (alpha*x + beta)**k = c_k * sum_j C(k,j) alpha^j beta^(k-j) x^j
            binom = mp.mpf(1)
            for j in range(k + 1):
                power_x[j] += power_t[k] * binom * (alpha ** j) * (beta ** (k - j))
                if j < k:
                    binom = binom * (k - j) / (j + 1)
        return [float(c) for c in power_x]

    def get_error(self):
        """Equioscillation error magnitude as a float."""
        return float(abs(self.E))