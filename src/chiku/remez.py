"""Iterative Remez (minimax) polynomial approximation in numpy.

Constructs the polynomial of degree ``n`` that minimizes the uniform error
against ``f`` on ``[a, b]``. Coefficients are stored as a Chebyshev series
in the standard variable ``t in [-1, 1]``; ``predict`` accepts the
original variable ``x``.

Implementation notes
--------------------
* The (n + 2) x (n + 2) linear system per iteration is solved with
  ``scipy.linalg.solve``, with an ``lstsq`` fallback for ill-conditioned
  inputs.
* Extrema of the residual on a dense grid are detected by sign changes of
  the discrete derivative, with explicit handling of bins that contain a
  discontinuity (both sides of the bin are kept).
* Reference points are reduced to a strictly alternating-sign sequence; if
  more than ``n + 2`` candidates remain, a sliding window picks the one
  that maximizes the minimum |residual|.
* The solver tracks the best iterate by max-residual and runs several
  restarts from perturbed initial nodes if convergence stalls.
"""

import numpy as np
from scipy.linalg import solve as _solve, lstsq as _lstsq, LinAlgError


def _chebvander(t, n):
    """Vandermonde-like matrix V[i, k] = T_k(t[i]) via the recurrence."""
    t = np.asarray(t, dtype=float)
    V = np.empty((t.size, n + 1), dtype=float)
    V[:, 0] = 1.0
    if n >= 1:
        V[:, 1] = t
    for k in range(2, n + 1):
        V[:, k] = 2.0 * t * V[:, k - 1] - V[:, k - 2]
    return V


def _chebval(t, c):
    """Clenshaw evaluation of the Chebyshev series at t."""
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=float)
    if c.size == 0:
        return np.zeros_like(t)
    if c.size == 1:
        return np.full_like(t, c[0])
    b1 = np.zeros_like(t)
    b2 = np.zeros_like(t)
    for k in range(len(c) - 1, 0, -1):
        b1, b2 = c[k] + 2.0 * t * b1 - b2, b1
    return c[0] + t * b1 - b2


def _cheb_to_power(coeffs):
    """Convert Chebyshev-series coeffs (in t) to power-series coeffs (in t)."""
    n = len(coeffs)
    poly = np.zeros(n, dtype=float)
    if n == 0:
        return poly
    # Build T_k as power-series rows iteratively.
    # T_0 = [1], T_1 = [0, 1], T_{k+1} = 2*x*T_k - T_{k-1}
    T_prev = np.array([1.0])
    T_curr = np.array([0.0, 1.0])
    poly[0] += coeffs[0]
    if n > 1:
        poly[1] += coeffs[1]
    for k in range(2, n):
        T_next = np.zeros(k + 1, dtype=float)
        T_next[1:] += 2.0 * T_curr
        T_next[: len(T_prev)] -= T_prev
        T_prev, T_curr = T_curr, T_next
        poly[: len(T_curr)] += coeffs[k] * T_curr
    return poly


class RemezSolver:

    def __init__(self, f, degree, frange=(-1.0, 1.0), max_iter=50, tol=1e-10,
                 grid_size=None, n_restarts=4, verbose=False):
        self.f = f
        self.n = int(degree)
        self.a, self.b = float(frange[0]), float(frange[1])
        if self.b <= self.a:
            raise ValueError("require frange[1] > frange[0]")
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbose = bool(verbose)
        self.n_restarts = int(n_restarts)
        self.grid_size = int(grid_size) if grid_size else max(8000, 400 * (self.n + 2))

        self.coeffs = None
        self.E = None
        self.x_std = None
        self._converged = False
        self._solve_with_restarts()

    # ---- domain map ----
    def _to_standard(self, x):
        return (2.0 * x - (self.a + self.b)) / (self.b - self.a)

    def _from_standard(self, t):
        return 0.5 * (self.b - self.a) * t + 0.5 * (self.a + self.b)

    def _f_std(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        x = self._from_standard(t)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            y = np.asarray(self.f(x), dtype=float)
        if not np.all(np.isfinite(y)):
            finite = y[np.isfinite(y)]
            if finite.size == 0:
                raise ValueError("f returned no finite values on the domain")
            ymin, ymax = finite.min(), finite.max()
            y = np.where(
                np.isfinite(y),
                y,
                np.where(np.nan_to_num(y, nan=0.0) > 0, ymax, ymin),
            )
        return y

    def _initial_nodes(self, seed=0):
        k = np.arange(self.n + 2)
        nodes = np.cos(np.pi * k / (self.n + 1))
        if seed > 0:
            rng = np.random.default_rng(seed)
            jitter = rng.uniform(-0.05, 0.05, size=nodes.shape)
            jitter[0] = jitter[-1] = 0.0
            nodes = nodes + jitter * (1.0 - np.abs(nodes))
        return np.sort(nodes)

    def _solve_step(self, x_std):
        N = self.n + 2
        V = _chebvander(x_std, self.n)
        signs = ((-1.0) ** np.arange(N)).reshape(-1, 1)
        A = np.hstack([V, signs])
        b = self._f_std(x_std)
        try:
            sol = _solve(A, b)
        except LinAlgError:
            sol, *_ = _lstsq(A, b)
        return sol[:-1], float(sol[-1])

    def _find_extrema(self, coeffs):
        ng = self.grid_size
        t = np.linspace(-1.0, 1.0, ng + 1)
        r = self._f_std(t) - _chebval(t, coeffs)

        d = np.diff(r)
        abs_d = np.abs(d)
        interior = np.where(d[:-1] * d[1:] < 0)[0] + 1

        # Discontinuity detection: bin with disproportionately large change.
        median_d = float(np.median(abs_d)) + 1e-300
        jump_bins = np.where(abs_d > 100.0 * median_d)[0]

        candidates = []
        for i in interior:
            lo, hi = t[i - 1], t[i + 1]
            fine = np.linspace(lo, hi, 201)
            rf = self._f_std(fine) - _chebval(fine, coeffs)
            j = int(np.argmax(np.abs(rf)))
            candidates.append((fine[j], rf[j]))
        for i in jump_bins:
            candidates.append((t[i], r[i]))
            candidates.append((t[i + 1], r[i + 1]))

        candidates.sort(key=lambda p: p[0])
        pts = [t[0]] + [p for p, _ in candidates] + [t[-1]]
        vals = [r[0]] + [v for _, v in candidates] + [r[-1]]
        return np.array(pts), np.array(vals)

    def _select_alternating(self, pts, rvs, N):
        t_alt, v_alt = [pts[0]], [rvs[0]]
        for tt, vv in zip(pts[1:], rvs[1:]):
            same = (np.sign(vv) == np.sign(v_alt[-1])) and v_alt[-1] != 0
            if same:
                if abs(vv) > abs(v_alt[-1]):
                    t_alt[-1], v_alt[-1] = tt, vv
            else:
                t_alt.append(tt)
                v_alt.append(vv)
        if len(t_alt) <= N:
            return np.array(t_alt)
        best, score = None, -np.inf
        for s in range(len(t_alt) - N + 1):
            m = min(abs(v) for v in v_alt[s : s + N])
            if m > score:
                score, best = m, t_alt[s : s + N]
        return np.array(best)

    def _solve_once(self, x_std):
        N = self.n + 2
        best = {"coeffs": None, "E": None, "x_std": None, "abs_E": np.inf}
        prev_E = None
        for it in range(self.max_iter):
            coeffs, E = self._solve_step(x_std)
            r_here = self._f_std(x_std) - _chebval(x_std, coeffs)
            here_max = float(np.max(np.abs(r_here)))
            if here_max < best["abs_E"]:
                best.update(coeffs=coeffs, E=E, x_std=x_std.copy(), abs_E=here_max)

            pts, rvs = self._find_extrema(coeffs)
            new_x = self._select_alternating(pts, rvs, N)
            if len(new_x) < N:
                break

            rn = self._f_std(new_x) - _chebval(new_x, coeffs)
            r_abs = np.abs(rn)
            max_e, min_e = float(r_abs.max()), float(r_abs.min())
            ratio = (max_e - min_e) / max_e if max_e > 0 else 0.0
            if self.verbose:
                print(f"iter {it:3d}  E={E:+.3e}  max|r|={max_e:.3e}  ratio={ratio:.2e}")
            if max_e > 0 and ratio < self.tol:
                best.update(coeffs=coeffs, E=E, x_std=new_x.copy(), abs_E=max_e)
                self._converged = True
                return best
            if prev_E is not None and abs(E - prev_E) < self.tol * max(1.0, abs(E)):
                break
            prev_E = E
            x_std = new_x
        return best

    def _solve_with_restarts(self):
        best = {"coeffs": None, "abs_E": np.inf}
        for r in range(self.n_restarts):
            attempt = self._solve_once(self._initial_nodes(seed=r))
            if attempt["coeffs"] is not None and attempt["abs_E"] < best["abs_E"]:
                best = attempt
            if self._converged:
                break
        if best["coeffs"] is None:
            raise RuntimeError("Remez failed on all restarts")
        self.coeffs = np.asarray(best["coeffs"], dtype=float)
        self.E = best["E"]
        self.x_std = best["x_std"]

    # ---- public API ----
    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __setitem__(self, idx, val):
        self.coeffs[idx] = val

    def predict(self, x):
        t = self._to_standard(np.asarray(x, dtype=float))
        return _chebval(t, self.coeffs)

    def get_coeffs(self):
        """Power-series coefficients in the original variable x.

        Returns ``[c0, c1, ..., cn]`` such that
        ``p(x) = c0 + c1*x + ... + cn*x**n``.
        """
        # 1. Convert Chebyshev coefficients (in t) to power-series in t.
        power_t = _cheb_to_power(self.coeffs)
        # 2. Substitute t = alpha*x + beta where t = (2x - (a+b))/(b-a).
        alpha = 2.0 / (self.b - self.a)
        beta = -(self.a + self.b) / (self.b - self.a)
        n = len(power_t)
        power_x = np.zeros(n, dtype=float)
        for k in range(n):
            if power_t[k] == 0.0:
                continue
            # c_k * (alpha*x + beta)**k = c_k * sum_j C(k,j) alpha^j beta^(k-j) x^j
            binom = 1.0
            for j in range(k + 1):
                power_x[j] += power_t[k] * binom * (alpha ** j) * (beta ** (k - j))
                if j < k:
                    binom = binom * (k - j) / (j + 1)
        return power_x

    def print_coeffs(self):
        print(self.get_coeffs())

    def get_error(self):
        return float(abs(self.E)) if self.E is not None else float("nan")

    @property
    def converged(self):
        return self._converged