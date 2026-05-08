"""mpFourier: truncated Fourier series on a finite interval in arbitrary precision.

Computes coefficients ``a_0, {a_k}, {b_k}`` for k = 1..K such that

    f(x) ~ a_0 + sum_k [ a_k cos(k pi x / L) + b_k sin(k pi x / L) ]

where ``L = b - a``. Coefficients are obtained by ``mpmath.quad`` quadrature
at the current ``mp.dps``.
"""

import mpmath as mp


class mpFourier:
    """Fourier series approximation in arbitrary precision.

    Parameters
    ----------
    func : callable
        Target function. Must accept and return mpmath types.
    domain : tuple of (mpf or float), optional
        Interval ``(a, b)``. Default ``(-1, 1.5)``.
    degree : int, optional
        Number of harmonics ``K`` (cosines and sines computed for k=1..K).
        Default 4.

    Attributes
    ----------
    a0 : mpf
        Constant (DC) term.
    ak : list of mpf
        Cosine coefficients for k = 1..K.
    bk : list of mpf
        Sine coefficients for k = 1..K.
    """

    def __init__(self, func=lambda x: x**2 - 4*x + 1, domain=(-1, 1.5), degree=4):
        self.func = func
        self.a = mp.mpf(domain[0])
        self.b = mp.mpf(domain[1])
        self.degree = int(degree)
        self.L = self.b - self.a
        if self.L <= 0:
            raise ValueError("require domain[1] > domain[0]")

        # a0 = (1/(2L)) * integral_a^b f(x) dx
        self.a0 = mp.quad(self.func, [self.a, self.b]) / (2 * self.L)

        self.ak = []
        self.bk = []
        pi = mp.pi
        for k in range(1, self.degree + 1):
            ck = (k * pi) / self.L
            ak_val = mp.quad(lambda x, ck=ck: self.func(x) * mp.cos(ck * x),
                             [self.a, self.b]) / self.L
            bk_val = mp.quad(lambda x, ck=ck: self.func(x) * mp.sin(ck * x),
                             [self.a, self.b]) / self.L
            self.ak.append(ak_val)
            self.bk.append(bk_val)

    def __len__(self):
        # total scalar coefficients: a0 plus K cosines plus K sines
        return 1 + 2 * self.degree

    def __getitem__(self, idx):
        # 0 -> a0, 1..K -> ak, K+1..2K -> bk
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

    def predict(self, x):
        """Evaluate the truncated Fourier series at x."""
        x = mp.mpf(x) if not isinstance(x, mp.mpc) else x
        result = self.a0
        pi = mp.pi
        for k in range(1, self.degree + 1):
            phase = (k * pi * x) / self.L
            result += self.ak[k - 1] * mp.cos(phase)
            result += self.bk[k - 1] * mp.sin(phase)
        return result