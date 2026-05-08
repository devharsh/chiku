"""mpPade: Pade rational approximation in arbitrary precision via mpmath.

Constructs the [pd/qd] Pade approximant ``P(x) / Q(x)`` from a Taylor
coefficient list. Q is normalized so that ``Q(0) = 1``.

Coefficients are determined from the Pade equations

    sum_{i=0}^{pd+qd} a_i x^i  *  Q(x)  -  P(x)  =  O(x^(pd+qd+1))

which gives a linear system for the unknowns
``[p_0, ..., p_pd, q_1, ..., q_qd]``. The system is solved with
``mpmath.lu_solve`` at the current ``mp.dps`` precision.
"""

import mpmath as mp


class mpPade:
    """Pade approximant in arbitrary precision.

    Parameters
    ----------
    fcoeffs : sequence of mpf or float
        Taylor coefficients ``[a_0, a_1, ..., a_{pd+qd}]`` of the target
        function, in ascending power order.
    pd : int, optional
        Numerator degree. Default 3.
    qd : int, optional
        Denominator degree. Default 3.

    Attributes
    ----------
    p : list of mpf
        Numerator coefficients ``[p_0, p_1, ..., p_pd]``, ascending.
    q : list of mpf
        Denominator coefficients ``[1, q_1, ..., q_qd]``, ascending.
    """

    def __init__(
        self,
        fcoeffs=(0.622459, 0.235004, -0.0287784, -0.0160595,
                 0.00436483, 0.00113017, -0.000542105),
        pd=3,
        qd=3,
    ):
        self.pd = int(pd)
        self.qd = int(qd)

        if len(fcoeffs) < self.pd + self.qd + 1:
            raise ValueError(
                "need at least pd + qd + 1 = {} Taylor coefficients, got {}"
                .format(self.pd + self.qd + 1, len(fcoeffs))
            )

        a = [mp.mpf(c) for c in fcoeffs[: self.pd + self.qd + 1]]

        N = self.pd + self.qd + 1
        # Unknowns: [p_0, ..., p_pd, q_1, ..., q_qd]  (q_0 fixed at 1)
        # Equation k (k = 0..pd+qd):
        #   sum_{i=0}^{min(k, qd)} a_{k-i} q_i  =  p_k  (with p_k = 0 for k > pd)
        # i.e.  p_k - sum_{i=1}^{min(k,qd)} a_{k-i} q_i = a_k
        A = mp.matrix(N, N)
        b = mp.matrix(N, 1)
        for k in range(N):
            # numerator side: p_k coefficient is +1 if k <= pd
            if k <= self.pd:
                A[k, k] = mp.mpf(1)
            # denominator side: -a_{k-i} for q_i, i = 1..min(k, qd)
            for i in range(1, min(k, self.qd) + 1):
                A[k, self.pd + i] = -a[k - i]
            b[k] = a[k]

        sol = mp.lu_solve(A, b)
        self.p = [sol[i] for i in range(self.pd + 1)]
        self.q = [mp.mpf(1)] + [sol[self.pd + 1 + i] for i in range(self.qd)]

    def __len__(self):
        return len(self.p) + len(self.q) - 1  # total free coefficients

    def __getitem__(self, idx):
        # combined view: first the numerator, then the denominator (q_1..q_qd)
        if idx < len(self.p):
            return self.p[idx]
        return self.q[idx - len(self.p) + 1]

    def __setitem__(self, idx, val):
        if idx < len(self.p):
            self.p[idx] = val
        else:
            self.q[idx - len(self.p) + 1] = val

    def predict(self, x):
        """Evaluate ``P(x) / Q(x)`` at x."""
        x = mp.mpf(x) if not isinstance(x, mp.mpc) else x
        # Horner in ascending order
        nom = mp.mpf(0)
        x_pow = mp.mpf(1)
        for c in self.p:
            nom += c * x_pow
            x_pow *= x
        den = mp.mpf(0)
        x_pow = mp.mpf(1)
        for c in self.q:
            den += c * x_pow
            x_pow *= x
        if den == 0:
            den = mp.mpf(1)
        return nom / den