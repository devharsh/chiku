import unittest
import random

import mpmath as mp

from chiku.mpmath import (
    mpdigits,
    mptaylor,
    mppade,
    mpchebyshev,
    mpfourier,
    mpremez,
)


def sigmoid_mp(x):
    return 1 / (1 + mp.exp(-x))


class Test_MPMath(unittest.TestCase):
    def test_mptaylor(self):
        self.poly = mptaylor.mpTaylor(sigmoid_mp, fpoint=0, degree=6)
        self.rnum = random.random()
        self.assertAlmostEqual(
            float(self.poly.predict(self.rnum)),
            float(sigmoid_mp(mp.mpf(self.rnum))),
            2,
        )

    def test_mppade(self):
        self.t = mptaylor.mpTaylor(sigmoid_mp, fpoint=0, degree=6)
        self.poly = mppade.mpPade(list(self.t.coeffs), pd=3, qd=3)
        self.rnum = random.random()
        self.assertAlmostEqual(
            float(self.poly.predict(self.rnum)),
            float(sigmoid_mp(mp.mpf(self.rnum))),
            2,
        )

    def test_mpchebyshev(self):
        self.poly = mpchebyshev.mpChebyshev(
            sigmoid_mp, domain=(-1, 1), degree=6
        )
        self.rnum = random.random()
        self.assertAlmostEqual(
            float(self.poly.predict(self.rnum)),
            float(sigmoid_mp(mp.mpf(self.rnum))),
            2,
        )

    def test_mpfourier(self):
        def f(x):
            return x * x

        self.poly = mpfourier.mpFourier(f, domain=(-1, 5), degree=70)
        self.assertEqual(round(float(self.poly.predict(2))), f(2))

    def test_mpremez(self):
        self.poly = mpremez.mpRemez(sigmoid_mp, degree=4, domain=(-1, 1))
        self.rnum = random.random()
        self.assertAlmostEqual(
            float(self.poly.predict(self.rnum)),
            float(sigmoid_mp(mp.mpf(self.rnum))),
            2,
        )

    def test_mpdigits(self):
        d = mpdigits.mpdigits()
        one = d.get_d(1)
        sqrt = d.sqrt()

        def f(x):
            return sqrt((one + 2 * x) / (one + x))

        self.t = mptaylor.mpTaylor(f, fpoint=0, degree=6)
        self.p = mppade.mpPade(list(self.t.coeffs), pd=3, qd=3)
        self.x = 10
        self.assertAlmostEqual(
            float(self.p.predict(self.x)), float(f(mp.mpf(self.x))), 2
        )


if __name__ == "__main__":
    unittest.main()