from mpmath import *


class mpfourier:

    def __init__(self, f=lambda x: x**2 - 4*x + 1, r=[-1, 1.5], d=4):
        mp.dps = 15
        mp.pretty = True

        self.r = r
        self.cs = fourier(f, r, d)

        nprint(self.cs[0])
        nprint(self.cs[1])


    def predict(self, x):
        return fourierval(self.cs, self.r, x)
