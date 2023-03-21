from mpmath import *


class mppade:

    def __init__(self, fa=[0.622459, 0.235004, -0.0287784, -0.0160595, 0.00436483, 0.00113017, -0.000542105], pd=3, qd=3):
        mp.dps = 15
        mp.pretty = True

        self.p, self.q = pade(fa, pd, qd)


    def predict(self, x):
        return polyval(self.p[::-1], x)/polyval(self.q[::-1], x)
