from mpmath import *


class mpchebyshev:

    def __init__(self, f=cos, r=[1,2], d=5):
        mp.dps = 15
        mp.pretty = True
        self.poly, self.err = chebyfit(f, r, d, error=True)
        nprint(self.poly)
        nprint(self.err, 12)

    def __len__(self):
        return len(self.poly)

    def __getitem__(self, idx):
        return self.poly[idx]

    def __setitem__(self, idx, val):
        self.poly[idx] = val

    def predict(self, x):
        return nprint(polyval(self.poly, x), 12)
