from mpmath import *


class mptaylor:

    def __init__(self, f=sin, p=0, d=5):
        mp.dps = 15
        mp.pretty = True

        self.poly = taylor(f, p, d)
        nprint(chop(self.poly))


    def __len__(self):
        return len(self.poly)

    def __getitem__(self, idx):
        return self.poly[idx]

    def __setitem__(self, idx, val):
        self.poly[idx] = val


    def predict(self, x):
        return polyval(self.poly[::-1], x - p)
