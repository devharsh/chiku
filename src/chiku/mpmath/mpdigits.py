from mpmath import *


class mpdigits:
    def __init__(self, x=0):
        self.x =  x

    def get_d(self, d=pi):
        return mpf(d)

    def sin(self):
        return sin

    def cos(self):
        return cos

    def tan(self):
        return tan

    def sinh(self):
        return sinh

    def cosh(self):
        return cosh

    def tanh(self):
        return tanh

    def exp(self):
        return exp

    def sqrt(self):
        return sqrt

    def quad(self):
        return quad
