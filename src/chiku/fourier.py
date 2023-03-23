import math
from scipy.integrate import quad


class fourier:

    def __init__(self, f, a, b, k):
        self.L = b - a

        def integrand(x):
            return f(x)
        val, err = quad(integrand, a, b)
        self.a0 = (1/(2*self.L))*val

        self.ak = []
        for i in range(k):
            def integrand(x):
                return f(x)*math.cos(((i+1)*math.pi*x)/self.L)
            val, err = quad(integrand, a, b)
            self.ak.append((1/self.L)*val)

        self.bk = []
        for i in range(k):
            def integrand(x):
                return f(x)*math.sin(((i+1)*math.pi*x)/self.L)
            val, err = quad(integrand, a, b)
            self.bk.append((1/self.L)*val)


    def predict(self, x):
        res = self.a0

        for k in range(len(self.ak)):
            res += self.ak[k] * math.cos((math.pi*(k+1)*x)/self.L)

        for k in range(len(self.bk)):
            res += self.bk[k] * math.sin((math.pi*(k+1)*x)/self.L)

        return res
