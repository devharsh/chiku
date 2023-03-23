from scipy import linalg


class remez:

    def __init__(self, f, cheb, degree):

        if len(cheb)-2 != degree or len(cheb)%2 != 0:
            exit("order mismatch")

        matP = []
        matY = []
        ed = 1

        for x in cheb:
            matY.append(f(x))

            arrP = [1]
            for d in range(degree):
                arrP.append(x**(d+1))
            arrP.append((-1)**(1+ed))

            matP.append(arrP)
            ed += 1

        self.coeffs = linalg.solve(matP, matY)


    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __setitem__(self, idx, val):
        self.coeffs[idx] = val


    def predict(self, x):
        res = self.coeffs[-1]

        for i in range(len(self.coeffs)-1):
            res += self.coeffs[i] * (x**i)

        return res
