import unittest
import random

import numpy as np

from chiku import chebyshev


class Test_Chebyshev(unittest.TestCase):
    def test_chebyshev(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self.poly = chebyshev.chebyshev(sigmoid, degree=5, frange=(-1, 1))
        self.rnum = random.random()
        self.assertAlmostEqual(
            float(self.poly.predict(self.rnum)), sigmoid(self.rnum), 2
        )


if __name__ == "__main__":
    unittest.main()