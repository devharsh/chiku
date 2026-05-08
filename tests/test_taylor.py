import unittest
import random

import numpy as np

from chiku import taylor


class Test_Taylor(unittest.TestCase):
    def test_taylor(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self.poly = taylor.taylor(sigmoid, degree=5, frange=(-1, 1))
        self.rnum = random.random()
        self.assertAlmostEqual(
            float(self.poly.predict(self.rnum)), sigmoid(self.rnum), 2
        )


if __name__ == "__main__":
    unittest.main()