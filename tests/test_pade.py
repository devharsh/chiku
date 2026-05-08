import unittest
import random

import numpy as np

from chiku import pade


class Test_Pade(unittest.TestCase):
    def test_pade(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self.poly = pade.pade(sigmoid, pd=3, qd=3, frange=(-1, 1))
        self.rnum = random.random()
        self.assertAlmostEqual(
            float(self.poly.predict(self.rnum)), sigmoid(self.rnum), 2
        )


if __name__ == "__main__":
    unittest.main()