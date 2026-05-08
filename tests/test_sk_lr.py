import unittest
import random

import numpy as np

from chiku import sk_lr


class Test_SK_LR(unittest.TestCase):
    def test_sk_lr(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self.poly = sk_lr.sk_lr(
            sigmoid, degree=[1, 2, 3, 4, 5], frange=(-1, 1)
        )
        self.rnum = random.random()
        self.assertAlmostEqual(
            float(self.poly.predict(self.rnum)), sigmoid(self.rnum), 2
        )


if __name__ == "__main__":
    unittest.main()
