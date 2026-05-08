import unittest
import random

import numpy as np

from chiku import tf_ann


class Test_TF_ANN(unittest.TestCase):
    def test_tf_ann(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        self.poly = tf_ann.tf_ann(
            sigmoid,
            degree=[1, 2, 3, 4, 5],
            frange=(-1, 1),
            epochs=256,
            verbose=0,
            seed=0,
        )
        self.rnum = random.random()
        self.assertAlmostEqual(
            float(self.poly.predict(self.rnum)), sigmoid(self.rnum), 2
        )


if __name__ == "__main__":
    unittest.main()