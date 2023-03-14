import unittest
from chiku import chebyshev
import numpy as np
import random

class Test_Chebyshev(unittest.TestCase):
	def test_chebyshev(self):
		def sigmoid(x):
			return 1/(1+np.exp(-x))

		self.poly = chebyshev.chebyshev(sigmoid)
		self.rnum = random.random()
		self.assertAlmostEqual(self.poly.predict(self.rnum), sigmoid(self.rnum), 2)

if __name__ == '__main__':
	unittest.main()
