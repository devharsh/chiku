import unittest
from chiku import chebyshev, remez
import numpy as np
import random

class Test_Remez(unittest.TestCase):
	def test_remez(self):
		def sigmoid(x):
			return 1/(1+np.exp(-x))

		self.cpoly = chebyshev.chebyshev(sigmoid, degree=6, frange=[-2,2])
		self.poly = remez.remez(sigmoid, self.cpoly.get_coeffs(), degree=4)

		self.rnum = random.random()
		self.assertAlmostEqual(self.poly.predict(self.rnum), sigmoid(self.rnum), 2)

if __name__ == '__main__':
	unittest.main()
