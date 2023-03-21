import unittest
from chiku import taylor,pade
import numpy as np
import random

class Test_Pade(unittest.TestCase):
	def test_pade(self):
		def sigmoid(x):
			return 1/(1+np.exp(-x))

		self.t_poly = taylor.taylor(sigmoid)
		self.poly = pade.pade(self.t_poly.get_coeffs(), 3, 3)

		self.rnum = random.random()
		self.assertAlmostEqual(self.poly.predict(self.rnum), sigmoid(self.rnum), 2)

if __name__ == '__main__':
	unittest.main()
