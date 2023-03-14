import unittest
from chiku import taylor
import numpy as np
import random

class Test_Taylor(unittest.TestCase):
	def test_taylor(self):
		def sigmoid(x):
			return 1/(1+np.exp(-x))

		self.poly = taylor.taylor(sigmoid)
		self.rnum = random.random()
		self.assertAlmostEqual(self.poly.predict(self.rnum), sigmoid(self.rnum), 2)

if __name__ == '__main__':
	unittest.main()
