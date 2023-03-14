import unittest
from chiku import tf_ann
import numpy as np
import random

class Test_TF_ANN(unittest.TestCase):
	def test_tf_ann(self):
		def sigmoid(x):
			return 1/(1+np.exp(-x))

		self.poly = tf_ann.tf_ann()
		self.poly.fit(sigmoid, epochs=256)

		self.rnum = random.random()
		self.assertAlmostEqual(self.poly.predict(self.rnum), sigmoid(self.rnum), 2)

if __name__ == '__main__':
	unittest.main()
