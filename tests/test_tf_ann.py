import unittest
from chiku import tf_ann
import numpy as np

class Test_TF_ANN(unittest.TestCase):
	def sigmoid(x):
		return 1/(1+np.exp(-x))
	
	ary = np.linspace(-1, 1, 11)
	print(ary)
	
	poly = tf_ann.tf_ann()
	poly.fit(sigmoid)
	
	tx = []
	ty = []
	
	for i in ary:
		tx.append(poly.predict(i))
		ty.append(sigmoid(i))
	
	def test_tf_ann(self):
		self.assertAlmostEqual(tx, ty, 8)
		
if __name__ == '__main__':
	unittest.main()
	