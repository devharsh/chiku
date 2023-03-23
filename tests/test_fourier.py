import unittest
from chiku import fourier


class Test_Fourier(unittest.TestCase):
	def test_fourier(self):
		def f(x):
			return x*x

		self.poly = fourier.fourier(f, -1, 5, 70)
		self.assertEqual(round(self.poly.predict(2)), f(2))

if __name__ == '__main__':
	unittest.main()
