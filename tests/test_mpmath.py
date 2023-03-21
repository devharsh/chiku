import unittest
from chiku.mpmath import mpdigits,mptaylor,mppade


class Test_MPMath(unittest.TestCase):
	def test_mpmath(self):
		d = mpdigits.mpdigits()
		one = d.get_d(1)

		def f(x):
			sqrt = d.sqrt()
			return sqrt((one + 2*x)/(one + x))

		self.t = mptaylor.mptaylor(f, 0, 6)

		self.p = mppade.mppade(self.t, 3, 3)
		self.x = 10

		self.assertAlmostEqual(self.p.predict(self.x), f(self.x), 2)

if __name__ == '__main__':
	unittest.main()
