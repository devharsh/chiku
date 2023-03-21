import math


class chebyshev:

	def __init__(self, funcx=math.log, degree=5, frange=[-1,1]):
		self.a = frange[0]
		self.b = frange[1]

		bma = 0.5 * (self.b - self.a)
		bpa = 0.5 * (self.b + self.a)
		f = [funcx(math.cos(math.pi * (k + 0.5) / degree) * bma + bpa) for k in range(degree)]
		fac = 2.0 / degree
		self.c = [fac * sum([f[k] * math.cos(math.pi * j * (k + 0.5) / degree)
			for k in range(degree)]) for j in range(degree)]
		print(self.c)


	def __len__(self):
		return len(self.c)

	def __getitem__(self, idx):
		return self.c[idx]

	def __setitem__(self, idx, val):
		self.c[idx] = val


	def predict(self, x):
		y = (2.0 * x - self.a - self.b) * (1.0 / (self.b - self.a))
		y2 = 2.0 * y
		(d, dd) = (self.c[-1], 0)             # Special case first step for efficiency
		for cj in self.c[-2:0:-1]:            # Clenshaw's recurrence
			(d, dd) = (y2 * d - dd + cj, d)

		return y * d - dd + 0.5 * self.c[0]   # Last step is different
