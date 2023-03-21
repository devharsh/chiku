import math
import numdifftools as nd


class taylor:

	def __init__(self, funcx=math.log, degree=5, fpoint=0):
		self.degree = degree
		self.fpoint = fpoint
		self.coeffs = []

		for i in range(0, self.degree+1):
			ider = nd.Derivative(funcx, n=i)
			self.coeffs.append(ider(self.fpoint)/math.factorial(i))

	def __len__(self):
		return len(self.coeffs)

	def __getitem__(self, idx):
		return self.coeffs[idx]

	def __setitem__(self, idx, val):
		self.coeffs[idx] = val

	def print_equation(self):
		eqn_string = ""
		for i in range(self.degree + 1):
			if self.coeffs[i] != 0:
				eqn_string += str(self.coeffs[i]) + ("(x-{})^{}".format(self.fpoint, i) if i > 0 else "") + " + "
		eqn_string = eqn_string[:-3] if eqn_string.endswith(" + ") else eqn_string
		print(eqn_string)

	def get_coeffs(self):
		return self.coeffs

	def print_coeffs(self):
		print(self.coeffs)

	def predict(self, x):
		fx = 0
		for i in range(len(self.coeffs)):
			fx += self.coeffs[i] * ((x - self.fpoint)**i)  # coefficient * nth term
		return fx

	def approximate_derivative(self, x):
		value = 0
		for i in range(1, len(self.coeffs)): # skip the first value (constant) as the derivative is 0
			value += self.coeffs[i] * i * ((x - self.fpoint)**(i-1)) # differentiate each term: x^n => n*x^(n-1)
		return value

	def approximate_integral(self, x0, x1):
		value = 0
		for i in range(len(self.coeffs)):
			value += ((self.coeffs[i] * (1/(i+1)) * ((x1 - self.fpoint)**(i+1))) -
			(self.coeffs[i] * (1/(i+1)) * ((x0 - self.fpoint)**(i+1)))) # integrate each term: x^n => (1/n+1)*x^(n+1)
		return value
