from sympy import symbols, solve


class pade:

	def __init__(self, fcoeffs=[0.622459, 0.235004, -0.0287784, -0.0160595, 0.00436483, 0.00113017, -0.000542105], pd=4, qd=4):

		ta = fcoeffs
		tl = len(ta)
		aa = []
		al = pd
		ba = ["1"]
		bl = qd

		for a in range(al):
			aa.append("a"+str(a))

		for b in range(1,bl):
			ba.append("b"+str(b))

		tb = []

		for b in range(bl):
			temp = []
			for t in range(tl):
				temp.append(str(ta[t])+"*"+ba[b])
			tb.append(temp)

		eq = []

		for i in range(tl+bl-1):
			if i == 0:
				ia = [(0,0)]
			else:
				ia = list(set(ix))

			ix = []
			str_eq = ""

			for s in ia:
				ri,ci = s[0],s[1]
				str_eq += "(" + tb[ri][ci] + ") + "

				if ri+1 < bl:
					ix.append((ri+1,ci))
				if ci+1 < tl:
					ix.append((ri,ci+1))

			eq.append(str_eq[:-2])

		feq = []
		for e in range(len(eq)):
			if e < al:
				feq.append(eq[e] + "- " + aa[e])
			else:
				feq.append(eq[e])

		syma = []
		for a in aa:
			syma.append(a)
		for b in range(1,bl):
			syma.append(ba[b])

		solution = solve(feq[:(al+bl-1)], syma)
		print("Solution:", solution)

		self.pa = []
		self.qa = [1]

		for a in aa:
			self.pa.append(solution[symbols(a)])
		for b in range(1,bl):
			self.qa.append(solution[symbols(ba[b])])


	def get_p_coeffs(self):
		return self.pa

	def get_q_coeffs(self):
		return self.qa


	def predict(self, x):
		nom = 0
		den = 0

		for i in range(len(self.pa)):
			nom += self.pa[i]*(x**i)
		for i in range(len(self.qa)):
			den += self.qa[i]*(x**i)

		if den == 0:
			den = 1

		return nom/den
