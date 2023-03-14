import numpy as np
from tensorflow import keras


class tf_ann:

	def __init__(self, degree=5, optimizer='adam', loss=keras.losses.MeanSquaredError()):
		self.degree = degree
		self.model = keras.Sequential([
			keras.layers.Dense(1, input_dim=self.degree, activation=keras.activations.linear),
			keras.layers.Dense(1, activation=keras.activations.linear)
			])
		self.model.compile(optimizer=optimizer, loss=loss)
		self.model.summary()


	def fit(self, funcx=np.log, frange=[-1,1], points=2**11, batch_size=32, epochs=512):
		self.ary = np.linspace(frange[0], frange[1], points)
		x_train = []
		y_train = []

		for i in self.ary:
			xal = []
			for d in range(self.degree):
				xal.append(i**(d+1))
			x_train.append(xal)
			y_train.append(funcx(i))

		history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

		w0, b0 = self.model.layers[0].get_weights()
		w1, b1 = self.model.layers[1].get_weights()
		wts = w0 * w1
		bias = (b0*w1) + b1

		self.coeffs = []
		self.coeffs.append(bias[0][0])
		for w in wts:
			self.coeffs.append(w[0])

		return self.coeffs


	def predict(self, x):
		result = 0
		for i in range(len(self.coeffs)):
			result += self.coeffs[i] * (x**i)
		return result
