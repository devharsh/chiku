"""Polynomial approximation via a small linear MLP.

Two-layer Keras model with linear activations on both hidden and output
layers. Because both layers are linear, the composed model is itself a
polynomial whose coefficients can be recovered analytically from the
weights. Training fits these coefficients to ``f``.

Inputs are the requested powers of ``x`` (``self.degree``); both inputs
and the target are standardized before training so the loss does not
overflow on wide intervals or large function ranges.
"""

import numpy as np
from tensorflow import keras


class tf_ann:

    def __init__(self, f, degree=(1, 2, 3, 4, 5), frange=(-1.0, 1.0),
                 points=2 ** 12, batch_size=64, epochs=100, lr=0.001,
                 verbose=0, seed=0):
        self.f = f
        self.degree = list(degree)
        self.a, self.b = float(frange[0]), float(frange[1])
        if self.b <= self.a:
            raise ValueError("require frange[1] > frange[0]")
        self.lr = float(lr)

        keras.utils.set_random_seed(int(seed))

        self.model = keras.Sequential([
            keras.layers.Input(shape=(len(self.degree),)),
            keras.layers.Dense(1, activation=keras.activations.linear),
            keras.layers.Dense(1, activation=keras.activations.linear),
        ])
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.lr),
            loss="mse",
        )

        # ---- build training data ----
        xs = np.linspace(self.a, self.b, int(points)).astype(np.float64)
        # Power basis features in float64 to avoid overflow in intermediates;
        # for x in [-100, 100] and d=7 the raw values reach 1e14, fine in
        # float64 but immediate inf in float32.
        X_raw = np.stack([xs ** d for d in self.degree], axis=1)
        y_raw = np.array([float(f(x)) for x in xs], dtype=np.float64)

        # Standardize features and target. Without this the loss overflows
        # to inf on the first forward pass and gradients become NaN.
        self.x_mean = X_raw.mean(axis=0)
        self.x_std = X_raw.std(axis=0)
        self.x_std = np.where(self.x_std < 1e-12, 1.0, self.x_std)
        self.y_mean = float(y_raw.mean())
        self.y_std = float(y_raw.std())
        if self.y_std < 1e-12:
            self.y_std = 1.0

        X_train = ((X_raw - self.x_mean) / self.x_std).astype(np.float32)
        y_train = ((y_raw - self.y_mean) / self.y_std).astype(np.float32)

        self.history = self.model.fit(
            X_train, y_train,
            batch_size=int(batch_size),
            epochs=int(epochs),
            verbose=int(verbose),
        )

        # ---- collapse two linear layers into one affine map ----
        # On standardized inputs z = (X_raw - mu) / sigma:
        #     y_std_hat = W . z + B,   where W = w0 * w1, B = b0*w1 + b1.
        w0, b0 = self.model.layers[0].get_weights()   # (n_feat, 1), (1,)
        w1, b1 = self.model.layers[1].get_weights()   # (1, 1),     (1,)
        W = (w0 * w1).flatten().astype(np.float64)
        B = float((b0 * w1 + b1).flatten()[0])

        # Undo standardization: y_hat = y_mean + y_std * (W . z + B).
        # Coefficient on x^{d_j} is y_std * W_j / sigma_j; constant absorbs
        # y_mean, y_std*B, and the cross terms in mu.
        feat_coeffs = self.y_std * W / self.x_std
        const_term = (
            self.y_mean
            + self.y_std * B
            - float(np.sum(feat_coeffs * self.x_mean))
        )

        max_deg = max(self.degree)
        coeffs = np.zeros(max_deg + 1, dtype=float)
        coeffs[0] = float(const_term)
        for i, d in enumerate(self.degree):
            coeffs[d] += float(feat_coeffs[i])
        self.coeffs = coeffs

    def __len__(self):
        return len(self.coeffs)

    def __getitem__(self, idx):
        return self.coeffs[idx]

    def __setitem__(self, idx, val):
        self.coeffs[idx] = val

    def get_coeffs(self):
        return np.array(self.coeffs, dtype=float)

    def print_coeffs(self):
        print(self.coeffs)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        powers = np.array([x ** i for i in range(len(self.coeffs))])
        return np.tensordot(self.coeffs, powers, axes=1)