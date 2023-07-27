## chiku
### Efficient Probabilistic Polynomial Function Approximation Python Library.


Complex (non-linear) functions like Sigmoid ( $\sigma(x)$ ) and Hyperbolic Tangent ( $\tanh{x}$ ) can be computed with Fully Homomorphic Encryption (FHE) in an encrypted domain using piecewise-linear functions (a linear approximation of $\sigma(x) = 0.5 + 0.25x$ can be derived from the first two terms of Taylor series $\frac{1}{2} + \frac{1}{4}x$ ) or polynomial approximations like Taylor, Pade, Chebyshev, Remez, and Fourier series. These deterministic approaches yield the same polynomial for the same function. In contrast, we propose to use Artificial Neural Network ( $ANN$ ) to derive the approximation polynomial probabilistically, where the coefficients are based on the initial weights and convergence of the $ANN$ model. Our scheme is publicly available here as an open-source Python package.

Library | Taylor | Fourier | Pade | Chebyshev | Remez | ANN
--------|--------|---------|------|-----------|-------|-----
[numpy](https://github.com/numpy/numpy)||||:heavy_check_mark:||
[scipy](https://github.com/scipy/scipy)|:heavy_check_mark:||:heavy_check_mark:|||
[mpmath](https://github.com/mpmath/mpmath)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:||
[chiku](https://github.com/devharsh/chiku)|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:

The table above compares our library with other popular Python packages for numerical analysis. While the $mpmath$ library provides Taylor, Pade, Fourier, and Chebyshev approximations, a user has to transform the functions to suit the $mpmath$ datatypes (e.g., $mpf$ for real float and $mpc$ for complex values). In contrast, our library requires no modifications and can approximate arbitrary functions. Additionally, we provide Remez approximation along with the other methods supported by the $mpmath$.

While $ANN$ are known for their universal function approximation properties, they are often treated as a black box and used to calculate the output value. We propose to use a basic 3-layer perceptron consisting of an input layer, a hidden layer, and an output layer; both hidden and output layers have linear activations to generate the coefficients for an approximation polynomial of a given order. In this architecture, the input layer is dynamic, with the input nodes corresponding to the desired polynomial degrees. While having a variable number of hidden layers is possible, we fix it to a single layer with a single node to minimize the computation.

![Polynomial approximation using ANN](https://github.com/devharsh/chiku/blob/main/ANN-approximation.drawio.png "Polynomial approximation using ANN")

We show coefficient calculations for a third-order polynomial $d=3$ for a univariate function $f(x) = y$ for an input $x$, actual output $y$, and predicted output $y_{out}$.

Input layer weights are

$\{w_1, w_2, \ldots, w_d\} = \{w_1, w_2, w_3\} = \{x, x^2, x^3\}$

and biases are $\{b_1, b_2, b_3\} = b_h$. Thus the output of the hidden layer is

$y_h = w_1 x + w_2 x^2 + w_3 x^3 + b_h$

The predicted output is calculated by

$y_{out} = w_{out} \cdot y_h + b_{out}$
$= w_1 w_{out} x + w_2 w_{out} x^2 + w_3 w_{out} x^3 + (b_h w_{out} + b_{out})$

where the layer weights $\{w_1 w_{out}, w_2 w_{out}, w_3 w_{out}\}$ are the coefficients for the approximating polynomial of order-3

and the constant term is $b_h w_{out} + b_{out}$.

Our polynomial approximation approach using $ANN$ can generate polynomials with specified degrees. E.g., a user can generate a complete third-order polynomial for $\sin(x)$, which yields a polynomial
$-0.0931199x^3 - 0.001205849x^2 + 0.85615075x + 0.0009873845$
in the interval $[-\pi,\pi]$. Meanwhile, a user may want to optimize the above polynomial by eliminating the coefficients for $x^2$ to reduce costly multiplications in FHE, which yields the following:
$-0.09340597x^3 + 0.8596622x + 0.0005142888.$
