## Chapter 2: Feedforward Networks

### Linear Regression
$$f(x) = w^Tx + b$$

$$SSE[w,b](S) = \frac{1}{2}\sum_{i=1}^n (y_i - f(x_i))^2$$

$$\hat{w} = argmin_{w,b} SSE[w,b](S) = (X^TX)^{-1}X^Ty$$

### Regularized Linear Regression

Ridge Regression:
$$\hat{w} = argmin_{w,b} SSE[w,b](S) + \lambda ||w||_2^2$$

$$w = (X^TX + \lambda I)^{-1}X^Ty$$

Lasso Regression:
$$\hat{w} = argmin_{w,b} SSE[w,b](S) + \lambda ||w||_1$$

### Logistic Regression

$$\ell(y,z) = -y \log(\sigma(z)) - (1-y) \log(1-\sigma(z))$$

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

$$\hat{w} = argmin_{w,b} \sum_{i=1}^n \ell(y_i, f(x_i))$$

### Layer & Units
Units within a hidden layer are interchangeable. 

### Autencoder
MSE on reconstruction error.

$$\ell(y,z) = \frac{1}{2}\sum_{i=1}^n (y_i - z_i)^2$$

Linear autoencoders perform PCA.

### Residual Layers
$$F[W,b](x) = x + [\phi(Wx + b)]$$

DenseNet: Takes in all previous layers and outputs a new layer.

### Projections
You can also change the dimensionality of the residual layer with a projection matrix.

$$F[W,b](x) = Vx + [\phi(Wx + b)]$$

### Universal Approximation
MLP w/ single hidden layer can approximate any function.

Uniform Approximation,

$$||f||_{\infty} = \sup_{x \in [a,b]} |f(x)|$$

We say $f$ is uniformly approximated by $g_n$ if $||f-g_n||_{\infty} \rightarrow 0$ as $n \rightarrow \infty$.

### Weierstrass Theorem
Polynomials can uniformly approximate any continuous function on a closed interval.

### Dimension Lifting
Leshno's Theorem: Any function can be approximated by a function with a single hidden layer with $n$ units.

$$\text{span}(\{\phi(ax+b): a,b \in \mathbb{R}\})$$

universally approximates $C(\mathbb{R})$. The lifting theorem then allows us to lift the dimension of the function space to a higher dimension, i.e. $\text{span}(\{\phi(ax+b): a,b \in \mathbb{R}\})$ universally approximates $C(\mathbb{R}^n)$.

### Montufar Theorem
Consider a ReLU MLP with $L$ hidden layers of width $m > n$. The dimensionality is $n$. The number of linear regions is lower bounded by

$$R(m,L) \geq R(m) \frac{m}{n}^{n(L-1)}$$

where $R(m)$ is the number of connected regions of $\mathbb{R}^m$ spanned by $m$ linear functions.

$$R(m) = \sum_{i=0}^{\min(n,m)} \binom{m}{i}$$

### Shekhtman's Theorem
Piecewise linear functions are dense in $C([0,1])$.

