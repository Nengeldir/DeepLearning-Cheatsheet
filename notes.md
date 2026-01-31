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

## Feedforward Networks
### Regression Models
#### Linear Regression
##### The simplest model predicts a response $y$ based on input $x$ assuming a linear relationship $f[w](x) = w'x$.
##### It typically minimizes the Mean Squared Error (MSE): $h[w] = \frac{1}{2s} \sum (w'x_i - y_i)^2$.
#### Logistic Regression
##### Utilizes the logistic function $\sigma(z) = \frac{1}{1 + e^{-z}}$ to model probabilities.
##### It minimizes the cross-entropy loss (negative log-likelihood), which acts as a surrogate loss for the 0/1 classification error.
##### The logistic function has the property $\sigma(-z) = 1 - \sigma(z)$ and its derivative is $\sigma(z)(1-\sigma(z))$.

### Layers & Units
#### Feedforward Architecture
##### The network processes inputs through a sequence of hidden layer transformations (forward propagation) to extract features of increasing complexity,.
##### The map is a composition of layer functions: $G = F_L \circ \dots \circ F_1$.
#### Layers
##### A layer is a parameterized map $F[\theta](x) = \phi(Wx + b)$ combining an affine transformation with a pointwise non-linearity.
##### This design allows "all units to be created equal" by sharing a single scalar activation function $\phi$.
#### Units (Neurons)
##### Units are the "malleable atoms" of the network, defined as ridge functions $f[w, b](x) = \phi(w'x + b)$,.
##### They possess an invariance where the function value is constant along directions orthogonal to the weight vector $w$.
#### Symmetries
##### Networks exhibit permutation symmetry; the map is invariant if units within a hidden layer are permuted (along with their corresponding weights).
##### Consequently, parameterizations in feedforward networks are never unique.
#### Nesting
##### A layer of width $m$ can be embedded into a layer of width $m+1$ by adding "barren" (unused) units or by cloning existing units.
#### Outputs and Losses
##### Softmax is used for multi-class classification to map pre-activations to a probability simplex.
##### Models are assessed on Risk (expected loss), distinguishing between training risk (empirical) and test/population risk (expected loss on future data).

### Linear & Residual Networks
#### Linear Networks
##### Composed of linear maps ($\phi = id$), they are analytically simple but do not gain representational power from depth because affine maps are closed under composition,.
##### Their primary use is dimensionality reduction (contraction).
#### Autoencoders
##### Linear autoencoders with a bottleneck layer ($m < n$) essentially perform Principal Component Analysis (PCA) when minimizing squared reconstruction error,.
##### They project data to the subspace spanned by the principal components.
#### Residual Networks (ResNets)
##### Residual layers learn an incremental improvement: $F(x) = x + [\phi(Wx+b) - \phi(0)]$.
##### They utilize "skip connections" to propagate inputs forward, allowing the layer to be initialized near the identity map.
##### This architecture enabled the training of very deep networks (e.g., 100+ layers) by improving gradient flow.
#### Projections and DenseNets
##### Residual blocks can use linear projections ($Vx$) instead of identity shortcuts to change dimensionality.
##### DenseNets feed the output of *all* upstream layers into the current layer, accumulating input dimensionality.

### Sigmoid Networks
#### Activation Functions
##### Common choices include the logistic function and the hyperbolic tangent ($\tanh$).
##### $\tanh$ is representationaly equivalent to the logistic function, related by scaling and shifting: $\tanh(z) = 2\sigma(2z) - 1$.
#### Universal Approximation
##### A single hidden layer MLP with sufficiently large width ($m \to \infty$) can uniformly approximate any continuous function on a compact domain,.
##### This relies on theorems stating that non-polynomial smooth activation functions are universal approximators.
#### Barron’s Theorem
##### This theorem relates approximation error to the number of units ($m$), showing an error rate of $O(1/m)$,.
##### This rate is independent of input dimension $n$ (avoiding the curse of dimensionality), provided the function's Fourier transform satisfies specific regularity conditions.

### ReLU Networks
#### ReLU Activation
##### The Rectified Linear Unit is defined as $\phi(z) = \max\{0, z\}$.
##### It is piecewise linear; to address the lack of gradient in the negative regime, variants like Leaky ReLU are used.
#### Complexity and Zoning
##### A ReLU layer partitions the input space into convex polytopes (cells) via hyperplanes defined by weights.
##### Within each cell, the network mapping is affine.
##### Zaslavsky’s Theorem bounds the number of linear regions created by $m$ hyperplanes; while polynomial in width, the number of regions can grow exponentially with depth.
#### Universality
##### ReLU networks with a single hidden layer are universal function approximators.
##### This is proven using the fact that piecewise linear functions can be constructed using differences of convex functions (specifically, two max operations).
