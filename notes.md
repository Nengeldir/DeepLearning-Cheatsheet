## Gradient-Based Learning
### Backpropagation
#### Per Layer Gradients
##### Parameters are attached to layers, affecting the local mapping of activations $x_{l-1} \mapsto x_l$.
##### Partial derivatives reflect this locality; for example, $\frac{\partial x_l}{\partial w_l}$ depends on the activation function derivative and the input from the previous layer.
#### Chain Rule Application
##### To compute how downstream loss changes with respect to parameters, the chain rule combines the local gradient with the downstream signal.
##### This introduces the "delta" term ($\delta$), defined as the sensitivity of the loss to the unit activation: $\delta_{li} \equiv \frac{\partial h}{\partial x_{li}}$.
#### The Backpropagation Algorithm
##### A dynamic programming algorithm that functions in two passes: a forward pass to compute activations and a backward pass to propagate error.
##### $\delta$ terms are computed recursively from output to input using layer Jacobi matrices: $\delta_l = [\frac{\partial F_{l+1}}{\partial x_l}]' \delta_{l+1}$.
##### Deep learning frameworks implement this by defining vector-Jacobian products for each operator to avoid materializing full Jacobian matrices.
#### Computational Graphs
##### Models are represented as Directed Acyclic Graphs (DAGs) where nodes are tensors and edges are functions.
##### Automatic differentiation traces this graph; if a node has multiple outgoing edges, gradients from all branches are accumulated.

### Gradient Descent
#### Steepest Descent
##### The standard method to evolve parameters to minimize training risk $h[\theta]$.
##### The update rule is $\theta_{t+1} = \theta_t - \eta \nabla h(\theta_t)$, where $\eta$ is the step size (learning rate).
#### Gradient Flow
##### Gradient descent can be viewed as the numerical integration (discretization) of the continuous-time Ordinary Differential Equation (ODE) $\dot{x} = -\nabla f(x)$.
##### The trajectory depends heavily on initialization, especially in non-convex landscapes.
#### Convergence and Smoothness
##### If the objective function is $L$-smooth (Lipschitz continuous gradient), a step size of $\eta = 1/L$ guarantees strict progress in every step.
##### Under these conditions, gradient descent finds $\epsilon$-critical points in $O(\epsilon^{-2})$ steps.
#### Saddle Points
##### Non-convex objectives in Deep Learning contain saddle points which can slow down convergence.
##### Strategies to escape saddle points include adding noise to the gradient updates.

### Acceleration and Adaptivity
#### Momentum (Heavy Ball Method)
##### Addresses the issue of vanishing gradients or slow progress in flat directions.
##### Adds an extrapolation term based on the previous update: $\theta_{t+1} = \theta_t - \eta \nabla h(\theta_t) + \beta(\theta_t - \theta_{t-1})$.
##### If the gradient is constant, momentum boosts the effective step size by a factor of $1/(1-\beta)$.
#### Nesterov Acceleration
##### A theoretical improvement over standard momentum where the gradient is evaluated at the extrapolated (look-ahead) position rather than the current position.
#### AdaGrad
##### An adaptive method that adjusts the learning rate per parameter based on the history of gradient magnitudes.
##### It divides the step size by the square root of the sum of squared past gradients, effectively giving larger updates to parameters with historically small derivatives.
#### Adam
##### A popular optimizer that combines ideas from momentum and adaptivity.
##### It maintains exponential moving averages of both the gradients and their squares (variance).

### Stochastic Gradient Descent (SGD)
#### Mini-batches
##### Calculating full batch gradients is computationally prohibitive for large datasets.
##### SGD approximates the gradient using a small random subset (mini-batch) of size $r \ll s$.
#### Update Rule
##### The parameters are updated using the gradient estimated from the mini-batch: $\theta_{t+1} = \theta_t - \eta \nabla h_{batch}(\theta_t)$.
#### Bias and Variance
##### The stochastic update direction is unbiased, meaning its expectation equals the full batch gradient.
##### Convergence is criticaly dependent on controlling the variance of these stochastic gradients, specifically around the minima.


## Chapter 3: Gradient Based Learning
### Lipschitz Continuity

$$||f(x) - f(y)|| \leq L||x-y||$$

The largest step size is $\frac{1}{L}$ it arrives at an $\epsilon$-optimum in $t = \frac{2L}{\epsilon^2}$ steps.

### Polyak Lojasiewicz Condition

$$\frac{1}{2}||\nabla f(x)||^2 \leq \mu (f(x) - f^*)$$

where $\mu > 0$ is the Polyak-Lojasiewicz constant.

If $f$ is Lipschitz continuous and satisfies the Polyak-Lojasiewicz condition, then the optimal step size is $\eta = \frac{1}{L}$ and converges to an $\epsilon$-optimum with geometric rate: 

$$f(x_t) - f^* \leq (1 - \frac{\mu}{L})^t(f(x_0) - f^*)$$

### Momentum
$$\theta^{t+1} = \theta^t - \eta \nabla f(\theta^t) + \beta (\theta^t - \theta^{t-1})$$

with $\beta \in [0,1)$.

### Nesterov Accelerated Gradient
$$\vartheta^{t+1} = \theta^t + \beta(\theta^t - \theta^{t-1})$$

$$\theta^{t+1} = \vartheta^{t+1} - \eta \nabla f(\vartheta^{t+1})$$

### Adaptive Learning Rate
Adaptive learning rate methods adjust the learning rate based on the history of the gradient. 

$$\theta^{t+1}_i = \theta^t_i - \eta_i^t \nabla f(\theta^t_i)$$

$$\eta_i^t = \frac{\eta}{\sqrt{\sum_{i=1}^t \nabla f(\theta^t_i)^2}}$$

### Adam and RMSprop
$$g^t_i = \beta g_i^{t-1} + (1-\beta) \nabla f(\theta^t_i)$$

$$g_i^0 = \nabla f(\theta^0_i)$$

$$\gamma_i^t = \alpha\gamma_i^{t-1} + (1-\alpha) \nabla f(\theta^t_i)^2$$

$$\gamma_i^0 = \nabla f(\theta^0_i)^2$$

$$\theta^{t+1}_i = \theta^t_i - \eta_i^t g_i^t$$

### Stochastic Gradient Descent
We use a subset of the data to compute the gradient. This intreoduces noise into the gradient, which can help escape local minima.
The convergence of SGD with functions that are Lipschitz continuous and satisfy the Polyak-Lojasiewicz condition is 

$$\mathbb{E}[f(x_t) - f^*] \leq \mathcal{O}(\frac{1}{\sqrt{t}})$$
