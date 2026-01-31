## Connectionism
### Overview
#### Connectionism is a movement in cognitive science aiming to explain intellectual abilities using artificial neural networks.
#### Deep Learning (DL) focuses on Deep Neural Networks (DNN), comprising high-dimensional models composed of simple building blocks that are trainable with simple methods like stochastic gradient descent.
#### Early inspiration came from biological systems, specifically the structure of interconnected neurons (e.g., work by Ramón y Cajal).

### Intermezzo (~1950)
#### The Quest for Learning
##### Early Boolean logic models of neural networks were crude and failed to explain the ability to adapt and learn.
##### Alan Turing envisioned "unorganized machines" that could learn from experience, moving beyond fixed logic.

### Perceptron (1958, 1969)
#### Pattern Recognition
##### The perceptron was the first major breakthrough, introducing a universal machine capable of learning to recognize patterns from examples (supervised learning).
##### Patterns are represented as feature vectors $x \in \mathbb{R}^n$ with binary class membership $y \in \{-1, +1\}$.
#### Threshold Unit
##### It utilized a linear threshold unit defined as $f[w,b](x) = \text{sign}(x \cdot w + b)$ where $x \cdot w = \sum^n_{i=1} x_i w_i$.
##### The sign function returns $+1$ if the input is positive, $0$ if zero, and $-1$ otherwise.
#### Perceptron Learning
##### Parameters are adapted only when an example is misclassified using the update: $w_{new} = w_{old} + y_i x_i$ (and similarly for the bias $b$).
##### The algorithm is guaranteed to converge to a solution if the data is linearly separable (Novikoff's Theorem).
#### Capacity and Shattering
##### Cover’s function counting theorem describes the capacity of the perceptron; a perceptron in $n$ dimensions can arbitrarily classify $2n$ random patterns.
##### Asymptotic Shattering: The probability of finding a linear separator drops sharply when the number of samples $s$ exceeds $2n$.
#### Deep Perceptrons
##### Rosenblatt realized single units were insufficient and envisioned multi-layer networks with feature extractors.
##### Lacking a learning rule for hidden layers, early deep models used fixed (random) weights for the association units, training only the final response unit,.

### Parallel Distributed Processing (1986-1987)
#### The PDP Framework
##### Rumelhart, Hinton, and McClelland consolidated connectionist models into a general framework, offering a "LEGO toolbox" for designing models.
##### Key concepts included processing units, propagation rules, connectivity patterns, and a general learning rule based on experience,.
#### Activation Functions
##### To enable learning, threshold functions were replaced with differentiable non-linearities, such as generalized affine maps.
#### Delta Rule
##### Derived by differentiating the squared loss, the delta rule updates weights based on the error, input, and the sensitivity (derivative) of the activation function.
#### Generalized Delta Rule (Backpropagation)
##### The framework generalized the delta rule to deep architectures by using error backpropagation to automatically derive training signals ($\delta$ terms) for hidden units.
##### This allows for local parameter updates per unit based on backpropagated deltas.
#### Multilayer Perceptron (MLP)
##### A classic architecture featuring an input layer, hidden layers with sigmoid activations ($\phi(z) = \frac{1}{1+e^{-z}}$), and an output layer.
##### While MLPs are universal function approximators (as width $\to \infty$), this theoretical capability does not guarantee they are efficient to train or the best architecture for all tasks.

### Hopfield Networks (1982)
#### Associative Memory
##### These networks model how systems can retrieve memories (patterns) in an associative manner, functioning as collective computation.
#### Hebbian Learning
##### Weights are determined by the correlation of patterns: $w_{ij} = \frac{1}{n} \sum x_i x_j$.
##### This follows the principle that "neurons that fire together wire together," creating positive coupling for similar states and negative coupling for opposite states.
#### Energy Landscape
##### The network dynamics minimize an energy function $E(x) = -\frac{1}{2} x^\top W x$.
##### Provided the weight matrix is symmetric with a zero diagonal, the dynamics will converge to a stable state (local minimum) without oscillating.

## Chapter 1: Connectionism
### McCulloch & Pitts (1943)
One of the first approaches to model functions of nervous systesm with an abstract mathematical model.

Neurons are abstracted as linear threshold units, which receive and integrate a large number of inputs and produce a boolean output.

**MP-Neuron** Inputs $x \in \{0, 1\}^n$, synapses $\sigma \in \{-1, 1\}^n$, threshold $\theta \in \mathbb{R}$, 

$$f[\sigma, \theta](x) = \begin{cases} 1 & \text{if } \sum_{i=1}^n \sigma_i x_i \geq \theta \\ 0 & \text{otherwise} \end{cases}$$

The MP-Neuron lacked the ability to learn, as the weights were fixed and the threshold was a constant. It was also not able to model non-linear functions.

**Alan Turing** envisioned "unorganized machines" that could learn from experience, moving beyond fixed logic.

**Claude Shannon** created the Theseus machine, it was able to learn to navigate a maze. 

**Marvin Minsky** built SNARC (Stochastic Neural Analog Reiforcement Calcualtor) in 1952. 

**Rosenblatt** built the Perceptron in 1958. It was able to learn to recognize patterns from examples. It's function is:

$$f[w,b] = sign(xw + b)$$

where $xw = \sum^n_{i=1} x_i w_i$ and $sign(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ -1 & \text{if } z < 0 \\ 0 & \text{if } z = 0 \end{cases}$

It's learning is done if a sample is misclassified, updating the weights as $w_{new} = w_{old} + y_i x_i$. For the bias $b$, we update as $b_{new} = b_{old} + y_i$.

Novikoff's Theorem guarantees convergence of the Perceptron algorithm if the data is linearly separable. 

Cover's Theorem; Let $S \subseteq \mathbb{R}^n$ be a set of $s$ points in general position. The maximum number of dichotomies of $S$ that can be realized by a linear classifier is $C(s + 1,n) = 2\sum^{n}_{i=0} \binom{s}{i}$. S are the data points, n is the dimension of the data points. $C$ can be interpreted as the number of shatterings of $S$ by linear classifiers.
