## Chapter 5: Recurrent Neural Networks
### Simple Recurrent Networks
The problem with CNNs is that we need to specify the filter widths in advance. This constrains the range and type of patterns the network can learn. RNNs are more successful with temporal or sequence data. 

Assume we have an observation sequence $x_1, x_2, \dots, x_T$. A RNN introduces a complementary sequence of activations 

$$z_1, z_2, \dots, z_T$$

where $z_t$ is the hidden state at time $t$. The hidden state at time $t$ is given by:

$$z_t = f(z_{t-1}, x_t)$$

where $f$ is a learnable non-linear function. The initial hidden state is given by:

$$z_0 = f(z_{-1}, x_0)$$

where $z_{-1}$ is the initial hidden state. 

Optionally an RNN may produce outputs via a map $G$, inducing an output sequence $\hat{y}_1, \hat{y}_2, \dots, \hat{y}_T$ where

$$\hat{y}_t = G(z_t) = \psi(Wz_t$$

For an autoregressive RNN this would $\hat{y}_t \approx x_{t+1}$ for $t=1, \dots, T-1$. The network is trained to minimize the error between the predicted and the actual output.

#### Parametrization

One can define F to be a linear function with a non-linear activation function. 

$$F[U,V](z,x) = \phi(Uz + Vx)$$

where $U$ and $V$ are learnable weight matrices and $\phi$ is a non-linear activation function. For a finite sequence of length $T$ we can unroll the RNN to reveal the underlying structure of the network. This corresponds to a deep feedforward network with shared weights. 

#### Backpropagation Through Time

Since the RNN is a feedforward network, we can use backpropagation to train it. However, we need to be careful about the shared weights.

The starting points are the error signals at the outputs

$$\delta_k^t = \frac{\partial h}{\partial \hat{y}_k^t}$$

$$\frac{\partial h}{\partial w_{ki}} = \sum_{t=1}^T \frac{\partial h}{\partial \hat{y}_k^t} \frac{\partial \psi(Wz_t)}{\partial w_{ki}}z_i^t$$

The sum emerges from the fact that the weights are shared across time steps.

For the matrix V, we have

$$\frac{\partial h}{\partial v_{ij}} = \sum_{s=t}^T\delta_k^s\sum^m_{j=1}\frac{\partial \hat{y_k^s}}{\partial z_j^s}\frac{\partial z_j^s}{\partial z_i^t}$$

where $\frac{\partial \hat{y_k^s}}{\partial z_j^s} = \frac{\partial \psi(Wz_s)}{\partial z_j^s}w_{kj}$. 

#### Gradient Norms
The problem with simple RNNs is that the gradient norm can explode or vanish during backpropagation through time. This is because the gradient is multiplied by the weight matrix $U$ at each time step. This can be shown by looking at the largest singular value of the weight matrix $U$. 

#### Bidirectional RNNs
A bidirectional RNN is a RNN that processes the input sequence in both forward and backward directions. The hidden state does not necessarily have to follow the temporal order of the input sequence. One can alternatively define a reversed hidden state as:

$$\tilde{z}^t = \phi(\tilde{U}z^{t+1} + \tilde{V}x^t)$$

and the output map as:

$$\hat{y}^t = \psi(Wz^t + \tilde{W}\tilde{z}^t)$$

#### Deep RNNs
Another modification is to add depth to the hidden state representation to increase the representational power of the network. This can be done by stacking RNNs on top of each other as the picture depicts below.

![Stacked RNN](image.png)

### Gated Memory
To make training of RNNs more stable, we can use gates to control the flow of information. This controls when memory should be kept or forgotten. 

#### Gating
A gating unit takes the following non-linear form of a pointwise multiplication:

$$z^+ = \sigma(G\Xi) \odot z$$

where $\sigma$ is the sigmoid function and $\odot$ is the pointwise multiplication $G$ is the gate matrix and $\Xi$ is the input matrix. 


#### Long Short-Term Memory (LSTM)
A prominent example of a gated RNN is the Long Short-Term Memory (LSTM) network.

It uses two gating mechanisms for its memory: a forget gate and a storage gate:

$$C^t = \underbrace{\underbrace{\sigma(F\tilde{x}^t)}_{\text{Forget Gate}}
\odot \underbrace{C^{t-1}}_{\text{Previous Memory}}}_{\text{Forgetting}} + \underbrace{\underbrace{\sigma(I\tilde{x}^t)}_{\text{Input Gate}} \odot \underbrace{\tanh(\tilde{C}\tilde{x}^t)}_{\text{Candidate Memory}}}_{\text{Storing}}$$

In the equation above, we used the same notation as in the picture below. We define $\tilde{x}^t = [x_t, H_{t-1}]^T$ where $H_{t-1}$ is the hidden state of the previous time step.

![LSTM](image-1.png)

#### Gated Recurrent Unit (GRU)
Another example of a gated RNN is the Gated Recurrent Unit (GRU). It simplifies the architecture of LSTMs by merging the forget and storage gates into a single convex combination:

$$H^t = \underbrace{(1- \sigma(G[x_t, H_{t-1}]))}_{\text{Update Gate}} \odot \underbrace{H^{t-1}}_{\text{Previous Hidden State}} + \underbrace{\sigma(G[x_t, H_{t-1}])}_{\text{Update Gate}} \odot \underbrace{\tilde{H}^t}_{\text{Candidate Hidden State}}$$

where we define $\tilde{H}^t = \tanh(V[\xi_t \odot H_{t-1}, x_t])$.

![alt text](image-3.png)

#### Were RNNs all we needed?

Despite the success of RNNs, they were eventually replaced by Transformers for many NLP tasks. They are sequential in nature, which means they cannot be parallelized. This makes them slow to train on large datasets. 

### Linear Recurrent Models
New models like S4 and Mamba try to combine the benefits of RNNs and Transformers. 

However, we will focus on Linear Recurrent Units (LRUs).

The hidden state of an LRU is given by:

$$z^{t+1} = Az^t + Bx^t$$

where $A$ and $B$ are learnable matrices. 

We can diagonalize the evolution matrix $A$ over the complex numbers $\mathbb{C}$ as $A = P\Lambda P^{-1}$ where $\Lambda$ is a diagonal matrix with the eigenvalues of $A$ on the diagonal. 

We can then perform perform a change of basis

$$\xi^{t+1} = \Lambda \xi^t + P^{-1}Bx^t$$

where $\xi^t = P^{-1}z^t$ is the hidden state in the diagonal basis.

The stability of the recurrence relation depends on the eigenvalues of $A$. To keep it stable we need to initialize the eigenvalues to be inside the unit circle. 

### Sequence Learning
we can use RNNs as generative models for sequences. 

#### RNN models for sequences
Todo

#### Teacher Forcing
Todo

#### Sequence-to-Sequence Models
Todo

## Recurrent Neural Networks
### Simple Recurrent Networks
#### Concept and Definition
##### RNNs offer modeling flexibility for sequential data, unlike CNNs which require specifying filter widths in advance.
##### The model introduces a sequence of hidden activations (states) $z_t$ to model the unobserved evolving state of a system based on an observation sequence $x_t$.
##### The discrete time evolution is defined as $z_t \equiv F[\theta](z_{t-1}, x_t)$, where $F$ is a parameterized, time-invariant evolution operator.
#### Output and Unrolling
##### RNNs can produce outputs via a map $y_t = G[\vartheta](z_t)$.
##### The network can be "unrolled" in time, making it equivalent to a feedforward network with $T$ hidden layers, sharing parameters across all layers.
##### The hidden states act as a noisy memory or data summary, compressing the history.
#### Backpropagation Through Time (BPTT)
##### Training utilizes backpropagation by propagating error signals backwards through time.
##### Because weights are shared across time steps, the gradients for weights are sums of contributions from each time step.
##### The gradient computation involves the Jacobian of the state evolution.
#### Vanishing and Exploding Gradients
##### Simple RNNs struggle with gradients due to the repeated multiplication of the Jacobian matrix.
##### If the spectral norm of the weight matrix is $< 1$, gradients vanish exponentially (making long-range dependencies hard to learn); if $> 1$, gradients explode.
#### Structural Variants
##### **Bidirectional RNNs:** Process the sequence in both forward and reverse directions to capture context from both the past and the future.
##### **Deep RNNs:** Stack layers horizontally (hierarchical hidden states) or use deep MLPs for the state transition function to increase feature extraction power.

### Gated Memory
#### Motivation
##### To model long-range dependencies, RNNs must reliably memorize features without the gradient instability of simple RNNs.
##### Gating units control when memory is preserved or overwritten using multiplicative gating.
#### Long Short-Term Memory (LSTM)
##### The classical gated RNN, using two main gates: a forget gate and a storage gate.
##### It maintains a cell state updated via $z_t \equiv \sigma(F\tilde{x}_t) * z_{t-1} + \sigma(G\tilde{x}_t) * \tanh(V\tilde{x}_t)$.
##### It involves complex internal architecture with control signals and multiple weight matrices.
#### Gated Recurrent Unit (GRU)
##### A simplification of the LSTM that combines forgetting and storage into a single convex combination: $z_t = (1-\sigma)*z_{t-1} + \sigma*\tilde{z}_t$.
##### It reduces the number of weight matrices compared to LSTM while retaining effectiveness.
#### Minimal Gated Units
##### Recent work suggests removing dependencies on previous hidden states from the gating mechanism to allow for fast parallel processing (parallel prefix scan) during training.

### Linear Recurrent Models
#### Revival of RNNs
##### While Transformers replaced RNNs in many fields, new linear recurrent models (like S4, Mamba, LRU) aim to solve the lack of parallelization in classical RNNs.
#### Linear Recurrent Unit (LRU)
##### The hidden state evolution is a discrete-time linear system: $z_{t+1} = Az_t + Bx_t$.
##### The evolution matrix $A$ is diagonalized over complex numbers ($A = P\Lambda P^{-1}$), allowing for a change of basis.
#### Stability and Parameterization
##### Stability requires the modulus of eigenvalues to be bounded ($\le 1$).
##### Parameters are initialized using exponentials of logarithms to easily control the range and stability.
#### Universality
##### Despite the linearity of the recurrence, LRUs combined with a non-linear MLP output map are provably universal sequence-to-sequence approximators.

### Sequence Learning
#### Generative Modeling
##### RNNs can model the probability of an output sequence $y$ given input $x$ by factoring it into conditional probabilities: $p(y_{1:T}|x_{1:T}) \approx \prod p(y_t | x_{1:t}, y_{1:t-1})$.
##### To capture dependencies between outputs, the generated output $y_{t-1}$ is often fed back into the RNN state for the next step.
#### Teacher Forcing
##### During training, instead of feeding back the model's own (potentially erroneous) prediction, the ground truth target $y^*$ is fed back.
##### This simplifies learning but creates a discrepancy between training and testing (exposure bias).
#### Sequence-to-Sequence (Seq2Seq)
##### An Encoder-Decoder architecture where an encoder RNN maps the input sequence to a fixed latent vector $z$, and a decoder RNN generates the output sequence from $z$.
##### Used extensively for tasks like machine translation.
#### Attention Mechanisms
##### To improve upon the fixed-vector bottleneck of Seq2Seq, attention mechanisms compute mixing weights to selectively focus on different encoder states $z_{\tau}$ at each decoding step.
##### The decoder learns to "align" with relevant parts of the input via a feedforward network trained jointly with the RNNs.
