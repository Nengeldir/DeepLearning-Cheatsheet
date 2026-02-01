
## Chapter 4: Convolutional Networks
### Convolutions
#### Integral Operators
defined by a kernel and an interval

$$G: \mathbb{R}^2 \rightarrow \mathbb{R}$$

$$(Tf)(u) = \int^{t_2}_{t_1}G(u,t)f(t)dt$$

An example of an integral operator is the fourier transform. It is defined as:

$$\hat{f}(\xi) = \int_{-\infty}^{\infty} f(x)e^{-2\pi i x \xi}dx$$

#### Convolutions
Given two functions $f,g$ their convolution is defined as:

$$(f*g)(u) = \int_{-\infty}^{\infty} g(u-t)f(t)dt$$

This corresponds to an integral operator with kernel $G(u,t) = g(u-t)$.

Convolution is shift-equivariant, i.e.
$$f_{\Delta}(t) = f(t+\Delta)$$
$$f_\Delta*g = (f*g)_\Delta$$

Convolution is also commutative and associative.

#### Fourier Transform
Convolution operators can be computed efficiently using the Fourier transform.

$$\mathcal{F}[f*g] = \mathcal{F}[f]\mathcal{F}[g]$$

The result can be mapped back using the inverse Fourier transform.

$$\mathcal{F}^{-1}[\mathcal{F}[f*g]] = f*g$$

#### Linear Shift Equivariant Transforms
Any linear, translation invariant transform can be represented as a convolution.

#### Discrete Convolution
In practice we deal with discrete data, so we use discrete convolution.

$$(f*g)[u] = \sum_\infty^\infty f[t]g[u-t]$$

#### Cross Correlation
In the discrete case cross correlation is defined as:

$$(g \star f)[u] = \sum_\infty^\infty g[t]f[u+t]$$

This is also called a sliding inner product. The difference to normal convolution is that the kernel is not flipped. This however makes correlations not commutative.

#### Toeplitz Matrices
When f and g have finite support, the convolution can be represented as a matrix multiplication with a Toeplitz matrix.

$$(f \star g) = \begin{pmatrix}
    g_1 & 0 & 0 & \dots & 0 & 0 \\
    g_2 & g_1 & 0 & \dots  & 0 & 0 \\
    g_3 & g_2 & g_1 & \dots & 0 & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
    0 & 0 & 0 & \dots & g_m & g_{m-1} \\
    0 & 0 & 0 & \dots & 0 & g_m
\end{pmatrix} \begin{pmatrix}
    f_1 \\
    f_2 \\
    f_3 \\
    \vdots \\
    f_n
\end{pmatrix}$$

#### Convolutional Networks
Goal of Convolutional Networks is to exploit the translational equivariance of the data, locality, and scale. This is more efficient than a fully connected network. We learn a a set of kernel functions or filters from the data.

Discrete convolutions in higher dimensions (e.g. 2D) is defined as:

$$(F\star G)[i,j] = \sum_{k=-\infty}^{\infty}\sum_{l=-\infty}^{\infty} F[i-k,j-l]G[k,l]$$

#### Border Handling

We can use zero padding of the signal. This is also called same padding. Alternatively, we can use valid padding, which means that we only compute the convolution where the kernel is fully contained in the signal.

#### Receptive Field

The nesting (or stacking) of convolutional layers increases the receptive field of the network.

#### Weight Sharing

We can exploit the translational equivariance of the data by sharing weights across the kernel. This means that we use the same kernel for all positions in the input signal.

#### Non Linearity

To make the network learn complex patterns we need to introduce non-linearity. We can do this by applying a non-linear function to the output of the convolution. As was seen with MLP.

### Architectures

#### LeNet

LeNet is a convolutional neural network architecture that was developed by Yann LeCun in the 1990s. It is a simple yet effective architecture that was used to classify handwritten digits. 

#### AlexNet

AlexNet is a convolutional neural network architecture that was developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012. It is a deep convolutional neural network that was used to classify images in the ImageNet Large Scale Visual Recognition Challenge. 

#### VGG

VGG is a convolutional neural network architecture that was developed by the Visual Geometry Group at the University of Oxford in 2014. It is a deep convolutional neural network that was used to classify images in the ImageNet Large Scale Visual Recognition Challenge. 

#### Inception Network

Inception Network is a convolutional neural network architecture that was developed by Google in 2014. It is a deep convolutional neural network that was used to classify images in the ImageNet Large Scale Visual Recognition Challenge. 

#### ResNet

ResNet is a convolutional neural network architecture that was developed by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in 2015. It is a deep convolutional neural network that was used to classify images in the ImageNet Large Scale Visual Recognition Challenge. 


#### U-Nets

U-Nets is a convolutional neural network architecture that was developed by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015. It is a deep convolutional neural network that was used to classify images in the ImageNet Large Scale Visual Recognition Challenge. 

### Natural Language
#### Embeddings

Word embeddings are a way to represent words as vectors in a continuous vector space. 

#### Convnets for Language

Convolutional Neural Networks can be used for Natural Language Processing tasks. 


#### Cross-Correlation
##### A sibling to convolution defined as $(f \star g)(y) = \int f(x)g(y+x)dx$, often used interchangeably in deep learning frameworks despite not flipping the kernel,.
#### Toeplitz Matrices
##### Discrete convolution can be represented as matrix multiplication using a Toeplitz matrix, highlighting that convolutions are linear operations with parameter sharing (diagonal patterns in the matrix).

### Architecture Design
#### Goals
##### The architecture aims to exploit translation equivariance, locality, and scale to process signals more efficiently than fully connected networks,.
#### Connectivity and Topology
##### Unlike dense layers, ConvNets use sparse (local) connectivity where units are arranged on a grid inheriting the topology of the input (1D, 2D, 3D),.
#### Padding
##### Border handling strategies include "same padding" (padding with zeros to maintain size) and "valid padding" (only retaining fully contained windows, reducing size),.
#### Receptive Fields
##### The receptive field is the region of the input that influences a specific unit; due to the stacking of layers, receptive fields grow with the depth of the network,.
#### Weight Sharing
##### Parameters (kernel weights) are shared across the entire input grid, meaning loss gradients are computed by summing contributions from all spatial locations.
#### Channels
##### Networks learn multiple kernels (filters) per layer, creating multiple "channels" (feature maps). Connections are typically local spatially but fully connected across input channels,.
#### Pooling and Strides
##### Pooling (e.g., max-pooling) aggregates information locally to provide invariance to small translations, often combined with down-sampling (strides) to reduce resolution,.

### Computer Vision Models
#### Pyramidal Architecture
##### A standard design pattern that successively reduces spatial resolution (via pooling/strides) while increasing the number of channels (width),.
#### LeNet5 (1990s)
##### A classical architecture by LeCun using 5x5 convolutions and subsampling for digit recognition.
#### AlexNet (2012)
##### A deep CNN that popularized deep learning in vision, featuring a pyramidal structure and ReLU activations, with significant parameter count in the final dense layers.
#### VGG
##### An architecture emphasizing depth by using very small ($3 \times 3$) convolution kernels to build large receptive fields with fewer parameters than large kernels,.
#### Inception Network
##### Uses "Inception blocks" containing multiple processing paths (e.g., $1 \times 1$, $3 \times 3$, $5 \times 5$ convolutions) in parallel.
##### Utilizes $1 \times 1$ convolutions for dimensionality reduction (bottleneck) to manage computational cost,,.
#### ResNets (Residual Networks)
##### Introduced "skip connections" (adding the input to the output of a block: $y = x + F(x)$) to solve the vanishing gradient problem, enabling the training of very deep networks (100+ layers).
##### Validated on ImageNet, showing that without residual connections, deeper networks often performed worse than shallower ones.
#### U-Nets
##### An encoder-decoder architecture designed for image segmentation, featuring skip connections that concatenate features between the contracting (encoder) and expanding (decoder) paths to preserve spatial detail,.

### Sequence & Audio Processing
#### Embeddings
##### For discrete sequence data (NLP), symbols are mapped to dense vectors (embeddings) via a lookup table before processing,.
#### 1D Convolutions
##### CNNs can be applied to language by convolving over the time/sequence dimension, often using max-over-time pooling to handle variable lengths.
#### WaveNet
##### A generative model for raw audio that uses dilated convolutions (atrous convolutions) to exponentially increase the receptive field, handling long-range dependencies without excessive depth.
