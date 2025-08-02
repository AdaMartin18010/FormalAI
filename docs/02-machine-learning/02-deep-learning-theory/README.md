# 2.2 深度学习理论 / Deep Learning Theory

## 概述 / Overview

深度学习理论研究神经网络的理论性质、表达能力、优化理论和泛化能力，为现代AI系统提供数学基础。

Deep learning theory studies the theoretical properties, expressive power, optimization theory, and generalization capabilities of neural networks, providing mathematical foundations for modern AI systems.

## 目录 / Table of Contents

- [2.2 深度学习理论 / Deep Learning Theory](#22-深度学习理论--deep-learning-theory)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [1. 神经网络表达能力 / Neural Network Expressive Power](#1-神经网络表达能力--neural-network-expressive-power)
    - [1.1 通用逼近定理 / Universal Approximation Theorem](#11-通用逼近定理--universal-approximation-theorem)
    - [1.2 深度优势 / Depth Advantage](#12-深度优势--depth-advantage)
    - [1.3 宽度-深度权衡 / Width-Depth Trade-off](#13-宽度-深度权衡--width-depth-trade-off)
  - [2. 优化理论 / Optimization Theory](#2-优化理论--optimization-theory)
    - [2.1 梯度下降收敛性 / Gradient Descent Convergence](#21-梯度下降收敛性--gradient-descent-convergence)
    - [2.2 随机梯度下降 / Stochastic Gradient Descent](#22-随机梯度下降--stochastic-gradient-descent)
    - [2.3 自适应优化器 / Adaptive Optimizers](#23-自适应优化器--adaptive-optimizers)
  - [3. 损失景观 / Loss Landscape](#3-损失景观--loss-landscape)
    - [3.1 局部最小值 / Local Minima](#31-局部最小值--local-minima)
    - [3.2 鞍点 / Saddle Points](#32-鞍点--saddle-points)
    - [3.3 平坦最小值 / Flat Minima](#33-平坦最小值--flat-minima)
  - [4. 初始化理论 / Initialization Theory](#4-初始化理论--initialization-theory)
    - [4.1 Xavier初始化 / Xavier Initialization](#41-xavier初始化--xavier-initialization)
    - [4.2 He初始化 / He Initialization](#42-he初始化--he-initialization)
    - [4.3 正交初始化 / Orthogonal Initialization](#43-正交初始化--orthogonal-initialization)
  - [5. 正则化理论 / Regularization Theory](#5-正则化理论--regularization-theory)
    - [5.1 权重衰减 / Weight Decay](#51-权重衰减--weight-decay)
    - [5.2 Dropout理论 / Dropout Theory](#52-dropout理论--dropout-theory)
    - [5.3 批归一化 / Batch Normalization](#53-批归一化--batch-normalization)
  - [6. 残差网络理论 / Residual Network Theory](#6-残差网络理论--residual-network-theory)
    - [6.1 残差连接 / Residual Connections](#61-残差连接--residual-connections)
    - [6.2 梯度流 / Gradient Flow](#62-梯度流--gradient-flow)
    - [6.3 特征复用 / Feature Reuse](#63-特征复用--feature-reuse)
  - [7. 注意力机制理论 / Attention Mechanism Theory](#7-注意力机制理论--attention-mechanism-theory)
    - [7.1 自注意力 / Self-Attention](#71-自注意力--self-attention)
    - [7.2 多头注意力 / Multi-Head Attention](#72-多头注意力--multi-head-attention)
    - [7.3 位置编码 / Positional Encoding](#73-位置编码--positional-encoding)
  - [8. 图神经网络理论 / Graph Neural Network Theory](#8-图神经网络理论--graph-neural-network-theory)
    - [8.1 消息传递 / Message Passing](#81-消息传递--message-passing)
    - [8.2 图卷积 / Graph Convolution](#82-图卷积--graph-convolution)
    - [8.3 图注意力 / Graph Attention](#83-图注意力--graph-attention)
  - [9. 生成对抗网络理论 / Generative Adversarial Network Theory](#9-生成对抗网络理论--generative-adversarial-network-theory)
    - [9.1 纳什均衡 / Nash Equilibrium](#91-纳什均衡--nash-equilibrium)
    - [9.2 模式崩塌 / Mode Collapse](#92-模式崩塌--mode-collapse)
    - [9.3 梯度消失 / Gradient Vanishing](#93-梯度消失--gradient-vanishing)
  - [10. 强化学习理论 / Reinforcement Learning Theory](#10-强化学习理论--reinforcement-learning-theory)
    - [10.1 策略梯度 / Policy Gradient](#101-策略梯度--policy-gradient)
    - [10.2 价值函数逼近 / Value Function Approximation](#102-价值函数逼近--value-function-approximation)
    - [10.3 Actor-Critic方法 / Actor-Critic Methods](#103-actor-critic方法--actor-critic-methods)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：神经网络前向传播](#rust实现神经网络前向传播)
    - [Haskell实现：梯度下降优化](#haskell实现梯度下降优化)
  - [参考文献 / References](#参考文献--references)

---

## 1. 神经网络表达能力 / Neural Network Expressive Power

### 1.1 通用逼近定理 / Universal Approximation Theorem

**Cybenko定理 / Cybenko's Theorem:**

设 $\sigma$ 是连续sigmoidal函数，则对于任意连续函数 $f: [0,1]^n \rightarrow \mathbb{R}$ 和 $\epsilon > 0$，存在单隐层神经网络 $N$ 使得：

Let $\sigma$ be a continuous sigmoidal function, then for any continuous function $f: [0,1]^n \rightarrow \mathbb{R}$ and $\epsilon > 0$, there exists a single-hidden-layer neural network $N$ such that:

$$\sup_{x \in [0,1]^n} |f(x) - N(x)| < \epsilon$$

**形式化定义 / Formal Definition:**

$$N(x) = \sum_{i=1}^N \alpha_i \sigma(w_i^T x + b_i)$$

其中 $\alpha_i, b_i \in \mathbb{R}$，$w_i \in \mathbb{R}^n$。

where $\alpha_i, b_i \in \mathbb{R}$ and $w_i \in \mathbb{R}^n$.

### 1.2 深度优势 / Depth Advantage

**Telgarsky定理 / Telgarsky's Theorem:**

存在函数族，深度网络需要指数级更少的参数来达到相同的逼近精度。

There exist function families where deep networks require exponentially fewer parameters to achieve the same approximation accuracy.

**形式化表述 / Formal Statement:**

对于某些函数 $f$，如果浅层网络需要 $\Omega(2^n)$ 个参数，则深度网络只需要 $O(n)$ 个参数。

For some functions $f$, if shallow networks require $\Omega(2^n)$ parameters, then deep networks only need $O(n)$ parameters.

### 1.3 宽度-深度权衡 / Width-Depth Trade-off

**宽度定理 / Width Theorem:**

对于任意连续函数，存在宽度为 $O(n)$ 的浅层网络可以逼近。

For any continuous function, there exists a shallow network with width $O(n)$ that can approximate it.

**深度定理 / Depth Theorem:**

对于某些函数族，深度网络比浅层网络更有效。

For some function families, deep networks are more efficient than shallow networks.

---

## 2. 优化理论 / Optimization Theory

### 2.1 梯度下降收敛性 / Gradient Descent Convergence

**凸优化收敛 / Convex Optimization Convergence:**

对于凸函数 $f$，梯度下降以 $O(1/t)$ 速率收敛：

For convex function $f$, gradient descent converges at rate $O(1/t)$:

$$f(x_t) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2t}$$

其中 $L$ 是Lipschitz常数。

where $L$ is the Lipschitz constant.

**强凸函数收敛 / Strongly Convex Function Convergence:**

对于强凸函数，收敛速率为 $O((1-\mu/L)^t)$：

For strongly convex functions, convergence rate is $O((1-\mu/L)^t)$:

$$f(x_t) - f(x^*) \leq (1-\mu/L)^t (f(x_0) - f(x^*))$$

### 2.2 随机梯度下降 / Stochastic Gradient Descent

**SGD收敛 / SGD Convergence:**

对于凸函数，SGD以 $O(1/\sqrt{t})$ 速率收敛：

For convex functions, SGD converges at rate $O(1/\sqrt{t})$:

$$\mathbb{E}[f(\bar{x}_t)] - f(x^*) \leq \frac{G\|x_0 - x^*\|}{\sqrt{t}}$$

其中 $G$ 是梯度范数上界。

where $G$ is the upper bound on gradient norm.

**方差减少 / Variance Reduction:**

SVRG算法以 $O(1/t)$ 速率收敛：

SVRG algorithm converges at rate $O(1/t)$:

$$\mathbb{E}[f(x_t)] - f(x^*) \leq \frac{4L\|x_0 - x^*\|^2}{t}$$

### 2.3 自适应优化器 / Adaptive Optimizers

**Adam算法 / Adam Algorithm:**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$x_t = x_{t-1} - \frac{\alpha}{\sqrt{v_t} + \epsilon} m_t$$

**收敛性分析 / Convergence Analysis:**

对于凸函数，Adam以 $O(1/\sqrt{t})$ 速率收敛。

For convex functions, Adam converges at rate $O(1/\sqrt{t})$.

---

## 3. 损失景观 / Loss Landscape

### 3.1 局部最小值 / Local Minima

**局部最小值定义 / Local Minimum Definition:**

点 $x^*$ 是局部最小值，如果存在邻域 $N(x^*)$ 使得：

Point $x^*$ is a local minimum if there exists neighborhood $N(x^*)$ such that:

$$f(x^*) \leq f(x) \quad \forall x \in N(x^*)$$

**局部最小值密度 / Local Minimum Density:**

对于随机初始化的神经网络，局部最小值数量随网络大小指数增长。

For randomly initialized neural networks, the number of local minima grows exponentially with network size.

### 3.2 鞍点 / Saddle Points

**鞍点定义 / Saddle Point Definition:**

点 $x^*$ 是鞍点，如果 $\nabla f(x^*) = 0$ 且 Hessian矩阵 $H(x^*)$ 有正负特征值。

Point $x^*$ is a saddle point if $\nabla f(x^*) = 0$ and Hessian matrix $H(x^*)$ has both positive and negative eigenvalues.

**鞍点逃离 / Saddle Point Escaping:**

二阶方法可以快速逃离鞍点：

Second-order methods can quickly escape saddle points:

$$x_{t+1} = x_t - \eta H^{-1}(x_t) \nabla f(x_t)$$

### 3.3 平坦最小值 / Flat Minima

**平坦最小值定义 / Flat Minimum Definition:**

平坦最小值具有小的Hessian特征值，对应更好的泛化能力。

Flat minima have small Hessian eigenvalues, corresponding to better generalization.

**泛化界 / Generalization Bounds:**

$$\mathbb{E}[L(f)] \leq \hat{L}(f) + O\left(\sqrt{\frac{\text{tr}(H)}{n}}\right)$$

其中 $H$ 是损失函数的Hessian矩阵。

where $H$ is the Hessian matrix of the loss function.

---

## 4. 初始化理论 / Initialization Theory

### 4.1 Xavier初始化 / Xavier Initialization

**Xavier初始化 / Xavier Initialization:**

对于第 $l$ 层，权重方差为：

For layer $l$, weight variance is:

$$\text{Var}(W^l) = \frac{2}{n_{l-1} + n_l}$$

其中 $n_l$ 是第 $l$ 层的神经元数量。

where $n_l$ is the number of neurons in layer $l$.

**理论依据 / Theoretical Basis:**

保持每层输入输出的方差相等：

Maintain equal variance of input and output for each layer:

$$\text{Var}(y^l) = \text{Var}(y^{l-1})$$

### 4.2 He初始化 / He Initialization

**He初始化 / He Initialization:**

对于ReLU激活函数：

For ReLU activation function:

$$\text{Var}(W^l) = \frac{2}{n_{l-1}}$$

**理论依据 / Theoretical Basis:**

考虑ReLU的零化效应：

Consider the zeroing effect of ReLU:

$$\text{Var}(y^l) = \frac{1}{2} \text{Var}(y^{l-1})$$

### 4.3 正交初始化 / Orthogonal Initialization

**正交初始化 / Orthogonal Initialization:**

$$W = U \Sigma V^T$$

其中 $U, V$ 是正交矩阵，$\Sigma$ 是奇异值矩阵。

where $U, V$ are orthogonal matrices and $\Sigma$ is the singular value matrix.

**优势 / Advantages:**

- 保持梯度范数
- 避免梯度消失/爆炸
- 提高训练稳定性

- Maintain gradient norm
- Avoid gradient vanishing/explosion
- Improve training stability

---

## 5. 正则化理论 / Regularization Theory

### 5.1 权重衰减 / Weight Decay

**L2正则化 / L2 Regularization:**

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \sum_{i,j} W_{i,j}^2$$

**梯度更新 / Gradient Update:**

$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial W} = \frac{\partial \mathcal{L}}{\partial W} + \lambda W$$

**理论效果 / Theoretical Effect:**

权重衰减等价于在每次更新时缩小权重：

Weight decay is equivalent to shrinking weights at each update:

$$W_{t+1} = (1-\lambda) W_t - \eta \nabla \mathcal{L}(W_t)$$

### 5.2 Dropout理论 / Dropout Theory

**Dropout机制 / Dropout Mechanism:**

训练时以概率 $p$ 随机置零神经元：

During training, randomly zero neurons with probability $p$:

$$y = \frac{1}{1-p} \cdot \text{mask} \odot x$$

**理论解释 / Theoretical Explanation:**

Dropout等价于集成学习：

Dropout is equivalent to ensemble learning:

$$\mathbb{E}[f(x)] = \frac{1}{2^n} \sum_{m \in \{0,1\}^n} f_m(x)$$

### 5.3 批归一化 / Batch Normalization

**批归一化公式 / Batch Normalization Formula:**

$$\text{BN}(x) = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$

其中 $\mu_B, \sigma_B^2$ 是批次统计量。

where $\mu_B, \sigma_B^2$ are batch statistics.

**理论优势 / Theoretical Advantages:**

- 减少内部协变量偏移
- 允许更大的学习率
- 提高训练稳定性

- Reduce internal covariate shift
- Allow larger learning rates
- Improve training stability

---

## 6. 残差网络理论 / Residual Network Theory

### 6.1 残差连接 / Residual Connections

**残差块 / Residual Block:**

$$y = F(x, W) + x$$

其中 $F(x, W)$ 是残差函数。

where $F(x, W)$ is the residual function.

**梯度流 / Gradient Flow:**

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(\frac{\partial F}{\partial x} + I\right)$$

### 6.2 梯度流 / Gradient Flow

**梯度消失缓解 / Gradient Vanishing Mitigation:**

残差连接提供了恒等映射的捷径：

Residual connections provide shortcuts for identity mapping:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \cdot \prod_{i=l}^{L-1} \left(\frac{\partial F_i}{\partial x_i} + I\right)$$

### 6.3 特征复用 / Feature Reuse

**特征复用机制 / Feature Reuse Mechanism:**

浅层特征可以直接传递到深层：

Shallow features can be directly passed to deep layers:

$$x_L = x_0 + \sum_{i=1}^L F_i(x_{i-1})$$

---

## 7. 注意力机制理论 / Attention Mechanism Theory

### 7.1 自注意力 / Self-Attention

**注意力公式 / Attention Formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q, K, V$ 分别是查询、键、值矩阵。

where $Q, K, V$ are query, key, value matrices respectively.

**理论分析 / Theoretical Analysis:**

注意力权重表示token间的相关性：

Attention weights represent token correlations:

$$a_{ij} = \frac{\exp(q_i^T k_j / \sqrt{d_k})}{\sum_l \exp(q_i^T k_l / \sqrt{d_k})}$$

### 7.2 多头注意力 / Multi-Head Attention

**多头注意力 / Multi-Head Attention:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

其中每个头为：

where each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 7.3 位置编码 / Positional Encoding

**正弦位置编码 / Sinusoidal Positional Encoding:**

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

**理论性质 / Theoretical Properties:**

- 可以处理任意长度的序列
- 具有相对位置信息
- 可以外推到更长序列

- Can handle sequences of arbitrary length
- Has relative position information
- Can extrapolate to longer sequences

---

## 8. 图神经网络理论 / Graph Neural Network Theory

### 8.1 消息传递 / Message Passing

**消息传递框架 / Message Passing Framework:**

$$h_v^{(l+1)} = \text{UPDATE}^{(l)}\left(h_v^{(l)}, \text{AGGREGATE}^{(l)}\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

其中 $\mathcal{N}(v)$ 是节点 $v$ 的邻居。

where $\mathcal{N}(v)$ are the neighbors of node $v$.

### 8.2 图卷积 / Graph Convolution

**图卷积公式 / Graph Convolution Formula:**

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$$

其中 $\tilde{A} = A + I$ 是带自环的邻接矩阵。

where $\tilde{A} = A + I$ is the adjacency matrix with self-loops.

### 8.3 图注意力 / Graph Attention

**图注意力 / Graph Attention:**

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_j]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_k]))}$$

---

## 9. 生成对抗网络理论 / Generative Adversarial Network Theory

### 9.1 纳什均衡 / Nash Equilibrium

**GAN目标函数 / GAN Objective:**

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]$$

**纳什均衡 / Nash Equilibrium:**

在纳什均衡点，生成器和判别器都无法通过单方面改变策略来改善性能。

At Nash equilibrium, neither generator nor discriminator can improve performance by unilaterally changing strategy.

### 9.2 模式崩塌 / Mode Collapse

**模式崩塌定义 / Mode Collapse Definition:**

生成器只生成数据分布的一部分模式：

Generator only generates a subset of data distribution modes:

$$p_g(x) \neq p_{\text{data}}(x)$$

**理论分析 / Theoretical Analysis:**

模式崩塌源于判别器的过度自信：

Mode collapse stems from discriminator overconfidence:

$$D(x) \rightarrow 1 \quad \forall x$$

### 9.3 梯度消失 / Gradient Vanishing

**梯度消失问题 / Gradient Vanishing Problem:**

当判别器过于强大时，生成器梯度消失：

When discriminator is too strong, generator gradient vanishes:

$$\nabla_G V(D, G) \rightarrow 0$$

**解决方案 / Solutions:**

- Wasserstein GAN
- Gradient penalty
- Spectral normalization

---

## 10. 强化学习理论 / Reinforcement Learning Theory

### 10.1 策略梯度 / Policy Gradient

**策略梯度定理 / Policy Gradient Theorem:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(\tau) R(\tau)]$$

其中 $\tau$ 是轨迹，$R(\tau)$ 是回报。

where $\tau$ is trajectory and $R(\tau)$ is reward.

**REINFORCE算法 / REINFORCE Algorithm:**

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) R_t$$

### 10.2 价值函数逼近 / Value Function Approximation

**价值函数 / Value Function:**

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s\right]$$

**Q函数 / Q-Function:**

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s, a_0 = a\right]$$

### 10.3 Actor-Critic方法 / Actor-Critic Methods

**Actor-Critic框架 / Actor-Critic Framework:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) A^\pi(s, a)]$$

其中优势函数为：

where advantage function is:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

---

## 代码示例 / Code Examples

### Rust实现：神经网络前向传播

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    activation: Activation,
}

#[derive(Debug, Clone)]
enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
}

impl Layer {
    fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<Vec<f64>> = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-1.0..1.0) * (2.0 / input_size as f64).sqrt())
                    .collect()
            })
            .collect();
        
        let biases: Vec<f64> = (0..output_size)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        
        Layer {
            weights,
            biases,
            activation,
        }
    }
    
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; self.weights.len()];
        
        for (i, (weights, bias)) in self.weights.iter().zip(&self.biases).enumerate() {
            let sum: f64 = weights.iter().zip(input).map(|(w, x)| w * x).sum();
            output[i] = self.activation.apply(sum + bias);
        }
        
        output
    }
}

impl Activation {
    fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Linear => x,
        }
    }
    
    fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Sigmoid => {
                let s = self.apply(x);
                s * (1.0 - s)
            },
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Linear => 1.0,
        }
    }
}

#[derive(Debug)]
struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    fn new(layer_sizes: Vec<usize>, activations: Vec<Activation>) -> Self {
        let mut layers = Vec::new();
        
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Layer::new(
                layer_sizes[i],
                layer_sizes[i + 1],
                activations[i].clone(),
            ));
        }
        
        NeuralNetwork { layers }
    }
    
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        
        current
    }
    
    fn backward(&self, input: &[f64], target: &[f64]) -> Vec<Vec<Vec<f64>>> {
        // 前向传播
        let mut activations = vec![input.to_vec()];
        let mut z_values = Vec::new();
        
        for layer in &self.layers {
            let z: Vec<f64> = layer.weights.iter().zip(&layer.biases).map(|(weights, bias)| {
                weights.iter().zip(&activations.last().unwrap()).map(|(w, x)| w * x).sum::<f64>() + bias
            }).collect();
            z_values.push(z.clone());
            
            let activation: Vec<f64> = z.iter().map(|&z_val| layer.activation.apply(z_val)).collect();
            activations.push(activation);
        }
        
        // 反向传播
        let mut gradients = Vec::new();
        let mut delta = activations.last().unwrap().iter().zip(target).map(|(a, t)| a - t).collect::<Vec<f64>>();
        
        for (layer_idx, layer) in self.layers.iter().enumerate().rev() {
            let mut layer_gradients = Vec::new();
            
            for (neuron_idx, (weights, bias)) in layer.weights.iter().zip(&layer.biases).enumerate() {
                let mut weight_gradients = Vec::new();
                
                for (input_idx, input_val) in activations[layer_idx].iter().enumerate() {
                    weight_gradients.push(delta[neuron_idx] * input_val);
                }
                
                layer_gradients.push(weight_gradients);
            }
            
            gradients.push(layer_gradients);
            
            if layer_idx > 0 {
                let mut new_delta = vec![0.0; activations[layer_idx].len()];
                
                for (input_idx, _) in activations[layer_idx].iter().enumerate() {
                    for (neuron_idx, weights) in layer.weights.iter().enumerate() {
                        new_delta[input_idx] += delta[neuron_idx] * weights[input_idx] * 
                            layer.activation.derivative(z_values[layer_idx - 1][input_idx]);
                    }
                }
                delta = new_delta;
            }
        }
        
        gradients.reverse();
        gradients
    }
}

fn main() {
    // 创建神经网络：2输入 -> 3隐藏 -> 1输出
    let network = NeuralNetwork::new(
        vec![2, 3, 1],
        vec![Activation::ReLU, Activation::Sigmoid],
    );
    
    // 测试前向传播
    let input = vec![0.5, 0.3];
    let output = network.forward(&input);
    
    println!("输入: {:?}", input);
    println!("输出: {:?}", output);
    
    // 测试反向传播
    let target = vec![0.8];
    let gradients = network.backward(&input, &target);
    
    println!("梯度数量: {}", gradients.len());
}
```

### Haskell实现：梯度下降优化

```haskell
import Data.List (foldl')
import System.Random

-- 神经网络类型定义
data Layer = Layer {
    weights :: [[Double]],
    biases :: [Double],
    activation :: Activation
} deriving Show

data Activation = ReLU | Sigmoid | Tanh | Linear deriving Show

data NeuralNetwork = NeuralNetwork {
    layers :: [Layer]
} deriving Show

-- 激活函数
applyActivation :: Activation -> Double -> Double
applyActivation ReLU x = max 0 x
applyActivation Sigmoid x = 1 / (1 + exp (-x))
applyActivation Tanh x = tanh x
applyActivation Linear x = x

applyActivationDerivative :: Activation -> Double -> Double
applyActivationDerivative ReLU x = if x > 0 then 1 else 0
applyActivationDerivative Sigmoid x = let s = applyActivation Sigmoid x in s * (1 - s)
applyActivationDerivative Tanh x = 1 - (tanh x) ^ 2
applyActivationDerivative Linear _ = 1

-- 前向传播
forward :: NeuralNetwork -> [Double] -> [Double]
forward network input = foldl' (\acc layer -> forwardLayer layer acc) input (layers network)

forwardLayer :: Layer -> [Double] -> [Double]
forwardLayer layer input = map (\i -> applyActivation (activation layer) (sum (zipWith (*) (weights layer !! i) input) + biases layer !! i)) [0..length (weights layer) - 1]

-- 损失函数
mseLoss :: [Double] -> [Double] -> Double
mseLoss predicted target = sum (zipWith (\p t -> (p - t) ^ 2) predicted target) / fromIntegral (length predicted)

-- 梯度下降优化
gradientDescent :: NeuralNetwork -> [Double] -> [Double] -> Double -> NeuralNetwork
gradientDescent network input target learningRate = 
    let gradients = computeGradients network input target
        updatedLayers = zipWith (updateLayer learningRate) (layers network) gradients
    in network { layers = updatedLayers }

-- 计算梯度
computeGradients :: NeuralNetwork -> [Double] -> [Double] -> [[[Double]]]
computeGradients network input target = 
    let (activations, zValues) = forwardPass network input
        delta = zipWith (-) (last activations) target
    in backwardPass network activations zValues delta

-- 前向传播（保存中间值）
forwardPass :: NeuralNetwork -> [Double] -> ([[Double]], [[Double]])
forwardPass network input = 
    let layers_list = layers network
        (activations, zValues) = foldl' (\(acts, zs) layer -> 
            let z = zipWith (\weights bias -> sum (zipWith (*) weights (last acts)) + bias) (weights layer) (biases layer)
                activation = map (applyActivation (activation layer)) z
            in (acts ++ [activation], zs ++ [z])) ([input], []) layers_list
    in (activations, zValues)

-- 反向传播
backwardPass :: NeuralNetwork -> [[Double]] -> [[Double]] -> [Double] -> [[[Double]]]
backwardPass network activations zValues initialDelta = 
    let layers_list = reverse (layers network)
        (_, gradients) = foldl' (\(delta, grads) (layer, layerIdx) -> 
            let layerGradients = computeLayerGradients layer (activations !! layerIdx) delta
                newDelta = if layerIdx > 0 then computeNewDelta layer (activations !! (layerIdx - 1)) delta (zValues !! (layerIdx - 1)) else []
            in (newDelta, grads ++ [layerGradients])) (initialDelta, []) (zip layers_list [length layers_list - 1, length layers_list - 2..0])
    in reverse gradients

-- 计算层梯度
computeLayerGradients :: Layer -> [Double] -> [Double] -> [[Double]]
computeLayerGradients layer activation delta = 
    map (\neuronIdx -> 
        map (\inputIdx -> delta !! neuronIdx * (activation !! inputIdx)) [0..length activation - 1]
    ) [0..length (weights layer) - 1]

-- 计算新的delta
computeNewDelta :: Layer -> [Double] -> [Double] -> [Double] -> [Double]
computeNewDelta layer prevActivation delta prevZ = 
    map (\inputIdx -> 
        sum (zipWith (\neuronIdx weight -> 
            delta !! neuronIdx * weight * applyActivationDerivative (activation layer) (prevZ !! inputIdx)
        ) [0..length (weights layer) - 1] (map (!! inputIdx) (weights layer)))
    ) [0..length prevActivation - 1]

-- 更新层参数
updateLayer :: Double -> Layer -> [[Double]] -> Layer
updateLayer learningRate layer gradients = 
    let updatedWeights = zipWith (\weights gradient -> 
            zipWith (\w g -> w - learningRate * g) weights gradient
        ) (weights layer) gradients
    in layer { weights = updatedWeights }

-- 训练函数
train :: NeuralNetwork -> [[Double]] -> [[Double]] -> Double -> Int -> NeuralNetwork
train network inputs targets learningRate epochs = 
    foldl' (\net epoch -> 
        foldl' (\acc (input, target) -> gradientDescent acc input target learningRate) net (zip inputs targets)
    ) network [1..epochs]

-- 创建简单网络
createNetwork :: NeuralNetwork
createNetwork = NeuralNetwork {
    layers = [
        Layer {
            weights = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            biases = [0.1, 0.2, 0.3],
            activation = ReLU
        },
        Layer {
            weights = [[0.1, 0.2, 0.3]],
            biases = [0.1],
            activation = Sigmoid
        }
    ]
}

-- 主函数
main :: IO ()
main = do
    let network = createNetwork
        inputs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        targets = [[0.8], [0.9], [0.7]]
        learningRate = 0.01
        epochs = 1000
    
    let trainedNetwork = train network inputs targets learningRate epochs
    
    putStrLn "训练后的网络:"
    print trainedNetwork
    
    putStrLn "\n预测结果:"
    mapM_ (\input -> do
        let output = forward trainedNetwork input
        putStrLn $ "输入: " ++ show input ++ " -> 输出: " ++ show output
    ) inputs
```

---

## 参考文献 / References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.
3. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
4. Vaswani, A., et al. (2017). Attention is all you need. *NIPS*.
5. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.
6. Goodfellow, I., et al. (2014). Generative adversarial nets. *NIPS*.
7. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.

---

*本模块为FormalAI提供了深度学习的理论基础，涵盖了从表达能力到优化理论的各个方面，为现代AI系统的设计和分析提供了数学工具。*
