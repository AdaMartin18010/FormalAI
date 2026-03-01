# AI/ML 模型方法与建模理论全面指南

> **对标课程**: Stanford CS229 (Machine Learning), CS230 (Deep Learning), CS224N (NLP with Deep Learning), CMU 10-701
>
> **适用对象**: 机器学习工程师、数据科学家、AI研究人员

---

## 目录

- [AI/ML 模型方法与建模理论全面指南](#aiml-模型方法与建模理论全面指南)
  - [目录](#目录)
  - [1. 监督学习方法](#1-监督学习方法)
    - [1.1 线性回归与逻辑回归](#11-线性回归与逻辑回归)
      - [1.1.1 线性回归 (Linear Regression)](#111-线性回归-linear-regression)
      - [1.1.2 逻辑回归 (Logistic Regression)](#112-逻辑回归-logistic-regression)
    - [1.2 支持向量机 (SVM)](#12-支持向量机-svm)
      - [1.2.1 硬间隔SVM](#121-硬间隔svm)
      - [1.2.2 软间隔SVM（含松弛变量）](#122-软间隔svm含松弛变量)
      - [1.2.3 核方法](#123-核方法)
    - [1.3 决策树与集成学习](#13-决策树与集成学习)
      - [1.3.1 决策树 (Decision Tree)](#131-决策树-decision-tree)
      - [1.3.2 随机森林 (Random Forest)](#132-随机森林-random-forest)
      - [1.3.3 梯度提升树 (Gradient Boosting)](#133-梯度提升树-gradient-boosting)
    - [1.4 XGBoost、LightGBM、CatBoost](#14-xgboostlightgbmcatboost)
      - [1.4.1 XGBoost (eXtreme Gradient Boosting)](#141-xgboost-extreme-gradient-boosting)
      - [1.4.2 LightGBM](#142-lightgbm)
      - [1.4.3 CatBoost](#143-catboost)
    - [1.5 神经网络基础](#15-神经网络基础)
      - [1.5.1 多层感知机 (MLP)](#151-多层感知机-mlp)
      - [1.5.2 优化算法](#152-优化算法)
  - [2. 深度学习架构](#2-深度学习架构)
    - [2.1 卷积神经网络 (CNN)](#21-卷积神经网络-cnn)
      - [2.1.1 卷积操作](#211-卷积操作)
      - [2.1.2 经典CNN架构](#212-经典cnn架构)
      - [2.1.3 残差连接 (ResNet)](#213-残差连接-resnet)
      - [2.1.4 批归一化 (Batch Normalization)](#214-批归一化-batch-normalization)
    - [2.2 循环神经网络 (RNN)](#22-循环神经网络-rnn)
      - [2.2.1 基础RNN](#221-基础rnn)
      - [2.2.2 LSTM (Long Short-Term Memory)](#222-lstm-long-short-term-memory)
      - [2.2.3 GRU (Gated Recurrent Unit)](#223-gru-gated-recurrent-unit)
    - [2.3 Transformer架构](#23-transformer架构)
      - [2.3.1 自注意力机制 (Self-Attention)](#231-自注意力机制-self-attention)
      - [2.3.2 Transformer编码器](#232-transformer编码器)
      - [2.3.3 Transformer解码器](#233-transformer解码器)
      - [2.3.4 BERT (Bidirectional Encoder Representations)](#234-bert-bidirectional-encoder-representations)
      - [2.3.5 GPT系列 (Generative Pre-trained Transformer)](#235-gpt系列-generative-pre-trained-transformer)
      - [2.3.6 Vision Transformer (ViT)](#236-vision-transformer-vit)
  - [3. 无监督学习方法](#3-无监督学习方法)
    - [3.1 聚类算法](#31-聚类算法)
      - [3.1.1 K-Means聚类](#311-k-means聚类)
      - [3.1.2 DBSCAN (Density-Based Spatial Clustering)](#312-dbscan-density-based-spatial-clustering)
      - [3.1.3 层次聚类](#313-层次聚类)
      - [3.1.4 高斯混合模型 (GMM)](#314-高斯混合模型-gmm)
    - [3.2 降维方法](#32-降维方法)
      - [3.2.1 主成分分析 (PCA)](#321-主成分分析-pca)
      - [3.2.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)](#322-t-sne-t-distributed-stochastic-neighbor-embedding)
      - [3.2.3 UMAP (Uniform Manifold Approximation and Projection)](#323-umap-uniform-manifold-approximation-and-projection)
      - [3.2.4 自编码器 (Autoencoder)](#324-自编码器-autoencoder)
    - [3.3 异常检测](#33-异常检测)
      - [3.3.1 统计方法](#331-统计方法)
      - [3.3.2 基于距离的方法](#332-基于距离的方法)
      - [3.3.3 基于重构的方法](#333-基于重构的方法)
  - [4. 强化学习](#4-强化学习)
    - [4.1 基础概念](#41-基础概念)
    - [4.2 基于价值的方法](#42-基于价值的方法)
      - [4.2.1 Q-Learning](#421-q-learning)
      - [4.2.2 DQN (Deep Q-Network)](#422-dqn-deep-q-network)
    - [4.3 策略梯度方法](#43-策略梯度方法)
      - [4.3.1 策略梯度定理](#431-策略梯度定理)
      - [4.3.2 REINFORCE](#432-reinforce)
      - [4.3.3 Actor-Critic](#433-actor-critic)
      - [4.3.4 A3C (Asynchronous Advantage Actor-Critic)](#434-a3c-asynchronous-advantage-actor-critic)
      - [4.3.5 PPO (Proximal Policy Optimization)](#435-ppo-proximal-policy-optimization)
      - [4.3.6 SAC (Soft Actor-Critic)](#436-sac-soft-actor-critic)
    - [4.4 基于模型的强化学习](#44-基于模型的强化学习)
      - [4.4.1 模型学习](#441-模型学习)
      - [4.4.2 Dyna-Q](#442-dyna-q)
      - [4.4.3 MBPO (Model-Based Policy Optimization)](#443-mbpo-model-based-policy-optimization)
  - [5. 生成式AI](#5-生成式ai)
    - [5.1 变分自编码器 (VAE)](#51-变分自编码器-vae)
      - [5.1.1 基本架构](#511-基本架构)
      - [5.1.2 ELBO (Evidence Lower Bound)](#512-elbo-evidence-lower-bound)
      - [5.1.3 VAE变体](#513-vae变体)
    - [5.2 生成对抗网络 (GAN)](#52-生成对抗网络-gan)
      - [5.2.1 基本框架](#521-基本框架)
      - [5.2.2 训练技巧](#522-训练技巧)
      - [5.2.3 DCGAN (Deep Convolutional GAN)](#523-dcgan-deep-convolutional-gan)
      - [5.2.4 StyleGAN](#524-stylegan)
      - [5.2.5 CycleGAN](#525-cyclegan)
    - [5.3 扩散模型 (Diffusion Models)](#53-扩散模型-diffusion-models)
      - [5.3.1 前向扩散过程](#531-前向扩散过程)
      - [5.3.2 反向去噪过程](#532-反向去噪过程)
      - [5.3.3 DDPM (Denoising Diffusion Probabilistic Models)](#533-ddpm-denoising-diffusion-probabilistic-models)
      - [5.3.4 Stable Diffusion / Latent Diffusion](#534-stable-diffusion--latent-diffusion)
      - [5.3.5 扩散模型 vs GAN vs VAE](#535-扩散模型-vs-gan-vs-vae)
    - [5.4 流模型 (Flow-based Models)](#54-流模型-flow-based-models)
      - [5.4.1 可逆变换](#541-可逆变换)
      - [5.4.2 RealNVP](#542-realnvp)
      - [5.4.3 Glow](#543-glow)
    - [5.5 大语言模型 (LLM)](#55-大语言模型-llm)
      - [5.5.1 规模定律 (Scaling Laws)](#551-规模定律-scaling-laws)
      - [5.5.2 预训练策略](#552-预训练策略)
      - [5.5.3 微调方法](#553-微调方法)
      - [5.5.4 指令微调 (Instruction Tuning)](#554-指令微调-instruction-tuning)
      - [5.5.5 RLHF (Reinforcement Learning from Human Feedback)](#555-rlhf-reinforcement-learning-from-human-feedback)
      - [5.5.6 LLM推理优化](#556-llm推理优化)
  - [6. 建模方法论](#6-建模方法论)
    - [6.1 特征工程](#61-特征工程)
      - [6.1.1 特征类型](#611-特征类型)
      - [6.1.2 数值特征处理](#612-数值特征处理)
      - [6.1.3 类别特征编码](#613-类别特征编码)
      - [6.1.4 特征交互](#614-特征交互)
      - [6.1.5 特征选择](#615-特征选择)
    - [6.2 模型选择策略](#62-模型选择策略)
      - [6.2.1 问题类型与模型选择](#621-问题类型与模型选择)
      - [6.2.2 偏差-方差权衡](#622-偏差-方差权衡)
      - [6.2.3 学习曲线分析](#623-学习曲线分析)
    - [6.3 超参数优化](#63-超参数优化)
      - [6.3.1 网格搜索 (Grid Search)](#631-网格搜索-grid-search)
      - [6.3.2 随机搜索 (Random Search)](#632-随机搜索-random-search)
      - [6.3.3 贝叶斯优化](#633-贝叶斯优化)
      - [6.3.4 早停与剪枝](#634-早停与剪枝)
    - [6.4 交叉验证与模型评估](#64-交叉验证与模型评估)
      - [6.4.1 交叉验证方法](#641-交叉验证方法)
      - [6.4.2 分类评估指标](#642-分类评估指标)
      - [6.4.3 回归评估指标](#643-回归评估指标)
      - [6.4.4 模型校准](#644-模型校准)
    - [6.5 集成学习方法](#65-集成学习方法)
      - [6.5.1 Bagging](#651-bagging)
      - [6.5.2 Boosting](#652-boosting)
      - [6.5.3 Stacking](#653-stacking)
      - [6.5.4 集成策略对比](#654-集成策略对比)
    - [6.6 模型部署与监控](#66-模型部署与监控)
      - [6.6.1 模型序列化](#661-模型序列化)
      - [6.6.2 模型监控指标](#662-模型监控指标)
  - [附录](#附录)
    - [A. 常用数学符号表](#a-常用数学符号表)
    - [B. 常用损失函数速查](#b-常用损失函数速查)
    - [C. 推荐学习资源](#c-推荐学习资源)

---

## 1. 监督学习方法

### 1.1 线性回归与逻辑回归

#### 1.1.1 线性回归 (Linear Regression)

**数学原理**

线性回归假设目标变量 $y$ 与输入特征 $\mathbf{x}$ 之间存在线性关系：

$$y = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$

其中 $\mathbf{w} \in \mathbb{R}^d$ 是权重向量，$b \in \mathbb{R}$ 是偏置项。

**损失函数（均方误差 MSE）**

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^{m} (y^{(i)} - \mathbf{w}^T \mathbf{x}^{(i)} - b)^2$$

**闭式解（正规方程）**

$$\hat{\mathbf{w}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

**梯度下降更新规则**

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j} = w_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

#### 1.1.2 逻辑回归 (Logistic Regression)

**Sigmoid函数**

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**假设函数**

$$h_\theta(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

**对数似然损失（交叉熵）**

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right]$$

**多分类扩展（Softmax回归）**

$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

---

### 1.2 支持向量机 (SVM)

#### 1.2.1 硬间隔SVM

**优化目标**

$$\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2$$

约束条件：$y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1, \quad \forall i$

#### 1.2.2 软间隔SVM（含松弛变量）

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^{m} \xi_i$$

约束条件：

- $y^{(i)}(\mathbf{w}^T \mathbf{x}^{(i)} + b) \geq 1 - \xi_i$
- $\xi_i \geq 0$

#### 1.2.3 核方法

**核技巧**：通过核函数 $K(\mathbf{x}, \mathbf{x}')$ 隐式计算高维特征空间的内积

| 核函数 | 表达式 | 适用场景 |
|--------|--------|----------|
| 线性核 | $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$ | 线性可分数据 |
| 多项式核 | $K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^T \mathbf{x}' + r)^d$ | 多项式关系 |
| RBF/Gaussian核 | $K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma ||\mathbf{x} - \mathbf{x}'||^2\right)$ | 非线性数据 |
| Sigmoid核 | $K(\mathbf{x}, \mathbf{x}') = \tanh(\gamma \mathbf{x}^T \mathbf{x}' + r)$ | 神经网络类似 |

**对偶问题**

$$\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} K(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$$

约束：$0 \leq \alpha_i \leq C$, $\sum_{i=1}^{m} \alpha_i y^{(i)} = 0$

---

### 1.3 决策树与集成学习

#### 1.3.1 决策树 (Decision Tree)

**信息增益（ID3算法）**

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

其中熵 $H(S) = -\sum_{i} p_i \log_2 p_i$

**基尼不纯度（CART算法）**

$$Gini(S) = 1 - \sum_{i} p_i^2$$

**分裂标准对比**

| 标准 | 公式 | 特点 |
|------|------|------|
| 信息增益 | $IG = H_{parent} - H_{children}$ | 偏向多值属性 |
| 信息增益率 | $IGR = \frac{IG}{SplitInfo}$ | 修正多值偏向 |
| 基尼指数 | $Gini = 1 - \sum p_i^2$ | 计算高效，CART默认 |

#### 1.3.2 随机森林 (Random Forest)

**算法原理**

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x})$$

其中 $T_b$ 是第 $b$ 棵决策树，通过Bootstrap采样构建。

**Bagging（Bootstrap Aggregating）**

$$Var(\bar{X}) = \frac{\sigma^2}{n} + \frac{n-1}{n}\rho\sigma^2$$

**特征重要性计算**

$$Importance(X_j) = \frac{1}{B} \sum_{b=1}^{B} \left( \text{impurity\_decrease}(X_j, T_b) \right)$$

**超参数**

| 参数 | 说明 | 典型值 |
|------|------|--------|
| n_estimators | 树的数量 | 100-500 |
| max_depth | 最大深度 | 10-30 |
| min_samples_split | 分裂最小样本数 | 2-10 |
| max_features | 特征采样比例 | sqrt, log2 |

#### 1.3.3 梯度提升树 (Gradient Boosting)

**加法模型**

$$F_M(\mathbf{x}) = \sum_{m=1}^{M} \gamma_m h_m(\mathbf{x})$$

**前向分步算法**

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \gamma_m h_m(\mathbf{x})$$

**负梯度（伪残差）**

$$r_{im} = -\left[ \frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F=F_{m-1}}$$

---

### 1.4 XGBoost、LightGBM、CatBoost

#### 1.4.1 XGBoost (eXtreme Gradient Boosting)

**目标函数（二阶泰勒展开）**

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

其中：

- $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$（一阶梯度）
- $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$（二阶梯度/Hessian）
- $\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$（正则化项）

**结构分数**

$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma$$

**关键特性**

- 列采样（Column Subsampling）
- 行采样（Row Subsampling）
- 缺失值处理
- 并行计算优化

#### 1.4.2 LightGBM

**直方图算法**

将连续特征离散化为 $k$ 个bin，大幅减少计算量：

$$H[j] = \sum_{i: x_i \in bin_j} g_i, \quad H_{cnt}[j] = \sum_{i: x_i \in bin_j} 1$$

**Leaf-wise生长策略**

$$\text{选择分裂增益最大的叶子节点进行分裂}$$

**GOSS (Gradient-based One-Side Sampling)**

保留梯度大的样本，随机采样梯度小的样本：

$$\tilde{g}_i = \begin{cases} g_i & \text{if } i \in \text{Top}_a \\ \frac{1-a}{b} g_i & \text{if } i \in \text{Random}_b \end{cases}$$

**EFB (Exclusive Feature Bundling)**

将互斥特征捆绑在一起，减少特征数量。

#### 1.4.3 CatBoost

**Ordered Target Statistics**

解决预测偏移（Prediction Shift）问题：

$$\hat{x}_{k}^{i} = \frac{\sum_{j=1}^{i-1} \mathbb{1}_{x_{k}^{j} = x_{k}^{i}} \cdot y^j + a \cdot P}{\sum_{j=1}^{i-1} \mathbb{1}_{x_{k}^{j} = x_{k}^{i}} + a}$$

**对称树（Oblivious Trees）**

所有叶子节点使用相同的分裂条件，加速预测。

**三种梯度提升算法对比**

| 特性 | XGBoost | LightGBM | CatBoost |
|------|---------|----------|----------|
| 分裂算法 | 预排序 | 直方图 | 直方图 |
| 树生长 | Level-wise | Leaf-wise | Symmetric |
| 类别特征 | 需编码 | 需编码 | 原生支持 |
| 缺失值 | 自动处理 | 自动处理 | 自动处理 |
| 速度 | 中等 | 最快 | 快 |
| 内存 | 中等 | 最低 | 低 |
| GPU支持 | 是 | 是 | 是 |

---

### 1.5 神经网络基础

#### 1.5.1 多层感知机 (MLP)

**前向传播**

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

**激活函数**

| 激活函数 | 公式 | 导数 | 特点 |
|----------|------|------|------|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | 输出(0,1)，梯度消失 |
| Tanh | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - \tanh^2(z)$ | 输出(-1,1)，零中心化 |
| ReLU | $\text{ReLU}(z) = \max(0, z)$ | $\mathbb{1}_{z>0}$ | 计算快，可能死亡 |
| Leaky ReLU | $\max(\alpha z, z)$ | $\alpha \text{ if } z<0 \text{ else } 1$ | 解决死亡ReLU |
| GELU | $z \cdot \Phi(z)$ | 复杂 | Transformer首选 |
| Swish | $z \cdot \sigma(z)$ | 复杂 | 自门控机制 |

**反向传播（链式法则）**

$$\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{\partial L}{\partial \mathbf{z}^{[l]}} \cdot \frac{\partial \mathbf{z}^{[l]}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} \cdot (\mathbf{a}^{[l-1]})^T$$

**权重初始化**

| 方法 | 公式 | 适用场景 |
|------|------|----------|
| Xavier/Glorot | $W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}\right]$ | Tanh, Sigmoid |
| He初始化 | $W \sim \mathcal{N}(0, \sqrt{\frac{2}{n_{in}}})$ | ReLU |
| Kaiming Uniform | $W \sim U\left[-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right]$ | ReLU |

#### 1.5.2 优化算法

**SGD with Momentum**

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_\theta L(\theta_{t-1})$$

$$\theta_t = \theta_{t-1} - \alpha \mathbf{v}_t$$

**Adam (Adaptive Moment Estimation)**

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \mathbf{g}_t$$

$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) \mathbf{g}_t^2$$

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

**学习率调度**

| 策略 | 公式 | 特点 |
|------|------|------|
| Step Decay | $\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/s \rfloor}$ | 阶梯下降 |
| Exponential | $\alpha_t = \alpha_0 \cdot e^{-kt}$ | 平滑下降 |
| Cosine Annealing | $\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max}-\alpha_{min})(1+\cos(\frac{t}{T}\pi))$ | 周期性 |
| Warmup | 线性增加到初始值 | 训练初期稳定 |

---

## 2. 深度学习架构

### 2.1 卷积神经网络 (CNN)

#### 2.1.1 卷积操作

**2D卷积**

$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m, n)$$

**输出尺寸计算**

$$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$

其中：$I$=输入尺寸, $K$=核大小, $P$=填充, $S$=步长

#### 2.1.2 经典CNN架构

| 架构 | 年份 | 关键创新 | 参数量 |
|------|------|----------|--------|
| LeNet-5 | 1998 | 首个成功CNN | 60K |
| AlexNet | 2012 | ReLU+Dropout+GPU | 60M |
| VGGNet | 2014 | 3×3小卷积核 | 138M |
| ResNet | 2015 | 残差连接 | 25M-60M |
| DenseNet | 2017 | 密集连接 | 少 |
| EfficientNet | 2019 | 复合缩放 | 5M-66M |

#### 2.1.3 残差连接 (ResNet)

**残差块**

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

**为什么有效**

- 解决梯度消失：$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} (1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}})$
- 恒等映射容易学习：令 $\mathcal{F} = 0$
- 网络可以更深：ResNet-152, ResNet-1001

#### 2.1.4 批归一化 (Batch Normalization)

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

**优点**

- 加速训练收敛
- 允许更高学习率
- 减少对初始化的依赖
- 轻微正则化效果

---

### 2.2 循环神经网络 (RNN)

#### 2.2.1 基础RNN

**前向传播**

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

$$\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

**BPTT (Backpropagation Through Time)**

$$\frac{\partial L}{\partial \mathbf{W}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial \mathbf{W}}$$

**梯度问题**

$$\frac{\partial L}{\partial \mathbf{h}_1} = \frac{\partial L}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$$

- 梯度消失：$||\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}|| < 1$
- 梯度爆炸：$||\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}|| > 1$

#### 2.2.2 LSTM (Long Short-Term Memory)

**门控机制**

$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(遗忘门)}$$

$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(输入门)}$$

$$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C) \quad \text{(候选状态)}$$

$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \quad \text{(细胞状态)}$$

$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(输出门)}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad \text{(隐藏状态)}$$

#### 2.2.3 GRU (Gated Recurrent Unit)

**简化门控**

$$\mathbf{z}_t = \sigma(\mathbf{W}_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(更新门)}$$

$$\mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(重置门)}$$

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

**LSTM vs GRU对比**

| 特性 | LSTM | GRU |
|------|------|-----|
| 门数量 | 3个 (f, i, o) | 2个 (z, r) |
| 细胞状态 | 有 | 无（合并到隐藏状态） |
| 参数量 | 更多 | 更少（约25%） |
| 训练速度 | 较慢 | 较快 |
| 性能 | 相当 | 相当 |
| 适用 | 长序列 | 短序列/资源受限 |

---

### 2.3 Transformer架构

#### 2.3.1 自注意力机制 (Self-Attention)

**Scaled Dot-Product Attention**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：

- $Q \in \mathbb{R}^{n \times d_k}$：Query矩阵
- $K \in \mathbb{R}^{m \times d_k}$：Key矩阵
- $V \in \mathbb{R}^{m \times d_v}$：Value矩阵
- $\sqrt{d_k}$：缩放因子，防止softmax梯度消失

**多头注意力 (Multi-Head Attention)**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

#### 2.3.2 Transformer编码器

**结构组件**

```
Input Embedding + Positional Encoding
        ↓
    [Multi-Head Attention]
        ↓
    [Add & Norm] (残差连接 + Layer Norm)
        ↓
    [Feed Forward]
        ↓
    [Add & Norm]
        ↓
    (重复N次)
```

**位置编码**

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**Layer Normalization**

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

#### 2.3.3 Transformer解码器

**结构组件**

```
Output Embedding + Positional Encoding
        ↓
    [Masked Multi-Head Attention] (自回归掩码)
        ↓
    [Add & Norm]
        ↓
    [Multi-Head Attention] (编码器-解码器注意力)
        ↓
    [Add & Norm]
        ↓
    [Feed Forward]
        ↓
    [Add & Norm]
        ↓
    (重复N次)
        ↓
    [Linear + Softmax]
```

**因果/掩码注意力**

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M_{ij} = \begin{cases} 0 & i \geq j \\ -\infty & i < j \end{cases}$（上三角掩码）

#### 2.3.4 BERT (Bidirectional Encoder Representations)

**预训练任务**

1. **Masked Language Model (MLM)**
   - 随机mask 15%的token
   - 预测被mask的token
   - 损失函数：$\mathcal{L}_{MLM} = -\mathbb{E}_{x \in \mathcal{D}} \log P(x_{masked} | x_{\backslash masked})$

2. **Next Sentence Prediction (NSP)**
   - 判断句子B是否是句子A的下一句
   - 50%正例，50%负例

**BERT变体**

| 模型 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|----------|----------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

**微调策略**

```python
# 分类任务
[CLS] token output → Linear → Softmax

# 序列标注
每个token output → Linear per token → Softmax

# 问答
Start/End logits → Softmax over sequence
```

#### 2.3.5 GPT系列 (Generative Pre-trained Transformer)

**架构特点**

- 仅使用Transformer解码器
- 因果/自回归语言建模
- 从左到右的单向注意力

**GPT-2/GPT-3规模**

| 模型 | 层数 | 维度 | 头数 | 参数量 |
|------|------|------|------|--------|
| GPT-2 Small | 12 | 768 | 12 | 117M |
| GPT-2 Medium | 24 | 1024 | 16 | 345M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 175B |

**GPT-3上下文学习**

$$P(y|x, C) = \prod_{i=1}^{N} P(y_i | y_{<i}, x, C)$$

其中 $C$ 是上下文示例（In-context learning）。

#### 2.3.6 Vision Transformer (ViT)

**图像分块**

$$\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$$

其中：$N = HW/P^2$ 是patch数量，$P$ 是patch大小，$C$ 是通道数

**架构**

```
Image → Patch Embedding + [CLS] token + Position Embedding
                    ↓
            Transformer Encoder × L
                    ↓
            [CLS] token → MLP Head → Class
```

**ViT vs CNN对比**

| 特性 | ViT | CNN |
|------|-----|-----|
| 归纳偏置 | 较少（需大量数据） | 平移等变性 |
| 全局感受野 | 第一层即全局 | 逐层扩大 |
| 数据效率 | 需要大数据集 | 数据效率高 |
| 计算效率 | 大图像更优 | 小图像更优 |
| 可解释性 | 注意力可视化 | 特征可视化 |

---


## 3. 无监督学习方法

### 3.1 聚类算法

#### 3.1.1 K-Means聚类

**目标函数**

$$J = \sum_{i=1}^{k} \sum_{\mathbf{x} \in C_i} ||\mathbf{x} - \boldsymbol{\mu}_i||^2$$

其中 $C_i$ 是第 $i$ 个簇，$\boldsymbol{\mu}_i$ 是簇中心。

**Lloyd算法**

```
1. 随机初始化k个簇中心
2. 重复直到收敛：
   a. 分配步骤：将每个点分配到最近的中心
      C_i = {x : ||x - μ_i|| ≤ ||x - μ_j||, ∀j}
   b. 更新步骤：重新计算簇中心
      μ_i = (1/|C_i|) Σ_{x∈C_i} x
```

**K-Means++初始化**

$$D(\mathbf{x})^2 = \min_{\boldsymbol{\mu} \in \mathcal{M}} ||\mathbf{x} - \boldsymbol{\mu}||^2$$

选择下一个中心的概率：$P(\mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}'} D(\mathbf{x}')^2}$

**时间复杂度**：$O(n \cdot k \cdot d \cdot i)$，其中 $i$ 是迭代次数

#### 3.1.2 DBSCAN (Density-Based Spatial Clustering)

**核心概念**

- **核心点**：$\epsilon$邻域内至少有 $MinPts$ 个点
- **边界点**：在核心点的 $\epsilon$ 邻域内但自身不是核心点
- **噪声点**：既不是核心点也不是边界点

**算法流程**

```
1. 标记所有点为核心点、边界点或噪声点
2. 删除噪声点
3. 为距离在ε内的核心点之间创建边
4. 每个连通分量形成一个簇
5. 将边界点分配给相邻核心点的簇
```

**参数选择**

| 参数 | 作用 | 选择方法 |
|------|------|----------|
| ε | 邻域半径 | k-distance图拐点 |
| MinPts | 最小点数 | 通常 ≥ 维度+1 |

**K-Means vs DBSCAN对比**

| 特性 | K-Means | DBSCAN |
|------|---------|--------|
| 簇形状 | 球形 | 任意形状 |
| 簇数量 | 需预设 | 自动确定 |
| 噪声处理 | 无 | 有 |
| 可扩展性 | 好 | 中等 |
| 对异常值敏感 | 是 | 否 |
| 时间复杂度 | O(n) | O(n log n) |

#### 3.1.3 层次聚类

**凝聚式（Agglomerative）**

```
1. 每个点作为一个簇
2. 重复直到只剩一个簇：
   a. 计算所有簇对之间的距离
   b. 合并距离最近的两个簇
```

**链接准则**

| 方法 | 距离定义 | 特点 |
|------|----------|------|
| 单链接 | $d(C_i, C_j) = \min_{x∈C_i, y∈C_j} d(x,y)$ | 可发现非球形簇，链式效应 |
| 全链接 | $d(C_i, C_j) = \max_{x∈C_i, y∈C_j} d(x,y)$ | 紧凑簇，对噪声敏感 |
| 平均链接 | $d(C_i, C_j) = \frac{1}{|C_i||C_j|}\sum_{x∈C_i}\sum_{y∈C_j} d(x,y)$ | 平衡方案 |
| Ward | 最小化合并后的方差增加 | 倾向于等大小簇 |

**树状图（Dendrogram）**

通过切割树状图在不同高度得到不同数量的簇。

#### 3.1.4 高斯混合模型 (GMM)

**概率模型**

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

其中：

- $\pi_k$：混合系数，$\sum_k \pi_k = 1$
- $\mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$：第 $k$ 个高斯分布

**EM算法**

**E步（期望）**：计算后验概率（责任）

$$\gamma(z_{nk}) = \frac{\pi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**M步（最大化）**：更新参数

$$N_k = \sum_{n=1}^{N} \gamma(z_{nk})$$

$$\boldsymbol{\mu}_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) \mathbf{x}_n$$

$$\boldsymbol{\Sigma}_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T$$

$$\pi_k^{new} = \frac{N_k}{N}$$

**K-Means vs GMM**

| 特性 | K-Means | GMM |
|------|---------|-----|
| 簇形状 | 球形 | 椭圆形 |
| 软分配 | 否 | 是（概率） |
| 协方差 | 各向同性 | 可学习 |
| 计算成本 | 低 | 高 |
| 收敛保证 | 是 | 局部最优 |

---

### 3.2 降维方法

#### 3.2.1 主成分分析 (PCA)

**目标**

找到投影方向 $\mathbf{w}$ 使得投影后的方差最大：

$$\max_{\mathbf{w}} \mathbf{w}^T \mathbf{S} \mathbf{w} \quad \text{s.t.} \quad \mathbf{w}^T \mathbf{w} = 1$$

其中 $\mathbf{S} = \frac{1}{N} \sum_{n=1}^{N} (\mathbf{x}_n - \bar{\mathbf{x}})(\mathbf{x}_n - \bar{\mathbf{x}})^T$ 是协方差矩阵。

**求解**

PCA解是协方差矩阵的前 $k$ 个最大特征值对应的特征向量：

$$\mathbf{S}\mathbf{w}_i = \lambda_i \mathbf{w}_i, \quad \lambda_1 \geq \lambda_2 \geq ... \geq \lambda_d$$

**投影与重构**

$$\mathbf{z} = \mathbf{W}^T (\mathbf{x} - \bar{\mathbf{x}})$$

$$\hat{\mathbf{x}} = \mathbf{W}\mathbf{z} + \bar{\mathbf{x}}$$

**方差保留率**

$$\text{保留方差比例} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

**SVD实现**

$$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

PCA基向量是 $\mathbf{V}$ 的列。

#### 3.2.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

**高维空间相似度**

$$p_{j|i} = \frac{\exp(-||\mathbf{x}_i - \mathbf{x}_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||\mathbf{x}_i - \mathbf{x}_k||^2 / 2\sigma_i^2)}$$

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$

**低维空间相似度（t分布）**

$$q_{ij} = \frac{(1 + ||\mathbf{y}_i - \mathbf{y}_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||\mathbf{y}_k - \mathbf{y}_l||^2)^{-1}}$$

**KL散度目标**

$$C = KL(P||Q) = \sum_{i} \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**梯度**

$$\frac{\partial C}{\partial \mathbf{y}_i} = 4 \sum_{j} (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + ||\mathbf{y}_i - \mathbf{y}_j||^2)^{-1}$$

**关键参数**

| 参数 | 作用 | 建议值 |
|------|------|--------|
| perplexity | 有效邻居数 | 5-50 |
| learning_rate | 梯度步长 | 10-1000 |
| n_iter | 迭代次数 | 1000+ |
| early_exaggeration | 早期放大 | 4-12 |

#### 3.2.3 UMAP (Uniform Manifold Approximation and Projection)

**核心思想**

假设数据均匀分布在黎曼流形上，寻找保持拓扑结构的低维表示。

**高维相似度**

$$p_{j|i} = \exp\left(\frac{-d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i}{\sigma_i}\right)$$

其中 $\rho_i$ 是到最近邻居的距离。

**低维相似度**

$$q_{ij} = \left(1 + a||\mathbf{y}_i - \mathbf{y}_j||^{2b}\right)^{-1}$$

**交叉熵损失**

$$C = \sum_{i,j} \left[ p_{ij} \log\frac{p_{ij}}{q_{ij}} + (1-p_{ij}) \log\frac{1-p_{ij}}{1-q_{ij}} \right]$$

**t-SNE vs UMAP对比**

| 特性 | t-SNE | UMAP |
|------|-------|------|
| 速度 | 慢 | 快 |
| 全局结构 | 差 | 好 |
| 可重复性 | 随机初始化 | 更稳定 |
| 超参数敏感 | 是 | 是 |
| 新数据投影 | 不支持 | 支持（transform） |

#### 3.2.4 自编码器 (Autoencoder)

**架构**

```
Input x → Encoder → Latent z → Decoder → Reconstruction x̂
```

**损失函数**

$$L(\mathbf{x}, \hat{\mathbf{x}}) = ||\mathbf{x} - \hat{\mathbf{x}}||^2 = ||\mathbf{x} - f_{decoder}(f_{encoder}(\mathbf{x}))||^2$$

**变体**

| 变体 | 特点 | 应用 |
|------|------|------|
| Denoising AE | 输入加噪声 | 特征学习 |
| Sparse AE | 稀疏约束 | 特征选择 |
| Contractive AE | 雅可比惩罚 | 鲁棒表示 |
| VAE | 概率编码 | 生成模型 |

---

### 3.3 异常检测

#### 3.3.1 统计方法

**Z-Score**

$$z = \frac{x - \mu}{\sigma}$$

$|z| > 3$ 视为异常

**IQR方法**

$$IQR = Q_3 - Q_1$$

异常值：$x < Q_1 - 1.5 \cdot IQR$ 或 $x > Q_3 + 1.5 \cdot IQR$

#### 3.3.2 基于距离的方法

**LOF (Local Outlier Factor)**

$$LOF_k(p) = \frac{\sum_{o \in N_k(p)} \frac{lrd_k(o)}{lrd_k(p)}}{|N_k(p)|}$$

其中 $lrd$ 是局部可达密度：

$$lrd_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} reach\text{-}dist_k(p, o)}$$

**孤立森林 (Isolation Forest)**

异常点更容易被孤立：

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

其中 $h(x)$ 是孤立 $x$ 所需的平均路径长度。

#### 3.3.3 基于重构的方法

**Autoencoder异常检测**

$$\text{Anomaly Score}(\mathbf{x}) = ||\mathbf{x} - \hat{\mathbf{x}}||^2$$

**PCA异常检测**

$$\text{Anomaly Score}(\mathbf{x}) = ||\mathbf{x} - \mathbf{W}\mathbf{W}^T\mathbf{x}||^2$$

---

## 4. 强化学习

### 4.1 基础概念

**马尔可夫决策过程 (MDP)**

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

- $\mathcal{S}$：状态空间
- $\mathcal{A}$：动作空间
- $\mathcal{P}(s'|s,a)$：状态转移概率
- $\mathcal{R}(s,a)$：奖励函数
- $\gamma \in [0,1]$：折扣因子

**价值函数**

状态价值函数：

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s\right]$$

动作价值函数：

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t r_t | s_0 = s, a_0 = a\right]$$

**贝尔曼方程**

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')]$$

$$Q^\pi(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

**最优价值函数**

$$V^*(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \max_{a'} Q^*(s',a')]$$

### 4.2 基于价值的方法

#### 4.2.1 Q-Learning

**更新规则**

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**收敛条件**：所有状态-动作对无限次访问，$\alpha$ 逐渐减小。

#### 4.2.2 DQN (Deep Q-Network)

**损失函数**

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D} \left[\left(r + \gamma \max_{a'} Q(s',a'; \theta^-) - Q(s,a; \theta)\right)^2\right]$$

**关键技术**

| 技术 | 作用 |
|------|------|
| 经验回放 | 打破样本相关性 |
| 目标网络 | 稳定学习目标 |
| ε-贪婪 | 探索-利用平衡 |

**Double DQN**

解决Q值过估计问题：

$$Y_t = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-)$$

**Dueling DQN**

分离状态价值和优势：

$$Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')$$

架构：

```
Input → Conv Layers → FC Layers
                      ↓
              ┌───────┴───────┐
              ↓               ↓
         Value Stream    Advantage Stream
              ↓               ↓
            V(s)           A(s,a)
              └───────┬───────┘
                      ↓
                  Q(s,a) = V(s) + A(s,a) - mean(A)
```

### 4.3 策略梯度方法

#### 4.3.1 策略梯度定理

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\right]$$

#### 4.3.2 REINFORCE

**梯度估计**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是累积回报。

**带基线的REINFORCE**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))\right]$$

#### 4.3.3 Actor-Critic

**架构**

- **Actor**：策略网络 $\pi_\theta(a|s)$
- **Critic**：价值网络 $V_w(s)$ 或 $Q_w(s,a)$

**更新规则**

Critic更新（TD误差）：

$$\delta_t = r_t + \gamma V_w(s_{t+1}) - V_w(s_t)$$

$$w \leftarrow w + \alpha_w \delta_t \nabla_w V_w(s_t)$$

Actor更新：

$$\theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(a_t|s_t)$$

#### 4.3.4 A3C (Asynchronous Advantage Actor-Critic)

**异步训练**

- 多个worker并行
- 每个worker独立探索
- 定期同步全局网络

**优势函数**

$$A(s_t, a_t) = Q(s_t, a_t) - V(s_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$

#### 4.3.5 PPO (Proximal Policy Optimization)

**目标函数**

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是重要性采样比率。

**完整目标**

$$L^{CLIP+VF+S}(\theta) = \mathbb{E}_t \left[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\right]$$

**PPO vs TRPO**

| 特性 | TRPO | PPO |
|------|------|-----|
| 约束 | KL散度约束 | 裁剪目标 |
| 计算 | 复杂（共轭梯度） | 简单（SGD） |
| 实现难度 | 高 | 低 |
| 性能 | 好 | 相当/更好 |

#### 4.3.6 SAC (Soft Actor-Critic)

**最大熵强化学习**

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$

**软Q函数**

$$Q_{soft}^\pi(s,a) = r(s,a) + \gamma \mathbb{E}_{s' \sim p}[V_{soft}^\pi(s')]$$

**软价值函数**

$$V_{soft}^\pi(s) = \mathbb{E}_{a \sim \pi}\left[Q_{soft}^\pi(s,a) - \alpha \log \pi(a|s)\right]$$

**策略优化**

$$\pi^* = \arg\min_\pi D_{KL}\left(\pi(\cdot|s) \Big\| \frac{\exp(Q_{soft}^\pi(s,\cdot)/\alpha)}{Z_{soft}^\pi(s)}\right)$$

### 4.4 基于模型的强化学习

#### 4.4.1 模型学习

学习环境模型 $\hat{p}(s', r | s, a)$：

$$L_{model} = \mathbb{E}_{(s,a,s',r) \sim D}\left[||s' - \hat{f}(s,a)||^2 + (r - \hat{r}(s,a))^2\right]$$

#### 4.4.2 Dyna-Q

```
1. 从真实环境采样 (s, a, r, s')
2. 更新Q值：Q(s,a) += α[r + γ max_a' Q(s',a') - Q(s,a)]
3. 学习模型：存储/更新模型 p̂(s'|s,a), r̂(s,a)
4. 从模型采样n次，更新Q值（规划）
5. 重复
```

#### 4.4.3 MBPO (Model-Based Policy Optimization)

**伪代码**

```
1. 初始化策略 π_θ，模型 p̂_φ
2. 从真实环境收集初始数据 D_env
3. 循环：
   a. 在 D_env 上训练模型 p̂_φ
   b. 从模型生成k步rollouts，存入 D_model
   c. 在 D_env ∪ D_model 上训练策略 π_θ
   d. 在真实环境执行 π_θ，收集数据加入 D_env
```

**模型集成**

使用多个模型减少模型误差：

$$p̂(s'|s,a) = \frac{1}{M} \sum_{i=1}^{M} p̂_i(s'|s,a)$$

---


## 5. 生成式AI

### 5.1 变分自编码器 (VAE)

#### 5.1.1 基本架构

**编码器**：$q_\phi(\mathbf{z}|\mathbf{x})$（近似后验）

**解码器**：$p_\theta(\mathbf{x}|\mathbf{z})$（似然）

**重参数化技巧**

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

#### 5.1.2 ELBO (Evidence Lower Bound)

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - KL(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$$

**重构项**：$\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]$

**KL散度项**：$KL(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})) = \frac{1}{2}\sum_{j=1}^{J} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)$

#### 5.1.3 VAE变体

| 变体 | 特点 | 应用 |
|------|------|------|
| β-VAE | 加权KL项 | 解耦表示学习 |
| CVAE | 条件VAE | 条件生成 |
| VQ-VAE | 离散隐变量 | 图像/音频生成 |
| Hierarchical VAE | 多层隐变量 | 高质量生成 |

### 5.2 生成对抗网络 (GAN)

#### 5.2.1 基本框架

**minimax游戏**

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

**最优判别器**

$$D^*(\mathbf{x}) = \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}$$

**最优生成器**

当 $p_g = p_{data}$ 时，$D^*(\mathbf{x}) = 1/2$

#### 5.2.2 训练技巧

| 技巧 | 说明 |
|------|------|
| 标签平滑 | 使用0.9而非1作为真实标签 |
| 噪声输入 | 给判别器输入添加噪声 |
| 历史平均 | 使用生成器参数的历史平均 |
| 谱归一化 | 限制判别器的Lipschitz常数 |

#### 5.2.3 DCGAN (Deep Convolutional GAN)

**架构原则**

- 用步长卷积替代池化（生成器）
- 使用BatchNorm
- 移除全连接隐藏层
- 生成器使用ReLU，输出层使用Tanh
- 判别器使用LeakyReLU

#### 5.2.4 StyleGAN

**映射网络**

$$\mathbf{w} = f(\mathbf{z}), \quad \mathbf{w} \in \mathcal{W}$$

**自适应实例归一化 (AdaIN)**

$$\text{AdaIN}(\mathbf{x}_i, \mathbf{y}) = \mathbf{y}_{s,i} \cdot \frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} + \mathbf{y}_{b,i}$$

**渐进式增长**

从低分辨率开始，逐步增加层数以生成更高分辨率图像。

#### 5.2.5 CycleGAN

**循环一致性损失**

$$\mathcal{L}_{cyc}(G, F) = \mathbb{E}_{\mathbf{x}}[||F(G(\mathbf{x})) - \mathbf{x}||_1] + \mathbb{E}_{\mathbf{y}}[||G(F(\mathbf{y})) - \mathbf{y}||_1]$$

**完整目标**

$$\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{GAN}(G, D_Y) + \mathcal{L}_{GAN}(F, D_X) + \lambda \mathcal{L}_{cyc}(G, F)$$

### 5.3 扩散模型 (Diffusion Models)

#### 5.3.1 前向扩散过程

**马尔可夫链**

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

**任意时刻的闭式解**

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

**重参数化**

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

#### 5.3.2 反向去噪过程

**学习目标**

$$\mathcal{L}_{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)||^2\right]$$

**等价形式**

$$\mathcal{L}_{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[||\mathbf{x}_0 - \mathbf{x}_\theta(\mathbf{x}_t, t)||^2\right]$$

#### 5.3.3 DDPM (Denoising Diffusion Probabilistic Models)

**训练算法**

```
重复直到收敛：
  1. 采样 x_0 ~ q(x_0)
  2. 采样 t ~ Uniform({1, ..., T})
  3. 采样 ε ~ N(0, I)
  4. 计算梯度下降：∇_θ ||ε - ε_θ(√ᾱ_t x_0 + √(1-ᾱ_t)ε, t)||²
```

**采样算法**

```
1. 采样 x_T ~ N(0, I)
2. 对于 t = T, ..., 1：
   a. 如果 t > 1，采样 z ~ N(0, I)，否则 z = 0
   b. x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))ε_θ(x_t, t)) + σ_t z
3. 返回 x_0
```

#### 5.3.4 Stable Diffusion / Latent Diffusion

**核心思想**

在潜在空间而非像素空间进行扩散：

$$\mathcal{L}_{LDM} = \mathbb{E}_{\mathcal{E}(\mathbf{x}), \boldsymbol{\epsilon} \sim \mathcal{N}(0,1), t}\left[||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)||^2\right]$$

其中 $\mathbf{z} = \mathcal{E}(\mathbf{x})$ 是编码器输出的潜在表示。

**条件生成**

$$\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{y})$$

通过cross-attention机制注入条件信息 $\mathbf{y}$（文本、类别等）。

**架构**

```
Text Encoder (CLIP) → Condition Embedding
                              ↓
Image → VAE Encoder → Latent Space → U-Net with Cross-Attention → Denoised Latent
                              ↓
                         VAE Decoder → Generated Image
```

#### 5.3.5 扩散模型 vs GAN vs VAE

| 特性 | GAN | VAE | Diffusion |
|------|-----|-----|-----------|
| 训练稳定性 | 差 | 好 | 好 |
| 生成质量 | 高 | 中等 | 高 |
| 多样性 | 可能模式坍塌 | 好 | 好 |
| 采样速度 | 快（单次前向） | 快 | 慢（多步迭代） |
| 条件控制 | 需要额外设计 | 需要条件VAE | 天然支持 |
| 可逆性 | 否 | 是 | 是 |
| 隐空间 | 无 | 有 | 有（噪声） |

### 5.4 流模型 (Flow-based Models)

#### 5.4.1 可逆变换

**变量替换公式**

$$p_x(\mathbf{x}) = p_z(f(\mathbf{x})) \left| \det \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} \right|$$

**对数似然**

$$\log p_x(\mathbf{x}) = \log p_z(f(\mathbf{x})) + \log \left| \det \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} \right|$$

#### 5.4.2 RealNVP

**仿射耦合层**

$$\mathbf{y}_{1:d} = \mathbf{x}_{1:d}$$

$$\mathbf{y}_{d+1:D} = \mathbf{x}_{d+1:D} \odot \exp(s(\mathbf{x}_{1:d})) + t(\mathbf{x}_{1:d})$$

**雅可比行列式**

$$\det \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \exp\left(\sum_j s(\mathbf{x}_{1:d})_j\right)$$

#### 5.4.3 Glow

**1×1可逆卷积**

$$\mathbf{y}_{i,j} = \mathbf{W} \mathbf{x}_{i,j}$$

雅可比行列式：$\det \mathbf{W}^h$（$h \times w$ 空间维度）

**可分离卷积**

结合actnorm、可逆1×1卷积和仿射耦合层。

### 5.5 大语言模型 (LLM)

#### 5.5.1 规模定律 (Scaling Laws)

**性能与规模的关系**

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

其中：

- $N$：模型参数量
- $D$：训练token数
- $N_c, D_c$：临界值
- $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$

**Chinchilla最优**

$$N_{opt} \propto D^{0.5}, \quad D_{opt} \propto N^{0.5}$$

对于给定计算量 $C = 6ND$，最优配置满足 $N \approx D^{0.5}$。

#### 5.5.2 预训练策略

**自回归语言建模**

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

**掩码语言建模（BERT风格）**

$$\mathcal{L} = -\mathbb{E}_{\mathbf{x} \sim D} \sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\backslash \mathcal{M}}; \theta)$$

**Prefix LM（T5风格）**

$$\mathcal{L} = -\sum_{t=|prefix|+1}^{T} \log P(x_t | x_{\leq t-1}; \theta)$$

#### 5.5.3 微调方法

**全参数微调 (Full Fine-tuning)**

更新所有参数：$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{task}$

**参数高效微调 (PEFT)**

| 方法 | 原理 | 可训练参数比例 |
|------|------|----------------|
| LoRA | 低秩适配 $W' = W + BA$ | 0.1-1% |
| Prefix Tuning | 训练前缀嵌入 | 0.1% |
| Prompt Tuning | 训练软提示 | <0.01% |
| Adapter | 插入小型适配器层 | 1-5% |

**LoRA (Low-Rank Adaptation)**

$$W' = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$

#### 5.5.4 指令微调 (Instruction Tuning)

**格式**

```
指令：{instruction}
输入：{input}
输出：{output}
```

**FLAN (Fine-tuned LAnguage Net)**

将任务转化为自然语言指令格式进行微调。

#### 5.5.5 RLHF (Reinforcement Learning from Human Feedback)

**三阶段流程**

```
1. 预训练 → SFT (Supervised Fine-Tuning)
                    ↓
2. 训练奖励模型 RM(s,a) = E[human preference]
                    ↓
3. PPO优化：max E[RM(s,a)] - β KL(π||π_ref)
```

**PPO目标**

$$\mathcal{L}_{PPO} = \mathbb{E}_{(s,a) \sim \pi_\theta}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)} A(s,a), \text{clip}(...) A(s,a)\right)\right]$$

**KL惩罚**

$$\mathcal{L} = \mathcal{L}_{PPO} - \beta \mathbb{E}_{s \sim D}\left[KL(\pi_\theta(\cdot|s) || \pi_{ref}(\cdot|s))\right]$$

#### 5.5.6 LLM推理优化

**KV Cache**

缓存键值对避免重复计算：

$$\text{Attention}(Q, K_{cache}, V_{cache})$$

**量化**

| 方法 | 精度 | 压缩比 | 质量损失 |
|------|------|--------|----------|
| INT8 | 8-bit | 2x | 极小 |
| INT4 | 4-bit | 4x | 小 |
| GPTQ | 4-bit | 4x | 小 |
| AWQ | 4-bit | 4x | 很小 |

**投机解码 (Speculative Decoding)**

使用小模型生成候选token，大模型并行验证。

---


## 6. 建模方法论

### 6.1 特征工程

#### 6.1.1 特征类型

| 类型 | 描述 | 处理方法 |
|------|------|----------|
| 数值特征 | 连续或离散数值 | 归一化、标准化、分箱 |
| 类别特征 | 有限离散值 | One-hot, Label, Target编码 |
| 文本特征 | 字符串 | TF-IDF, Word2Vec, BERT |
| 时间特征 | 日期时间 | 提取年/月/日/小时等 |
| 地理特征 | 坐标/位置 | 距离计算、聚类 |

#### 6.1.2 数值特征处理

**标准化 (Standardization)**

$$x' = \frac{x - \mu}{\sigma}$$

**归一化 (Min-Max Scaling)**

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Robust Scaling**

$$x' = \frac{x - \text{median}}{IQR}$$

**对数变换**

$$x' = \log(x + 1)$$

适用于右偏分布。

**分箱 (Binning)**

- 等宽分箱：$bin_i = [x_{min} + i \cdot w, x_{min} + (i+1) \cdot w)$
- 等频分箱：每个bin样本数相等
- 基于聚类的分箱

#### 6.1.3 类别特征编码

| 编码方法 | 公式 | 适用场景 |
|----------|------|----------|
| One-Hot | $n$ 维向量，第 $i$ 位为1 | 低基数类别 |
| Label | 整数映射 | 树模型 |
| Ordinal | 有序映射 | 有序类别 |
| Target | $x_i' = P(y=1|x=x_i)$ | 高基数类别 |
| Frequency | $x_i' = count(x_i)$ | 通用 |

**Target Encoding**

$$\hat{x}_i = \frac{\sum_{j \in D_{train}} \mathbb{1}_{x_j = x_i} \cdot y_j + \alpha \cdot \bar{y}}{\sum_{j \in D_{train}} \mathbb{1}_{x_j = x_i} + \alpha}$$

其中 $\alpha$ 是平滑参数，$\bar{y}$ 是全局均值。

#### 6.1.4 特征交互

**多项式特征**

$$\phi(\mathbf{x}) = [1, x_1, x_2, ..., x_1^2, x_1x_2, ..., x_n^d]$$

**特征交叉**

$$x_{ij} = x_i \times x_j$$

**自动特征工程**

- Featuretools：自动特征合成
- AutoCross：自动交叉特征搜索

#### 6.1.5 特征选择

**过滤法 (Filter)**

| 方法 | 度量 | 适用 |
|------|------|------|
| 方差阈值 | $Var(X)$ | 去除低方差特征 |
| 相关系数 | $\rho(X, y)$ | 线性关系 |
| 互信息 | $I(X; Y)$ | 非线性关系 |
| 卡方检验 | $\chi^2$ | 分类特征 |
| F检验 | ANOVA F-value | 数值特征 |

**包装法 (Wrapper)**

- 前向选择：从空集开始，逐步添加最优特征
- 后向消除：从全集开始，逐步移除最差特征
- 递归特征消除 (RFE)

**嵌入法 (Embedded)**

- L1正则化（Lasso）：自动特征选择
- 树模型的特征重要性

### 6.2 模型选择策略

#### 6.2.1 问题类型与模型选择

| 问题类型 | 推荐模型 | 备选方案 |
|----------|----------|----------|
| 二分类（小数据） | 逻辑回归、SVM | 随机森林、XGBoost |
| 二分类（大数据） | XGBoost、LightGBM | 神经网络 |
| 多分类 | XGBoost、神经网络 | 随机森林 |
| 回归（线性关系） | 线性回归、Ridge | Elastic Net |
| 回归（非线性） | XGBoost、神经网络 | 随机森林 |
| 时间序列 | ARIMA、LSTM | Prophet、Transformer |
| 图像分类 | CNN、ViT | ResNet、EfficientNet |
| NLP | Transformer、BERT | LSTM+Attention |
| 推荐系统 | 矩阵分解、DeepFM | 双塔模型 |

#### 6.2.2 偏差-方差权衡

**误差分解**

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{(f(x) - \mathbb{E}[\hat{f}(x)])^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{Variance}} + \underbrace{\sigma^2_\epsilon}_{\text{Noise}}$$

**诊断方法**

| 情况 | 训练误差 | 验证误差 | 解决方案 |
|------|----------|----------|----------|
| 高偏差 | 高 | 高 | 增加模型复杂度、更多特征 |
| 高方差 | 低 | 高 | 正则化、更多数据、简化模型 |
| 正常 | 低 | 略高 | 模型合适 |

#### 6.2.3 学习曲线分析

```python
# 伪代码
for n in [100, 200, 500, 1000, 5000, 10000]:
    X_train_n = X_train[:n]
    y_train_n = y_train[:n]

    model.fit(X_train_n, y_train_n)
    train_score = model.score(X_train_n, y_train_n)
    val_score = model.score(X_val, y_val)

    plot(n, train_score, 'train')
    plot(n, val_score, 'val')
```

**曲线解读**

- 训练误差 ↑，验证误差 ↓ → 需要更多数据
- 两者差距大 → 高方差（过拟合）
- 两者都高 → 高偏差（欠拟合）

### 6.3 超参数优化

#### 6.3.1 网格搜索 (Grid Search)

```python
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3]
}
# 总组合数：3 × 4 × 3 = 36
```

**复杂度**：$O(\prod_{i=1}^{k} n_i)$，$n_i$ 是第 $i$ 个参数的选择数

#### 6.3.2 随机搜索 (Random Search)

```python
param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.001, 0.5)
}
# 采样n_iter次
```

**优势**

- 在相同预算下探索更多参数空间
- 对重要参数更有效

#### 6.3.3 贝叶斯优化

**高斯过程代理模型**

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

**采集函数**

| 采集函数 | 公式 | 特点 |
|----------|------|------|
| EI (Expected Improvement) | $\mathbb{E}[\max(0, f(x^*) - f(x))]$ | 平衡探索-利用 |
| PI (Probability of Improvement) | $P(f(x) \geq f(x^+) + \xi)$ | 偏向利用 |
| UCB (Upper Confidence Bound) | $\mu(x) + \kappa \sigma(x)$ | 显式平衡 |

**TPE (Tree-structured Parzen Estimator)**

$$p(\mathbf{x}|y) = \begin{cases} l(\mathbf{x}) & \text{if } y < y^* \\ g(\mathbf{x}) & \text{if } y \geq y^* \end{cases}$$

**优化库对比**

| 库 | 算法 | 特点 |
|----|------|------|
| Optuna | TPE, CMA-ES | 高效、易用 |
| Hyperopt | TPE, Random | 成熟稳定 |
| Ray Tune | 多种 | 分布式支持 |
| Scikit-optimize | GP, RF | 轻量级 |

#### 6.3.4 早停与剪枝

**Hyperband**

1. 随机采样配置
2. 用少量资源快速评估
3. 淘汰表现差的配置
4. 对优秀配置分配更多资源

**Successive Halving**

$$r_i = r_{min} \cdot \eta^i, \quad n_i = \lfloor n_{max} / \eta^i \rfloor$$

### 6.4 交叉验证与模型评估

#### 6.4.1 交叉验证方法

**K折交叉验证**

$$CV_{(k)} = \frac{1}{k} \sum_{i=1}^{k} \mathcal{L}(\mathcal{A}(D_{-i}), D_i)$$

**分层K折**

保持每折中类别比例与整体一致。

**留一法 (LOOCV)**

$$CV_{(n)} = \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(\mathcal{A}(D_{-i}), (x_i, y_i))$$

**时间序列交叉验证**

```
Fold 1: [train] [val]
Fold 2: [train    ] [val]
Fold 3: [train        ] [val]
```

确保验证集始终在训练集之后。

#### 6.4.2 分类评估指标

| 指标 | 公式 | 适用场景 |
|------|------|----------|
| 准确率 | $\frac{TP+TN}{TP+TN+FP+FN}$ | 平衡数据集 |
| 精确率 | $\frac{TP}{TP+FP}$ | 关注假阳性 |
| 召回率 | $\frac{TP}{TP+FN}$ | 关注假阴性 |
| F1-Score | $\frac{2 \cdot P \cdot R}{P + R}$ | 平衡P和R |
| AUC-ROC | ROC曲线下面积 | 排序能力 |
| AUC-PR | PR曲线下面积 | 不平衡数据 |
| Log Loss | $-\sum y \log(\hat{y})$ | 概率校准 |

**混淆矩阵**

| 实际\预测 | 正类 | 负类 |
|-----------|------|------|
| 正类 | TP | FN |
| 负类 | FP | TN |

**多分类指标**

- Macro：各类别指标的平均
- Micro：全局计算
- Weighted：按支持度加权平均

#### 6.4.3 回归评估指标

| 指标 | 公式 | 特点 |
|------|------|------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 对大误差敏感 |
| RMSE | $\sqrt{MSE}$ | 与目标同量纲 |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 鲁棒 |
| MAPE | $\frac{100\%}{n}\sum|\frac{y_i - \hat{y}_i}{y_i}|$ | 相对误差 |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | 解释方差比例 |
| Huber Loss | 混合MSE和MAE | 鲁棒且可导 |

#### 6.4.4 模型校准

**可靠性图**

将预测概率分桶，比较平均预测概率与实际准确率。

**Platt Scaling**

$$P(y=1|\mathbf{x}) = \sigma(a \cdot f(\mathbf{x}) + b)$$

**Isotonic Regression**

非参数单调校准：$\hat{p} = g(f(\mathbf{x}))$，其中 $g$ 是单调函数。

### 6.5 集成学习方法

#### 6.5.1 Bagging

**Bootstrap采样**

$$P(\text{样本}i\text{在bootstrap中}) = 1 - (1 - \frac{1}{n})^n \approx 0.632$$

**Out-of-Bag估计**

$$\widehat{Err}_{oob} = \frac{1}{n} \sum_{i=1}^{n} L(y_i, \hat{f}^{-i}(x_i))$$

#### 6.5.2 Boosting

**AdaBoost**

权重更新：

$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(\alpha_t \cdot \mathbb{1}_{y_i \neq h_t(x_i)})$$

其中 $\alpha_t = \frac{1}{2} \ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)$

**Gradient Boosting**

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

其中 $h_m$ 拟合负梯度：$-g_m(x) = -\left[\frac{\partial L(y, F(x))}{\partial F(x)}\right]_{F=F_{m-1}}$

#### 6.5.3 Stacking

**两层架构**

```
Level 0 (Base Models):
  - Model A: Random Forest
  - Model B: XGBoost
  - Model C: Neural Network
              ↓
        Meta-features
              ↓
Level 1 (Meta Learner):
  - Logistic Regression / Linear Regression
```

**防止数据泄露**

使用K折交叉验证生成元特征：

```python
for fold in k_folds:
    base_model.fit(train[fold])
    meta_features[val[fold]] = base_model.predict(val[fold])
```

#### 6.5.4 集成策略对比

| 方法 | 基学习器 | 训练方式 | 代表算法 |
|------|----------|----------|----------|
| Bagging | 同质，强学习器 | 并行，Bootstrap | Random Forest |
| Boosting | 同质，弱学习器 | 串行，加权 | XGBoost, AdaBoost |
| Stacking | 异质 | 并行+元学习器 | 任意组合 |
| Voting | 异质 | 并行 | 简单平均/加权 |

### 6.6 模型部署与监控

#### 6.6.1 模型序列化

| 格式 | 适用框架 | 特点 |
|------|----------|------|
| Pickle | Python通用 | 简单但不安全 |
| Joblib | sklearn | 高效处理numpy数组 |
| ONNX | 跨框架 | 可移植 |
| SavedModel | TensorFlow | 生产标准 |
| TorchScript | PyTorch | 部署优化 |

#### 6.6.2 模型监控指标

| 指标 | 说明 | 告警阈值 |
|------|------|----------|
| 预测延迟 | P99响应时间 | > 100ms |
| 吞吐量 | QPS | < 目标值 |
| 预测分布漂移 | PSI/KS统计量 | PSI > 0.2 |
| 特征漂移 | 特征分布变化 | 显著变化 |
| 模型性能 | AUC/Accuracy下降 | > 5% |

**PSI (Population Stability Index)**

$$PSI = \sum_{i}(Actual_i - Expected_i) \times \ln\left(\frac{Actual_i}{Expected_i}\right)$$

- PSI < 0.1：无显著变化
- 0.1 ≤ PSI < 0.25：轻微变化
- PSI ≥ 0.25：显著变化

---

## 附录

### A. 常用数学符号表

| 符号 | 含义 |
|------|------|
| $\mathbf{x}, \mathbf{y}$ | 向量 |
| $\mathbf{X}, \mathbf{W}$ | 矩阵 |
| $\mathcal{N}(\mu, \sigma^2)$ | 正态分布 |
| $\mathbb{E}[X]$ | 期望 |
| $\text{Var}(X)$ | 方差 |
| $\nabla_\theta L$ | 关于 $\theta$ 的梯度 |
| $\odot$ | Hadamard积（逐元素乘） |
| $\sigma(\cdot)$ | Sigmoid函数 |
| $\text{softmax}(\cdot)$ | Softmax函数 |
| $||\cdot||_2$ | L2范数 |
| $||\cdot||_1$ | L1范数 |

### B. 常用损失函数速查

| 任务 | 损失函数 | 公式 |
|------|----------|------|
| 回归 | MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
| 回归 | MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ |
| 回归 | Huber | $\sum L_\delta(y_i - \hat{y}_i)$ |
| 二分类 | BCE | $-\sum[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$ |
| 多分类 | CE | $-\sum y_i \log(\hat{y}_i)$ |
| 多分类 | Focal Loss | $-\sum(1-\hat{y}_i)^\gamma y_i \log(\hat{y}_i)$ |

### C. 推荐学习资源

**课程**

- Stanford CS229: Machine Learning
- Stanford CS230: Deep Learning
- Stanford CS224N: NLP with Deep Learning
- Stanford CS231n: Convolutional Neural Networks
- Berkeley CS285: Deep Reinforcement Learning
- CMU 10-701/715: Introduction to Machine Learning

**书籍**

- 《Pattern Recognition and Machine Learning》- Bishop
- 《Deep Learning》- Goodfellow, Bengio, Courville
- 《Reinforcement Learning: An Introduction》- Sutton & Barto
- 《The Elements of Statistical Learning》- Hastie, Tibshirani, Friedman

**框架文档**

- PyTorch: <https://pytorch.org/docs/>
- TensorFlow: <https://www.tensorflow.org/api_docs>
- Scikit-learn: <https://scikit-learn.org/stable/>
- Hugging Face: <https://huggingface.co/docs>

---

*文档版本: 1.0*
*最后更新: 2024*
*对标课程: Stanford CS229, CS230, CS224N, CMU 10-701*
