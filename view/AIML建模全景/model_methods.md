# AI/ML 模型方法与算法 Comprehensive Analysis

> **文档版本**: 1.0
> **更新日期**: 2025年
> **参考标准**: Stanford CS229/CS230, MIT 6.034, Deep Learning (Goodfellow et al.)

---

## 目录

- [AI/ML 模型方法与算法 Comprehensive Analysis](#aiml-模型方法与算法-comprehensive-analysis)
  - [目录](#目录)
  - [1. 监督学习 (Supervised Learning)](#1-监督学习-supervised-learning)
    - [1.1 线性回归与逻辑回归](#11-线性回归与逻辑回归)
      - [1.1.1 线性回归 (Linear Regression)](#111-线性回归-linear-regression)
      - [1.1.2 逻辑回归 (Logistic Regression)](#112-逻辑回归-logistic-regression)
      - [1.1.3 正则化技术](#113-正则化技术)
    - [1.2 决策树与集成方法](#12-决策树与集成方法)
      - [1.2.1 决策树 (Decision Tree)](#121-决策树-decision-tree)
      - [1.2.2 随机森林 (Random Forest)](#122-随机森林-random-forest)
      - [1.2.3 XGBoost (eXtreme Gradient Boosting)](#123-xgboost-extreme-gradient-boosting)
      - [1.2.4 LightGBM](#124-lightgbm)
      - [1.2.5 CatBoost](#125-catboost)
    - [1.3 支持向量机 (Support Vector Machine)](#13-支持向量机-support-vector-machine)
      - [1.3.1 线性SVM (硬间隔)](#131-线性svm-硬间隔)
      - [1.3.2 软间隔SVM](#132-软间隔svm)
      - [1.3.3 对偶问题](#133-对偶问题)
      - [1.3.4 核方法 (Kernel Methods)](#134-核方法-kernel-methods)
    - [1.4 神经网络基础](#14-神经网络基础)
      - [1.4.1 多层感知机 (MLP)](#141-多层感知机-mlp)
      - [1.4.2 激活函数](#142-激活函数)
      - [1.4.3 反向传播 (Backpropagation)](#143-反向传播-backpropagation)
      - [1.4.4 优化算法](#144-优化算法)
  - [2. 深度学习 (Deep Learning)](#2-深度学习-deep-learning)
    - [2.1 卷积神经网络 (CNN) 架构演进](#21-卷积神经网络-cnn-架构演进)
      - [2.1.1 卷积操作基础](#211-卷积操作基础)
      - [2.1.2 经典架构演进](#212-经典架构演进)
      - [2.1.3 现代CNN组件](#213-现代cnn组件)
    - [2.2 RNN与序列模型](#22-rnn与序列模型)
      - [2.2.1 循环神经网络 (RNN)](#221-循环神经网络-rnn)
      - [2.2.2 LSTM (Long Short-Term Memory)](#222-lstm-long-short-term-memory)
      - [2.2.3 GRU (Gated Recurrent Unit)](#223-gru-gated-recurrent-unit)
      - [2.2.4 Seq2Seq与注意力机制](#224-seq2seq与注意力机制)
    - [2.3 Transformer架构](#23-transformer架构)
      - [2.3.1 自注意力机制 (Self-Attention)](#231-自注意力机制-self-attention)
      - [2.3.2 Transformer编码器](#232-transformer编码器)
      - [2.3.3 Transformer解码器](#233-transformer解码器)
      - [2.3.4 BERT (Bidirectional Encoder Representations from Transformers)](#234-bert-bidirectional-encoder-representations-from-transformers)
      - [2.3.5 GPT系列 (Generative Pre-trained Transformer)](#235-gpt系列-generative-pre-trained-transformer)
      - [2.3.6 T5 (Text-to-Text Transfer Transformer)](#236-t5-text-to-text-transfer-transformer)
    - [2.4 生成模型](#24-生成模型)
      - [2.4.1 变分自编码器 (VAE)](#241-变分自编码器-vae)
      - [2.4.2 生成对抗网络 (GAN)](#242-生成对抗网络-gan)
      - [2.4.3 扩散模型 (Diffusion Models)](#243-扩散模型-diffusion-models)
      - [2.4.4 流模型 (Flow-based Models)](#244-流模型-flow-based-models)
  - [3. 无监督学习 (Unsupervised Learning)](#3-无监督学习-unsupervised-learning)
    - [3.1 聚类算法](#31-聚类算法)
      - [3.1.1 K-means聚类](#311-k-means聚类)
      - [3.1.2 高斯混合模型 (GMM)](#312-高斯混合模型-gmm)
      - [3.1.3 DBSCAN (Density-Based Spatial Clustering)](#313-dbscan-density-based-spatial-clustering)
      - [3.1.4 层次聚类 (Hierarchical Clustering)](#314-层次聚类-hierarchical-clustering)
    - [3.2 降维技术](#32-降维技术)
      - [3.2.1 主成分分析 (PCA)](#321-主成分分析-pca)
      - [3.2.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)](#322-t-sne-t-distributed-stochastic-neighbor-embedding)
      - [3.2.3 UMAP (Uniform Manifold Approximation and Projection)](#323-umap-uniform-manifold-approximation-and-projection)
      - [3.2.4 自编码器 (Autoencoder)](#324-自编码器-autoencoder)
    - [3.3 异常检测](#33-异常检测)
      - [3.3.1 Isolation Forest](#331-isolation-forest)
      - [3.3.2 One-Class SVM](#332-one-class-svm)
      - [3.3.3 Deep SVDD (Deep Support Vector Data Description)](#333-deep-svdd-deep-support-vector-data-description)
  - [4. 强化学习 (Reinforcement Learning)](#4-强化学习-reinforcement-learning)
    - [4.1 基础概念](#41-基础概念)
    - [4.2 基础算法](#42-基础算法)
      - [4.2.1 动态规划](#421-动态规划)
      - [4.2.2 蒙特卡洛方法 (Monte Carlo)](#422-蒙特卡洛方法-monte-carlo)
      - [4.2.3 时序差分学习 (Temporal Difference)](#423-时序差分学习-temporal-difference)
      - [4.2.4 DQN (Deep Q-Network)](#424-dqn-deep-q-network)
    - [4.3 策略梯度方法](#43-策略梯度方法)
      - [4.3.1 REINFORCE](#431-reinforce)
      - [4.3.2 Actor-Critic](#432-actor-critic)
      - [4.3.3 PPO (Proximal Policy Optimization)](#433-ppo-proximal-policy-optimization)
    - [4.4 模型基础方法](#44-模型基础方法)
      - [4.4.1 MCTS (Monte Carlo Tree Search)](#441-mcts-monte-carlo-tree-search)
      - [4.4.2 MuZero](#442-muzero)
  - [5. 图神经网络 (Graph Neural Networks)](#5-图神经网络-graph-neural-networks)
    - [5.1 图基础](#51-图基础)
    - [5.2 GCN (Graph Convolutional Network)](#52-gcn-graph-convolutional-network)
    - [5.3 GraphSAGE](#53-graphsage)
    - [5.4 GAT (Graph Attention Network)](#54-gat-graph-attention-network)
    - [5.5 其他GNN变体](#55-其他gnn变体)
    - [5.6 图嵌入](#56-图嵌入)
      - [5.6.1 DeepWalk](#561-deepwalk)
      - [5.6.2 Node2Vec](#562-node2vec)
    - [5.7 图神经网络任务](#57-图神经网络任务)
  - [6. 多模态与前沿技术](#6-多模态与前沿技术)
    - [6.1 多模态学习](#61-多模态学习)
      - [6.1.1 CLIP (Contrastive Language-Image Pre-training)](#611-clip-contrastive-language-image-pre-training)
      - [6.1.2 DALL-E](#612-dall-e)
      - [6.1.3 Stable Diffusion](#613-stable-diffusion)
    - [6.2 大语言模型 (LLM)](#62-大语言模型-llm)
      - [6.2.1 预训练策略](#621-预训练策略)
      - [6.2.2 缩放定律 (Scaling Laws)](#622-缩放定律-scaling-laws)
      - [6.2.3 高效微调方法](#623-高效微调方法)
      - [6.2.4 RLHF (Reinforcement Learning from Human Feedback)](#624-rlhf-reinforcement-learning-from-human-feedback)
    - [6.3 Agent架构](#63-agent架构)
      - [6.3.1 ReAct (Reasoning + Acting)](#631-react-reasoning--acting)
      - [6.3.2 Plan-and-Solve](#632-plan-and-solve)
      - [6.3.3 Multi-Agent系统](#633-multi-agent系统)
    - [6.4 RAG (Retrieval-Augmented Generation)](#64-rag-retrieval-augmented-generation)
      - [6.4.1 基础架构](#641-基础架构)
      - [6.4.2 DPR (Dense Passage Retrieval)](#642-dpr-dense-passage-retrieval)
      - [6.4.3 高级RAG技术](#643-高级rag技术)
  - [7. 模型对比矩阵](#7-模型对比矩阵)
    - [7.1 监督学习模型对比](#71-监督学习模型对比)
    - [7.2 深度学习架构对比](#72-深度学习架构对比)
    - [7.3 序列模型对比](#73-序列模型对比)
    - [7.4 生成模型对比](#74-生成模型对比)
    - [7.5 聚类算法对比](#75-聚类算法对比)
    - [7.6 强化学习算法对比](#76-强化学习算法对比)
    - [7.7 图神经网络对比](#77-图神经网络对比)
  - [8. 模型选型决策树](#8-模型选型决策树)
    - [8.1 通用机器学习问题选型](#81-通用机器学习问题选型)
    - [8.2 深度学习任务选型](#82-深度学习任务选型)
    - [8.3 强化学习选型](#83-强化学习选型)
    - [8.4 大语言模型应用选型](#84-大语言模型应用选型)
  - [9. 实践指南](#9-实践指南)
    - [9.1 超参数调优](#91-超参数调优)
      - [9.1.1 学习率调优](#911-学习率调优)
      - [9.1.2 批大小选择](#912-批大小选择)
      - [9.1.3 正则化参数](#913-正则化参数)
      - [9.1.4 集成方法超参数](#914-集成方法超参数)
    - [9.2 数据预处理](#92-数据预处理)
      - [9.2.1 数值特征](#921-数值特征)
      - [9.2.2 类别特征](#922-类别特征)
      - [9.2.3 文本预处理](#923-文本预处理)
    - [9.3 评估指标](#93-评估指标)
      - [9.3.1 分类任务](#931-分类任务)
      - [9.3.2 回归任务](#932-回归任务)
      - [9.3.3 排序任务](#933-排序任务)
      - [9.3.4 生成任务](#934-生成任务)
      - [9.3.5 聚类评估](#935-聚类评估)
    - [9.4 交叉验证策略](#94-交叉验证策略)
    - [9.5 模型诊断](#95-模型诊断)
      - [9.5.1 过拟合检测](#951-过拟合检测)
      - [9.5.2 欠拟合检测](#952-欠拟合检测)
      - [9.5.3 学习曲线分析](#953-学习曲线分析)
    - [9.6 部署优化](#96-部署优化)
      - [9.6.1 模型压缩](#961-模型压缩)
      - [9.6.2 推理优化](#962-推理优化)
  - [10. 总结](#10-总结)

---

## 1. 监督学习 (Supervised Learning)

### 1.1 线性回归与逻辑回归

#### 1.1.1 线性回归 (Linear Regression)

**数学原理**:

线性回归假设目标变量 $y$ 与输入特征 $x$ 之间存在线性关系：

$$y = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$

其中 $\mathbf{w} \in \mathbb{R}^d$ 为权重向量，$b$ 为偏置项。

**损失函数 - 均方误差 (MSE)**:

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i - b)^2$$

矩阵形式：
$$\mathcal{L}(\mathbf{w}) = \frac{1}{n} ||\mathbf{y} - \mathbf{X}\mathbf{w}||_2^2$$

**解析解 (正规方程)**:

$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

**梯度下降更新规则**:

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} \mathcal{L}$$

其中梯度：
$$\nabla_{\mathbf{w}} \mathcal{L} = -\frac{2}{n} \mathbf{X}^T (\mathbf{y} - \mathbf{X}\mathbf{w})$$

#### 1.1.2 逻辑回归 (Logistic Regression)

**Sigmoid 函数**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**预测概率**:

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

**损失函数 - 交叉熵损失 (Cross-Entropy Loss)**:

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

多分类扩展 (Softmax Regression)：
$$P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

**梯度推导**:

$$\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_{ij}$$

#### 1.1.3 正则化技术

**L1 正则化 (Lasso)**:

$$\mathcal{L}_{L1}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda ||\mathbf{w}||_1 = \mathcal{L}(\mathbf{w}) + \lambda \sum_{j=1}^{d} |w_j|$$

- 产生稀疏解，实现特征选择
- 不可导点使用次梯度 (subgradient)

**L2 正则化 (Ridge)**:

$$\mathcal{L}_{L2}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda ||\mathbf{w}||_2^2 = \mathcal{L}(\mathbf{w}) + \lambda \sum_{j=1}^{d} w_j^2$$

- 权重衰减，防止过拟合
- 有解析解：$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$

**弹性网络 (Elastic Net)**:

$$\mathcal{L}_{EN}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda_1 ||\mathbf{w}||_1 + \lambda_2 ||\mathbf{w}||_2^2$$

### 1.2 决策树与集成方法

#### 1.2.1 决策树 (Decision Tree)

**分裂准则**:

**信息增益 (ID3算法)**:

$$IG(D, A) = H(D) - H(D|A) = H(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} H(D_v)$$

其中熵：$H(D) = -\sum_{k=1}^{K} p_k \log_2 p_k$

**信息增益率 (C4.5算法)**:

$$Gain\_ratio(D, A) = \frac{IG(D, A)}{H_A(D)}$$

其中 $H_A(D) = -\sum_{v} \frac{|D_v|}{|D|} \log_2 \frac{|D_v|}{|D|}$

**基尼指数 (CART分类树)**:

$$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$$

$$Gini\_index(D, A) = \sum_{v} \frac{|D_v|}{|D|} Gini(D_v)$$

**CART回归树**:

最小化平方误差：
$$\min_{j,s} \left[ \min_{c_1} \sum_{x_i \in R_1(j,s)} (y_i - c_1)^2 + \min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2 \right]$$

**剪枝策略**

- 预剪枝：限制树深度、最小叶节点样本数、最小信息增益
- 后剪枝：代价复杂度剪枝 $C_\alpha(T) = C(T) + \alpha |T|$

#### 1.2.2 随机森林 (Random Forest)

**算法原理**

1. **Bagging (Bootstrap Aggregating)**
   - 从训练集中有放回地抽取 $n$ 个样本构建子训练集
   - 重复 $B$ 次得到 $B$ 个不同的训练集

2. **随机特征选择**
   - 每个节点分裂时，随机选择 $m$ 个特征 ($m \approx \sqrt{d}$)
   - 从这 $m$ 个特征中选择最优分裂特征

**预测聚合**

分类：$\hat{y} = \arg\max_k \sum_{b=1}^{B} \mathbb{1}(h_b(\mathbf{x}) = k)$

回归：$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} h_b(\mathbf{x})$

**Out-of-Bag (OOB) 估计**

$$OOB\_error = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(y_i \neq \hat{y}_i^{OOB})$$

**特征重要性**

$$Importance(x_j) = \frac{1}{B} \sum_{b=1}^{B} \left( IG_b^{before} - IG_b^{after\_permutation} \right)$$

#### 1.2.3 XGBoost (eXtreme Gradient Boosting)

**目标函数**

$$\mathcal{L}(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

其中 $\Omega(f) = \gamma T + \frac{1}{2} \lambda ||\mathbf{w}||^2$ 为正则化项

**加法模型**

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(\mathbf{x}_i)$$

**二阶泰勒展开**

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

其中：

- $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ (一阶梯度)
- $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ (二阶梯度/Hessian)

**最优叶子权重**

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

**分裂增益**

$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

**系统优化**

- 列块存储 (Column Block)
- 缓存感知访问 (Cache-aware Access)
- 核外计算 (Out-of-core Computation)

#### 1.2.4 LightGBM

**核心创新**

1. **基于梯度的单边采样 (GOSS)**
   - 保留梯度大的样本，对梯度小的样本随机采样
   - 保持数据分布的同时减少计算量

2. **互斥特征捆绑 (EFB)**
   - 将互斥特征捆绑在一起，减少特征维度
   - 适用于高维稀疏特征

3. **直方图算法**
   - 将连续特征离散化为 $k$ 个bin
   - 时间复杂度从 $O(\#data \times \#feature)$ 降到 $O(\#bin \times \#feature)$

4. **叶子优先 (Leaf-wise) 生长策略**
   - 相比层级优先 (Level-wise)，减少分裂次数
   - 需要控制最大深度防止过拟合

#### 1.2.5 CatBoost

**核心特性**

1. **有序提升 (Ordered Boosting)**
   - 解决预测偏移 (Prediction Shift) 问题
   - 使用排序统计量计算梯度

2. **有序目标统计 (Ordered Target Statistics)**
   - 处理类别特征的目标编码
   - 避免目标泄漏

$$\hat{x}_{ik} = \frac{\sum_{j=1}^{n} \mathbb{1}(x_{jk} = x_{ik}) \cdot y_j + a \cdot p}{\sum_{j=1}^{n} \mathbb{1}(x_{jk} = x_{ik}) + a}$$

1. **对称树 (Symmetric Trees)**
   - 同一层使用相同的分裂条件
   - 加速预测，减少过拟合

### 1.3 支持向量机 (Support Vector Machine)

#### 1.3.1 线性SVM (硬间隔)

**优化目标**

$$\min_{\mathbf{w}, b} \frac{1}{2} ||\mathbf{w}||^2$$

约束条件：$y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad i = 1, ..., n$

#### 1.3.2 软间隔SVM

**引入松弛变量**

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^{n} \xi_i$$

约束条件：

- $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i$
- $\xi_i \geq 0$

**合页损失 (Hinge Loss)**

$$\mathcal{L}_{hinge}(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})$$

#### 1.3.3 对偶问题

**拉格朗日函数**

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2} ||\mathbf{w}||^2 - \sum_{i=1}^{n} \alpha_i [y_i(\mathbf{w}^T \mathbf{x}_i + b) - 1]$$

**对偶问题**

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j$$

约束条件：

- $\sum_{i=1}^{n} \alpha_i y_i = 0$
- $\alpha_i \geq 0$

**KKT条件**

- $\alpha_i = 0$: 样本在间隔外，非支持向量
- $0 < \alpha_i < C$: 样本在间隔边界上
- $\alpha_i = C$: 样本在间隔内或被误分类

#### 1.3.4 核方法 (Kernel Methods)

**核技巧**

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$$

**常用核函数**

| 核函数 | 表达式 | 适用场景 |
|--------|--------|----------|
| 线性核 | $K(\mathbf{u}, \mathbf{v}) = \mathbf{u}^T \mathbf{v}$ | 高维数据，线性可分 |
| 多项式核 | $K(\mathbf{u}, \mathbf{v}) = (\gamma \mathbf{u}^T \mathbf{v} + r)^d$ | 多项式特征交互 |
| RBF/高斯核 | $K(\mathbf{u}, \mathbf{v}) = \exp(-\gamma ||\mathbf{u} - \mathbf{v}||^2)$ | 通用，非线性边界 |
| Sigmoid核 | $K(\mathbf{u}, \mathbf{v}) = \tanh(\gamma \mathbf{u}^T \mathbf{v} + r)$ | 神经网络类似 |

**Mercer定理**

核函数 $K$ 有效当且仅当其对应的核矩阵是半正定的。

**SMO算法 (Sequential Minimal Optimization)**

- 每次选择两个拉格朗日乘子进行优化
- 解析求解两个变量的二次规划问题
- 时间复杂度：$O(n^2)$ 到 $O(n^3)$

### 1.4 神经网络基础

#### 1.4.1 多层感知机 (MLP)

**前向传播**

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

**通用近似定理**

具有至少一个隐藏层的前馈网络，配合非线性激活函数，可以以任意精度近似任何连续函数。

#### 1.4.2 激活函数

| 激活函数 | 公式 | 导数 | 特性 |
|----------|------|------|------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | 输出(0,1)，梯度消失 |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | 输出(-1,1)，零中心化 |
| ReLU | $\text{ReLU}(x) = \max(0, x)$ | $\mathbb{1}(x > 0)$ | 计算快，缓解梯度消失 |
| Leaky ReLU | $\max(\alpha x, x)$ | $\alpha$ (x<0), 1 (x>0) | 解决神经元死亡 |
| PReLU | $\max(\alpha x, x)$ (可学习$\alpha$) | - | 自适应负斜率 |
| ELU | $x$ (x≥0), $\alpha(e^x-1)$ (x<0) | - | 平滑负值 |
| GELU | $x \cdot \Phi(x)$ | - | Transformer首选 |
| Swish | $x \cdot \sigma(x)$ | - | 自门控机制 |
| Softmax | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | - | 多分类输出层 |

#### 1.4.3 反向传播 (Backpropagation)

**链式法则**

$$\frac{\partial \mathcal{L}}{\partial w_{ij}^{[l]}} = \frac{\partial \mathcal{L}}{\partial z_i^{[l]}} \cdot \frac{\partial z_i^{[l]}}{\partial w_{ij}^{[l]}} = \delta_i^{[l]} \cdot a_j^{[l-1]}$$

**误差反向传播**

输出层：$\delta^{[L]} = \nabla_{\mathbf{a}} \mathcal{L} \odot g'^{[L]}(\mathbf{z}^{[L]})$

隐藏层：$\delta^{[l]} = (\mathbf{W}^{[l+1]})^T \delta^{[l+1]} \odot g'^{[l]}(\mathbf{z}^{[l]})$

**权重更新**

$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \cdot \delta^{[l]} (\mathbf{a}^{[l-1]})^T$$
$$\mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \cdot \delta^{[l]}$$

#### 1.4.4 优化算法

**SGD with Momentum**

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla_{\mathbf{w}} \mathcal{L}_t$$
$$\mathbf{w}_t = \mathbf{w}_{t-1} - \eta \mathbf{v}_t$$

**AdaGrad**

$$\mathbf{r}_t = \mathbf{r}_{t-1} + (\nabla_{\mathbf{w}} \mathcal{L}_t)^2$$
$$\mathbf{w}_t = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{r}_t + \epsilon}} \odot \nabla_{\mathbf{w}} \mathcal{L}_t$$

**RMSProp**

$$\mathbf{r}_t = \rho \mathbf{r}_{t-1} + (1-\rho)(\nabla_{\mathbf{w}} \mathcal{L}_t)^2$$
$$\mathbf{w}_t = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{r}_t + \epsilon}} \odot \nabla_{\mathbf{w}} \mathcal{L}_t$$

**Adam (Adaptive Moment Estimation)**

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \nabla_{\mathbf{w}} \mathcal{L}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) (\nabla_{\mathbf{w}} \mathcal{L}_t)^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$
$$\mathbf{w}_t = \mathbf{w}_{t-1} - \frac{\eta \cdot \hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}$$

推荐参数：$\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---


## 2. 深度学习 (Deep Learning)

### 2.1 卷积神经网络 (CNN) 架构演进

#### 2.1.1 卷积操作基础

**二维卷积**

$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) \cdot K(m, n)$$

**卷积层参数计算**

- 输出尺寸：$O = \lfloor \frac{I - K + 2P}{S} \rfloor + 1$
- 参数量：$(K_h \times K_w \times C_{in} + 1) \times C_{out}$

**池化操作**

- 最大池化：$y = \max_{i \in R} x_i$
- 平均池化：$y = \frac{1}{|R|} \sum_{i \in R} x_i$
- 全局平均池化：$y_c = \frac{1}{H \times W} \sum_{i,j} x_{c,i,j}$

#### 2.1.2 经典架构演进

**LeNet-5 (1998)**

- 结构：Conv → Pool → Conv → Pool → FC → FC → Output
- 创新：使用卷积提取局部特征，参数共享

**AlexNet (2012)**

- 8层网络，ReLU激活，Dropout正则化
- 创新：
  - ReLU解决梯度消失
  - GPU并行训练
  - 数据增强、Dropout
  - 局部响应归一化 (LRN)

**VGGNet (2014)**

- VGG-16/19：使用3×3小卷积核堆叠
- 创新：
  - 小卷积核(3×3)替代大卷积核
  - 网络深度增加到16-19层
  - 参数量：138M

**ResNet (2015)**

- 残差学习框架，解决深层网络退化问题
- 残差块：$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$
- 变体：
  - ResNet-18/34/50/101/152
  - Bottleneck设计：1×1 → 3×3 → 1×1
  - 参数量：ResNet-50约25.6M

**DenseNet (2017)**

- 密集连接：每层与前面所有层连接
- 公式：$\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_{l-1}])$
- 优点：特征重用，梯度流动更好
- 缺点：显存消耗大

**EfficientNet (2019)**

- 复合缩放：统一缩放深度、宽度、分辨率
- 复合系数：$d = \alpha^\phi$, $w = \beta^\phi$, $r = \gamma^\phi$
- EfficientNet-B0到B7，ImageNet准确率从77.3%到84.3%

**Vision Transformer (ViT) (2020)**

- 将图像分割为patches，作为序列输入Transformer
- 公式：$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_p^1\mathbf{E}; ...; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{pos}$
- 需要大规模数据预训练 (JFT-300M)

**Swin Transformer (2021)**

- 层次化Vision Transformer
- 移位窗口 (Shifted Window) 自注意力
- 线性复杂度：$O(M^2 \cdot n)$，$M$为窗口大小

#### 2.1.3 现代CNN组件

**批归一化 (Batch Normalization)**

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

- 训练时使用batch统计量
- 测试时使用移动平均统计量

**层归一化 (Layer Normalization)**

$$\hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$$

- 对每个样本的所有特征进行归一化
- 适用于RNN和Transformer

**组归一化 (Group Normalization)**

- 将通道分组归一化
- 小batch时比BN更稳定

**Dropout**

$$r_j^{(l)} \sim \text{Bernoulli}(p)$$
$$\tilde{\mathbf{y}}^{(l)} = \mathbf{r}^{(l)} \odot \mathbf{y}^{(l)}$$

测试时：$\mathbf{W}_{test}^{(l)} = p \cdot \mathbf{W}^{(l)}$

**Inception模块**

- 并行使用不同尺寸的卷积核
- 1×1卷积降维
- 结构：1×1, 3×3, 5×5卷积 + 3×3池化

**空洞卷积 (Dilated Convolution)**

$$(I *_d K)(i, j) = \sum_{m} \sum_{n} I(i+d \cdot m, j+d \cdot n) \cdot K(m, n)$$

- 扩大感受野不增加参数
- 语义分割常用

### 2.2 RNN与序列模型

#### 2.2.1 循环神经网络 (RNN)

**基本RNN**

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$
$$\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

**BPTT (Backpropagation Through Time)**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}}$$

**梯度问题**

- 梯度消失：长期依赖难以学习
- 梯度爆炸：参数更新不稳定

#### 2.2.2 LSTM (Long Short-Term Memory)

**门控机制**

遗忘门：$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$

输入门：$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$

候选状态：$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)$

细胞状态：$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$

输出门：$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$

隐藏状态：$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$

**LSTM变体**

- Peephole LSTM：门控连接细胞状态
- Coupled LSTM：遗忘门和输入门联动

#### 2.2.3 GRU (Gated Recurrent Unit)

**简化门控**

重置门：$\mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t])$

更新门：$\mathbf{z}_t = \sigma(\mathbf{W}_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t])$

候选状态：$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$

隐藏状态：$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$

**LSTM vs GRU**

- LSTM：3个门，2个状态，参数量更大
- GRU：2个门，1个状态，参数量更小，训练更快

#### 2.2.4 Seq2Seq与注意力机制

**编码器-解码器框架**

编码器：$\mathbf{h}_t = \text{RNN}_{enc}(\mathbf{x}_t, \mathbf{h}_{t-1})$

上下文向量：$\mathbf{c} = \mathbf{h}_T$ 或 $\mathbf{c} = q(\{\mathbf{h}_1, ..., \mathbf{h}_T\})$

解码器：$\mathbf{s}_t = \text{RNN}_{dec}(\mathbf{y}_{t-1}, \mathbf{s}_{t-1}, \mathbf{c})$

**注意力机制 (Bahdanau Attention)**

对齐分数：$e_{tj} = a(\mathbf{s}_{t-1}, \mathbf{h}_j)$

注意力权重：$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T} \exp(e_{tk})}$

上下文向量：$\mathbf{c}_t = \sum_{j=1}^{T} \alpha_{tj} \mathbf{h}_j$

**Luong Attention**

- Dot：$score(s_t, h_j) = s_t^T h_j$
- General：$score(s_t, h_j) = s_t^T W h_j$
- Concat：$score(s_t, h_j) = v^T \tanh(W_s s_t + W_h h_j)$

### 2.3 Transformer架构

#### 2.3.1 自注意力机制 (Self-Attention)

**Scaled Dot-Product Attention**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：

- $Q \in \mathbb{R}^{n \times d_k}$：查询矩阵
- $K \in \mathbb{R}^{m \times d_k}$：键矩阵
- $V \in \mathbb{R}^{m \times d_v}$：值矩阵

**多头注意力 (Multi-Head Attention)**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

**复杂度分析**

| 层类型 | 每层复杂度 | 顺序操作 | 最大路径长度 |
|--------|-----------|----------|-------------|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |

#### 2.3.2 Transformer编码器

**编码器层结构**

```
Input → Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm → Output
```

**位置编码 (Positional Encoding)**

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**前馈网络 (FFN)**

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

或 GELU 版本：$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$

#### 2.3.3 Transformer解码器

**解码器层结构**

```
Input → Masked Multi-Head Attention → Add & Norm
      → Multi-Head Attention (Encoder-Decoder) → Add & Norm
      → Feed Forward → Add & Norm → Output
```

**Masked Self-Attention**

$$\text{Attention}(Q, K, V)_{ij} = \begin{cases} \text{softmax}(\frac{QK^T}{\sqrt{d_k}})_{ij} & i \geq j \\ 0 & i < j \end{cases}$$

#### 2.3.4 BERT (Bidirectional Encoder Representations from Transformers)

**预训练任务**

1. **Masked Language Model (MLM)**
   - 随机mask 15%的token
   - 80%替换为[MASK]，10%替换为随机token，10%保持不变
   - 损失：$\mathcal{L}_{MLM} = -\mathbb{E}_{x \in \mathcal{D}} \log P(x | \hat{x})$

2. **Next Sentence Prediction (NSP)**
   - 判断句子B是否是句子A的下一句
   - 50%正例，50%负例

**模型变体**

- BERT-Base：12层，768维，12头，110M参数
- BERT-Large：24层，1024维，16头，340M参数

**下游任务微调**

- 分类任务：[CLS] token输出接分类器
- 序列标注：每个token输出接分类器
- 问答任务：预测start/end位置

#### 2.3.5 GPT系列 (Generative Pre-trained Transformer)

**GPT-1 (2018)**

- 12层Transformer解码器
- 无监督预训练 + 有监督微调
- 117M参数

**GPT-2 (2019)**

- 15亿参数
- 零样本学习能力
- 多任务学习

**GPT-3 (2020)**

- 1750亿参数
- 上下文学习 (In-context Learning)
- Few-shot/One-shot/Zero-shot能力

**GPT-4 (2023)**

- 多模态能力（文本+图像）
- 更强的推理和安全性

**InstructGPT/GPT-3.5**

- SFT (Supervised Fine-Tuning)
- RLHF (Reinforcement Learning from Human Feedback)

#### 2.3.6 T5 (Text-to-Text Transfer Transformer)

**统一框架**

- 所有NLP任务统一为text-to-text格式
- 输入："translate English to German: ..."
- 输出：目标文本

**模型结构**

- Encoder-Decoder架构
- 相对位置编码
- T5-Base: 220M, T5-Large: 770M, T5-11B: 11B

**预训练策略**

- Span Corruption：随机mask连续span
- 替换为唯一sentinel token

### 2.4 生成模型

#### 2.4.1 变分自编码器 (VAE)

**模型结构**

编码器：$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \boldsymbol{\sigma}_\phi^2(\mathbf{x}))$

解码器：$p_\theta(\mathbf{x}|\mathbf{z})$

**重参数化技巧**

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

**ELBO (Evidence Lower Bound)**

$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}))$$

**KL散度解析解**

$$D_{KL}(q||p) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$

**VAE变体**

- β-VAE：加权KL项，学习解耦表示
- CVAE：条件VAE
- VQ-VAE：离散隐变量

#### 2.4.2 生成对抗网络 (GAN)

**minimax 博弈**

$$\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

**训练过程**

1. 固定G，更新D：
   $$\max_D \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

2. 固定D，更新G：
   $$\min_G \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$
   或等价地：$\max_G \mathbb{E}_{\mathbf{z} \sim p_z}[\log D(G(\mathbf{z}))]$

**理论分析**

全局最优时：$p_g = p_{data}$，$D^*(\mathbf{x}) = \frac{1}{2}$

**损失函数变体**

| GAN变体 | 判别器损失 | 生成器损失 |
|---------|-----------|-----------|
| Standard GAN | $-\log D(x) - \log(1-D(G(z)))$ | $\log(1-D(G(z)))$ |
| Non-saturating | 同上 | $-\log D(G(z))$ |
| WGAN | $D(x) - D(G(z))$ | $-D(G(z))$ |
| LSGAN | $(D(x)-1)^2 + D(G(z))^2$ | $(D(G(z))-1)^2$ |
| Hinge | $\max(0, 1-D(x)) + \max(0, 1+D(G(z)))$ | $-D(G(z))$ |

**WGAN (Wasserstein GAN)**

$$W(p_{data}, p_g) = \inf_{\gamma \in \Pi} \mathbb{E}_{(x,y) \sim \gamma}[||x-y||]$$

近似：$\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x \sim p_{data}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$

约束：$|D(x) - D(y)| \leq K|x-y|$ (Lipschitz约束)

WGAN-GP：梯度惩罚代替权重裁剪
$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x}}[(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2]$$

**条件GAN (cGAN)**

$$\min_G \max_D V(D, G) = \mathbb{E}_{x,y}[\log D(x|y)] + \mathbb{E}_{z,y}[\log(1 - D(G(z|y)|y))]$$

**StyleGAN**

- 映射网络：$z \rightarrow w$ (解耦隐空间)
- 自适应实例归一化 (AdaIN)
- 渐进式增长训练

#### 2.4.3 扩散模型 (Diffusion Models)

**前向扩散过程**

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

累积形式：$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$

其中 $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

**反向去噪过程**

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

**训练目标**

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ ||\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, t)||^2 \right]$$

**DDPM采样**

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$$

**DDIM (去噪扩散隐式模型)**

- 确定性采样，加速推理
- 从1000步减少到50步

**Stable Diffusion**

- 在隐空间进行扩散 (LDM)
- VAE编码器：$\mathcal{E}(x) = z$
- U-Net噪声预测网络
- CLIP文本编码器条件

**Classifier-Free Guidance (CFG)**

$$\hat{\epsilon}_\theta = \epsilon_{unc} + s \cdot (\epsilon_{cond} - \epsilon_{unc})$$

$s$为引导尺度，通常7.5-15

#### 2.4.4 流模型 (Flow-based Models)

**可逆变换**

$$\mathbf{x} = f(\mathbf{z}), \quad \mathbf{z} = f^{-1}(\mathbf{x})$$

**变量替换公式**

$$p_x(\mathbf{x}) = p_z(f^{-1}(\mathbf{x})) \left| \det \frac{\partial f^{-1}}{\partial \mathbf{x}} \right|$$

**对数似然**

$$\log p_x(\mathbf{x}) = \log p_z(\mathbf{z}) + \sum_{i=1}^{K} \log \left| \det \frac{\partial f_i}{\partial \mathbf{h}_{i-1}} \right|$$

**RealNVP**

- 仿射耦合层
- 计算高效：三角雅可比矩阵

**Glow**

- 1×1可逆卷积
- 可学习置换

---


## 3. 无监督学习 (Unsupervised Learning)

### 3.1 聚类算法

#### 3.1.1 K-means聚类

**目标函数**

$$J = \sum_{i=1}^{n} \sum_{k=1}^{K} r_{ik} ||\mathbf{x}_i - \boldsymbol{\mu}_k||^2$$

其中 $r_{ik} \in \{0, 1\}$ 为分配指示变量

**算法步骤**

1. **初始化**：随机选择K个中心点
2. **E步 (分配)**：$r_{ik} = \begin{cases} 1 & k = \arg\min_j ||\mathbf{x}_i - \boldsymbol{\mu}_j||^2 \\ 0 & \text{otherwise} \end{cases}$
3. **M步 (更新)**：$\boldsymbol{\mu}_k = \frac{\sum_i r_{ik} \mathbf{x}_i}{\sum_i r_{ik}}$
4. 重复2-3直到收敛

**K-means++初始化**

1. 随机选择第一个中心
2. 对每个点计算到最近中心的距离 $D(x)$
3. 以概率 $\frac{D(x)^2}{\sum_x D(x)^2}$ 选择新中心
4. 重复直到K个中心

**复杂度**

- 时间：$O(n \cdot K \cdot d \cdot I)$，$I$为迭代次数
- 空间：$O(n \cdot d + K \cdot d)$

**优缺点**

- 优点：简单高效，可扩展
- 缺点：需要预设K，对初始值敏感，假设球形簇

#### 3.1.2 高斯混合模型 (GMM)

**概率模型**

$$p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

其中 $\sum_k \pi_k = 1$

**EM算法**

**E步**：计算后验概率（责任）
$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

**M步**：更新参数
$$N_k = \sum_{i=1}^{n} \gamma_{ik}$$
$$\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} \mathbf{x}_i$$
$$\boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$$
$$\pi_k = \frac{N_k}{n}$$

**对数似然**

$$\ln p(X|\pi, \mu, \Sigma) = \sum_{i=1}^{n} \ln \left\{ \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right\}$$

**与K-means关系**

- K-means是GMM的硬分配极限情况
- 当协方差矩阵趋近于0时，GMM退化为K-means

#### 3.1.3 DBSCAN (Density-Based Spatial Clustering)

**核心概念**

- **核心点**：$\epsilon$邻域内至少有MinPts个点
- **边界点**：在核心点的$\epsilon$邻域内但不是核心点
- **噪声点**：既不是核心点也不是边界点

**算法步骤**

1. 标记所有点为核心点、边界点或噪声点
2. 删除噪声点
3. 为距离在$\epsilon$内的核心点之间建立边
4. 每个连通组件形成一个簇
5. 将边界点分配到相邻核心点的簇

**参数选择**

- $k$-距离图：选择拐点作为$\epsilon$
- MinPts通常设为维度数+1

**复杂度**

- 时间：$O(n \log n)$（使用空间索引）
- 空间：$O(n)$

**优缺点**

- 优点：自动确定簇数量，发现任意形状簇，对噪声鲁棒
- 缺点：参数敏感，高维数据效果差

#### 3.1.4 层次聚类 (Hierarchical Clustering)

**凝聚式 (Agglomerative)**

1. 每个点作为一个簇
2. 计算所有簇对之间的距离
3. 合并距离最近的两个簇
4. 重复直到所有点在一个簇中

**距离度量**

| 连接方式 | 公式 |
|---------|------|
| 单链接 | $d(A,B) = \min_{a \in A, b \in B} d(a,b)$ |
| 全链接 | $d(A,B) = \max_{a \in A, b \in B} d(a,b)$ |
| 平均链接 | $d(A,B) = \frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} d(a,b)$ |
| Ward | 最小化合并后的方差增量 |

**分裂式 (Divisive)**

- 从所有点在一个簇开始
- 递归分裂直到每个点单独成簇

**复杂度**

- 凝聚式：$O(n^2 \log n)$ 或 $O(n^2)$
- 分裂式：$O(2^n)$

### 3.2 降维技术

#### 3.2.1 主成分分析 (PCA)

**目标**

找到投影矩阵 $\mathbf{W}$，使得投影后的方差最大：
$$\max_{\mathbf{W}} \text{Tr}(\mathbf{W}^T \mathbf{S} \mathbf{W})$$
约束：$\mathbf{W}^T \mathbf{W} = \mathbf{I}$

其中样本协方差矩阵：$\mathbf{S} = \frac{1}{n-1} \sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^T$

**求解**

对协方差矩阵进行特征分解：
$$\mathbf{S} \mathbf{w}_i = \lambda_i \mathbf{w}_i$$

选择前 $k$ 个最大特征值对应的特征向量

**投影**

$$\mathbf{z}_i = \mathbf{W}^T (\mathbf{x}_i - \bar{\mathbf{x}})$$

**重构**

$$\hat{\mathbf{x}}_i = \mathbf{W} \mathbf{z}_i + \bar{\mathbf{x}}$$

**方差保留率**

$$\text{保留率} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

**SVD方法**

对数据矩阵 $\mathbf{X}$ 进行SVD分解：$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$

PCA投影矩阵：$\mathbf{W} = \mathbf{V}_k$

#### 3.2.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

**高维空间相似度**

$$p_{j|i} = \frac{\exp(-||\mathbf{x}_i - \mathbf{x}_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||\mathbf{x}_i - \mathbf{x}_k||^2 / 2\sigma_i^2)}$$

对称化：$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$

**低维空间相似度**

$$q_{ij} = \frac{(1 + ||\mathbf{y}_i - \mathbf{y}_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||\mathbf{y}_k - \mathbf{y}_l||^2)^{-1}}$$

**KL散度损失**

$$C = KL(P||Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**梯度**

$$\frac{\partial C}{\partial \mathbf{y}_i} = 4 \sum_{j \neq i} (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + ||\mathbf{y}_i - \mathbf{y}_j||^2)^{-1}$$

**Barnes-Hut 近似**

- 时间复杂度从 $O(n^2)$ 降到 $O(n \log n)$
- 使用四叉树/八叉树近似远距离点

**使用建议**

- 先用PCA降维到30-50维
- 困惑度 (perplexity) 通常5-50
- 学习率通常10-1000

#### 3.2.3 UMAP (Uniform Manifold Approximation and Projection)

**理论基础**

- 假设数据均匀分布在黎曼流形上
- 保持模糊单纯形集拓扑结构

**高维相似度**

$$p_{j|i} = \exp\left(-\frac{d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i}{\sigma_i}\right)$$

其中 $\rho_i$ 为到最近邻的距离

**低维相似度**

$$q_{ij} = (1 + a||\mathbf{y}_i - \mathbf{y}_j||^{2b})^{-1}$$

**交叉熵损失**

$$C = \sum_{i,j} \left[ p_{ij} \log \frac{p_{ij}}{q_{ij}} + (1-p_{ij}) \log \frac{1-p_{ij}}{1-q_{ij}} \right]$$

**与t-SNE对比**

| 特性 | t-SNE | UMAP |
|------|-------|------|
| 速度 | 慢 | 快 |
| 全局结构 | 可能丢失 | 更好地保留 |
| 超参数敏感 | 高 | 中等 |
| 可重复性 | 随机性大 | 更稳定 |
| 新数据投影 | 不支持 | 支持 |

#### 3.2.4 自编码器 (Autoencoder)

**结构**

编码器：$\mathbf{z} = f_\phi(\mathbf{x})$

解码器：$\hat{\mathbf{x}} = g_\theta(\mathbf{z})$

**损失函数**

$$\mathcal{L}(\phi, \theta) = \frac{1}{n} \sum_{i=1}^{n} ||\mathbf{x}_i - g_\theta(f_\phi(\mathbf{x}_i))||^2$$

**变体**

**去噪自编码器 (Denoising Autoencoder)**

- 输入加入噪声：$\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}$
- 重构原始输入

**稀疏自编码器 (Sparse Autoencoder)**

- 添加稀疏约束：$\mathcal{L} + \lambda \sum_j KL(\rho || \hat{\rho}_j)$

**收缩自编码器 (Contractive Autoencoder)**

- 添加雅可比惩罚：$\mathcal{L} + \lambda ||J_f(\mathbf{x})||_F^2$

### 3.3 异常检测

#### 3.3.1 Isolation Forest

**核心思想**

- 异常点更容易被孤立
- 随机选择特征和分裂值构建树

**路径长度**

- 异常点：短路径
- 正常点：长路径

**异常分数**

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

其中 $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$，$H$为调和数

**算法步骤**

1. 从训练集中随机采样子样本
2. 递归随机选择特征和分裂值构建孤立树
3. 计算每个点的平均路径长度
4. 计算异常分数

**复杂度**

- 训练：$O(t \cdot \psi \log \psi)$，$t$为树数量，$\psi$为子样本大小
- 预测：$O(t \log \psi)$

#### 3.3.2 One-Class SVM

**优化目标**

$$\min_{\mathbf{w}, \rho, \xi} \frac{1}{2} ||\mathbf{w}||^2 + \frac{1}{\nu n} \sum_{i=1}^{n} \xi_i - \rho$$

约束：$\mathbf{w}^T \phi(\mathbf{x}_i) \geq \rho - \xi_i$，$\xi_i \geq 0$

**对偶问题**

$$\min_{\boldsymbol{\alpha}} \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j K(\mathbf{x}_i, \mathbf{x}_j)$$

约束：$0 \leq \alpha_i \leq \frac{1}{\nu n}$，$\sum_i \alpha_i = 1$

**决策函数**

$$f(\mathbf{x}) = \text{sgn}\left(\sum_{i} \alpha_i K(\mathbf{x}_i, \mathbf{x}) - \rho\right)$$

**参数 $\nu$**

- 上界：异常点比例
- 下界：支持向量比例

#### 3.3.3 Deep SVDD (Deep Support Vector Data Description)

**目标**

学习一个神经网络，将数据映射到超球体内：

$$\min_{\mathcal{W}} \frac{1}{n} \sum_{i=1}^{n} ||\phi(\mathbf{x}_i; \mathcal{W}) - \mathbf{c}||^2 + \frac{\lambda}{2} \sum_{\ell=1}^{L} ||\mathbf{W}^\ell||_F^2$$

**异常分数**

$$s(\mathbf{x}) = ||\phi(\mathbf{x}; \mathcal{W}) - \mathbf{c}||^2$$

**软边界版本**

$$\min_{\mathcal{W}, \xi} \frac{1}{n} \sum_{i=1}^{n} \xi_i + \frac{\lambda}{2} \sum_{\ell=1}^{L} ||\mathbf{W}^\ell||_F^2$$

约束：$||\phi(\mathbf{x}_i; \mathcal{W}) - \mathbf{c}||^2 \leq R^2 + \xi_i$，$\xi_i \geq 0$

---

## 4. 强化学习 (Reinforcement Learning)

### 4.1 基础概念

**马尔可夫决策过程 (MDP)**

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

- $\mathcal{S}$：状态空间
- $\mathcal{A}$：动作空间
- $\mathcal{P}(s'|s,a)$：状态转移概率
- $\mathcal{R}(s,a,s')$：奖励函数
- $\gamma \in [0,1]$：折扣因子

**策略 (Policy)**

$$\pi(a|s) = P(A_t = a | S_t = s)$$

**价值函数**

状态价值函数：
$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0 = s \right]$$

动作价值函数：
$$Q^\pi(s,a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0 = s, A_0 = a \right]$$

**贝尔曼方程**

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a) [r + \gamma V^\pi(s')]$$

$$Q^\pi(s,a) = \sum_{s', r} p(s', r|s, a) [r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')]$$

**最优价值函数**

$$V^*(s) = \max_\pi V^\pi(s)$$
$$Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

**贝尔曼最优方程**

$$V^*(s) = \max_a \sum_{s', r} p(s', r|s, a) [r + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s', r} p(s', r|s, a) [r + \gamma \max_{a'} Q^*(s', a')]$$

### 4.2 基础算法

#### 4.2.1 动态规划

**策略迭代 (Policy Iteration)**

1. **策略评估**：迭代计算 $V^\pi$
   $$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s, a) [r + \gamma V_k(s')]$$

2. **策略改进**：贪心更新策略
   $$\pi'(s) = \arg\max_a \sum_{s', r} p(s', r|s, a) [r + \gamma V^\pi(s')]$$

3. 重复直到收敛

**价值迭代 (Value Iteration)**

$$V_{k+1}(s) = \max_a \sum_{s', r} p(s', r|s, a) [r + \gamma V_k(s')]$$

#### 4.2.2 蒙特卡洛方法 (Monte Carlo)

**首次访问MC**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [G_t - Q(S_t, A_t)]$$

其中 $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$

**增量更新**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} [G_t - Q(S_t, A_t)]$$

**MC控制**

- 策略评估：MC采样估计Q值
- 策略改进：$\epsilon$-贪心策略

#### 4.2.3 时序差分学习 (Temporal Difference)

**TD(0)**

$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

**SARSA (On-Policy)**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

**Q-Learning (Off-Policy)**

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

**SARSA vs Q-Learning**

| 特性 | SARSA | Q-Learning |
|------|-------|------------|
| 策略类型 | On-policy | Off-policy |
| 探索性 | 保守 | 激进 |
| 收敛性 | 稳定 | 可能不稳定 |

#### 4.2.4 DQN (Deep Q-Network)

**神经网络近似**

$$Q(s, a; \theta) \approx Q^*(s, a)$$

**损失函数**

$$L_i(\theta_i) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_i^-) - Q(s, a; \theta_i) \right)^2 \right]$$

**关键技术**

1. **经验回放 (Experience Replay)**
   - 存储转移 $(s, a, r, s')$ 到回放缓冲区
   - 随机采样打破相关性

2. **目标网络 (Target Network)**
   - 使用单独的网络计算目标Q值
   - 定期同步参数：$\theta^- \leftarrow \theta$

3. **$\epsilon$-贪心探索**
   $$\pi(a|s) = \begin{cases} \arg\max_a Q(s,a) & \text{prob } 1-\epsilon \\ \text{random} & \text{prob } \epsilon \end{cases}$$

**Double DQN**

解决Q值过估计问题：
$$Y_t = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta_t^-)$$

**Dueling DQN**

分离状态价值和优势：
$$Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a')$$

**Prioritized Experience Replay**

按TD误差优先级采样：
$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

其中 $p_i = |\delta_i| + \epsilon$

### 4.3 策略梯度方法

#### 4.3.1 REINFORCE

**策略梯度定理**

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

**带基线的REINFORCE**

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) \right]$$

基线 $b(s_t)$ 通常使用价值函数估计

#### 4.3.2 Actor-Critic

**框架**

- **Actor**：策略网络 $\pi_\theta(a|s)$
- **Critic**：价值网络 $V_w(s)$ 或 $Q_w(s,a)$

**A2C (Advantage Actor-Critic)**

Critic更新：
$$\delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$$
$$w \leftarrow w + \alpha_w \delta_t \nabla_w V_w(S_t)$$

Actor更新：
$$\theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(A_t|S_t)$$

**A3C (Asynchronous Advantage Actor-Critic)**

- 多个并行worker
- 异步更新全局网络
- 减少数据相关性

#### 4.3.3 PPO (Proximal Policy Optimization)

**目标**

防止策略更新过大：

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

**优势函数估计 (GAE)**

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

**完整目标**

$$L^{CLIP+VF+S}(\theta) = \mathbb{E}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

其中：

- $L_t^{VF} = (V_\theta(s_t) - V_t^{targ})^2$ (价值函数损失)
- $S$ 为策略熵（鼓励探索）

### 4.4 模型基础方法

#### 4.4.1 MCTS (Monte Carlo Tree Search)

**四个步骤**

1. **选择 (Selection)**：从根节点选择子节点直到叶节点
   - UCB1：$\text{UCB1} = \bar{X}_j + 2C_p \sqrt{\frac{2 \ln n}{n_j}}$

2. **扩展 (Expansion)**：扩展叶节点的一个子节点

3. **模拟 (Simulation)**：从扩展节点随机模拟到终止

4. **反向传播 (Backpropagation)**：更新路径上节点的统计量

**AlphaGo/AlphaZero中的MCTS**

$$U(s,a) = c_{puct} P(a|s) \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

$$Q(s,a) = \frac{W(s,a)}{N(s,a)}$$

选择动作：$a = \arg\max_a (Q(s,a) + U(s,a))$

#### 4.4.2 MuZero

**三个神经网络**

1. **表征网络 (Representation)**：$h_\theta(s^0) = h_\theta(o_1, ..., o_t)$
2. **动态网络 (Dynamics)**：$r^k, s^k = g_\theta(s^{k-1}, a^k)$
3. **预测网络 (Prediction)**：$p^k, v^k = f_\theta(s^k)$

**损失函数**

$$l_t(\theta) = \sum_{k=0}^{K} \left( l^r(u_{t+k}, r_t^k) + l^v(z_{t+k}, v_t^k) + l^p(\pi_{t+k}, p_t^k) \right)$$

---


## 5. 图神经网络 (Graph Neural Networks)

### 5.1 图基础

**图的表示**

$G = (V, E)$，其中：

- $V$：节点集合，$|V| = n$
- $E$：边集合，$|E| = m$

**邻接矩阵**

$$A_{ij} = \begin{cases} 1 & (i,j) \in E \\ 0 & \text{otherwise} \end{cases}$$

**度矩阵**

$$D_{ii} = \sum_j A_{ij}$$

**归一化邻接矩阵**

$$\hat{A} = D^{-1/2} A D^{-1/2}$$

或带自环：$\tilde{A} = A + I$，$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$

**图拉普拉斯矩阵**

- 未归一化：$L = D - A$
- 对称归一化：$L_{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$
- 随机游走归一化：$L_{rw} = D^{-1} L = I - D^{-1} A$

### 5.2 GCN (Graph Convolutional Network)

**谱图卷积**

$$\mathbf{x} *_G \mathbf{g} = U g_\theta(\Lambda) U^T \mathbf{x}$$

其中 $L = U \Lambda U^T$ 为拉普拉斯矩阵的特征分解

**Chebyshev多项式近似**

$$g_\theta(\Lambda) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})$$

其中 $\tilde{\Lambda} = 2\Lambda/\lambda_{max} - I$

**一阶近似 (K=1)**

$$\mathbf{H}^{(l+1)} = \sigma\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

或等价地：
$$\mathbf{H}^{(l+1)} = \sigma\left(\hat{A} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

**完整GCN公式**

$$Z = f(X, A) = \text{softmax}\left(\hat{A} \text{ReLU}(\hat{A} X W^{(0)}) W^{(1)}\right)$$

**半监督损失**

$$\mathcal{L} = -\sum_{i \in \mathcal{Y}_L} \sum_{f=1}^{F} Y_{if} \ln Z_{if}$$

### 5.3 GraphSAGE

**采样与聚合**

1. **邻居采样**：对每个节点采样固定数量邻居
2. **聚合邻居特征**：
   - Mean：$\mathbf{h}_{\mathcal{N}(v)}^k = \text{mean}(\{\mathbf{h}_u^{k-1}, \forall u \in \mathcal{N}(v)\})$
   - LSTM：$\mathbf{h}_{\mathcal{N}(v)}^k = \text{LSTM}(\{\mathbf{h}_u^{k-1}, \forall u \in \mathcal{N}(v)\})$
   - Pooling：$\mathbf{h}_{\mathcal{N}(v)}^k = \gamma(\{Q\mathbf{h}_u^{k-1}, \forall u \in \mathcal{N}(v)\})$

3. **更新节点表示**：
$$\mathbf{h}_v^k = \sigma\left(\mathbf{W}^k \cdot \text{CONCAT}(\mathbf{h}_v^{k-1}, \mathbf{h}_{\mathcal{N}(v)}^k)\right)$$

4. **归一化**：$\mathbf{h}_v^k \leftarrow \frac{\mathbf{h}_v^k}{||\mathbf{h}_v^k||_2}$

**归纳学习**

训练时：
$$\mathcal{L} = -\sum_{v \in \mathcal{V}_L} \log P(y_v | z_v)$$

测试时：可直接应用于未见过的图

### 5.4 GAT (Graph Attention Network)

**注意力机制**

$$e_{ij} = \text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}\mathbf{h}_i || \mathbf{W}\mathbf{h}_j])$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

**多头注意力**

$$\mathbf{h}_i' = \|_{k=1}^{K} \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j\right)$$

或平均：
$$\mathbf{h}_i' = \sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \mathbf{h}_j\right)$$

**优势**

- 邻居权重可解释
- 适用于异质图
- 计算高效

### 5.5 其他GNN变体

**GIN (Graph Isomorphism Network)**

$$\mathbf{h}_v^{(k)} = \text{MLP}^{(k)}\left((1 + \epsilon^{(k)}) \cdot \mathbf{h}_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)}\right)$$

理论证明：GIN与WL图同构测试一样强大

**MPNN (Message Passing Neural Network)**

消息传递：$\mathbf{m}_v^{(t+1)} = \sum_{w \in \mathcal{N}(v)} M_t(\mathbf{h}_v^{(t)}, \mathbf{h}_w^{(t)}, \mathbf{e}_{vw})$

更新：$\mathbf{h}_v^{(t+1)} = U_t(\mathbf{h}_v^{(t)}, \mathbf{m}_v^{(t+1)})$

读出：$\hat{y} = R(\{\mathbf{h}_v^{(T)} | v \in G\})$

**Graph Transformer**

将自注意力扩展到图：
$$\text{Attn}_{ij} = \frac{(W_Q \mathbf{h}_i)^T (W_K \mathbf{h}_j)}{\sqrt{d_k}} + b_{ij}$$

其中 $b_{ij}$ 为边相关的偏置

### 5.6 图嵌入

#### 5.6.1 DeepWalk

**随机游走**

从每个节点开始生成随机游走序列

**Skip-gram模型**

$$\min_{\Phi} -\log P(\{v_{i-w}, ..., v_{i+w}\} | \Phi(v_i))$$

**层次Softmax**

$$P(v_i | \Phi(v_j)) = \prod_{k=1}^{\lceil \log |V| \rceil} P(b_k | \Phi(v_j))$$

#### 5.6.2 Node2Vec

**有偏随机游走**

返回参数 $p$，进出参数 $q$

$$
P(c_i = x | c_{i-1} = v) = \begin{cases}
\frac{\pi_{vx}}{Z} & \text{if } (v,x) \in E \\
0 & \text{otherwise}
\end{cases}
$$

其中 $\pi_{vx} = \alpha_{pq}(t,x) \cdot w_{vx}$

$$
\alpha_{pq}(t,x) = \begin{cases}
\frac{1}{p} & \text{if } d_{tx} = 0 \\
1 & \text{if } d_{tx} = 1 \\
\frac{1}{q} & \text{if } d_{tx} = 2
\end{cases}
$$

**BFS vs DFS**

- $p$小，$q$大：DFS-like，学习同质性 (homophily)
- $p$大，$q$小：BFS-like，学习结构等价 (structural equivalence)

### 5.7 图神经网络任务

| 任务类型 | 输出 | 示例 |
|---------|------|------|
| 节点分类 | 每个节点的标签 | 社交网络用户分类 |
| 链接预测 | 边存在的概率 | 推荐系统 |
| 图分类 | 整个图的标签 | 分子性质预测 |
| 图生成 | 生成新图 | 分子设计 |

---

## 6. 多模态与前沿技术

### 6.1 多模态学习

#### 6.1.1 CLIP (Contrastive Language-Image Pre-training)

**模型架构**

- 文本编码器：Transformer
- 图像编码器：ResNet或Vision Transformer

**对比学习目标**

$$\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_I + \mathcal{L}_T \right)$$

其中：
$$\mathcal{L}_I = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\mathbf{x}_i \cdot \mathbf{y}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{x}_i \cdot \mathbf{y}_j / \tau)}$$

**零样本分类**

$$p(y=k|x) = \frac{\exp(\cos(f_I(x), f_T(t_k)) / \tau)}{\sum_{j=1}^{K} \exp(\cos(f_I(x), f_T(t_j)) / \tau)}$$

#### 6.1.2 DALL-E

**离散VAE (dVAE)**

- 将256×256图像压缩为32×32的token
- 词汇表大小8192

**自回归生成**

$$p_{\theta}(x, y) = p_\theta(y) \prod_{i=1}^{n} p_\theta(x_i | x_{<i}, y)$$

#### 6.1.3 Stable Diffusion

**隐空间扩散**

1. **VAE编码**：$z = \mathcal{E}(x)$
2. **扩散过程**：在隐空间进行
3. **U-Net去噪**：$\epsilon_\theta(z_t, t, c)$
4. **VAE解码**：$x = \mathcal{D}(z_0)$

**条件机制**

- 文本条件：CLIP文本编码器
- 分类器无关引导 (CFG)

### 6.2 大语言模型 (LLM)

#### 6.2.1 预训练策略

**自回归语言建模 (GPT)**

$$\mathcal{L} = -\sum_{i=1}^{N} \log P(x_i | x_{<i})$$

**掩码语言建模 (BERT)**

$$\mathcal{L} = -\mathbb{E}_{x \in \mathcal{D}} \sum_{i \in \mathcal{M}} \log P(x_i | x_{\backslash \mathcal{M}})$$

**前缀语言建模 (T5)**

$$\mathcal{L} = -\sum_{i=m+1}^{n} \log P(x_i | x_{\leq m}, x_{m+1:i-1})$$

#### 6.2.2 缩放定律 (Scaling Laws)

**Chinchilla定律**

最优计算量：$C \approx 6ND$，其中$N$为参数量，$D$为token数

最优配置：

- 参数量 $N_{opt} \propto C^{0.5}$
- 数据量 $D_{opt} \propto C^{0.5}$

**损失预测**

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

其中 $\alpha \approx 0.34$, $\beta \approx 0.28$

#### 6.2.3 高效微调方法

**LoRA (Low-Rank Adaptation)**

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$

训练时冻结$W$，只训练$A$和$B$

**QLoRA**

- 4-bit量化基模型
- 双量化 (Double Quantization)
- 分页优化器 (Paged Optimizers)
- NF4量化：4-bit Normal Float

**Prefix Tuning**

在输入前添加可学习的prefix embedding：
$$[P_1, ..., P_k, x_1, ..., x_n]$$

**Prompt Tuning**

$$h = [h_1, ..., h_k, e(x_1), ..., e(x_n)]$$

#### 6.2.4 RLHF (Reinforcement Learning from Human Feedback)

**三阶段流程**

1. **SFT (Supervised Fine-Tuning)**
   - 使用人工标注数据微调
   - 最大化似然：$\max_\pi \mathbb{E}_{(x,y) \sim \mathcal{D}} [\log \pi(y|x)]$

2. **奖励模型训练**
   - 收集偏好数据：$(x, y_w, y_l)$
   - Bradley-Terry模型：
   $$\mathcal{L}_R = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

3. **RL优化**
   - PPO目标：
   $$\mathcal{L}_{RL} = \mathbb{E}_{(x,y) \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right]$$

   - KL惩罚防止偏离参考模型太远

**DPO (Direct Preference Optimization)**

直接优化策略，无需显式奖励模型：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) \right]$$

### 6.3 Agent架构

#### 6.3.1 ReAct (Reasoning + Acting)

**交替执行**

$$
\text{Thought}_1 \rightarrow \text{Action}_1 \rightarrow \text{Observation}_1 \rightarrow \text{Thought}_2 \rightarrow ...
$$

**Prompt模板**

```
Question: {question}
Thought 1: {reasoning about what to do}
Action 1: {action to take}
Observation 1: {result of action}
...
Thought N: {final reasoning}
Answer: {final answer}
```

#### 6.3.2 Plan-and-Solve

1. **规划阶段**：LLM生成执行计划
2. **执行阶段**：按计划逐步执行
3. **反思阶段**：检查结果，必要时重新规划

#### 6.3.3 Multi-Agent系统

- **角色分配**：不同Agent负责不同任务
- **通信机制**：Agent间信息交换
- **协调策略**：共识达成、投票机制

### 6.4 RAG (Retrieval-Augmented Generation)

#### 6.4.1 基础架构

$$
P(y|x) = \sum_{z \in \mathcal{R}(x)} P(z|x) P(y|x, z)
$$

**组件**

1. **检索器 (Retriever)**
   - 稠密检索：DPR (Dense Passage Retrieval)
   - 稀疏检索：BM25

2. **生成器 (Generator)**
   - Seq2Seq模型：BART、T5
   - 自回归模型：GPT

#### 6.4.2 DPR (Dense Passage Retrieval)

**双编码器**

- 问题编码器：$E_Q(q)$
- 段落编码器：$E_P(p)$

**相似度**

$$\text{sim}(q, p) = E_Q(q)^T E_P(p)$$

**训练目标**

负对数似然：
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, p^+))}{\exp(\text{sim}(q, p^+)) + \sum_{p^-} \exp(\text{sim}(q, p^-))}$$

#### 6.4.3 高级RAG技术

**查询重写**

- 使用LLM改写查询以提高检索质量

**重排序 (Re-ranking)**

- 使用交叉编码器对检索结果重排序
- ColBERT：延迟交互模型

**迭代RAG**

- 多轮检索-生成
- 每轮基于之前结果优化查询

**Self-RAG**

模型自反思：

- Retrieve token：是否需要检索
- IsRel token：检索内容是否相关
- IsSup token：是否支持生成
- IsUse token：是否有用

---


## 7. 模型对比矩阵

### 7.1 监督学习模型对比

| 模型 | 优点 | 缺点 | 适用数据 | 计算成本 | 可解释性 |
|------|------|------|----------|----------|----------|
| **线性回归** | 简单快速，可解释 | 只能拟合线性关系 | 数值型，线性可分 | 极低 O(d³) | 高 |
| **逻辑回归** | 概率输出，正则化 | 特征工程要求高 | 分类任务，中小规模 | 低 O(nd) | 高 |
| **决策树** | 直观可解释，非线性 | 容易过拟合 | 混合类型 | 低 O(n·d·log n) | 高 |
| **随机森林** | 准确率高，抗过拟合 | 训练慢，黑盒 | 各种类型 | 中 O(B·n·d·log n) | 中 |
| **XGBoost** | 准确率高，速度快 | 超参数调优复杂 | 表格数据，竞赛首选 | 中 | 中 |
| **LightGBM** | 极快，内存友好 | 小数据集可能过拟合 | 大规模数据 | 低-中 | 中 |
| **CatBoost** | 自动处理类别特征 | 训练稍慢 | 含类别特征的数据 | 中 | 中 |
| **SVM** | 高维有效，泛化好 | 大数据集慢，调参难 | 中小规模，高维 | 高 O(n²)~O(n³) | 低 |
| **MLP** | 万能近似，非线性 | 需要大量数据 | 大规模数据 | 高 | 低 |

### 7.2 深度学习架构对比

| 架构 | 核心创新 | 参数量 | 适用任务 | 计算效率 |
|------|----------|--------|----------|----------|
| **LeNet** | 卷积+池化 | 60K | 手写数字 | 高 |
| **AlexNet** | ReLU+Dropout | 60M | 图像分类 | 中 |
| **VGG** | 小卷积核堆叠 | 138M | 图像分类 | 低 |
| **ResNet** | 残差连接 | 25M-60M | 各种视觉任务 | 高 |
| **DenseNet** | 密集连接 | 小 | 特征重用场景 | 中 |
| **EfficientNet** | 复合缩放 | 5M-66M | 移动端/边缘 | 极高 |
| **ViT** | 自注意力 | 86M-632M | 大规模图像 | 低 |
| **Swin Transformer** | 移位窗口 | 88M-197M | 各种视觉任务 | 中 |

### 7.3 序列模型对比

| 模型 | 长期依赖 | 并行计算 | 训练速度 | 推理速度 | 适用场景 |
|------|----------|----------|----------|----------|----------|
| **RNN** | 差 | 否 | 慢 | 慢 | 简单序列 |
| **LSTM** | 较好 | 否 | 慢 | 慢 | 需要记忆的任务 |
| **GRU** | 较好 | 否 | 中 | 中 | LSTM的轻量替代 |
| **Transformer** | 优秀 | 是 | 快 | 慢(自回归) | 长序列，NLP |
| **Linear Attention** | 优秀 | 是 | 快 | 快 | 超长序列 |
| **Mamba/SSM** | 优秀 | 是 | 快 | 极快 | 超长序列，线性复杂度 |

### 7.4 生成模型对比

| 模型 | 训练稳定性 | 采样质量 | 采样速度 | 似然计算 | 模式覆盖 |
|------|------------|----------|----------|----------|----------|
| **VAE** | 高 | 中 | 快 | 是 | 好 |
| **GAN** | 低 | 高 | 快 | 否 | 差(模式坍塌) |
| **Diffusion** | 高 | 极高 | 慢 | 近似 | 极好 |
| **Flow** | 高 | 高 | 快 | 精确 | 好 |
| **Autoregressive** | 高 | 高 | 慢 | 精确 | 好 |

### 7.5 聚类算法对比

| 算法 | 簇形状 | 需预设K | 噪声鲁棒 | 时间复杂度 | 适用场景 |
|------|--------|---------|----------|------------|----------|
| **K-means** | 球形 | 是 | 差 | O(n·K·d·I) | 大规模，球形簇 |
| **GMM** | 椭球形 | 是 | 中 | O(n·K·d²·I) | 需要概率输出 |
| **DBSCAN** | 任意 | 否 | 强 | O(n log n) | 噪声多，任意形状 |
| **层次聚类** | 任意 | 否 | 中 | O(n²)~O(n³) | 小规模，需要层次 |
| **谱聚类** | 任意 | 是 | 中 | O(n³) | 非凸形状 |

### 7.6 强化学习算法对比

| 算法 | 样本效率 | 稳定性 | 连续动作 | 大规模状态 | 适用场景 |
|------|----------|--------|----------|------------|----------|
| **Q-Learning** | 低 | 中 | 否 | 否 | 离散状态动作 |
| **DQN** | 中 | 中 | 否 | 是 | Atari游戏 |
| **Policy Gradient** | 低 | 低 | 是 | 是 | 简单连续控制 |
| **A2C/A3C** | 中 | 中 | 是 | 是 | 并行环境 |
| **PPO** | 高 | 高 | 是 | 是 | 通用首选 |
| **SAC** | 高 | 高 | 是 | 是 | 连续控制 |
| **TD3** | 高 | 高 | 是 | 是 | 连续控制 |

### 7.7 图神经网络对比

| 模型 | 归纳能力 | 可解释性 | 计算复杂度 | 表达能力 | 适用场景 |
|------|----------|----------|------------|----------|----------|
| **GCN** | 否 | 低 | O(L·|E|) | 中 | 直推学习 |
| **GraphSAGE** | 是 | 低 | O(L·n·k) | 中 | 归纳学习 |
| **GAT** | 是 | 高 | O(L·|E|) | 高 | 需要注意力解释 |
| **GIN** | 是 | 低 | O(L·|E|) | 极高 | 图同构任务 |
| **GatedGCN** | 是 | 低 | O(L·|E|) | 高 | 边特征重要 |

---

## 8. 模型选型决策树

### 8.1 通用机器学习问题选型

```
开始
│
├─ 有标签数据？
│  ├─ 否 → 无监督学习
│  │  ├─ 需要降维？ → PCA/UMAP/t-SNE
│  │  ├─ 需要聚类？
│  │  │  ├─ 知道簇数量？ → K-means/GMM
│  │  │  └─ 不知道簇数量？ → DBSCAN/层次聚类
│  │  └─ 异常检测？ → Isolation Forest/One-Class SVM
│  │
│  └─ 是 → 监督学习
│     ├─ 预测目标是？
│     │  ├─ 连续值 → 回归
│     │  │  ├─ 数据量 < 100K → 线性回归/Ridge/Lasso
│     │  │  └─ 数据量 > 100K → XGBoost/LightGBM/神经网络
│     │  │
│     │  └─ 离散值 → 分类
│     │     ├─ 数据量 < 10K → SVM/逻辑回归
│     │     ├─ 数据量 10K-1M → 随机森林/XGBoost
│     │     └─ 数据量 > 1M → LightGBM/神经网络
│     │
│     └─ 需要可解释性？
│        ├─ 是 → 决策树/线性模型
│        └─ 否 → 集成方法/深度学习
```

### 8.2 深度学习任务选型

```
数据类型？
│
├─ 图像数据
│  ├─ 分类任务
│  │  ├─ 数据量 < 10K → 迁移学习(ResNet/EfficientNet)
│  │  └─ 数据量 > 10K → 训练ResNet/EfficientNet/ViT
│  ├─ 检测任务 → Faster R-CNN/YOLO/RT-DETR
│  ├─ 分割任务 → U-Net/DeepLab/Mask R-CNN
│  └─ 生成任务 → GAN/Diffusion/Flow
│
├─ 文本数据
│  ├─ 分类/NER → BERT/RoBERTa/DeBERTa
│  ├─ 生成 → GPT/T5/BART
│  ├─ 翻译 → mBART/M2M-100/NLLB
│  └─ 问答 → T5/BERT+SQuAD微调
│
├─ 序列/时间序列
│  ├─ 短序列 → LSTM/GRU
│  ├─ 长序列 → Transformer/Informer
│  └─ 极长序列 → Mamba/Linear Attention
│
└─ 图数据
   ├─ 节点分类 → GCN/GAT/GraphSAGE
   ├─ 链接预测 → GAE/SEAL
   ├─ 图分类 → GIN/DGCNN
   └─ 图生成 → GraphRNN/VGAE
```

### 8.3 强化学习选型

```
环境特征？
│
├─ 状态空间
│  ├─ 离散小规模 → Q-Learning/SARSA
│  └─ 连续/大规模 → DQN/PPO/SAC
│
├─ 动作空间
│  ├─ 离散 → DQN/A3C/PPO
│  └─ 连续 → PPO/SAC/TD3
│
├─ 样本获取成本
│  ├─ 低成本 → On-policy (PPO/A2C)
│  └─ 高成本 → Off-policy (SAC/TD3/DQN)
│
└─ 环境数量
   ├─ 单环境 → SAC/TD3
   └─ 多环境并行 → PPO/A3C
```

### 8.4 大语言模型应用选型

```
应用场景？
│
├─ 文本分类
│  ├─ 通用分类 → BERT-base微调
│  └─ 长文本 → Longformer/BigBird
│
├─ 文本生成
│  ├─ 通用生成 → GPT-3.5/4 API
│  ├─ 领域特定 → 领域数据微调
│  └─ 可控生成 → 使用CFG/PPLM
│
├─ 问答系统
│  ├─ 封闭域 → 微调BERT/RoBERTa
│  └─ 开放域 → RAG + LLM
│
├─ 代码生成
│  └─ CodeLlama/Codex/StarCoder
│
└─ 多模态
   ├─ 图文理解 → CLIP/LLaVA
   ├─ 图像生成 → DALL-E/Stable Diffusion
   └─ 视频理解 → Video-LLaMA/VideoChat
```

---

## 9. 实践指南

### 9.1 超参数调优

#### 9.1.1 学习率调优

**学习率范围测试**

1. 从很小的学习率开始
2. 每个batch线性增加学习率
3. 记录损失变化
4. 选择损失下降最快的区间

**学习率调度策略**

| 策略 | 公式 | 适用场景 |
|------|------|----------|
| Step Decay | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$ | 通用 |
| Exponential | $\eta_t = \eta_0 \cdot e^{-kt}$ | 通用 |
| Cosine Annealing | $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})(1+\cos(\frac{t}{T}\pi))$ | 现代首选 |
| Warm Restart | 周期性重启 | 逃离局部最优 |
| One Cycle | 先增后减 | 快速收敛 |

**推荐初始学习率**

- SGD: 0.01-0.1
- Adam: 1e-4 to 3e-4
- AdamW: 1e-5 to 1e-4

#### 9.1.2 批大小选择

**经验法则**

- 小batch (32-64)：更好的泛化，更多噪声
- 大batch (256+)：训练更快，需要调整学习率

**线性缩放规则**

$$\eta_{new} = \eta_{base} \times \frac{batch\_size_{new}}{batch\_size_{base}}$$

#### 9.1.3 正则化参数

**Dropout率**

- 输入层：0.2
- 隐藏层：0.3-0.5
- 大型网络：0.5

**L2正则化**

- 通常1e-4 to 1e-2
- 与Dropout配合使用

**早停 (Early Stopping)**

```python
patience = 10  # 容忍轮数
min_delta = 0.001  # 最小改善阈值
best_loss = float('inf')
counter = 0

for epoch in range(max_epochs):
    val_loss = validate()
    if val_loss < best_loss - min_delta:
        best_loss = val_loss
        counter = 0
        save_checkpoint()
    else:
        counter += 1
        if counter >= patience:
            break  # 早停
```

#### 9.1.4 集成方法超参数

**随机森林**

- n_estimators: 100-500
- max_depth: None (生长到叶节点纯净)
- min_samples_split: 2-10
- max_features: sqrt(n_features)

**XGBoost/LightGBM**

- learning_rate: 0.01-0.3
- n_estimators: 100-1000
- max_depth: 3-10
- subsample: 0.6-1.0
- colsample_bytree: 0.6-1.0
- reg_alpha/reg_lambda: 0-1

### 9.2 数据预处理

#### 9.2.1 数值特征

**标准化 (Standardization)**

$$x' = \frac{x - \mu}{\sigma}$$

适用：SVM、神经网络、PCA

**归一化 (Normalization)**

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

适用：KNN、神经网络、图像数据

**稳健标准化**

$$x' = \frac{x - \text{median}}{\text{IQR}}$$

适用：含异常值的数据

#### 9.2.2 类别特征

**One-Hot编码**

- 低基数 (< 10)
- 无顺序关系

**Label编码**

- 高基数
- 树模型

**Target编码**

$$\hat{x}_{ik} = \frac{\sum_{j=1}^{n} \mathbb{1}(x_{jk} = x_{ik}) \cdot y_j + a \cdot p}{\sum_{j=1}^{n} \mathbb{1}(x_{jk} = x_{ik}) + a}$$

- 高基数
- 需要正则化防止过拟合

**Embedding**

- 神经网络
- 高基数类别

#### 9.2.3 文本预处理

**Tokenization**

- Word-based: BPE, WordPiece, SentencePiece
- Character-based
- Subword-based (推荐)

**序列处理**

- Padding/Truncating
- Attention Mask
- Position Encoding

### 9.3 评估指标

#### 9.3.1 分类任务

| 指标 | 公式 | 适用场景 |
|------|------|----------|
| **Accuracy** | $\frac{TP+TN}{TP+TN+FP+FN}$ | 平衡数据集 |
| **Precision** | $\frac{TP}{TP+FP}$ | 假阳性代价高 |
| **Recall** | $\frac{TP}{TP+FN}$ | 假阴性代价高 |
| **F1-Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | 不平衡数据 |
| **AUC-ROC** | ROC曲线下面积 | 阈值无关 |
| **AUC-PR** | PR曲线下面积 | 极度不平衡 |
| **Log Loss** | $-\frac{1}{N}\sum(y\log(p) + (1-y)\log(1-p))$ | 概率校准 |
| **Cohen's Kappa** | $\frac{p_o - p_e}{1 - p_e}$ | 一致性评估 |

**多分类扩展**

- Macro: 各类别平均
- Micro: 全局计算
- Weighted: 按支持度加权

#### 9.3.2 回归任务

| 指标 | 公式 | 特性 |
|------|------|------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | 对大误差敏感 |
| **RMSE** | $\sqrt{MSE}$ | 与目标同单位 |
| **MAE** | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | 对异常值鲁棒 |
| **MAPE** | $\frac{100\%}{n}\sum|\frac{y_i - \hat{y}_i}{y_i}|$ | 百分比误差 |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | 解释方差比例 |
| **Huber Loss** | 混合MSE和MAE | 平衡两者优点 |

#### 9.3.3 排序任务

**NDCG (Normalized Discounted Cumulative Gain)**

$$DCG_p = \sum_{i=1}^{p} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$$nDCG_p = \frac{DCG_p}{IDCG_p}$$

**MAP (Mean Average Precision)**

$$AP = \frac{1}{|R|} \sum_{k=1}^{n} P(k) \cdot rel(k)$$

#### 9.3.4 生成任务

| 指标 | 说明 | 适用 |
|------|------|------|
| **BLEU** | n-gram精确度 | 机器翻译 |
| **ROUGE** | n-gram召回率 | 摘要 |
| **METEOR** | 同义词、词干 | 翻译 |
| **CIDEr** | TF-IDF加权 | 图像描述 |
| **SPICE** | 语义命题 | 图像描述 |
| **Perplexity** | 概率模型困惑度 | 语言模型 |

#### 9.3.5 聚类评估

| 指标 | 有标签 | 说明 |
|------|--------|------|
| **Silhouette Score** | 否 | 簇内紧密度vs簇间分离度 |
| **Calinski-Harabasz Index** | 否 | 方差比 |
| **Davies-Bouldin Index** | 否 | 簇内距离/簇间距离 |
| **Adjusted Rand Index** | 是 | 调整后的Rand指数 |
| **Normalized Mutual Info** | 是 | 归一化互信息 |
| **V-Measure** | 是 | 同质性和完整性 |

### 9.4 交叉验证策略

**K折交叉验证**

```python
from sklearn.model_selection import KFold, StratifiedKFold

# 回归任务
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 分类任务（保持类别比例）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**时间序列分割**

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
```

**分组交叉验证**

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
```

### 9.5 模型诊断

#### 9.5.1 过拟合检测

**症状**

- 训练损失持续下降，验证损失上升
- 训练准确率很高，测试准确率很低

**解决方案**

1. 增加数据
2. 数据增强
3. 正则化 (L1/L2/Dropout)
4. 早停
5. 简化模型
6. 集成方法

#### 9.5.2 欠拟合检测

**症状**

- 训练和验证损失都很高
- 模型无法捕捉数据模式

**解决方案**

1. 增加模型复杂度
2. 减少正则化
3. 特征工程
4. 更长的训练时间
5. 更好的优化器

#### 9.5.3 学习曲线分析

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# 绘制学习曲线
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
```

**曲线解读**

- 两条曲线都高且接近：模型合适
- 训练曲线高，验证曲线低：过拟合
- 两条曲线都低：欠拟合

### 9.6 部署优化

#### 9.6.1 模型压缩

| 技术 | 压缩比 | 精度损失 | 适用场景 |
|------|--------|----------|----------|
| **剪枝** | 2-10x | 小 | 稀疏模型 |
| **量化** | 2-4x | 小 | 边缘部署 |
| **知识蒸馏** | 可变 | 可控 | 学生网络 |
| **低秩分解** | 2-5x | 小 | 大矩阵 |

#### 9.6.2 推理优化

**批处理推理**

- 合并多个请求
- 提高GPU利用率

**模型缓存**

- 缓存常用输入的输出
- KV Cache for LLM

**动态批处理**

- 实时合并请求
- 控制延迟

---

## 10. 总结

本文档提供了AI/ML领域主要模型和方法的全面分析，涵盖：

1. **监督学习**：从传统机器学习到深度学习的完整谱系
2. **深度学习**：CNN、RNN、Transformer等核心架构
3. **无监督学习**：聚类、降维、异常检测方法
4. **强化学习**：从基础算法到现代策略梯度方法
5. **图神经网络**：处理图结构数据的专用方法
6. **多模态与前沿**：CLIP、LLM、Agent、RAG等最新技术

**关键要点**：

- 没有最好的模型，只有最适合的模型
- 理解数据特性是选型的第一步
- 从简单模型开始，逐步增加复杂度
- 始终关注过拟合和泛化能力
- 模型可解释性与性能需要权衡

**推荐学习路径**：

1. 掌握基础：线性回归、逻辑回归、决策树
2. 理解集成：随机森林、XGBoost
3. 深入深度学习：CNN、RNN、Transformer
4. 探索前沿：LLM、多模态、Agent

---

*本文档持续更新，建议结合最新论文和实践进行调整。*
