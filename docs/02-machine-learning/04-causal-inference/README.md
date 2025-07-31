# 2.4 因果推理 / Causal Inference

## 概述 / Overview

因果推理理论研究如何从观察数据中识别和估计因果关系，为FormalAI提供因果发现和干预分析的数学基础。

Causal inference theory studies how to identify and estimate causal relationships from observational data, providing mathematical foundations for causal discovery and intervention analysis in FormalAI.

## 目录 / Table of Contents

- [2.4 因果推理 / Causal Inference](#24-因果推理--causal-inference)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [1. 因果图模型 / Causal Graph Models](#1-因果图模型--causal-graph-models)
    - [1.1 有向无环图 / Directed Acyclic Graphs](#11-有向无环图--directed-acyclic-graphs)
    - [1.2 路径分析 / Path Analysis](#12-路径分析--path-analysis)
    - [1.3 因果贝叶斯网络 / Causal Bayesian Networks](#13-因果贝叶斯网络--causal-bayesian-networks)
  - [2. 结构因果模型 / Structural Causal Models](#2-结构因果模型--structural-causal-models)
    - [2.1 SCM定义 / SCM Definition](#21-scm定义--scm-definition)
    - [2.2 干预 / Interventions](#22-干预--interventions)
    - [2.3 前门准则 / Front-Door Criterion](#23-前门准则--front-door-criterion)
  - [3. 因果发现 / Causal Discovery](#3-因果发现--causal-discovery)
    - [3.1 约束型方法 / Constraint-Based Methods](#31-约束型方法--constraint-based-methods)
    - [3.2 评分型方法 / Score-Based Methods](#32-评分型方法--score-based-methods)
    - [3.3 函数型方法 / Functional Methods](#33-函数型方法--functional-methods)
  - [4. 因果效应估计 / Causal Effect Estimation](#4-因果效应估计--causal-effect-estimation)
    - [4.1 平均因果效应 / Average Causal Effect](#41-平均因果效应--average-causal-effect)
    - [4.2 倾向得分 / Propensity Score](#42-倾向得分--propensity-score)
    - [4.3 工具变量 / Instrumental Variables](#43-工具变量--instrumental-variables)
  - [5. 反事实推理 / Counterfactual Reasoning](#5-反事实推理--counterfactual-reasoning)
    - [5.1 反事实定义 / Counterfactual Definition](#51-反事实定义--counterfactual-definition)
    - [5.2 反事实计算 / Counterfactual Computation](#52-反事实计算--counterfactual-computation)
    - [5.3 反事实公平性 / Counterfactual Fairness](#53-反事实公平性--counterfactual-fairness)
  - [6. 因果强化学习 / Causal Reinforcement Learning](#6-因果强化学习--causal-reinforcement-learning)
    - [6.1 因果MDP / Causal MDP](#61-因果mdp--causal-mdp)
    - [6.2 因果探索 / Causal Exploration](#62-因果探索--causal-exploration)
    - [6.3 因果策略梯度 / Causal Policy Gradient](#63-因果策略梯度--causal-policy-gradient)
  - [7. 因果机器学习 / Causal Machine Learning](#7-因果机器学习--causal-machine-learning)
    - [7.1 因果表示学习 / Causal Representation Learning](#71-因果表示学习--causal-representation-learning)
    - [7.2 因果迁移学习 / Causal Transfer Learning](#72-因果迁移学习--causal-transfer-learning)
    - [7.3 因果生成模型 / Causal Generative Models](#73-因果生成模型--causal-generative-models)
  - [8. 因果公平性 / Causal Fairness](#8-因果公平性--causal-fairness)
    - [8.1 因果公平性定义 / Causal Fairness Definitions](#81-因果公平性定义--causal-fairness-definitions)
    - [8.2 因果去偏见 / Causal Debiasing](#82-因果去偏见--causal-debiasing)
    - [8.3 因果公平性评估 / Causal Fairness Evaluation](#83-因果公平性评估--causal-fairness-evaluation)
  - [9. 因果解释性 / Causal Interpretability](#9-因果解释性--causal-interpretability)
    - [9.1 因果解释 / Causal Explanations](#91-因果解释--causal-explanations)
    - [9.2 因果注意力 / Causal Attention](#92-因果注意力--causal-attention)
    - [9.3 因果可解释性 / Causal Interpretability](#93-因果可解释性--causal-interpretability)
  - [10. 因果元学习 / Causal Meta-Learning](#10-因果元学习--causal-meta-learning)
    - [10.1 因果元学习 / Causal Meta-Learning](#101-因果元学习--causal-meta-learning)
    - [10.2 因果少样本学习 / Causal Few-Shot Learning](#102-因果少样本学习--causal-few-shot-learning)
    - [10.3 因果迁移学习 / Causal Transfer Learning](#103-因果迁移学习--causal-transfer-learning)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：因果发现算法](#rust实现因果发现算法)
    - [Haskell实现：因果效应估计](#haskell实现因果效应估计)
  - [参考文献 / References](#参考文献--references)

---

## 1. 因果图模型 / Causal Graph Models

### 1.1 有向无环图 / Directed Acyclic Graphs

**因果图 / Causal Graph:**

$G = (V, E)$ 其中 $V$ 是节点集合，$E$ 是有向边集合。

$G = (V, E)$ where $V$ is the set of nodes and $E$ is the set of directed edges.

**马尔可夫性质 / Markov Property:**

给定父节点，节点与其非后代节点条件独立。

Given parents, a node is conditionally independent of its non-descendants.

**因果马尔可夫假设 / Causal Markov Assumption:**

$$P(X_1, \ldots, X_n) = \prod_{i=1}^n P(X_i | \text{Pa}(X_i))$$

其中 $\text{Pa}(X_i)$ 是 $X_i$ 的父节点集合。

where $\text{Pa}(X_i)$ is the set of parents of $X_i$.

### 1.2 路径分析 / Path Analysis

**d-分离 / d-Separation:**

路径 $p$ 在节点集 $Z$ 下被d-分离，如果：

Path $p$ is d-separated by node set $Z$ if:

1. 路径包含链式结构 $A \rightarrow C \rightarrow B$ 且 $C \in Z$
2. 路径包含叉式结构 $A \leftarrow C \rightarrow B$ 且 $C \in Z$
3. 路径包含对撞结构 $A \rightarrow C \leftarrow B$ 且 $C \notin Z$ 且 $\text{Desc}(C) \cap Z = \emptyset$

**全局马尔可夫性质 / Global Markov Property:**

如果 $A$ 和 $B$ 在 $Z$ 下d-分离，那么 $A \perp\!\!\!\perp B | Z$。

If $A$ and $B$ are d-separated by $Z$, then $A \perp\!\!\!\perp B | Z$.

### 1.3 因果贝叶斯网络 / Causal Bayesian Networks

**因果贝叶斯网络 / Causal Bayesian Network:**

$B = (G, \Theta)$ 其中：

$B = (G, \Theta)$ where:

- $G$ 是因果图
- $\Theta$ 是条件概率参数

**参数化 / Parameterization:**

$$\theta_{ijk} = P(X_i = k | \text{Pa}(X_i) = j)$$

---

## 2. 结构因果模型 / Structural Causal Models

### 2.1 SCM定义 / SCM Definition

**结构因果模型 / Structural Causal Model:**

$M = (U, V, F, P(U))$ 其中：

$M = (U, V, F, P(U))$ where:

- $U$ 是外生变量集合 / set of exogenous variables
- $V$ 是内生变量集合 / set of endogenous variables
- $F$ 是结构函数集合 / set of structural functions
- $P(U)$ 是外生变量分布 / distribution of exogenous variables

**结构方程 / Structural Equations:**

$$X_i = f_i(\text{Pa}(X_i), U_i)$$

### 2.2 干预 / Interventions

**do-演算 / do-Calculus:**

$do(X = x)$ 表示将变量 $X$ 设置为值 $x$。

$do(X = x)$ means setting variable $X$ to value $x$.

**干预分布 / Interventional Distribution:**

$$P(Y | do(X = x)) = \sum_z P(Y | X = x, Z = z) P(Z = z)$$

**后门准则 / Backdoor Criterion:**

如果 $Z$ 满足后门准则，那么：

If $Z$ satisfies the backdoor criterion, then:

$$P(Y | do(X = x)) = \sum_z P(Y | X = x, Z = z) P(Z = z)$$

### 2.3 前门准则 / Front-Door Criterion

**前门路径 / Front-Door Path:**

$X \rightarrow M \rightarrow Y$ 其中 $M$ 是中介变量。

$X \rightarrow M \rightarrow Y$ where $M$ is a mediator.

**前门调整 / Front-Door Adjustment:**

$$P(Y | do(X = x)) = \sum_m P(M = m | X = x) \sum_{x'} P(Y | X = x', M = m) P(X = x')$$

---

## 3. 因果发现 / Causal Discovery

### 3.1 约束型方法 / Constraint-Based Methods

**PC算法 / PC Algorithm:**

1. 从完全无向图开始
2. 基于条件独立性测试删除边
3. 定向对撞结构
4. 应用定向规则

   1. Start with complete undirected graph
   2. Remove edges based on conditional independence tests
   3. Orient colliders
   4. Apply orientation rules

**SGS算法 / SGS Algorithm:**

PC算法的变体，使用不同的搜索策略。

Variant of PC algorithm with different search strategy.

### 3.2 评分型方法 / Score-Based Methods

**贝叶斯信息准则 / Bayesian Information Criterion:**

$$\text{BIC}(G) = \log P(D | G) - \frac{d}{2} \log n$$

其中 $d$ 是参数数量，$n$ 是样本数量。

where $d$ is the number of parameters and $n$ is the sample size.

**结构搜索 / Structure Search:**

$$\hat{G} = \arg\max_G \text{BIC}(G)$$

### 3.3 函数型方法 / Functional Methods

**ANM算法 / Additive Noise Model:**

$$Y = f(X) + N_Y$$

其中 $N_Y \perp\!\!\!\perp X$。

where $N_Y \perp\!\!\!\perp X$.

**LiNGAM算法 / Linear Non-Gaussian Additive Model:**

$$X_i = \sum_{j \in \text{Pa}(i)} b_{ij} X_j + N_i$$

其中 $N_i$ 是非高斯噪声。

where $N_i$ is non-Gaussian noise.

---

## 4. 因果效应估计 / Causal Effect Estimation

### 4.1 平均因果效应 / Average Causal Effect

**平均因果效应 / Average Causal Effect:**

$$\text{ACE} = \mathbb{E}[Y | do(X = 1)] - \mathbb{E}[Y | do(X = 0)]$$

**条件平均因果效应 / Conditional Average Causal Effect:**

$$\text{CACE} = \mathbb{E}[Y | do(X = 1), Z = z] - \mathbb{E}[Y | do(X = 0), Z = z]$$

### 4.2 倾向得分 / Propensity Score

**倾向得分 / Propensity Score:**

$$e(X) = P(T = 1 | X)$$

**倾向得分匹配 / Propensity Score Matching:**

$$\text{ACE} = \mathbb{E}_{e(X)}[\mathbb{E}[Y | T = 1, e(X)] - \mathbb{E}[Y | T = 0, e(X)]]$$

**倾向得分加权 / Propensity Score Weighting:**

$$\text{ACE} = \mathbb{E}\left[\frac{TY}{e(X)}\right] - \mathbb{E}\left[\frac{(1-T)Y}{1-e(X)}\right]$$

### 4.3 工具变量 / Instrumental Variables

**工具变量 / Instrumental Variable:**

$Z$ 是工具变量，如果：

$Z$ is an instrumental variable if:

1. $Z \rightarrow X$ (相关性)
2. $Z \perp\!\!\!\perp Y | X, U$ (排他性)
3. $Z \perp\!\!\!\perp U$ (独立性)

**两阶段最小二乘 / Two-Stage Least Squares:**

$$\hat{X} = \hat{\alpha}_0 + \hat{\alpha}_1 Z$$
$$\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 \hat{X}$$

---

## 5. 反事实推理 / Counterfactual Reasoning

### 5.1 反事实定义 / Counterfactual Definition

**反事实 / Counterfactual:**

"如果 $X$ 是 $x$ 而不是 $x'$，那么 $Y$ 会是 $y$"。

"If $X$ were $x$ instead of $x'$, then $Y$ would be $y$".

**反事实查询 / Counterfactual Query:**

$$P(Y_{X=x} = y | X = x', Y = y')$$

### 5.2 反事实计算 / Counterfactual Computation

**三步法 / Three-Step Method:**

1. **外推 / Extrapolation:** 估计外生变量 $U$
2. **干预 / Intervention:** 设置 $X = x$
3. **预测 / Prediction:** 计算 $Y$

**反事实分布 / Counterfactual Distribution:**

$$P(Y_{X=x} | X = x', Y = y') = \int P(Y_{X=x} | U) P(U | X = x', Y = y') dU$$

### 5.3 反事实公平性 / Counterfactual Fairness

**反事实公平性 / Counterfactual Fairness:**

$$\mathbb{E}[Y_{X=x} | A = a] = \mathbb{E}[Y_{X=x} | A = a']$$

其中 $A$ 是敏感属性。

where $A$ is the sensitive attribute.

---

## 6. 因果强化学习 / Causal Reinforcement Learning

### 6.1 因果MDP / Causal MDP

**因果MDP / Causal MDP:**

$M = (S, A, P, R, \gamma, C)$ 其中 $C$ 是因果图。

$M = (S, A, P, R, \gamma, C)$ where $C$ is the causal graph.

**因果转移函数 / Causal Transition Function:**

$$P(s' | s, a) = \prod_{i=1}^n P(s_i' | \text{Pa}(s_i'))$$

### 6.2 因果探索 / Causal Exploration

**因果UCB / Causal UCB:**

$$\text{UCB}(s, a) = \hat{Q}(s, a) + \sqrt{\frac{\log t}{N(s, a)}} + \text{CausalBonus}(s, a)$$

**因果奖励塑造 / Causal Reward Shaping:**

$$R'(s, a) = R(s, a) + \gamma \Phi(s') - \Phi(s)$$

其中 $\Phi$ 是因果势函数。

where $\Phi$ is the causal potential function.

### 6.3 因果策略梯度 / Causal Policy Gradient

**因果策略梯度 / Causal Policy Gradient:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s,a) \text{CausalWeight}(s,a)]$$

**因果权重 / Causal Weight:**

$$\text{CausalWeight}(s,a) = \frac{P(a | \text{Pa}(a))}{P(a | s)}$$

---

## 7. 因果机器学习 / Causal Machine Learning

### 7.1 因果表示学习 / Causal Representation Learning

**因果不变性 / Causal Invariance:**

$$P(Y | \text{Pa}(Y)) = P(Y | \text{Pa}(Y), E)$$

其中 $E$ 是环境变量。

where $E$ is the environment variable.

**因果表示 / Causal Representation:**

$$Z = f(X) \text{ s.t. } Z \text{ satisfies causal invariance}$$

### 7.2 因果迁移学习 / Causal Transfer Learning

**因果迁移 / Causal Transfer:**

$$\text{Transfer}(S, T) = \mathbb{E}_{x \sim P_T(x)}[f_S(x)]$$

其中 $f_S$ 是在源域 $S$ 上学习的因果模型。

where $f_S$ is the causal model learned on source domain $S$.

**因果域适应 / Causal Domain Adaptation:**

$$\min_f \mathcal{L}_{\text{source}}(f) + \lambda \mathcal{L}_{\text{causal}}(f)$$

### 7.3 因果生成模型 / Causal Generative Models

**因果GAN / Causal GAN:**

$$\min_G \max_D V(D, G) + \lambda \mathcal{L}_{\text{causal}}(G)$$

**因果VAE / Causal VAE:**

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{causal}}$$

---

## 8. 因果公平性 / Causal Fairness

### 8.1 因果公平性定义 / Causal Fairness Definitions

**反事实公平性 / Counterfactual Fairness:**

$$\mathbb{E}[Y_{X=x} | A = a] = \mathbb{E}[Y_{X=x} | A = a']$$

**路径特定公平性 / Path-Specific Fairness:**

$$\mathbb{E}[Y_{X=x, \pi} | A = a] = \mathbb{E}[Y_{X=x, \pi} | A = a']$$

其中 $\pi$ 是特定路径。

where $\pi$ is a specific path.

### 8.2 因果去偏见 / Causal Debiasing

**因果去偏见 / Causal Debiasing:**

$$\min_f \mathcal{L}_{\text{pred}}(f) + \lambda \mathcal{L}_{\text{fair}}(f)$$

**反事实数据增强 / Counterfactual Data Augmentation:**

$$D_{\text{aug}} = D \cup \{(x_{a \rightarrow a'}, y)\}$$

### 8.3 因果公平性评估 / Causal Fairness Evaluation

**因果公平性指标 / Causal Fairness Metrics:**

- 反事实公平性 / Counterfactual fairness
- 路径特定公平性 / Path-specific fairness
- 因果影响 / Causal influence

---

## 9. 因果解释性 / Causal Interpretability

### 9.1 因果解释 / Causal Explanations

**因果解释 / Causal Explanation:**

$$\text{Explanation}(x, y) = \{\text{Path}_1, \text{Path}_2, \ldots, \text{Path}_k\}$$

其中 $\text{Path}_i$ 是因果路径。

where $\text{Path}_i$ is a causal path.

**SHAP因果解释 / SHAP Causal Explanation:**

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$

### 9.2 因果注意力 / Causal Attention

**因果注意力 / Causal Attention:**

$$\alpha_{ij} = \frac{\exp(\text{CausalScore}(i,j))}{\sum_k \exp(\text{CausalScore}(i,k))}$$

**因果分数 / Causal Score:**

$$\text{CausalScore}(i,j) = \text{Attention}(i,j) \cdot \text{CausalWeight}(i,j)$$

### 9.3 因果可解释性 / Causal Interpretability

**因果可解释性 / Causal Interpretability:**

$$\text{Interpretability}(f) = \mathbb{E}_{x \sim P(x)}[\text{CausalExplanation}(f, x)]$$

---

## 10. 因果元学习 / Causal Meta-Learning

### 10.1 因果元学习 / Causal Meta-Learning

**因果元学习 / Causal Meta-Learning:**

$$\min_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})}[\mathcal{L}_{\text{causal}}(\theta, \mathcal{T})]$$

**因果快速适应 / Causal Fast Adaptation:**

$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\text{causal}}(\theta, \mathcal{T})$$

### 10.2 因果少样本学习 / Causal Few-Shot Learning

**因果原型网络 / Causal Prototype Network:**

$$c_k = \frac{1}{|S_k|} \sum_{(x_i, y_i) \in S_k} f_\phi(x_i)$$

**因果匹配网络 / Causal Matching Network:**

$$P(y | x, S) = \sum_{i=1}^k a(x, x_i) y_i$$

### 10.3 因果迁移学习 / Causal Transfer Learning

**因果迁移 / Causal Transfer:**

$$\mathcal{L}_{\text{transfer}} = \mathcal{L}_{\text{source}} + \lambda \mathcal{L}_{\text{causal}} + \mu \mathcal{L}_{\text{target}}$$

---

## 代码示例 / Code Examples

### Rust实现：因果发现算法

```rust
use std::collections::{HashMap, HashSet};
use ndarray::{Array2, Array1};

#[derive(Debug, Clone)]
struct CausalGraph {
    nodes: Vec<String>,
    edges: HashMap<(String, String), bool>,
    adjacency_matrix: Array2<bool>,
}

impl CausalGraph {
    fn new(nodes: Vec<String>) -> Self {
        let n = nodes.len();
        CausalGraph {
            nodes,
            edges: HashMap::new(),
            adjacency_matrix: Array2::from_elem((n, n), false),
        }
    }
    
    fn add_edge(&mut self, from: &str, to: &str) {
        let from_idx = self.nodes.iter().position(|x| x == from).unwrap();
        let to_idx = self.nodes.iter().position(|x| x == to).unwrap();
        
        self.edges.insert((from.to_string(), to.to_string()), true);
        self.adjacency_matrix[[from_idx, to_idx]] = true;
    }
    
    fn remove_edge(&mut self, from: &str, to: &str) {
        let from_idx = self.nodes.iter().position(|x| x == from).unwrap();
        let to_idx = self.nodes.iter().position(|x| x == to).unwrap();
        
        self.edges.remove(&(from.to_string(), to.to_string()));
        self.adjacency_matrix[[from_idx, to_idx]] = false;
    }
    
    fn get_parents(&self, node: &str) -> Vec<String> {
        let node_idx = self.nodes.iter().position(|x| x == node).unwrap();
        let mut parents = Vec::new();
        
        for (i, parent) in self.nodes.iter().enumerate() {
            if self.adjacency_matrix[[i, node_idx]] {
                parents.push(parent.clone());
            }
        }
        
        parents
    }
    
    fn get_children(&self, node: &str) -> Vec<String> {
        let node_idx = self.nodes.iter().position(|x| x == node).unwrap();
        let mut children = Vec::new();
        
        for (i, child) in self.nodes.iter().enumerate() {
            if self.adjacency_matrix[[node_idx, i]] {
                children.push(child.clone());
            }
        }
        
        children
    }
    
    fn is_d_separated(&self, x: &str, y: &str, z: &[String]) -> bool {
        // 简化的d-分离检查
        let x_parents = self.get_parents(x);
        let y_parents = self.get_parents(y);
        
        // 检查是否有共同父节点在Z中
        for parent in &x_parents {
            if z.contains(parent) && y_parents.contains(parent) {
                return true;
            }
        }
        
        false
    }
}

#[derive(Debug, Clone)]
struct PCAlgorithm {
    data: Array2<f64>,
    alpha: f64,
}

impl PCAlgorithm {
    fn new(data: Array2<f64>, alpha: f64) -> Self {
        PCAlgorithm { data, alpha }
    }
    
    fn independence_test(&self, x: usize, y: usize, z: &[usize]) -> bool {
        // 简化的独立性测试（卡方检验）
        let n = self.data.shape()[0];
        let mut contingency = Array2::zeros((2, 2));
        
        for i in 0..n {
            let x_val = if self.data[[i, x]] > 0.0 { 1 } else { 0 };
            let y_val = if self.data[[i, y]] > 0.0 { 1 } else { 0 };
            contingency[[x_val, y_val]] += 1.0;
        }
        
        // 计算卡方统计量
        let total = contingency.sum();
        let expected = Array2::zeros((2, 2));
        for i in 0..2 {
            for j in 0..2 {
                expected[[i, j]] = contingency.row(i).sum() * contingency.column(j).sum() / total;
            }
        }
        
        let chi_square = ((contingency - &expected).mapv(|x| x * x) / &expected).sum();
        chi_square < 3.841 // 0.05显著性水平
    }
    
    fn run(&self) -> CausalGraph {
        let n_vars = self.data.shape()[1];
        let mut graph = CausalGraph::new(
            (0..n_vars).map(|i| format!("X{}", i)).collect()
        );
        
        // 步骤1: 从完全无向图开始
        for i in 0..n_vars {
            for j in (i+1)..n_vars {
                graph.add_edge(&format!("X{}", i), &format!("X{}", j));
                graph.add_edge(&format!("X{}", j), &format!("X{}", i));
            }
        }
        
        // 步骤2: 基于独立性测试删除边
        let mut l = 0;
        while l < n_vars {
            for i in 0..n_vars {
                for j in (i+1)..n_vars {
                    let neighbors_i: Vec<usize> = (0..n_vars)
                        .filter(|&k| k != i && k != j && graph.edges.contains_key(&(format!("X{}", i), format!("X{}", k))))
                        .collect();
                    
                    if neighbors_i.len() >= l {
                        for subset in self.combinations(&neighbors_i, l) {
                            if self.independence_test(i, j, &subset) {
                                graph.remove_edge(&format!("X{}", i), &format!("X{}", j));
                                graph.remove_edge(&format!("X{}", j), &format!("X{}", i));
                                break;
                            }
                        }
                    }
                }
            }
            l += 1;
        }
        
        graph
    }
    
    fn combinations(&self, items: &[usize], k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        if items.is_empty() {
            return vec![];
        }
        
        let mut result = Vec::new();
        for i in 0..=items.len()-k {
            let mut combo = vec![items[i]];
            for sub_combo in self.combinations(&items[i+1..], k-1) {
                combo.extend(sub_combo);
                result.push(combo.clone());
                combo.truncate(1);
            }
        }
        result
    }
}

fn main() {
    // 生成示例数据
    let n_samples = 1000;
    let n_vars = 4;
    let mut data = Array2::zeros((n_samples, n_vars));
    
    // 模拟因果结构: X0 -> X1 -> X3, X0 -> X2 -> X3
    for i in 0..n_samples {
        let x0 = rand::random::<f64>();
        let x1 = x0 + 0.5 * rand::random::<f64>();
        let x2 = x0 + 0.3 * rand::random::<f64>();
        let x3 = x1 + x2 + 0.2 * rand::random::<f64>();
        
        data[[i, 0]] = x0;
        data[[i, 1]] = x1;
        data[[i, 2]] = x2;
        data[[i, 3]] = x3;
    }
    
    let pc = PCAlgorithm::new(data, 0.05);
    let causal_graph = pc.run();
    
    println!("发现的因果图:");
    for edge in causal_graph.edges.keys() {
        println!("{} -> {}", edge.0, edge.1);
    }
}
```

### Haskell实现：因果效应估计

```haskell
import Data.List (foldl')
import Numeric.LinearAlgebra

-- 因果效应估计
data CausalEffect = CausalEffect {
    treatment :: String,
    outcome :: String,
    effect :: Double,
    confidence :: Double
} deriving Show

-- 倾向得分匹配
propensityScoreMatching :: Matrix Double -> Vector Double -> Vector Double -> CausalEffect
propensityScoreMatching covariates treatment outcome = 
    let -- 估计倾向得分
        propensityScores = estimatePropensityScore covariates treatment
        
        -- 匹配处理组和对照组
        matchedPairs = matchOnPropensityScore propensityScores treatment outcome
        
        -- 计算平均因果效应
        effect = calculateATE matchedPairs
        
        -- 计算置信区间
        confidence = calculateConfidence matchedPairs
    in CausalEffect "treatment" "outcome" effect confidence

-- 估计倾向得分
estimatePropensityScore :: Matrix Double -> Vector Double -> Vector Double
estimatePropensityScore covariates treatment = 
    let -- 逻辑回归估计
        n = rows covariates
        ones = konst 1.0 n
        designMatrix = fromColumns [ones, covariates]
        
        -- 最大似然估计
        coefficients = logisticRegression designMatrix treatment
        
        -- 计算倾向得分
        linearPredictor = designMatrix `mult` coefficients
        propensityScores = mapVector sigmoid linearPredictor
    in propensityScores

-- 逻辑回归
logisticRegression :: Matrix Double -> Vector Double -> Vector Double
logisticRegression x y = 
    let -- 迭代重加权最小二乘
        n = rows x
        p = cols x
        beta = konst 0.0 p
        
        -- 牛顿-拉夫森迭代
        finalBeta = newtonRaphson x y beta
    in finalBeta

-- 牛顿-拉夫森迭代
newtonRaphson :: Matrix Double -> Vector Double -> Vector Double -> Vector Double
newtonRaphson x y beta = 
    let maxIter = 100
        tolerance = 1e-6
    in iterate newtonStep beta !! maxIter
where
    newtonStep beta = 
        let -- 计算预测概率
            linearPredictor = x `mult` beta
            probabilities = mapVector sigmoid linearPredictor
            
            -- 计算梯度
            gradient = x `mult` (y - probabilities)
            
            -- 计算Hessian
            weights = mapVector (\p -> p * (1 - p)) probabilities
            hessian = x `mult` (asColumn weights * x)
            
            -- 更新参数
            hessianInv = inv hessian
            step = hessianInv `mult` gradient
        in beta + step

-- Sigmoid函数
sigmoid :: Double -> Double
sigmoid x = 1.0 / (1.0 + exp (-x))

-- 匹配函数
matchOnPropensityScore :: Vector Double -> Vector Double -> Vector Double -> [(Double, Double)]
matchOnPropensityScore propensityScores treatment outcome = 
    let -- 分离处理组和对照组
        treatedIndices = findIndices (> 0.5) treatment
        controlIndices = findIndices (<= 0.5) treatment
        
        treatedScores = subVector propensityScores treatedIndices
        controlScores = subVector propensityScores controlIndices
        treatedOutcomes = subVector outcome treatedIndices
        controlOutcomes = subVector outcome controlIndices
        
        -- 最近邻匹配
        matches = zipWith (\i j -> (treatedOutcomes ! i, controlOutcomes ! j)) 
                         [0..] (nearestNeighborMatching treatedScores controlScores)
    in matches

-- 最近邻匹配
nearestNeighborMatching :: Vector Double -> Vector Double -> [Int]
nearestNeighborMatching treatedScores controlScores = 
    let nTreated = dim treatedScores
        nControl = dim controlScores
        distances = matrix nTreated nControl $ \(i, j) -> 
            abs (treatedScores ! i - controlScores ! j)
        
        -- 贪心匹配
        matches = greedyMatching distances
    in matches

-- 贪心匹配
greedyMatching :: Matrix Double -> [Int]
greedyMatching distances = 
    let n = rows distances
        m = cols distances
        used = replicate m False
        
        findMatches i used
            | i >= n = []
            | otherwise = 
                let bestMatch = findBestMatch i used
                    newUsed = take bestMatch used ++ [True] ++ drop (bestMatch + 1) used
                in bestMatch : findMatches (i + 1) newUsed
    in findMatches 0 used
where
    findBestMatch i used = 
        let available = [j | j <- [0..m-1], not (used !! j)]
            distances_i = [distances ! (i, j) | j <- available]
            minIndex = argmin distances_i
        in available !! minIndex

-- 计算平均因果效应
calculateATE :: [(Double, Double)] -> Double
calculateATE matches = 
    let differences = map (\(y1, y0) -> y1 - y0) matches
    in sum differences / fromIntegral (length differences)

-- 计算置信区间
calculateConfidence :: [(Double, Double)] -> Double
calculateConfidence matches = 
    let differences = map (\(y1, y0) -> y1 - y0) matches
        mean = sum differences / fromIntegral (length differences)
        variance = sum (map (\d -> (d - mean) ^ 2) differences) / fromIntegral (length differences - 1)
        stdError = sqrt (variance / fromIntegral (length differences))
    in 1.96 * stdError -- 95%置信区间

-- 示例
main :: IO ()
main = do
    -- 生成模拟数据
    let n = 1000
        covariates = matrix n 3 $ \(i, j) -> 
            case j of
                0 -> 1.0  -- 截距
                1 -> fromIntegral (i `mod` 2)  -- 二元变量
                2 -> fromIntegral (i `mod` 3) / 2.0  -- 连续变量
                _ -> 0.0
        
        -- 生成处理变量
        treatment = vector $ take n $ cycle [1.0, 0.0, 1.0, 0.0]
        
        -- 生成结果变量
        outcome = vector $ take n $ cycle [1.2, 0.8, 1.5, 0.6]
    
    let causalEffect = propensityScoreMatching covariates treatment outcome
    
    putStrLn "因果效应估计结果:"
    print causalEffect
    putStrLn $ "平均因果效应: " ++ show (effect causalEffect)
    putStrLn $ "95%置信区间: ±" ++ show (confidence causalEffect)
```

---

## 参考文献 / References

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
2. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.
3. Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference*. MIT Press.
4. Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall.
5. Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference in Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
6. Kusner, M. J., et al. (2017). Counterfactual fairness. *NIPS*.
7. Zhang, J., & Bareinboim, E. (2018). Fairness in decision-making—the causal explanation formula. *AAAI*.
8. Schölkopf, B. (2019). Causality for machine learning. *arXiv*.

---

*本模块为FormalAI提供了全面的因果推理理论基础，涵盖了从因果发现到因果机器学习的完整理论体系。*
