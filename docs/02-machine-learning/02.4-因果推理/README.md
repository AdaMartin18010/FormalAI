# 2.4 因果推理理论 / Causal Inference Theory / Kausale Inferenztheorie / Théorie de l'inférence causale

[返回全局导航](../../GLOBAL_NAVIGATION.md) · [学习路径](../../LEARNING_PATH_DESIGN.md)

## 概述 / Overview

因果推理理论研究如何从观察数据中识别和估计因果关系，为AI系统的决策和干预提供理论基础。

Causal inference theory studies how to identify and estimate causal relationships from observational data, providing theoretical foundations for decision-making and intervention in AI systems.

### do-演算与调整公式 / Do-Calculus and Adjustment / Do-Kalkül und Adjustierung / Calcul do et ajustement

- 后门准则（Backdoor Criterion）: 集合 \(Z\) 阻断 \(X\to Y\) 所有后门路径且不包含 \(X\) 的后代
- 后门调整：

\[ P(y\mid do(x)) = \sum_{z} P(y\mid x,z) P(z) \]

- 前门调整（Frontdoor，存在适当中介 \(Z\) 时）：

\[ P(y\mid do(x)) = \sum_{z} P(z\mid x) \sum_{x'} P(y\mid z, x') P(x') \]

- do-演算规则（示意）：
  1) 插入/删除观测
  2) 插入/删除干预
  3) 交换干预与观测（满足相应d-分离条件）

## 目录 / Table of Contents

- [2.4 因果推理理论 / Causal Inference Theory / Kausale Inferenztheorie / Théorie de l'inférence causale](#24-因果推理理论--causal-inference-theory--kausale-inferenztheorie--théorie-de-linférence-causale)
  - [概述 / Overview](#概述--overview)
    - [do-演算与调整公式 / Do-Calculus and Adjustment / Do-Kalkül und Adjustierung / Calcul do et ajustement](#do-演算与调整公式--do-calculus-and-adjustment--do-kalkül-und-adjustierung--calcul-do-et-ajustement)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes](#相关章节--related-chapters--verwandte-kapitel--chapitres-connexes)
  - [1. 因果图模型 / Causal Graphical Models](#1-因果图模型--causal-graphical-models)
    - [1.1 有向无环图 / Directed Acyclic Graphs](#11-有向无环图--directed-acyclic-graphs)
    - [1.2 因果马尔可夫条件 / Causal Markov Condition](#12-因果马尔可夫条件--causal-markov-condition)
    - [1.3 因果忠实性 / Causal Faithfulness](#13-因果忠实性--causal-faithfulness)
  - [2. 结构因果模型 / Structural Causal Models](#2-结构因果模型--structural-causal-models)
    - [2.1 结构方程 / Structural Equations](#21-结构方程--structural-equations)
    - [2.2 反事实推理 / Counterfactual Reasoning](#22-反事实推理--counterfactual-reasoning)
    - [2.3 因果层次 / Causal Hierarchy](#23-因果层次--causal-hierarchy)
  - [3. 因果发现 / Causal Discovery](#3-因果发现--causal-discovery)
    - [3.1 PC算法 / PC Algorithm](#31-pc算法--pc-algorithm)
    - [3.2 GES算法 / GES Algorithm](#32-ges算法--ges-algorithm)
    - [3.3 约束学习 / Constraint-Based Learning](#33-约束学习--constraint-based-learning)
  - [4. 因果效应估计 / Causal Effect Estimation](#4-因果效应估计--causal-effect-estimation)
    - [4.1 平均因果效应 / Average Causal Effect](#41-平均因果效应--average-causal-effect)
    - [4.2 倾向得分 / Propensity Score](#42-倾向得分--propensity-score)
    - [4.3 工具变量 / Instrumental Variables](#43-工具变量--instrumental-variables)
  - [5. 因果机器学习 / Causal Machine Learning](#5-因果机器学习--causal-machine-learning)
    - [5.1 因果森林 / Causal Forests](#51-因果森林--causal-forests)
    - [5.2 因果神经网络 / Causal Neural Networks](#52-因果神经网络--causal-neural-networks)
    - [5.3 因果强化学习 / Causal Reinforcement Learning](#53-因果强化学习--causal-reinforcement-learning)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：因果发现算法](#rust实现因果发现算法)
    - [Haskell实现：因果效应估计](#haskell实现因果效应估计)
  - [参考文献 / References](#参考文献--references)
  - [2024/2025 最新进展 / Latest Updates / Neueste Entwicklungen / Derniers développements](#20242025-最新进展--latest-updates--neueste-entwicklungen--derniers-développements)
    - [大规模因果推理 / Large-Scale Causal Inference](#大规模因果推理--large-scale-causal-inference)
    - [深度因果推理 / Deep Causal Inference](#深度因果推理--deep-causal-inference)
    - [因果推理应用理论 / Causal Inference Application Theory](#因果推理应用理论--causal-inference-application-theory)
    - [大模型因果推理 / Large Model Causal Inference](#大模型因果推理--large-model-causal-inference)
    - [因果AI对齐 / Causal AI Alignment](#因果ai对齐--causal-ai-alignment)
    - [实用工具链 / Practical Toolchain](#实用工具链--practical-toolchain)
  - [进一步阅读（2025 持续滚动） / Further Reading (Rolling 2025)](#进一步阅读2025-持续滚动--further-reading-rolling-2025)

---

## 相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes

**前置依赖 / Prerequisites / Voraussetzungen / Prérequis:**

- [0.0 ZFC公理系统](../../00-foundations/00-mathematical-foundations/00-set-theory-zfc.md) - 提供集合论基础 / Provides set theory foundation
- [1.1 形式化逻辑基础](../01-foundations/01-formal-logic/README.md) - 提供逻辑基础 / Provides logical foundation
- [2.1 统计学习理论](01-statistical-learning-theory/README.md) - 提供统计基础 / Provides statistical foundation

**后续应用 / Applications / Anwendungen / Applications:**

- [6.2 公平性与偏见理论](../06-interpretable-ai/02-fairness-bias/README.md) - 提供因果基础 / Provides causal foundation
- [6.3 鲁棒性理论](../06-interpretable-ai/03-robustness-theory/README.md) - 提供推理基础 / Provides reasoning foundation

---

## 1. 因果图模型 / Causal Graphical Models

### 1.1 有向无环图 / Directed Acyclic Graphs

**因果图 / Causal Graph:**

因果图 $G = (V, E)$ 是一个有向无环图，其中：

Causal graph $G = (V, E)$ is a directed acyclic graph where:

- $V$ 是变量集合 / set of variables
- $E$ 是有向边集合 / set of directed edges

**父节点 / Parents:**

$$\text{Pa}(X) = \{Y : Y \rightarrow X \in E\}$$

**子节点 / Children:**

$$\text{Ch}(X) = \{Y : X \rightarrow Y \in E\}$$

### 1.2 因果马尔可夫条件 / Causal Markov Condition

**因果马尔可夫条件 / Causal Markov Condition:**

给定其父节点，每个变量独立于其非后代：

Given its parents, each variable is independent of its non-descendants:

$$X \perp\!\!\!\perp \text{NonDesc}(X) | \text{Pa}(X)$$

**全局马尔可夫条件 / Global Markov Condition:**

如果 $X$ 和 $Y$ 被 $Z$ d-分离，则 $X \perp\!\!\!\perp Y | Z$。

If $X$ and $Y$ are d-separated by $Z$, then $X \perp\!\!\!\perp Y | Z$.

### 1.3 因果忠实性 / Causal Faithfulness

**因果忠实性 / Causal Faithfulness:**

图中的独立性关系完全由d-分离决定：

Independence relations in the graph are completely determined by d-separation:

$$X \perp\!\!\!\perp Y | Z \Leftrightarrow \text{d-sep}(X, Y | Z)$$

---

## 2. 结构因果模型 / Structural Causal Models

### 2.1 结构方程 / Structural Equations

**结构方程 / Structural Equations:**

$$X_i = f_i(\text{Pa}(X_i), U_i)$$

其中 $U_i$ 是外生变量。

where $U_i$ are exogenous variables.

**递归性 / Recursiveness:**

$$X_i = f_i(X_1, \ldots, X_{i-1}, U_i)$$

### 2.2 反事实推理 / Counterfactual Reasoning

**反事实 / Counterfactual:**

$$Y_{X=x}(u) = Y(f_1(u), \ldots, f_{X=x}(u), \ldots, f_n(u))$$

其中 $f_{X=x}$ 是将 $X$ 设置为 $x$ 的干预函数。

where $f_{X=x}$ is the intervention function setting $X$ to $x$.

**do-演算 / do-Calculus:**

$$P(Y | do(X=x)) = \sum_z P(Y | X=x, Z=z) P(Z=z)$$

### 2.3 因果层次 / Causal Hierarchy

**关联层 / Association Layer:**

$$P(Y | X)$$

**干预层 / Intervention Layer:**

$$P(Y | do(X))$$

**反事实层 / Counterfactual Layer:**

$$P(Y_{X=x} | X=x', Y=y')$$

---

## 3. 因果发现 / Causal Discovery

### 3.1 PC算法 / PC Algorithm

**PC算法步骤 / PC Algorithm Steps:**

1. **骨架学习 / Skeleton Learning:**
   - 从完全无向图开始
   - 通过独立性测试删除边

2. **方向学习 / Orientation Learning:**
   - 使用v-结构规则
   - 应用方向传播规则

**独立性测试 / Independence Test:**

$$\text{Test}(X \perp\!\!\!\perp Y | Z)$$

### 3.2 GES算法 / GES Algorithm

**GES算法 / GES Algorithm:**

1. **前向阶段 / Forward Phase:**
   - 从空图开始
   - 贪婪添加边

2. **后向阶段 / Backward Phase:**
   - 贪婪删除边

**评分函数 / Scoring Function:**

$$S(G, D) = \log P(D | G) - \text{penalty}(G)$$

### 3.3 约束学习 / Constraint-Based Learning

**约束学习 / Constraint-Based Learning:**

基于独立性约束学习因果结构：

Learn causal structure based on independence constraints:

$$\mathcal{I} = \{(X, Y, Z) : X \perp\!\!\!\perp Y | Z\}$$

---

## 4. 因果效应估计 / Causal Effect Estimation

### 4.1 平均因果效应 / Average Causal Effect

**平均因果效应 / Average Causal Effect:**

$$\text{ACE} = \mathbb{E}[Y_{X=1}] - \mathbb{E}[Y_{X=0}]$$

**识别条件 / Identification Conditions:**

1. 无混淆假设 / No Confounding Assumption
2. 正概率假设 / Positivity Assumption
3. 稳定单元值假设 / Stable Unit Treatment Value Assumption

### 4.2 倾向得分 / Propensity Score

**倾向得分 / Propensity Score:**

$$e(X) = P(T=1 | X)$$

**倾向得分定理 / Propensity Score Theorem:**

$$T \perp\!\!\!\perp X | e(X)$$

**逆概率加权 / Inverse Probability Weighting:**

$$\text{ACE} = \mathbb{E}\left[\frac{TY}{e(X)}\right] - \mathbb{E}\left[\frac{(1-T)Y}{1-e(X)}\right]$$

### 4.3 工具变量 / Instrumental Variables

**工具变量 / Instrumental Variables:**

变量 $Z$ 是 $X$ 对 $Y$ 的工具变量，如果：

Variable $Z$ is an instrumental variable for $X$ on $Y$ if:

1. $Z \rightarrow X$ (相关性)
2. $Z \perp\!\!\!\perp Y | X$ (排他性)
3. $Z \perp\!\!\!\perp U$ (外生性)

**两阶段最小二乘 / Two-Stage Least Squares:**

$$\hat{X} = \hat{\alpha}_0 + \hat{\alpha}_1 Z$$
$$Y = \hat{\beta}_0 + \hat{\beta}_1 \hat{X}$$

---

## 5. 因果机器学习 / Causal Machine Learning

### 5.1 因果森林 / Causal Forests

**因果森林 / Causal Forests:**

$$\hat{\tau}(x) = \frac{1}{|\{i : X_i \in L(x)\}|} \sum_{i : X_i \in L(x)} \tau_i$$

其中 $L(x)$ 是包含 $x$ 的叶子节点。

where $L(x)$ is the leaf containing $x$.

**诚实估计 / Honest Estimation:**

使用不同样本进行分割和估计：

Use different samples for splitting and estimation.

### 5.2 因果神经网络 / Causal Neural Networks

**因果神经网络 / Causal Neural Networks:**

$$f_\theta(x, t) = \mu_\theta(x) + \tau_\theta(x) \cdot t$$

其中 $t$ 是处理指示符。

where $t$ is the treatment indicator.

**因果正则化 / Causal Regularization:**

$$\mathcal{L} = \mathcal{L}_{\text{pred}} + \lambda \mathcal{L}_{\text{causal}}$$

### 5.3 因果强化学习 / Causal Reinforcement Learning

**因果强化学习 / Causal Reinforcement Learning:**

使用因果模型指导探索：

Use causal models to guide exploration:

$$\pi(a|s) = \pi_{\text{base}}(a|s) + \pi_{\text{causal}}(a|s)$$

---

## 代码示例 / Code Examples

### Rust实现：因果发现算法

```rust
use std::collections::{HashMap, HashSet};
use rand::Rng;

#[derive(Debug, Clone)]
struct CausalGraph {
    nodes: Vec<String>,
    edges: HashMap<(String, String), bool>,
    adj_matrix: Vec<Vec<bool>>,
}

impl CausalGraph {
    fn new(nodes: Vec<String>) -> Self {
        let n = nodes.len();
        let mut edges = HashMap::new();
        let adj_matrix = vec![vec![false; n]; n];
        
        CausalGraph {
            nodes,
            edges,
            adj_matrix,
        }
    }
    
    fn add_edge(&mut self, from: &str, to: &str) {
        let from_idx = self.get_node_index(from);
        let to_idx = self.get_node_index(to);
        
        if from_idx.is_some() && to_idx.is_some() {
            let from_idx = from_idx.unwrap();
            let to_idx = to_idx.unwrap();
            
            self.edges.insert((from.to_string(), to.to_string()), true);
            self.adj_matrix[from_idx][to_idx] = true;
        }
    }
    
    fn get_node_index(&self, node: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n == node)
    }
    
    fn get_parents(&self, node: &str) -> Vec<String> {
        let node_idx = self.get_node_index(node);
        if node_idx.is_none() {
            return Vec::new();
        }
        
        let node_idx = node_idx.unwrap();
        let mut parents = Vec::new();
        
        for (i, _) in self.nodes.iter().enumerate() {
            if self.adj_matrix[i][node_idx] {
                parents.push(self.nodes[i].clone());
            }
        }
        
        parents
    }
    
    fn get_children(&self, node: &str) -> Vec<String> {
        let node_idx = self.get_node_index(node);
        if node_idx.is_none() {
            return Vec::new();
        }
        
        let node_idx = node_idx.unwrap();
        let mut children = Vec::new();
        
        for (i, _) in self.nodes.iter().enumerate() {
            if self.adj_matrix[node_idx][i] {
                children.push(self.nodes[i].clone());
            }
        }
        
        children
    }
    
    fn is_d_separated(&self, x: &str, y: &str, z: &[String]) -> bool {
        // 简化的d-分离检查
        // 检查是否存在从x到y的路径，该路径在给定z时被阻塞
        let x_idx = self.get_node_index(x);
        let y_idx = self.get_node_index(y);
        
        if x_idx.is_none() || y_idx.is_none() {
            return true;
        }
        
        let x_idx = x_idx.unwrap();
        let y_idx = y_idx.unwrap();
        
        // 检查直接连接
        if self.adj_matrix[x_idx][y_idx] || self.adj_matrix[y_idx][x_idx] {
            // 如果z包含中间节点，则路径被阻塞
            return z.contains(&x.to_string()) || z.contains(&y.to_string());
        }
        
        // 检查间接路径（简化版本）
        for (i, _) in self.nodes.iter().enumerate() {
            if i != x_idx && i != y_idx {
                if (self.adj_matrix[x_idx][i] && self.adj_matrix[i][y_idx]) ||
                   (self.adj_matrix[y_idx][i] && self.adj_matrix[i][x_idx]) {
                    // 如果z包含中间节点，则路径被阻塞
                    return z.contains(&self.nodes[i]);
                }
            }
        }
        
        true
    }
}

#[derive(Debug)]
struct PCAlgorithm {
    graph: CausalGraph,
    independence_tests: HashMap<(String, String, Vec<String>), bool>,
}

impl PCAlgorithm {
    fn new(nodes: Vec<String>) -> Self {
        let graph = CausalGraph::new(nodes);
        let independence_tests = HashMap::new();
        
        PCAlgorithm {
            graph,
            independence_tests,
        }
    }
    
    fn run(&mut self, data: &[Vec<f64>]) {
        // 步骤1：学习骨架（无向图）
        self.learn_skeleton(data);
        
        // 步骤2：学习方向
        self.orient_edges();
    }
    
    fn learn_skeleton(&mut self, data: &[Vec<f64>]) {
        let n_nodes = self.graph.nodes.len();
        
        // 从完全图开始
        for i in 0..n_nodes {
            for j in 0..n_nodes {
                if i != j {
                    self.graph.add_edge(&self.graph.nodes[i], &self.graph.nodes[j]);
                }
            }
        }
        
        // 逐步删除边
        for i in 0..n_nodes {
            for j in (i+1)..n_nodes {
                if self.test_independence(&self.graph.nodes[i], &self.graph.nodes[j], &[], data) {
                    // 删除边
                    self.graph.edges.remove(&(self.graph.nodes[i].clone(), self.graph.nodes[j].clone()));
                    self.graph.edges.remove(&(self.graph.nodes[j].clone(), self.graph.nodes[i].clone()));
                    self.graph.adj_matrix[i][j] = false;
                    self.graph.adj_matrix[j][i] = false;
                }
            }
        }
    }
    
    fn test_independence(&mut self, x: &str, y: &str, z: &[String], data: &[Vec<f64>]) -> bool {
        // 简化的独立性测试（基于相关系数）
        let x_idx = self.graph.get_node_index(x);
        let y_idx = self.graph.get_node_index(y);
        
        if x_idx.is_none() || y_idx.is_none() {
            return true;
        }
        
        let x_idx = x_idx.unwrap();
        let y_idx = y_idx.unwrap();
        
        // 计算相关系数
        let correlation = self.calculate_correlation(data, x_idx, y_idx);
        
        // 如果相关系数接近0，则认为独立
        correlation.abs() < 0.1
    }
    
    fn calculate_correlation(&self, data: &[Vec<f64>], x_idx: usize, y_idx: usize) -> f64 {
        let n = data.len() as f64;
        
        let x_mean: f64 = data.iter().map(|row| row[x_idx]).sum::<f64>() / n;
        let y_mean: f64 = data.iter().map(|row| row[y_idx]).sum::<f64>() / n;
        
        let numerator: f64 = data.iter()
            .map(|row| (row[x_idx] - x_mean) * (row[y_idx] - y_mean))
            .sum();
        
        let x_var: f64 = data.iter().map(|row| (row[x_idx] - x_mean).powi(2)).sum();
        let y_var: f64 = data.iter().map(|row| (row[y_idx] - y_mean).powi(2)).sum();
        
        let denominator = (x_var * y_var).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    fn orient_edges(&mut self) {
        // 简化的方向学习
        // 在实际应用中，这里会使用更复杂的规则
        println!("Orienting edges...");
    }
}

fn main() {
    // 创建示例数据
    let nodes = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    let mut pc = PCAlgorithm::new(nodes);
    
    // 生成示例数据
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();
    
    for _ in 0..100 {
        let x = rng.gen::<f64>();
        let z = x + rng.gen::<f64>() * 0.1;
        let y = z + rng.gen::<f64>() * 0.1;
        data.push(vec![x, y, z]);
    }
    
    // 运行PC算法
    pc.run(&data);
    
    println!("PC Algorithm completed!");
    println!("Graph nodes: {:?}", pc.graph.nodes);
    println!("Graph edges: {:?}", pc.graph.edges);
}
```

### Haskell实现：因果效应估计

```haskell
import Data.List (foldl')
import Data.Map (Map)
import qualified Data.Map as Map
import System.Random

-- 数据结构
data Treatment = Control | Treatment deriving (Show, Eq)
data Observation = Observation {
    covariates :: [Double],
    treatment :: Treatment,
    outcome :: Double
} deriving Show

data CausalEffect = CausalEffect {
    ate :: Double,  -- Average Treatment Effect
    att :: Double,  -- Average Treatment Effect on Treated
    atc :: Double   -- Average Treatment Effect on Control
} deriving Show

-- 倾向得分模型
data PropensityModel = PropensityModel {
    coefficients :: [Double],
    intercept :: Double
} deriving Show

-- 创建倾向得分模型
createPropensityModel :: [Double] -> PropensityModel
createPropensityModel coeffs = PropensityModel {
    coefficients = coeffs,
    intercept = -0.5  -- 默认截距
}

-- 计算倾向得分
calculatePropensityScore :: PropensityModel -> [Double] -> Double
calculatePropensityScore model covariates =
    let linear = intercept model + sum (zipWith (*) (coefficients model) covariates)
    in 1.0 / (1.0 + exp (-linear))

-- 估计倾向得分模型
estimatePropensityModel :: [Observation] -> PropensityModel
estimatePropensityModel observations =
    let n = length observations
        treated = filter (\obs -> treatment obs == Treatment) observations
        control = filter (\obs -> treatment obs == Control) observations
        
        -- 简化的逻辑回归估计
        -- 在实际应用中，这里会使用更复杂的优化算法
        avgTreated = map mean (transpose (map covariates treated))
        avgControl = map mean (map covariates control)
        
        coeffs = zipWith (-) avgTreated avgControl
    in createPropensityModel coeffs
  where
    mean xs = sum xs / fromIntegral (length xs)
    transpose = foldr (zipWith (:)) (repeat [])

-- 计算平均因果效应
calculateATE :: [Observation] -> Double
calculateATE observations =
    let treated = filter (\obs -> treatment obs == Treatment) observations
        control = filter (\obs -> treatment obs == Control) observations
        
        treatedOutcomes = map outcome treated
        controlOutcomes = map outcome control
        
        avgTreated = sum treatedOutcomes / fromIntegral (length treatedOutcomes)
        avgControl = sum controlOutcomes / fromIntegral (length controlOutcomes)
    in avgTreated - avgControl

-- 使用倾向得分估计因果效应
estimateCausalEffectWithPropensity :: [Observation] -> CausalEffect
estimateCausalEffectWithPropensity observations =
    let propensityModel = estimatePropensityModel observations
        
        -- 计算每个观测的倾向得分
        observationsWithScore = map (\obs -> 
            (obs, calculatePropensityScore propensityModel (covariates obs))) observations
        
        -- 逆概率加权估计
        (weightedTreated, totalTreatedWeight) = foldl' 
            (\(sum, weight) (obs, score) -> 
                if treatment obs == Treatment
                then (sum + outcome obs / score, weight + 1.0 / score)
                else (sum, weight))
            (0.0, 0.0) observationsWithScore
        
        (weightedControl, totalControlWeight) = foldl' 
            (\(sum, weight) (obs, score) -> 
                if treatment obs == Control
                then (sum + outcome obs / (1.0 - score), weight + 1.0 / (1.0 - score))
                else (sum, weight))
            (0.0, 0.0) observationsWithScore
        
        ate = (weightedTreated / totalTreatedWeight) - (weightedControl / totalControlWeight)
        
        -- 简化的ATT和ATC计算
        att = ate  -- 简化假设
        atc = ate  -- 简化假设
    in CausalEffect { ate = ate, att = att, atc = atc }

-- 匹配估计
matchingEstimate :: [Observation] -> Double
matchingEstimate observations =
    let treated = filter (\obs -> treatment obs == Treatment) observations
        control = filter (\obs -> treatment obs == Control) observations
        
        -- 简化的最近邻匹配
        matchEffects = map (\treatedObs -> 
            let distances = map (\controlObs -> 
                    euclideanDistance (covariates treatedObs) (covariates controlObs)) control
                minDistance = minimum distances
                matchedControl = control !! (fromJust (elemIndex minDistance distances))
            in outcome treatedObs - outcome matchedControl) treated
        
        avgEffect = sum matchEffects / fromIntegral (length matchEffects)
    in avgEffect
  where
    euclideanDistance xs ys = sqrt (sum (zipWith (\x y -> (x - y) ^ 2) xs ys))
    fromJust (Just x) = x
    fromJust Nothing = error "No match found"

-- 工具变量估计
instrumentalVariableEstimate :: [Observation] -> [Double] -> Double
instrumentalVariableEstimate observations instruments =
    let -- 第一阶段：回归X对Z
        xValues = map (\obs -> head (covariates obs)) observations  -- 假设第一个协变量是X
        stage1Slope = calculateSlope instruments xValues
        
        -- 第二阶段：回归Y对预测的X
        yValues = map outcome observations
        predictedX = map (* stage1Slope) instruments
        stage2Slope = calculateSlope predictedX yValues
    in stage2Slope
  where
    calculateSlope xs ys =
        let n = fromIntegral (length xs)
            sumX = sum xs
            sumY = sum ys
            sumXY = sum (zipWith (*) xs ys)
            sumXX = sum (map (^ 2) xs)
            numerator = n * sumXY - sumX * sumY
            denominator = n * sumXX - sumX ^ 2
        in if denominator == 0 then 0 else numerator / denominator

-- 生成示例数据
generateObservationalData :: Int -> IO [Observation]
generateObservationalData n = do
    gen <- getStdGen
    let (observations, _) = foldl' 
            (\(obs, g) i -> 
                let (g1, g2) = split g
                    (x, g3) = randomR (-1.0, 1.0) g1
                    (z, g4) = randomR (-1.0, 1.0) g2
                    (u, g5) = randomR (-0.5, 0.5) g3
                    
                    -- 生成处理分配（基于协变量）
                    propensity = 1.0 / (1.0 + exp (-(x + z)))
                    (treatmentRandom, g6) = randomR (0.0, 1.0) g4
                    treatment = if treatmentRandom < propensity then Treatment else Control
                    
                    -- 生成结果
                    outcome = x + 2.0 * z + (if treatment == Treatment then 1.5 else 0.0) + u
                    
                    observation = Observation {
                        covariates = [x, z],
                        treatment = treatment,
                        outcome = outcome
                    }
                in (obs ++ [observation], g5)) 
            ([], gen) [1..n]
    return observations

-- 主函数
main :: IO ()
main = do
    putStrLn "生成观测数据..."
    observations <- generateObservationalData 1000
    
    putStrLn "计算因果效应..."
    
    -- 简单ATE
    let simpleATE = calculateATE observations
    putStrLn $ "简单ATE: " ++ show simpleATE
    
    -- 倾向得分估计
    let propensityATE = ate (estimateCausalEffectWithPropensity observations)
    putStrLn $ "倾向得分ATE: " ++ show propensityATE
    
    -- 匹配估计
    let matchingATE = matchingEstimate observations
    putStrLn $ "匹配ATE: " ++ show matchingATE
    
    -- 工具变量估计（使用第一个协变量作为工具）
    let instruments = map (\obs -> head (covariates obs)) observations
    let ivATE = instrumentalVariableEstimate observations instruments
    putStrLn $ "工具变量ATE: " ++ show ivATE
    
    putStrLn "\n因果效应估计完成！"
```

---

## 参考文献 / References

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
2. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.
3. Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference in Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
4. Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
5. Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *PNAS*.
6. Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *JASA*.

---

---

## 2024/2025 最新进展 / Latest Updates / Neueste Entwicklungen / Derniers développements

### 大规模因果推理 / Large-Scale Causal Inference

**2024年重要发展**:

- **高维因果发现**: 研究在高维数据中的因果发现算法和理论保证
- **时间序列因果推理**: 探索时间序列数据中的因果关系识别和估计
- **多变量因果推理**: 研究多变量系统中的复杂因果网络推断

**理论突破**:

- **因果表示学习**: 研究从观察数据中学习因果表示的理论框架
- **反事实推理**: 探索反事实推理的理论基础和计算方法
- **因果强化学习**: 研究结合因果推理的强化学习理论和算法

### 深度因果推理 / Deep Causal Inference

**前沿发展**:

- **神经因果模型**: 研究基于神经网络的因果模型和推理方法
- **因果生成模型**: 探索具有因果结构的生成模型理论
- **因果迁移学习**: 研究基于因果关系的迁移学习理论

### 因果推理应用理论 / Causal Inference Application Theory

**新兴应用领域**:

- **医疗AI**: 研究因果推理在医疗诊断和治疗中的理论基础
- **推荐系统**: 探索因果推理在推荐系统中的理论和实践
- **金融AI**: 研究因果推理在金融风险评估中的应用理论

### 大模型因果推理 / Large Model Causal Inference

**2024年重大进展**:

- **LLM中的因果推理**: 研究大语言模型如何理解和执行因果推理任务
- **因果提示工程**: 探索如何通过提示设计引导模型进行因果分析
- **多模态因果推理**: 研究结合文本、图像、音频的跨模态因果推理

**理论创新**:

- **因果注意力机制**: 设计能够识别因果关系的注意力模式
- **因果微调策略**: 开发针对因果推理任务的模型微调方法
- **因果评估指标**: 建立评估模型因果推理能力的标准化指标

### 因果AI对齐 / Causal AI Alignment

**前沿研究**:

- **因果价值学习**: 研究如何从人类反馈中学习因果偏好
- **因果安全机制**: 探索基于因果关系的AI安全防护理论
- **因果可解释性**: 开发因果推理的可解释性框架和工具

### 实用工具链 / Practical Toolchain

**2024年工具发展**:

- **因果发现平台**: 集成多种因果发现算法的统一平台
- **因果效应估计工具**: 提供倾向得分、工具变量等方法的实现
- **因果可视化**: 开发因果图、效应估计结果的可视化工具
- **因果基准测试**: 建立标准化的因果推理评估基准

---

*本模块为FormalAI提供了因果推理的理论基础，涵盖了从因果图模型到因果机器学习的各个方面，为AI系统的决策和干预提供了数学工具。*

---

## 进一步阅读（2025 持续滚动） / Further Reading (Rolling 2025)

- 年度权威索引：见 `docs/LATEST_UPDATES_INDEX.md` 的“权威索引（2025 持续滚动）”
- 来源类别锚点：
  - 顶尖大学课程：MIT/Stanford/CMU/Berkeley/Harvard（因果图、SCM、反事实、因果ML/因果RL）
  - A类会议/期刊：NeurIPS/ICML/UAI/AAAI/JMLR/PNAS 等
  - 标准与基准：NIST、ISO/IEC、W3C；数据许可、评测与显著性、模型/数据卡
  - 长期综述：Survey/Blueprint/Position（以期刊或arXiv正式版为准）

注：二手资料以一手论文与标准为准；引用需标注版本/日期。
