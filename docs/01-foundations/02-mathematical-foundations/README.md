# 1.2 数学基础 / Mathematical Foundations

## 概述 / Overview

数学基础为FormalAI提供严格的数学工具和理论框架，涵盖集合论、代数、拓扑学、微分几何、概率论、统计学、信息论和优化理论等核心领域。

Mathematical foundations provide rigorous mathematical tools and theoretical frameworks for FormalAI, covering core areas such as set theory, algebra, topology, differential geometry, probability theory, statistics, information theory, and optimization theory.

## 目录 / Table of Contents

- [1.2 数学基础 / Mathematical Foundations](#12-数学基础--mathematical-foundations)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [1. 集合论 / Set Theory](#1-集合论--set-theory)
    - [1.1 基本概念 / Basic Concepts](#11-基本概念--basic-concepts)
    - [1.2 关系与函数 / Relations and Functions](#12-关系与函数--relations-and-functions)
    - [1.3 基数与序数 / Cardinals and Ordinals](#13-基数与序数--cardinals-and-ordinals)
  - [2. 代数 / Algebra](#2-代数--algebra)
    - [2.1 群论 / Group Theory](#21-群论--group-theory)
    - [2.2 环论 / Ring Theory](#22-环论--ring-theory)
    - [2.3 线性代数 / Linear Algebra](#23-线性代数--linear-algebra)
  - [3. 拓扑学 / Topology](#3-拓扑学--topology)
    - [3.1 点集拓扑 / Point-Set Topology](#31-点集拓扑--point-set-topology)
    - [3.2 代数拓扑 / Algebraic Topology](#32-代数拓扑--algebraic-topology)
    - [3.3 微分拓扑 / Differential Topology](#33-微分拓扑--differential-topology)
  - [4. 微分几何 / Differential Geometry](#4-微分几何--differential-geometry)
    - [4.1 流形 / Manifolds](#41-流形--manifolds)
    - [4.2 切丛与余切丛 / Tangent and Cotangent Bundles](#42-切丛与余切丛--tangent-and-cotangent-bundles)
    - [4.3 黎曼几何 / Riemannian Geometry](#43-黎曼几何--riemannian-geometry)
  - [5. 概率论 / Probability Theory](#5-概率论--probability-theory)
    - [5.1 概率空间 / Probability Spaces](#51-概率空间--probability-spaces)
    - [5.2 随机变量 / Random Variables](#52-随机变量--random-variables)
    - [5.3 随机过程 / Stochastic Processes](#53-随机过程--stochastic-processes)
  - [6. 统计学 / Statistics](#6-统计学--statistics)
    - [6.1 描述统计 / Descriptive Statistics](#61-描述统计--descriptive-statistics)
    - [6.2 推断统计 / Inferential Statistics](#62-推断统计--inferential-statistics)
    - [6.3 贝叶斯统计 / Bayesian Statistics](#63-贝叶斯统计--bayesian-statistics)
  - [7. 信息论 / Information Theory](#7-信息论--information-theory)
    - [7.1 熵与信息 / Entropy and Information](#71-熵与信息--entropy-and-information)
    - [7.2 信道容量 / Channel Capacity](#72-信道容量--channel-capacity)
    - [7.3 编码理论 / Coding Theory](#73-编码理论--coding-theory)
  - [8. 优化理论 / Optimization Theory](#8-优化理论--optimization-theory)
    - [8.1 凸优化 / Convex Optimization](#81-凸优化--convex-optimization)
    - [8.2 非凸优化 / Non-Convex Optimization](#82-非凸优化--non-convex-optimization)
    - [8.3 随机优化 / Stochastic Optimization](#83-随机优化--stochastic-optimization)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：线性代数库](#rust实现线性代数库)
    - [Haskell实现：概率分布](#haskell实现概率分布)
  - [参考文献 / References](#参考文献--references)

---

## 1. 集合论 / Set Theory

### 1.1 基本概念 / Basic Concepts

**集合 / Set:**

集合是不同对象的无序集合：

A set is an unordered collection of distinct objects:

$$A = \{x : P(x)\}$$

其中 $P(x)$ 是谓词。

where $P(x)$ is a predicate.

**集合运算 / Set Operations:**

- **并集 / Union:** $A \cup B = \{x : x \in A \lor x \in B\}$
- **交集 / Intersection:** $A \cap B = \{x : x \in A \land x \in B\}$
- **差集 / Difference:** $A \setminus B = \{x : x \in A \land x \notin B\}$
- **补集 / Complement:** $A^c = \{x : x \notin A\}$

**幂集 / Power Set:**

$$\mathcal{P}(A) = \{B : B \subseteq A\}$$

### 1.2 关系与函数 / Relations and Functions

**关系 / Relation:**

$R \subseteq A \times B$ 是从 $A$ 到 $B$ 的关系。

$R \subseteq A \times B$ is a relation from $A$ to $B$.

**等价关系 / Equivalence Relation:**

关系 $R$ 是等价关系，如果：

Relation $R$ is an equivalence relation if:

1. **自反性 / Reflexivity:** $\forall x, xRx$
2. **对称性 / Symmetry:** $\forall x, y, xRy \Rightarrow yRx$
3. **传递性 / Transitivity:** $\forall x, y, z, xRy \land yRz \Rightarrow xRz$

**函数 / Function:**

函数 $f: A \rightarrow B$ 是满足以下条件的关系：

Function $f: A \rightarrow B$ is a relation satisfying:

$$\forall x \in A, \exists! y \in B, (x, y) \in f$$

### 1.3 基数与序数 / Cardinals and Ordinals

**基数 / Cardinal:**

集合 $A$ 的基数 $|A|$ 是其元素的数量。

The cardinality $|A|$ of set $A$ is the number of its elements.

**可数集 / Countable Set:**

集合 $A$ 是可数的，如果 $|A| \leq |\mathbb{N}|$。

Set $A$ is countable if $|A| \leq |\mathbb{N}|$.

**连续统假设 / Continuum Hypothesis:**

$$|\mathbb{R}| = 2^{|\mathbb{N}|} = \aleph_1$$

---

## 2. 代数 / Algebra

### 2.1 群论 / Group Theory

**群 / Group:**

群 $(G, \cdot)$ 是满足以下条件的集合和运算：

Group $(G, \cdot)$ is a set and operation satisfying:

1. **封闭性 / Closure:** $\forall a, b \in G, a \cdot b \in G$
2. **结合律 / Associativity:** $\forall a, b, c \in G, (a \cdot b) \cdot c = a \cdot (b \cdot c)$
3. **单位元 / Identity:** $\exists e \in G, \forall a \in G, e \cdot a = a \cdot e = a$
4. **逆元 / Inverse:** $\forall a \in G, \exists a^{-1} \in G, a \cdot a^{-1} = a^{-1} \cdot a = e$

**子群 / Subgroup:**

$H \subseteq G$ 是子群，如果 $(H, \cdot)$ 是群。

$H \subseteq G$ is a subgroup if $(H, \cdot)$ is a group.

**同态 / Homomorphism:**

函数 $\phi: G \rightarrow H$ 是同态，如果：

Function $\phi: G \rightarrow H$ is a homomorphism if:

$$\forall a, b \in G, \phi(a \cdot b) = \phi(a) \cdot \phi(b)$$

### 2.2 环论 / Ring Theory

**环 / Ring:**

环 $(R, +, \cdot)$ 是满足以下条件的集合和运算：

Ring $(R, +, \cdot)$ is a set and operations satisfying:

1. $(R, +)$ 是阿贝尔群
2. $(R, \cdot)$ 是幺半群
3. **分配律 / Distributivity:** $\forall a, b, c \in R, a \cdot (b + c) = a \cdot b + a \cdot c$

**理想 / Ideal:**

$I \subseteq R$ 是理想，如果：

$I \subseteq R$ is an ideal if:

1. $(I, +)$ 是子群
2. $\forall r \in R, \forall i \in I, r \cdot i, i \cdot r \in I$

### 2.3 线性代数 / Linear Algebra

**向量空间 / Vector Space:**

向量空间 $V$ 是满足以下条件的集合：

Vector space $V$ is a set satisfying:

1. $(V, +)$ 是阿贝尔群
2. **标量乘法 / Scalar Multiplication:** $\forall \alpha \in \mathbb{F}, \forall v \in V, \alpha v \in V$
3. **分配律 / Distributivity:** $\forall \alpha, \beta \in \mathbb{F}, \forall v \in V, (\alpha + \beta)v = \alpha v + \beta v$

**线性变换 / Linear Transformation:**

函数 $T: V \rightarrow W$ 是线性的，如果：

Function $T: V \rightarrow W$ is linear if:

$$\forall v_1, v_2 \in V, \forall \alpha \in \mathbb{F}, T(\alpha v_1 + v_2) = \alpha T(v_1) + T(v_2)$$

**特征值与特征向量 / Eigenvalues and Eigenvectors:**

$$\text{Av} = \lambda v$$

其中 $\lambda$ 是特征值，$v$ 是特征向量。

where $\lambda$ is eigenvalue and $v$ is eigenvector.

---

## 3. 拓扑学 / Topology

### 3.1 点集拓扑 / Point-Set Topology

**拓扑空间 / Topological Space:**

拓扑空间 $(X, \tau)$ 是集合 $X$ 和拓扑 $\tau$。

Topological space $(X, \tau)$ is a set $X$ and topology $\tau$.

**开集 / Open Set:**

集合 $U \subseteq X$ 是开集，如果 $U \in \tau$。

Set $U \subseteq X$ is open if $U \in \tau$.

**连续函数 / Continuous Function:**

函数 $f: X \rightarrow Y$ 是连续的，如果：

Function $f: X \rightarrow Y$ is continuous if:

$$\forall U \in \tau_Y, f^{-1}(U) \in \tau_X$$

### 3.2 代数拓扑 / Algebraic Topology

**同伦 / Homotopy:**

函数 $f, g: X \rightarrow Y$ 是同伦的，如果存在连续函数：

Functions $f, g: X \rightarrow Y$ are homotopic if there exists continuous function:

$$H: X \times [0,1] \rightarrow Y$$

使得 $H(x,0) = f(x)$ 和 $H(x,1) = g(x)$。

such that $H(x,0) = f(x)$ and $H(x,1) = g(x)$.

**基本群 / Fundamental Group:**

$$\pi_1(X, x_0) = \{\text{homotopy classes of loops at } x_0\}$$

### 3.3 微分拓扑 / Differential Topology

**微分流形 / Differentiable Manifold:**

微分流形是局部同胚于 $\mathbb{R}^n$ 的拓扑空间。

Differentiable manifold is a topological space locally homeomorphic to $\mathbb{R}^n$.

**切空间 / Tangent Space:**

$$T_p M = \{\text{derivations at } p\}$$

---

## 4. 微分几何 / Differential Geometry

### 4.1 流形 / Manifolds

**流形 / Manifold:**

$n$ 维流形是局部同胚于 $\mathbb{R}^n$ 的拓扑空间。

$n$-dimensional manifold is a topological space locally homeomorphic to $\mathbb{R}^n$.

**坐标图 / Coordinate Chart:**

$$(U, \phi): U \subseteq M \rightarrow \mathbb{R}^n$$

**光滑函数 / Smooth Function:**

函数 $f: M \rightarrow \mathbb{R}$ 是光滑的，如果：

Function $f: M \rightarrow \mathbb{R}$ is smooth if:

$$f \circ \phi^{-1}: \mathbb{R}^n \rightarrow \mathbb{R} \text{ is smooth}$$

### 4.2 切丛与余切丛 / Tangent and Cotangent Bundles

**切丛 / Tangent Bundle:**

$$TM = \bigcup_{p \in M} T_p M$$

**余切丛 / Cotangent Bundle:**

$$T^*M = \bigcup_{p \in M} T_p^* M$$

**微分形式 / Differential Form:**

$$\omega = \sum_{i_1 < \cdots < i_k} a_{i_1 \cdots i_k} dx^{i_1} \wedge \cdots \wedge dx^{i_k}$$

### 4.3 黎曼几何 / Riemannian Geometry

**黎曼度量 / Riemannian Metric:**

$$g: T_p M \times T_p M \rightarrow \mathbb{R}$$

**测地线 / Geodesic:**

$$\nabla_{\dot{\gamma}} \dot{\gamma} = 0$$

**曲率 / Curvature:**

$$R(X,Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z$$

---

## 5. 概率论 / Probability Theory

### 5.1 概率空间 / Probability Spaces

**概率空间 / Probability Space:**

概率空间 $(\Omega, \mathcal{F}, P)$ 包含：

Probability space $(\Omega, \mathcal{F}, P)$ contains:

- $\Omega$: 样本空间 / sample space
- $\mathcal{F}$: $\sigma$-代数 / $\sigma$-algebra
- $P$: 概率测度 / probability measure

**概率测度 / Probability Measure:**

$$P: \mathcal{F} \rightarrow [0,1]$$

满足：

satisfying:

1. $P(\Omega) = 1$
2. $P(\bigcup_{i=1}^{\infty} A_i) = \sum_{i=1}^{\infty} P(A_i)$ (可数可加性)

### 5.2 随机变量 / Random Variables

**随机变量 / Random Variable:**

$$X: \Omega \rightarrow \mathbb{R}$$

**分布函数 / Distribution Function:**

$$F_X(x) = P(X \leq x)$$

**期望 / Expectation:**

$$\mathbb{E}[X] = \int_{\Omega} X(\omega) dP(\omega)$$

**方差 / Variance:**

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$$

### 5.3 随机过程 / Stochastic Processes

**随机过程 / Stochastic Process:**

$$\{X_t : t \in T\}$$

**马尔可夫性质 / Markov Property:**

$$P(X_{t+1} | X_t, X_{t-1}, \ldots) = P(X_{t+1} | X_t)$$

**布朗运动 / Brownian Motion:**

$$W_t - W_s \sim \mathcal{N}(0, t-s)$$

---

## 6. 统计学 / Statistics

### 6.1 描述统计 / Descriptive Statistics

**样本均值 / Sample Mean:**

$$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$$

**样本方差 / Sample Variance:**

$$s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2$$

**样本相关系数 / Sample Correlation:**

$$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

### 6.2 推断统计 / Inferential Statistics

**置信区间 / Confidence Interval:**

$$P(\theta \in [L, U]) = 1 - \alpha$$

**假设检验 / Hypothesis Testing:**

$$H_0: \theta = \theta_0 \quad \text{vs} \quad H_1: \theta \neq \theta_0$$

**p值 / p-value:**

$$p = P(T \geq t_{\text{obs}} | H_0)$$

### 6.3 贝叶斯统计 / Bayesian Statistics

**贝叶斯定理 / Bayes' Theorem:**

$$P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}$$

**后验分布 / Posterior Distribution:**

$$P(\theta | D) \propto P(D | \theta) P(\theta)$$

**最大后验估计 / Maximum A Posteriori:**

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} P(\theta | D)$$

---

## 7. 信息论 / Information Theory

### 7.1 熵与信息 / Entropy and Information

**香农熵 / Shannon Entropy:**

$$H(X) = -\sum_{i=1}^n p_i \log p_i$$

**条件熵 / Conditional Entropy:**

$$H(X|Y) = -\sum_{i,j} p_{ij} \log p_{i|j}$$

**互信息 / Mutual Information:**

$$I(X;Y) = H(X) - H(X|Y)$$

### 7.2 信道容量 / Channel Capacity

**信道容量 / Channel Capacity:**

$$C = \max_{p(x)} I(X;Y)$$

**噪声信道定理 / Noisy Channel Theorem:**

对于任何 $\epsilon > 0$，存在编码方案使得：

For any $\epsilon > 0$, there exists coding scheme such that:

$$R < C - \epsilon \Rightarrow P_e < \epsilon$$

### 7.3 编码理论 / Coding Theory

**线性码 / Linear Code:**

$$C = \{c \in \mathbb{F}_q^n : c = mG\}$$

其中 $G$ 是生成矩阵。

where $G$ is generator matrix.

**汉明距离 / Hamming Distance:**

$$d_H(x,y) = |\{i : x_i \neq y_i\}|$$

---

## 8. 优化理论 / Optimization Theory

### 8.1 凸优化 / Convex Optimization

**凸函数 / Convex Function:**

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

**凸优化问题 / Convex Optimization Problem:**

$$\min_{x \in \mathcal{X}} f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, h_j(x) = 0$$

**KKT条件 / KKT Conditions:**

$$\nabla f(x^*) + \sum_i \lambda_i \nabla g_i(x^*) + \sum_j \mu_j \nabla h_j(x^*) = 0$$

### 8.2 非凸优化 / Non-Convex Optimization

**局部最优 / Local Optimum:**

$$\exists \epsilon > 0, \forall x \in B(x^*, \epsilon), f(x^*) \leq f(x)$$

**梯度下降 / Gradient Descent:**

$$x_{t+1} = x_t - \alpha \nabla f(x_t)$$

**随机梯度下降 / Stochastic Gradient Descent:**

$$x_{t+1} = x_t - \alpha \nabla f(x_t, \xi_t)$$

### 8.3 随机优化 / Stochastic Optimization

**随机优化问题 / Stochastic Optimization Problem:**

$$\min_{x} \mathbb{E}[f(x, \xi)]$$

**样本平均近似 / Sample Average Approximation:**

$$\min_{x} \frac{1}{N} \sum_{i=1}^N f(x, \xi_i)$$

---

## 代码示例 / Code Examples

### Rust实现：线性代数库

```rust
use std::ops::{Add, Sub, Mul, Div};

#[derive(Debug, Clone)]
struct Vector {
    data: Vec<f64>,
}

impl Vector {
    fn new(data: Vec<f64>) -> Self {
        Vector { data }
    }
    
    fn zeros(n: usize) -> Self {
        Vector { data: vec![0.0; n] }
    }
    
    fn ones(n: usize) -> Self {
        Vector { data: vec![1.0; n] }
    }
    
    fn dot(&self, other: &Vector) -> f64 {
        if self.data.len() != other.data.len() {
            panic!("Vector dimensions must match");
        }
        
        self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
    
    fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }
    
    fn normalize(&self) -> Vector {
        let norm = self.norm();
        if norm == 0.0 {
            return self.clone();
        }
        Vector {
            data: self.data.iter().map(|x| x / norm).collect()
        }
    }
}

impl Add for Vector {
    type Output = Vector;
    
    fn add(self, other: Vector) -> Vector {
        if self.data.len() != other.data.len() {
            panic!("Vector dimensions must match");
        }
        
        Vector {
            data: self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect()
        }
    }
}

impl Sub for Vector {
    type Output = Vector;
    
    fn sub(self, other: Vector) -> Vector {
        if self.data.len() != other.data.len() {
            panic!("Vector dimensions must match");
        }
        
        Vector {
            data: self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect()
        }
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;
    
    fn mul(self, scalar: f64) -> Vector {
        Vector {
            data: self.data.iter().map(|x| x * scalar).collect()
        }
    }
}

#[derive(Debug, Clone)]
struct Matrix {
    data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn new(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        
        // 检查所有行长度一致
        for row in &data {
            if row.len() != cols {
                panic!("All rows must have the same length");
            }
        }
        
        Matrix { data, rows, cols }
    }
    
    fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![0.0; cols]; rows],
            rows,
            cols,
        }
    }
    
    fn identity(n: usize) -> Self {
        let mut data = vec![vec![0.0; n]; n];
        for i in 0..n {
            data[i][i] = 1.0;
        }
        Matrix { data, rows: n, cols: n }
    }
    
    fn get(&self, i: usize, j: usize) -> f64 {
        if i >= self.rows || j >= self.cols {
            panic!("Index out of bounds");
        }
        self.data[i][j]
    }
    
    fn set(&mut self, i: usize, j: usize, value: f64) {
        if i >= self.rows || j >= self.cols {
            panic!("Index out of bounds");
        }
        self.data[i][j] = value;
    }
    
    fn transpose(&self) -> Matrix {
        let mut transposed = vec![vec![0.0; self.rows]; self.cols];
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed[j][i] = self.data[i][j];
            }
        }
        
        Matrix {
            data: transposed,
            rows: self.cols,
            cols: self.rows,
        }
    }
    
    fn determinant(&self) -> f64 {
        if self.rows != self.cols {
            panic!("Determinant only defined for square matrices");
        }
        
        if self.rows == 1 {
            return self.data[0][0];
        }
        
        if self.rows == 2 {
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0];
        }
        
        // 递归计算行列式
        let mut det = 0.0;
        for j in 0..self.cols {
            let minor = self.minor(0, j);
            det += self.data[0][j] * minor.determinant() * if j % 2 == 0 { 1.0 } else { -1.0 };
        }
        
        det
    }
    
    fn minor(&self, row: usize, col: usize) -> Matrix {
        let mut minor_data = Vec::new();
        
        for i in 0..self.rows {
            if i == row { continue; }
            let mut row_data = Vec::new();
            for j in 0..self.cols {
                if j == col { continue; }
                row_data.push(self.data[i][j]);
            }
            minor_data.push(row_data);
        }
        
        Matrix::new(minor_data)
    }
    
    fn inverse(&self) -> Option<Matrix> {
        if self.rows != self.cols {
            return None;
        }
        
        let det = self.determinant();
        if det == 0.0 {
            return None;
        }
        
        let mut adjoint = Matrix::zeros(self.rows, self.cols);
        
        for i in 0..self.rows {
            for j in 0..self.cols {
                let minor = self.minor(i, j);
                let cofactor = minor.determinant() * if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                adjoint.set(j, i, cofactor / det);
            }
        }
        
        Some(adjoint)
    }
}

impl Mul for Matrix {
    type Output = Matrix;
    
    fn mul(self, other: Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Matrix dimensions incompatible for multiplication");
        }
        
        let mut result = Matrix::zeros(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.set(i, j, sum);
            }
        }
        
        result
    }
}

// 特征值计算（简化版）
impl Matrix {
    fn eigenvalues(&self) -> Vec<f64> {
        if self.rows != self.cols {
            panic!("Eigenvalues only defined for square matrices");
        }
        
        // 简化的幂迭代方法
        let mut eigenvalues = Vec::new();
        let mut matrix = self.clone();
        
        for _ in 0..self.rows {
            // 使用幂迭代找到最大特征值
            let mut vector = Vector::ones(self.rows);
            
            for _ in 0..100 {
                let new_vector = matrix_vector_multiply(&matrix, &vector);
                let norm = new_vector.norm();
                if norm == 0.0 { break; }
                vector = new_vector.normalize();
            }
            
            let eigenvalue = matrix_vector_multiply(&matrix, &vector).dot(&vector);
            eigenvalues.push(eigenvalue);
            
            // 从矩阵中减去这个特征值的影响（简化）
            // 在实际实现中，这里应该使用更复杂的算法
        }
        
        eigenvalues
    }
}

fn matrix_vector_multiply(matrix: &Matrix, vector: &Vector) -> Vector {
    if matrix.cols != vector.data.len() {
        panic!("Matrix and vector dimensions incompatible");
    }
    
    let mut result = vec![0.0; matrix.rows];
    
    for i in 0..matrix.rows {
        for j in 0..matrix.cols {
            result[i] += matrix.data[i][j] * vector.data[j];
        }
    }
    
    Vector { data: result }
}

fn main() {
    // 测试向量运算
    let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
    
    println!("v1: {:?}", v1);
    println!("v2: {:?}", v2);
    println!("v1 + v2: {:?}", v1.clone() + v2.clone());
    println!("v1 · v2: {}", v1.dot(&v2));
    println!("||v1||: {}", v1.norm());
    
    // 测试矩阵运算
    let m1 = Matrix::new(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0]
    ]);
    
    let m2 = Matrix::new(vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0]
    ]);
    
    println!("\nm1: {:?}", m1);
    println!("m2: {:?}", m2);
    println!("m1 * m2: {:?}", m1.clone() * m2.clone());
    println!("det(m1): {}", m1.determinant());
    
    if let Some(inverse) = m1.inverse() {
        println!("m1^(-1): {:?}", inverse);
    }
    
    // 测试特征值
    let eigenvalues = m1.eigenvalues();
    println!("Eigenvalues of m1: {:?}", eigenvalues);
    
    println!("\n线性代数库演示完成！");
}
```

### Haskell实现：概率分布

```haskell
import Data.List (foldl')
import System.Random

-- 概率分布类型
data ProbabilityDistribution a = ProbabilityDistribution {
    samples :: [a],
    weights :: [Double]
} deriving Show

-- 连续分布类型
data ContinuousDistribution = ContinuousDistribution {
    pdf :: Double -> Double,  -- 概率密度函数
    cdf :: Double -> Double,  -- 累积分布函数
    mean :: Double,
    variance :: Double
} deriving Show

-- 离散分布类型
data DiscreteDistribution = DiscreteDistribution {
    pmf :: Int -> Double,     -- 概率质量函数
    support :: [Int],         -- 支撑集
    discreteMean :: Double,
    discreteVariance :: Double
} deriving Show

-- 创建均匀分布
uniformDistribution :: Double -> Double -> ContinuousDistribution
uniformDistribution a b = ContinuousDistribution {
    pdf = \x -> if x >= a && x <= b then 1.0 / (b - a) else 0.0,
    cdf = \x -> if x < a then 0.0 else if x > b then 1.0 else (x - a) / (b - a),
    mean = (a + b) / 2.0,
    variance = (b - a) ^ 2 / 12.0
}

-- 创建正态分布
normalDistribution :: Double -> Double -> ContinuousDistribution
normalDistribution mu sigma = ContinuousDistribution {
    pdf = \x -> (1.0 / (sigma * sqrt (2 * pi))) * exp (-((x - mu) ^ 2) / (2 * sigma ^ 2)),
    cdf = \x -> 0.5 * (1.0 + erf ((x - mu) / (sigma * sqrt 2))),
    mean = mu,
    variance = sigma ^ 2
}
  where
    erf :: Double -> Double
    erf x = 2.0 / sqrt pi * integrate (\t -> exp (-t ^ 2)) 0 x

-- 数值积分
integrate :: (Double -> Double) -> Double -> Double -> Double
integrate f a b =
    let n = 1000
        h = (b - a) / fromIntegral n
        xs = [a + fromIntegral i * h | i <- [0..n]]
    in h * sum [f x | x <- xs] - h * (f a + f b) / 2.0

-- 创建泊松分布
poissonDistribution :: Double -> DiscreteDistribution
poissonDistribution lambda = DiscreteDistribution {
    pmf = \k -> if k < 0 then 0.0 else (lambda ^ k * exp (-lambda)) / fromIntegral (factorial k),
    support = [0..],
    discreteMean = lambda,
    discreteVariance = lambda
}
  where
    factorial :: Int -> Int
    factorial 0 = 1
    factorial n = n * factorial (n - 1)

-- 创建二项分布
binomialDistribution :: Int -> Double -> DiscreteDistribution
binomialDistribution n p = DiscreteDistribution {
    pmf = \k -> if k < 0 || k > n then 0.0 
                else fromIntegral (choose n k) * p ^ k * (1 - p) ^ (n - k),
    support = [0..n],
    discreteMean = fromIntegral n * p,
    discreteVariance = fromIntegral n * p * (1 - p)
}
  where
    choose :: Int -> Int -> Int
    choose n k = factorial n `div` (factorial k * factorial (n - k))

-- 随机采样
sampleFromDistribution :: ProbabilityDistribution a -> IO a
sampleFromDistribution dist = do
    let totalWeight = sum (weights dist)
        normalizedWeights = map (/ totalWeight) (weights dist)
        cumulativeWeights = scanl1 (+) normalizedWeights
    
    random <- randomRIO (0.0, 1.0)
    
    let index = findIndex random cumulativeWeights
    return $ samples dist !! index
  where
    findIndex :: Double -> [Double] -> Int
    findIndex r weights = 
        case dropWhile (< r) (zip [0..] weights) of
            ((i, _):_) -> i
            [] -> length weights - 1

-- 蒙特卡洛积分
monteCarloIntegral :: (Double -> Double) -> Double -> Double -> Int -> IO Double
monteCarloIntegral f a b n = do
    samples <- replicateM n (randomRIO (a, b))
    let values = map f samples
        integral = (b - a) * sum values / fromIntegral n
    return integral

-- 期望值计算
expectation :: (a -> Double) -> ProbabilityDistribution a -> Double
expectation f dist =
    let weightedSum = sum [f x * w | (x, w) <- zip (samples dist) (weights dist)]
        totalWeight = sum (weights dist)
    in weightedSum / totalWeight

-- 方差计算
variance :: (a -> Double) -> ProbabilityDistribution a -> Double
variance f dist =
    let mu = expectation f dist
        squaredDiff = expectation (\x -> (f x - mu) ^ 2) dist
    in squaredDiff

-- 协方差计算
covariance :: (a -> Double) -> (a -> Double) -> ProbabilityDistribution a -> Double
covariance f g dist =
    let mu_f = expectation f dist
        mu_g = expectation g dist
        crossProduct = expectation (\x -> (f x - mu_f) * (g x - mu_g)) dist
    in crossProduct

-- 贝叶斯更新
bayesianUpdate :: ProbabilityDistribution Double -> (Double -> Double) -> ProbabilityDistribution Double
bayesianUpdate prior likelihood =
    let newWeights = zipWith (*) (weights prior) (map likelihood (samples prior))
        totalWeight = sum newWeights
        normalizedWeights = map (/ totalWeight) newWeights
    in ProbabilityDistribution (samples prior) normalizedWeights

-- 马尔可夫链蒙特卡洛 (MCMC)
mcmc :: (Double -> Double) -> Double -> Int -> IO [Double]
mcmc targetDistribution initialValue n = do
    let step current = do
            proposal <- randomRIO (current - 1.0, current + 1.0)
            acceptanceRatio <- randomRIO (0.0, 1.0)
            
            let ratio = targetDistribution proposal / targetDistribution current
            if acceptanceRatio < min 1.0 ratio
                then return proposal
                else return current
    
    foldM (\chain _ -> do
        newValue <- step (last chain)
        return $ chain ++ [newValue]
    ) [initialValue] [1..n]

-- 信息论函数
entropy :: ProbabilityDistribution a -> Double
entropy dist =
    let normalizedWeights = map (/ sum (weights dist)) (weights dist)
        logWeights = map (\w -> if w > 0 then -w * log w else 0.0) normalizedWeights
    in sum logWeights

klDivergence :: ProbabilityDistribution a -> ProbabilityDistribution a -> Double
klDivergence p q =
    let normalizedP = map (/ sum (weights p)) (weights p)
        normalizedQ = map (/ sum (weights q)) (weights q)
        klTerms = zipWith (\pi qi -> if pi > 0 && qi > 0 then pi * log (pi / qi) else 0.0) normalizedP normalizedQ
    in sum klTerms

-- 主函数
main :: IO ()
main = do
    putStrLn "概率论与统计学演示"
    
    -- 创建概率分布
    let uniform = uniformDistribution 0.0 1.0
        normal = normalDistribution 0.0 1.0
        poisson = poissonDistribution 3.0
        binomial = binomialDistribution 10 0.5
    
    putStrLn "\n连续分布:"
    putStrLn $ "均匀分布均值: " ++ show (mean uniform)
    putStrLn $ "均匀分布方差: " ++ show (variance uniform)
    putStrLn $ "正态分布均值: " ++ show (mean normal)
    putStrLn $ "正态分布方差: " ++ show (variance normal)
    
    putStrLn "\n离散分布:"
    putStrLn $ "泊松分布均值: " ++ show (discreteMean poisson)
    putStrLn $ "泊松分布方差: " ++ show (discreteVariance poisson)
    putStrLn $ "二项分布均值: " ++ show (discreteMean binomial)
    putStrLn $ "二项分布方差: " ++ show (discreteVariance binomial)
    
    -- 蒙特卡洛积分
    let f x = x ^ 2  -- 积分 x^2 from 0 to 1
    integral <- monteCarloIntegral f 0.0 1.0 10000
    putStrLn $ "\n蒙特卡洛积分 ∫x²dx from 0 to 1: " ++ show integral
    putStrLn $ "理论值: 1/3 ≈ " ++ show (1.0 / 3.0)
    
    -- 创建离散概率分布
    let dist = ProbabilityDistribution [1, 2, 3, 4, 5] [0.1, 0.2, 0.3, 0.2, 0.2]
    
    putStrLn "\n离散分布:"
    putStrLn $ "分布: " ++ show dist
    putStrLn $ "熵: " ++ show (entropy dist)
    
    -- 贝叶斯更新示例
    let prior = ProbabilityDistribution [0.1, 0.3, 0.5, 0.7, 0.9] [1.0, 1.0, 1.0, 1.0, 1.0]
        likelihood x = if x > 0.5 then 0.8 else 0.2
        posterior = bayesianUpdate prior likelihood
    
    putStrLn "\n贝叶斯更新:"
    putStrLn $ "先验: " ++ show (weights prior)
    putStrLn $ "后验: " ++ show (weights posterior)
    
    putStrLn "\n概率论与统计学演示完成！"
```

---

## 参考文献 / References

1. Halmos, P. R. (1974). *Naive Set Theory*. Springer.
2. Dummit, D. S., & Foote, R. M. (2004). *Abstract Algebra*. Wiley.
3. Munkres, J. R. (2000). *Topology*. Prentice Hall.
4. Do Carmo, M. P. (1992). *Riemannian Geometry*. Birkhäuser.
5. Billingsley, P. (1995). *Probability and Measure*. Wiley.
6. Casella, G., & Berger, R. L. (2002). *Statistical Inference*. Duxbury.
7. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.
8. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

---

*本模块为FormalAI提供了坚实的数学基础，涵盖了从集合论到优化理论的各个方面，为AI系统的设计和分析提供了数学工具。*
