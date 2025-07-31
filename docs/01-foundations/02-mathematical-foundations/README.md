# 1.2 数学基础 / Mathematical Foundations

## 概述 / Overview

数学基础为FormalAI提供严格的数学工具和理论框架，包括集合论、代数、分析、拓扑等核心数学分支。

Mathematical foundations provide rigorous mathematical tools and theoretical frameworks for FormalAI, including set theory, algebra, analysis, topology, and other core mathematical branches.

## 目录 / Table of Contents

- [1.2 数学基础 / Mathematical Foundations](#12-数学基础--mathematical-foundations)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [1. 集合论 / Set Theory](#1-集合论--set-theory)
    - [1.1 基本概念 / Basic Concepts](#11-基本概念--basic-concepts)
    - [1.2 集合运算 / Set Operations](#12-集合运算--set-operations)
    - [1.3 关系与函数 / Relations and Functions](#13-关系与函数--relations-and-functions)
    - [1.4 基数 / Cardinality](#14-基数--cardinality)
  - [2. 抽象代数 / Abstract Algebra](#2-抽象代数--abstract-algebra)
    - [2.1 群论 / Group Theory](#21-群论--group-theory)
    - [2.2 环论 / Ring Theory](#22-环论--ring-theory)
    - [2.3 模论 / Module Theory](#23-模论--module-theory)
  - [3. 线性代数 / Linear Algebra](#3-线性代数--linear-algebra)
    - [3.1 向量空间 / Vector Spaces](#31-向量空间--vector-spaces)
    - [3.2 线性变换 / Linear Transformations](#32-线性变换--linear-transformations)
    - [3.3 特征值与特征向量 / Eigenvalues and Eigenvectors](#33-特征值与特征向量--eigenvalues-and-eigenvectors)
  - [4. 泛函分析 / Functional Analysis](#4-泛函分析--functional-analysis)
    - [4.1 度量空间 / Metric Spaces](#41-度量空间--metric-spaces)
    - [4.2 巴拿赫空间 / Banach Spaces](#42-巴拿赫空间--banach-spaces)
    - [4.3 希尔伯特空间 / Hilbert Spaces](#43-希尔伯特空间--hilbert-spaces)
  - [5. 拓扑学 / Topology](#5-拓扑学--topology)
    - [5.1 拓扑空间 / Topological Spaces](#51-拓扑空间--topological-spaces)
    - [5.2 连续性与连通性 / Continuity and Connectedness](#52-连续性与连通性--continuity-and-connectedness)
  - [6. 测度论 / Measure Theory](#6-测度论--measure-theory)
    - [6.1 σ-代数 / σ-Algebra](#61-σ-代数--σ-algebra)
    - [6.2 测度 / Measure](#62-测度--measure)
    - [6.3 积分 / Integration](#63-积分--integration)
  - [7. 概率论 / Probability Theory](#7-概率论--probability-theory)
    - [7.1 概率空间 / Probability Space](#71-概率空间--probability-space)
    - [7.2 随机变量 / Random Variables](#72-随机变量--random-variables)
    - [7.3 条件概率 / Conditional Probability](#73-条件概率--conditional-probability)
  - [8. 信息论 / Information Theory](#8-信息论--information-theory)
    - [8.1 熵 / Entropy](#81-熵--entropy)
    - [8.2 互信息 / Mutual Information](#82-互信息--mutual-information)
  - [9. 优化理论 / Optimization Theory](#9-优化理论--optimization-theory)
    - [9.1 凸优化 / Convex Optimization](#91-凸优化--convex-optimization)
    - [9.2 拉格朗日乘数法 / Lagrange Multipliers](#92-拉格朗日乘数法--lagrange-multipliers)
  - [10. 范畴论 / Category Theory](#10-范畴论--category-theory)
    - [10.1 基本概念 / Basic Concepts](#101-基本概念--basic-concepts)
    - [10.2 自然变换 / Natural Transformations](#102-自然变换--natural-transformations)
    - [10.3 极限与余极限 / Limits and Colimits](#103-极限与余极限--limits-and-colimits)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：线性代数库](#rust实现线性代数库)
    - [Haskell实现：概率论基础](#haskell实现概率论基础)
  - [参考文献 / References](#参考文献--references)

---

## 1. 集合论 / Set Theory

### 1.1 基本概念 / Basic Concepts

**集合 / Set:**

集合是不同对象的无序聚集。

A set is an unordered collection of distinct objects.

**形式化定义 / Formal Definition:**

$$A = \{x \mid P(x)\}$$

其中 $P(x)$ 是谓词，表示对象 $x$ 满足的性质。

where $P(x)$ is a predicate representing the property that object $x$ satisfies.

### 1.2 集合运算 / Set Operations

**并集 / Union:**

$$A \cup B = \{x \mid x \in A \text{ or } x \in B\}$$

**交集 / Intersection:**

$$A \cap B = \{x \mid x \in A \text{ and } x \in B\}$$

**差集 / Difference:**

$$A \setminus B = \{x \mid x \in A \text{ and } x \notin B\}$$

**补集 / Complement:**

$$A^c = \{x \mid x \notin A\}$$

### 1.3 关系与函数 / Relations and Functions

**二元关系 / Binary Relation:**

$R \subseteq A \times B$ 是集合 $A$ 到集合 $B$ 的二元关系。

$R \subseteq A \times B$ is a binary relation from set $A$ to set $B$.

**函数 / Function:**

函数 $f: A \rightarrow B$ 是满足以下条件的二元关系：

A function $f: A \rightarrow B$ is a binary relation satisfying:

1. $\forall a \in A, \exists b \in B: (a,b) \in f$
2. $(a,b_1) \in f \land (a,b_2) \in f \Rightarrow b_1 = b_2$

**单射 / Injective:**

$$\forall a_1, a_2 \in A: f(a_1) = f(a_2) \Rightarrow a_1 = a_2$$

**满射 / Surjective:**

$$\forall b \in B, \exists a \in A: f(a) = b$$

**双射 / Bijective:**

函数既是单射又是满射。

A function that is both injective and surjective.

### 1.4 基数 / Cardinality

**等势 / Equinumerous:**

两个集合 $A$ 和 $B$ 等势，如果存在双射 $f: A \rightarrow B$。

Two sets $A$ and $B$ are equinumerous if there exists a bijection $f: A \rightarrow B$.

**基数 / Cardinality:**

集合 $A$ 的基数 $|A|$ 是与其等势的所有集合的等价类。

The cardinality $|A|$ of set $A$ is the equivalence class of all sets equinumerous to it.

**可数集 / Countable Set:**

集合 $A$ 是可数的，如果 $|A| \leq |\mathbb{N}|$。

A set $A$ is countable if $|A| \leq |\mathbb{N}|$.

---

## 2. 抽象代数 / Abstract Algebra

### 2.1 群论 / Group Theory

**群 / Group:**

群 $(G, \cdot)$ 是一个集合 $G$ 和二元运算 $\cdot$，满足：

A group $(G, \cdot)$ is a set $G$ with a binary operation $\cdot$ satisfying:

1. **结合律 / Associativity:** $(a \cdot b) \cdot c = a \cdot (b \cdot c)$
2. **单位元 / Identity:** $\exists e \in G: \forall a \in G, e \cdot a = a \cdot e = a$
3. **逆元 / Inverse:** $\forall a \in G, \exists a^{-1} \in G: a \cdot a^{-1} = a^{-1} \cdot a = e$

**子群 / Subgroup:**

$H \subseteq G$ 是 $G$ 的子群，如果 $H$ 在运算 $\cdot$ 下构成群。

$H \subseteq G$ is a subgroup of $G$ if $H$ forms a group under the operation $\cdot$.

**正规子群 / Normal Subgroup:**

$N \trianglelefteq G$ 如果 $\forall g \in G, gNg^{-1} = N$。

$N \trianglelefteq G$ if $\forall g \in G, gNg^{-1} = N$.

### 2.2 环论 / Ring Theory

**环 / Ring:**

环 $(R, +, \cdot)$ 是一个集合 $R$ 和两个二元运算 $+$ 和 $\cdot$，满足：

A ring $(R, +, \cdot)$ is a set $R$ with two binary operations $+$ and $\cdot$ satisfying:

1. $(R, +)$ 是阿贝尔群 / $(R, +)$ is an abelian group
2. **乘法结合律 / Multiplicative Associativity:** $(a \cdot b) \cdot c = a \cdot (b \cdot c)$
3. **分配律 / Distributivity:** $a \cdot (b + c) = a \cdot b + a \cdot c$ 和 $(a + b) \cdot c = a \cdot c + b \cdot c$

**域 / Field:**

域是交换环，其中非零元素都有乘法逆元。

A field is a commutative ring where every nonzero element has a multiplicative inverse.

### 2.3 模论 / Module Theory

**左模 / Left Module:**

$R$-模 $M$ 是阿贝尔群 $(M, +)$ 和标量乘法 $R \times M \rightarrow M$，满足：

An $R$-module $M$ is an abelian group $(M, +)$ with scalar multiplication $R \times M \rightarrow M$ satisfying:

1. $(r + s)m = rm + sm$
2. $r(m + n) = rm + rn$
3. $(rs)m = r(sm)$
4. $1m = m$

---

## 3. 线性代数 / Linear Algebra

### 3.1 向量空间 / Vector Spaces

**向量空间 / Vector Space:**

向量空间 $V$ 是域 $\mathbb{F}$ 上的阿贝尔群，配备标量乘法 $\mathbb{F} \times V \rightarrow V$。

A vector space $V$ is an abelian group over a field $\mathbb{F}$ equipped with scalar multiplication $\mathbb{F} \times V \rightarrow V$.

**线性无关 / Linear Independence:**

向量组 $\{v_1, \ldots, v_n\}$ 线性无关，如果：

A set of vectors $\{v_1, \ldots, v_n\}$ is linearly independent if:

$$\sum_{i=1}^n c_i v_i = 0 \Rightarrow c_i = 0 \text{ for all } i$$

**基 / Basis:**

向量空间 $V$ 的基是线性无关的生成集。

A basis of vector space $V$ is a linearly independent spanning set.

**维度 / Dimension:**

向量空间的维度是其基的基数。

The dimension of a vector space is the cardinality of its basis.

### 3.2 线性变换 / Linear Transformations

**线性变换 / Linear Transformation:**

$T: V \rightarrow W$ 是线性变换，如果：

$T: V \rightarrow W$ is a linear transformation if:

1. $T(v_1 + v_2) = T(v_1) + T(v_2)$
2. $T(cv) = cT(v)$

**核 / Kernel:**

$$\ker(T) = \{v \in V \mid T(v) = 0\}$$

**像 / Image:**

$$\text{im}(T) = \{T(v) \mid v \in V\}$$

**秩-零化度定理 / Rank-Nullity Theorem:**

$$\dim(V) = \dim(\ker(T)) + \dim(\text{im}(T))$$

### 3.3 特征值与特征向量 / Eigenvalues and Eigenvectors

**特征值 / Eigenvalue:**

$\lambda$ 是 $T$ 的特征值，如果存在非零向量 $v$ 使得：

$\lambda$ is an eigenvalue of $T$ if there exists a nonzero vector $v$ such that:

$$T(v) = \lambda v$$

**特征向量 / Eigenvector:**

满足上述条件的向量 $v$ 是特征向量。

The vector $v$ satisfying the above condition is an eigenvector.

**特征多项式 / Characteristic Polynomial:**

$$p_T(\lambda) = \det(T - \lambda I)$$

---

## 4. 泛函分析 / Functional Analysis

### 4.1 度量空间 / Metric Spaces

**度量 / Metric:**

度量 $d: X \times X \rightarrow \mathbb{R}$ 满足：

A metric $d: X \times X \rightarrow \mathbb{R}$ satisfies:

1. $d(x,y) \geq 0$ 且 $d(x,y) = 0 \Leftrightarrow x = y$
2. $d(x,y) = d(y,x)$
3. $d(x,z) \leq d(x,y) + d(y,z)$ (三角不等式)

**收敛 / Convergence:**

序列 $(x_n)$ 收敛到 $x$，如果：

A sequence $(x_n)$ converges to $x$ if:

$$\lim_{n \rightarrow \infty} d(x_n, x) = 0$$

**完备性 / Completeness:**

度量空间是完备的，如果每个柯西序列都收敛。

A metric space is complete if every Cauchy sequence converges.

### 4.2 巴拿赫空间 / Banach Spaces

**范数 / Norm:**

范数 $\|\cdot\|: V \rightarrow \mathbb{R}$ 满足：

A norm $\|\cdot\|: V \rightarrow \mathbb{R}$ satisfies:

1. $\|x\| \geq 0$ 且 $\|x\| = 0 \Leftrightarrow x = 0$
2. $\|\alpha x\| = |\alpha| \|x\|$
3. $\|x + y\| \leq \|x\| + \|y\|$

**巴拿赫空间 / Banach Space:**

完备的赋范向量空间。

A complete normed vector space.

### 4.3 希尔伯特空间 / Hilbert Spaces

**内积 / Inner Product:**

内积 $\langle \cdot, \cdot \rangle: H \times H \rightarrow \mathbb{C}$ 满足：

An inner product $\langle \cdot, \cdot \rangle: H \times H \rightarrow \mathbb{C}$ satisfies:

1. $\langle x, y \rangle = \overline{\langle y, x \rangle}$
2. $\langle \alpha x + \beta y, z \rangle = \alpha \langle x, z \rangle + \beta \langle y, z \rangle$
3. $\langle x, x \rangle \geq 0$ 且 $\langle x, x \rangle = 0 \Leftrightarrow x = 0$

**希尔伯特空间 / Hilbert Space:**

完备的内积空间。

A complete inner product space.

**正交性 / Orthogonality:**

$x \perp y$ 如果 $\langle x, y \rangle = 0$。

$x \perp y$ if $\langle x, y \rangle = 0$.

---

## 5. 拓扑学 / Topology

### 5.1 拓扑空间 / Topological Spaces

**拓扑 / Topology:**

集合 $X$ 上的拓扑 $\tau$ 是 $X$ 的子集族，满足：

A topology $\tau$ on a set $X$ is a family of subsets of $X$ satisfying:

1. $\emptyset, X \in \tau$
2. 任意并集属于 $\tau$ / arbitrary unions belong to $\tau$
3. 有限交集属于 $\tau$ / finite intersections belong to $\tau$

**开集 / Open Set:**

$U \subseteq X$ 是开集，如果 $U \in \tau$。

$U \subseteq X$ is open if $U \in \tau$.

**闭集 / Closed Set:**

$F \subseteq X$ 是闭集，如果 $X \setminus F$ 是开集。

$F \subseteq X$ is closed if $X \setminus F$ is open.

### 5.2 连续性与连通性 / Continuity and Connectedness

**连续函数 / Continuous Function:**

$f: X \rightarrow Y$ 连续，如果 $f^{-1}(U)$ 对每个开集 $U$ 都是开集。

$f: X \rightarrow Y$ is continuous if $f^{-1}(U)$ is open for every open set $U$.

**连通性 / Connectedness:**

拓扑空间 $X$ 连通，如果它不能表示为两个非空开集的不交并。

A topological space $X$ is connected if it cannot be written as the disjoint union of two nonempty open sets.

**紧致性 / Compactness:**

拓扑空间 $X$ 紧致，如果每个开覆盖都有有限子覆盖。

A topological space $X$ is compact if every open cover has a finite subcover.

---

## 6. 测度论 / Measure Theory

### 6.1 σ-代数 / σ-Algebra

**σ-代数 / σ-Algebra:**

集合 $X$ 上的 σ-代数 $\mathcal{A}$ 是 $X$ 的子集族，满足：

A σ-algebra $\mathcal{A}$ on a set $X$ is a family of subsets of $X$ satisfying:

1. $X \in \mathcal{A}$
2. $A \in \mathcal{A} \Rightarrow A^c \in \mathcal{A}$
3. $A_n \in \mathcal{A} \Rightarrow \bigcup_{n=1}^{\infty} A_n \in \mathcal{A}$

### 6.2 测度 / Measure

**测度 / Measure:**

测度 $\mu: \mathcal{A} \rightarrow [0, \infty]$ 满足：

A measure $\mu: \mathcal{A} \rightarrow [0, \infty]$ satisfies:

1. $\mu(\emptyset) = 0$
2. $\mu(\bigcup_{n=1}^{\infty} A_n) = \sum_{n=1}^{\infty} \mu(A_n)$ (σ-可加性)

**勒贝格测度 / Lebesgue Measure:**

$\mathbb{R}^n$ 上的标准测度。

The standard measure on $\mathbb{R}^n$.

### 6.3 积分 / Integration

**勒贝格积分 / Lebesgue Integral:**

$$\int_X f \, d\mu = \lim_{n \rightarrow \infty} \int_X f_n \, d\mu$$

其中 $(f_n)$ 是简单函数的递增序列。

where $(f_n)$ is an increasing sequence of simple functions.

---

## 7. 概率论 / Probability Theory

### 7.1 概率空间 / Probability Space

**概率空间 / Probability Space:**

$(\Omega, \mathcal{F}, P)$ 其中：

- $\Omega$ 是样本空间 / sample space
- $\mathcal{F}$ 是事件 σ-代数 / event σ-algebra
- $P$ 是概率测度 / probability measure

**概率测度 / Probability Measure:**

$P: \mathcal{F} \rightarrow [0,1]$ 满足：

$P: \mathcal{F} \rightarrow [0,1]$ satisfies:

1. $P(\Omega) = 1$
2. $P(\bigcup_{n=1}^{\infty} A_n) = \sum_{n=1}^{\infty} P(A_n)$ (对互斥事件)

### 7.2 随机变量 / Random Variables

**随机变量 / Random Variable:**

$X: \Omega \rightarrow \mathbb{R}$ 是随机变量，如果 $X^{-1}(B) \in \mathcal{F}$ 对所有博雷尔集 $B$。

$X: \Omega \rightarrow \mathbb{R}$ is a random variable if $X^{-1}(B) \in \mathcal{F}$ for all Borel sets $B$.

**期望 / Expectation:**

$$E[X] = \int_{\Omega} X \, dP$$

**方差 / Variance:**

$$\text{Var}(X) = E[(X - E[X])^2]$$

### 7.3 条件概率 / Conditional Probability

**条件概率 / Conditional Probability:**

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

**贝叶斯定理 / Bayes' Theorem:**

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

---

## 8. 信息论 / Information Theory

### 8.1 熵 / Entropy

**香农熵 / Shannon Entropy:**

$$H(X) = -\sum_{i=1}^n p_i \log p_i$$

其中 $p_i$ 是随机变量 $X$ 取第 $i$ 个值的概率。

where $p_i$ is the probability that random variable $X$ takes the $i$-th value.

**联合熵 / Joint Entropy:**

$$H(X,Y) = -\sum_{i,j} p_{ij} \log p_{ij}$$

**条件熵 / Conditional Entropy:**

$$H(X \mid Y) = H(X,Y) - H(Y)$$

### 8.2 互信息 / Mutual Information

**互信息 / Mutual Information:**

$$I(X;Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X)$$

**相对熵 / Relative Entropy (KL散度):**

$$D_{KL}(P \parallel Q) = \sum_i p_i \log \frac{p_i}{q_i}$$

---

## 9. 优化理论 / Optimization Theory

### 9.1 凸优化 / Convex Optimization

**凸集 / Convex Set:**

集合 $C$ 是凸的，如果：

A set $C$ is convex if:

$$\lambda x + (1-\lambda) y \in C \text{ for all } x,y \in C, \lambda \in [0,1]$$

**凸函数 / Convex Function:**

函数 $f$ 是凸的，如果：

A function $f$ is convex if:

$$f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y)$$

### 9.2 拉格朗日乘数法 / Lagrange Multipliers

**拉格朗日函数 / Lagrangian:**

$$L(x, \lambda) = f(x) + \sum_{i=1}^m \lambda_i g_i(x)$$

**KKT条件 / KKT Conditions:**

对于优化问题 $\min f(x)$ s.t. $g_i(x) \leq 0$：

For optimization problem $\min f(x)$ s.t. $g_i(x) \leq 0$:

1. $\nabla f(x) + \sum_{i=1}^m \lambda_i \nabla g_i(x) = 0$
2. $\lambda_i g_i(x) = 0$ (互补松弛性)
3. $\lambda_i \geq 0$

---

## 10. 范畴论 / Category Theory

### 10.1 基本概念 / Basic Concepts

**范畴 / Category:**

范畴 $\mathcal{C}$ 包含：

A category $\mathcal{C}$ consists of:

1. 对象集合 / a collection of objects
2. 态射集合 / a collection of morphisms
3. 复合运算 / composition operation
4. 单位态射 / identity morphisms

**函子 / Functor:**

函子 $F: \mathcal{C} \rightarrow \mathcal{D}$ 将对象和态射映射到目标范畴。

A functor $F: \mathcal{C} \rightarrow \mathcal{D}$ maps objects and morphisms to the target category.

### 10.2 自然变换 / Natural Transformations

**自然变换 / Natural Transformation:**

自然变换 $\eta: F \rightarrow G$ 是态射族 $\{\eta_A: F(A) \rightarrow G(A)\}$，满足：

A natural transformation $\eta: F \rightarrow G$ is a family of morphisms $\{\eta_A: F(A) \rightarrow G(A)\}$ satisfying:

$$G(f) \circ \eta_A = \eta_B \circ F(f)$$

### 10.3 极限与余极限 / Limits and Colimits

**极限 / Limit:**

对象 $L$ 和自然变换 $\pi: \Delta L \rightarrow F$ 构成 $F$ 的极限，如果：

Object $L$ and natural transformation $\pi: \Delta L \rightarrow F$ form a limit of $F$ if:

$$\text{Hom}(X, L) \cong \text{Cone}(X, F)$$

**余极限 / Colimit:**

对象 $C$ 和自然变换 $\iota: F \rightarrow \Delta C$ 构成 $F$ 的余极限。

Object $C$ and natural transformation $\iota: F \rightarrow \Delta C$ form a colimit of $F$.

---

## 代码示例 / Code Examples

### Rust实现：线性代数库

```rust
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone)]
struct Vector<T> {
    data: Vec<T>,
}

impl<T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Default> Vector<T> {
    fn new(data: Vec<T>) -> Self {
        Vector { data }
    }
    
    fn dot_product(&self, other: &Vector<T>) -> Option<T> {
        if self.data.len() != other.data.len() {
            return None;
        }
        
        let mut result = T::default();
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            result = result + *a * *b;
        }
        Some(result)
    }
    
    fn norm(&self) -> f64 
    where T: Into<f64> {
        let sum_squares: f64 = self.data.iter()
            .map(|&x| x.into().powi(2))
            .sum();
        sum_squares.sqrt()
    }
}

#[derive(Debug, Clone)]
struct Matrix<T> {
    data: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T: Copy + Add<Output = T> + Mul<Output = T> + Default> Matrix<T> {
    fn new(data: Vec<Vec<T>>) -> Option<Self> {
        if data.is_empty() || data[0].is_empty() {
            return None;
        }
        
        let rows = data.len();
        let cols = data[0].len();
        
        // 检查所有行长度一致
        if data.iter().any(|row| row.len() != cols) {
            return None;
        }
        
        Some(Matrix { data, rows, cols })
    }
    
    fn multiply(&self, other: &Matrix<T>) -> Option<Matrix<T>> {
        if self.cols != other.rows {
            return None;
        }
        
        let mut result = vec![vec![T::default(); other.cols]; self.rows];
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    result[i][j] = result[i][j] + self.data[i][k] * other.data[k][j];
                }
            }
        }
        
        Matrix::new(result)
    }
    
    fn determinant(&self) -> Option<T> 
    where T: Into<f64> + From<f64> {
        if self.rows != self.cols {
            return None;
        }
        
        // 简化的行列式计算（仅适用于2x2和3x3矩阵）
        match self.rows {
            2 => {
                let a = self.data[0][0];
                let b = self.data[0][1];
                let c = self.data[1][0];
                let d = self.data[1][1];
                Some(a * d - b * c)
            },
            3 => {
                // 3x3行列式计算
                let a = self.data[0][0];
                let b = self.data[0][1];
                let c = self.data[0][2];
                let d = self.data[1][0];
                let e = self.data[1][1];
                let f = self.data[1][2];
                let g = self.data[2][0];
                let h = self.data[2][1];
                let i = self.data[2][2];
                
                Some(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))
            },
            _ => None, // 更高维度的行列式计算需要更复杂的算法
        }
    }
}

fn main() {
    // 向量示例
    let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
    
    println!("向量点积: {:?}", v1.dot_product(&v2));
    println!("向量范数: {}", v1.norm());
    
    // 矩阵示例
    let m1 = Matrix::new(vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0]
    ]).unwrap();
    
    let m2 = Matrix::new(vec![
        vec![5.0, 6.0],
        vec![7.0, 8.0]
    ]).unwrap();
    
    println!("矩阵乘法: {:?}", m1.multiply(&m2));
    println!("矩阵行列式: {:?}", m1.determinant());
}
```

### Haskell实现：概率论基础

```haskell
import Data.List (foldl')
import System.Random

-- 概率分布
data Probability = Probability { prob :: Double } deriving Show

instance Num Probability where
    (Probability p1) + (Probability p2) = Probability (p1 + p2)
    (Probability p1) * (Probability p2) = Probability (p1 * p2)
    abs (Probability p) = Probability (abs p)
    signum (Probability p) = Probability (signum p)
    fromInteger n = Probability (fromInteger n)
    negate (Probability p) = Probability (negate p)

-- 随机变量
data RandomVariable a = RV { 
    outcomes :: [(a, Probability)],
    expectation :: Double,
    variance :: Double
} deriving Show

-- 创建随机变量
createRV :: [(a, Double)] -> RandomVariable a
createRV outcomes = RV {
    outcomes = map (\(x, p) -> (x, Probability p)) outcomes,
    expectation = sum [x * p | (x, p) <- outcomes],
    variance = sum [(x - expectation) ^ 2 * p | (x, p) <- outcomes]
}
where
    expectation = sum [x * p | (x, p) <- outcomes]

-- 伯努利分布
bernoulli :: Double -> RandomVariable Bool
bernoulli p = createRV [(True, p), (False, 1 - p)]

-- 二项分布
binomial :: Int -> Double -> RandomVariable Int
binomial n p = createRV [(k, prob) | k <- [0..n]]
where
    prob = fromIntegral (choose n k) * p^k * (1-p)^(n-k)
    choose n k = product [n-k+1..n] `div` product [1..k]

-- 正态分布近似
normalApprox :: Double -> Double -> RandomVariable Double
normalApprox mu sigma = createRV [(x, density x) | x <- [-3*sigma..3*sigma]]
where
    density x = exp (-((x - mu)^2) / (2 * sigma^2)) / (sigma * sqrt (2 * pi))

-- 条件概率
conditionalProb :: Eq a => RandomVariable a -> a -> a -> Double
conditionalProb rv a b = case lookup a (outcomes rv) of
    Just (Probability p_a) -> case lookup b (outcomes rv) of
        Just (Probability p_b) -> prob (Probability p_a * Probability p_b) / prob p_b
        Nothing -> 0
    Nothing -> 0

-- 信息熵
entropy :: RandomVariable a -> Double
entropy rv = -sum [p * logBase 2 p | (_, Probability p) <- outcomes rv, p > 0]

-- 互信息
mutualInformation :: (Eq a, Eq b) => RandomVariable (a, b) -> RandomVariable a -> RandomVariable b -> Double
mutualInformation joint_rv rv_x rv_y = entropy rv_x + entropy rv_y - entropy joint_rv

-- 示例
main :: IO ()
main = do
    let coin = bernoulli 0.5
    let dice = createRV [(i, 1/6) | i <- [1..6]]
    
    putStrLn "伯努利分布 (p=0.5):"
    print coin
    
    putStrLn "\n六面骰子分布:"
    print dice
    
    putStrLn $ "\n硬币熵: " ++ show (entropy coin)
    putStrLn $ "骰子熵: " ++ show (entropy dice)
```

---

## 参考文献 / References

1. Rudin, W. (1976). *Principles of Mathematical Analysis*. McGraw-Hill.
2. Axler, S. (2015). *Linear Algebra Done Right*. Springer.
3. Conway, J. B. (1990). *A Course in Functional Analysis*. Springer.
4. Munkres, J. R. (2000). *Topology*. Prentice Hall.
5. Folland, G. B. (1999). *Real Analysis: Modern Techniques and Their Applications*. Wiley.
6. Billingsley, P. (1995). *Probability and Measure*. Wiley.
7. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*. Wiley.
8. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
9. Mac Lane, S. (1998). *Categories for the Working Mathematician*. Springer.

---

*本模块为FormalAI提供了全面的数学基础，涵盖了从基础集合论到现代范畴论的完整数学工具集。*
