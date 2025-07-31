# 1.3 计算理论 / Computation Theory

## 概述 / Overview

计算理论研究计算的本质、能力和限制，为FormalAI提供计算复杂性和可计算性的理论基础。

Computation theory studies the nature, capabilities, and limitations of computation, providing theoretical foundations for computational complexity and computability in FormalAI.

## 目录 / Table of Contents

- [1.3 计算理论 / Computation Theory](#13-计算理论--computation-theory)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [1. 可计算性理论 / Computability Theory](#1-可计算性理论--computability-theory)
    - [1.1 图灵机 / Turing Machine](#11-图灵机--turing-machine)
    - [1.2 丘奇-图灵论题 / Church-Turing Thesis](#12-丘奇-图灵论题--church-turing-thesis)
    - [1.3 停机问题 / Halting Problem](#13-停机问题--halting-problem)
  - [2. 计算复杂性理论 / Computational Complexity Theory](#2-计算复杂性理论--computational-complexity-theory)
    - [2.1 时间复杂性 / Time Complexity](#21-时间复杂性--time-complexity)
    - [2.2 空间复杂性 / Space Complexity](#22-空间复杂性--space-complexity)
    - [2.3 P vs NP问题 / P vs NP Problem](#23-p-vs-np问题--p-vs-np-problem)
  - [3. 自动机理论 / Automata Theory](#3-自动机理论--automata-theory)
    - [3.1 有限自动机 / Finite Automata](#31-有限自动机--finite-automata)
    - [3.2 下推自动机 / Pushdown Automata](#32-下推自动机--pushdown-automata)
    - [3.3 图灵机 / Turing Machines](#33-图灵机--turing-machines)
  - [4. 形式语言理论 / Formal Language Theory](#4-形式语言理论--formal-language-theory)
    - [4.1 乔姆斯基层次结构 / Chomsky Hierarchy](#41-乔姆斯基层次结构--chomsky-hierarchy)
    - [4.2 文法 / Grammars](#42-文法--grammars)
    - [4.3 语言操作 / Language Operations](#43-语言操作--language-operations)
  - [5. 递归论 / Recursion Theory](#5-递归论--recursion-theory)
    - [5.1 原始递归函数 / Primitive Recursive Functions](#51-原始递归函数--primitive-recursive-functions)
    - [5.2 一般递归函数 / General Recursive Functions](#52-一般递归函数--general-recursive-functions)
    - [5.3 递归可枚举集 / Recursively Enumerable Sets](#53-递归可枚举集--recursively-enumerable-sets)
  - [6. 算法分析 / Algorithm Analysis](#6-算法分析--algorithm-analysis)
    - [6.1 渐近分析 / Asymptotic Analysis](#61-渐近分析--asymptotic-analysis)
    - [6.2 分治算法 / Divide and Conquer](#62-分治算法--divide-and-conquer)
    - [6.3 动态规划 / Dynamic Programming](#63-动态规划--dynamic-programming)
  - [7. 并行计算理论 / Parallel Computation Theory](#7-并行计算理论--parallel-computation-theory)
    - [7.1 PRAM模型 / PRAM Model](#71-pram模型--pram-model)
    - [7.2 并行复杂性 / Parallel Complexity](#72-并行复杂性--parallel-complexity)
    - [7.3 并行算法 / Parallel Algorithms](#73-并行算法--parallel-algorithms)
  - [8. 量子计算理论 / Quantum Computation Theory](#8-量子计算理论--quantum-computation-theory)
    - [8.1 量子比特 / Qubits](#81-量子比特--qubits)
    - [8.2 量子算法 / Quantum Algorithms](#82-量子算法--quantum-algorithms)
    - [8.3 量子复杂性 / Quantum Complexity](#83-量子复杂性--quantum-complexity)
  - [9. 随机计算 / Randomized Computation](#9-随机计算--randomized-computation)
    - [9.1 随机算法 / Randomized Algorithms](#91-随机算法--randomized-algorithms)
    - [9.2 随机复杂性类 / Randomized Complexity Classes](#92-随机复杂性类--randomized-complexity-classes)
    - [9.3 随机化技术 / Randomization Techniques](#93-随机化技术--randomization-techniques)
  - [10. 近似算法理论 / Approximation Algorithm Theory](#10-近似算法理论--approximation-algorithm-theory)
    - [10.1 近似比 / Approximation Ratio](#101-近似比--approximation-ratio)
    - [10.2 近似算法 / Approximation Algorithms](#102-近似算法--approximation-algorithms)
    - [10.3 不可近似性 / Inapproximability](#103-不可近似性--inapproximability)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：图灵机模拟器](#rust实现图灵机模拟器)
    - [Haskell实现：计算复杂性分析](#haskell实现计算复杂性分析)
  - [参考文献 / References](#参考文献--references)

---

## 1. 可计算性理论 / Computability Theory

### 1.1 图灵机 / Turing Machine

**图灵机定义 / Turing Machine Definition:**

图灵机 $M = (Q, \Sigma, \Gamma, \delta, q_0, q_{\text{accept}}, q_{\text{reject}})$ 其中：

Turing machine $M = (Q, \Sigma, \Gamma, \delta, q_0, q_{\text{accept}}, q_{\text{reject}})$ where:

- $Q$ 是状态集合 / set of states
- $\Sigma$ 是输入字母表 / input alphabet
- $\Gamma$ 是磁带字母表 / tape alphabet
- $\delta$ 是转移函数 / transition function
- $q_0$ 是初始状态 / initial state
- $q_{\text{accept}}$ 是接受状态 / accept state
- $q_{\text{reject}}$ 是拒绝状态 / reject state

**转移函数 / Transition Function:**

$$\delta: Q \times \Gamma \rightarrow Q \times \Gamma \times \{L, R\}$$

**配置 / Configuration:**

$(q, w, i)$ 其中 $q$ 是当前状态，$w$ 是磁带内容，$i$ 是读写头位置。

$(q, w, i)$ where $q$ is current state, $w$ is tape content, $i$ is head position.

### 1.2 丘奇-图灵论题 / Church-Turing Thesis

**论题陈述 / Thesis Statement:**

任何可计算的函数都可以由图灵机计算。

Any computable function can be computed by a Turing machine.

**等价模型 / Equivalent Models:**

- $\lambda$ 演算 / $\lambda$-calculus
- 递归函数 / Recursive functions
- 寄存器机器 / Register machines
- 细胞自动机 / Cellular automata

### 1.3 停机问题 / Halting Problem

**停机问题 / Halting Problem:**

给定图灵机 $M$ 和输入 $w$，判断 $M$ 在输入 $w$ 上是否停机。

Given Turing machine $M$ and input $w$, determine whether $M$ halts on input $w$.

**不可判定性 / Undecidability:**

停机问题是不可判定的。

The halting problem is undecidable.

**证明 / Proof:**

假设存在图灵机 $H$ 解决停机问题，构造矛盾：

Assume Turing machine $H$ solves the halting problem, construct contradiction:

$$
D(w) = \begin{cases}
\text{halt} & \text{if } H(w,w) = \text{reject} \\
\text{loop} & \text{if } H(w,w) = \text{accept}
\end{cases}
$$

---

## 2. 计算复杂性理论 / Computational Complexity Theory

### 2.1 时间复杂性 / Time Complexity

**时间复杂性 / Time Complexity:**

$T_M(n) = \max\{t_M(w): |w| = n\}$

其中 $t_M(w)$ 是图灵机 $M$ 在输入 $w$ 上的运行步数。

where $t_M(w)$ is the number of steps Turing machine $M$ takes on input $w$.

**时间复杂性类 / Time Complexity Classes:**

- $\text{P}$: 多项式时间可判定问题
- $\text{NP}$: 非确定性多项式时间可判定问题
- $\text{EXP}$: 指数时间可判定问题

- $\text{P}$: Polynomial time decidable problems
- $\text{NP}$: Nondeterministic polynomial time decidable problems
- $\text{EXP}$: Exponential time decidable problems

### 2.2 空间复杂性 / Space Complexity

**空间复杂性 / Space Complexity:**

$S_M(n) = \max\{s_M(w): |w| = n\}$

其中 $s_M(w)$ 是图灵机 $M$ 在输入 $w$ 上使用的磁带格子数。

where $s_M(w)$ is the number of tape cells used by Turing machine $M$ on input $w$.

**空间复杂性类 / Space Complexity Classes:**

- $\text{L}$: 对数空间可判定问题
- $\text{PSPACE}$: 多项式空间可判定问题
- $\text{EXPSPACE}$: 指数空间可判定问题

### 2.3 P vs NP问题 / P vs NP Problem

**NP完全性 / NP-Completeness:**

问题 $A$ 是NP完全的，如果：

Problem $A$ is NP-complete if:

1. $A \in \text{NP}$
2. 对于所有 $B \in \text{NP}$，$B \leq_p A$

**NP完全问题 / NP-Complete Problems:**

- SAT (布尔可满足性问题)
- 3-SAT
- 哈密顿路径问题
- 旅行商问题

- SAT (Boolean satisfiability problem)
- 3-SAT
- Hamiltonian path problem
- Traveling salesman problem

---

## 3. 自动机理论 / Automata Theory

### 3.1 有限自动机 / Finite Automata

**确定性有限自动机 / Deterministic Finite Automaton (DFA):**

$M = (Q, \Sigma, \delta, q_0, F)$ 其中：

$M = (Q, \Sigma, \delta, q_0, F)$ where:

- $Q$ 是状态集合 / set of states
- $\Sigma$ 是输入字母表 / input alphabet
- $\delta: Q \times \Sigma \rightarrow Q$ 是转移函数 / transition function
- $q_0$ 是初始状态 / initial state
- $F \subseteq Q$ 是接受状态集合 / set of accept states

**非确定性有限自动机 / Nondeterministic Finite Automaton (NFA):**

$$\delta: Q \times \Sigma \rightarrow 2^Q$$

**等价性 / Equivalence:**

对于每个NFA，存在等价的DFA。

For every NFA, there exists an equivalent DFA.

### 3.2 下推自动机 / Pushdown Automata

**下推自动机 / Pushdown Automaton (PDA):**

$M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$ 其中：

$M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$ where:

- $\Gamma$ 是栈字母表 / stack alphabet
- $Z_0$ 是初始栈符号 / initial stack symbol
- $\delta: Q \times \Sigma \times \Gamma \rightarrow 2^{Q \times \Gamma^*}$

**上下文无关语言 / Context-Free Languages:**

下推自动机识别的语言类。

The class of languages recognized by pushdown automata.

### 3.3 图灵机 / Turing Machines

**图灵机 / Turing Machine:**

最强大的自动机模型，可以识别递归可枚举语言。

The most powerful automaton model, can recognize recursively enumerable languages.

**丘奇-图灵论题 / Church-Turing Thesis:**

任何可计算的函数都可以由图灵机计算。

Any computable function can be computed by a Turing machine.

---

## 4. 形式语言理论 / Formal Language Theory

### 4.1 乔姆斯基层次结构 / Chomsky Hierarchy

**正则语言 / Regular Languages (Type 3):**

由正则表达式或有限自动机定义。

Defined by regular expressions or finite automata.

**上下文无关语言 / Context-Free Languages (Type 2):**

由上下文无关文法定义。

Defined by context-free grammars.

**上下文相关语言 / Context-Sensitive Languages (Type 1):**

由上下文相关文法定义。

Defined by context-sensitive grammars.

**递归可枚举语言 / Recursively Enumerable Languages (Type 0):**

由图灵机识别。

Recognized by Turing machines.

### 4.2 文法 / Grammars

**上下文无关文法 / Context-Free Grammar:**

$G = (V, \Sigma, R, S)$ 其中：

$G = (V, \Sigma, R, S)$ where:

- $V$ 是变元集合 / set of variables
- $\Sigma$ 是终结符集合 / set of terminals
- $R$ 是产生式规则集合 / set of production rules
- $S$ 是起始符号 / start symbol

**产生式规则 / Production Rules:**

$A \rightarrow \alpha$ 其中 $A \in V$，$\alpha \in (V \cup \Sigma)^*$

$A \rightarrow \alpha$ where $A \in V$ and $\alpha \in (V \cup \Sigma)^*$

### 4.3 语言操作 / Language Operations

**并集 / Union:**

$L_1 \cup L_2 = \{w: w \in L_1 \text{ or } w \in L_2\}$

**连接 / Concatenation:**

$L_1 \circ L_2 = \{w_1w_2: w_1 \in L_1, w_2 \in L_2\}$

**克林星 / Kleene Star:**

$L^* = \bigcup_{i=0}^{\infty} L^i$

---

## 5. 递归论 / Recursion Theory

### 5.1 原始递归函数 / Primitive Recursive Functions

**基本函数 / Basic Functions:**

- 零函数：$Z(x) = 0$
- 后继函数：$S(x) = x + 1$
- 投影函数：$P_i^n(x_1, \ldots, x_n) = x_i$

- Zero function: $Z(x) = 0$
- Successor function: $S(x) = x + 1$
- Projection function: $P_i^n(x_1, \ldots, x_n) = x_i$

**组合 / Composition:**

$f(x_1, \ldots, x_n) = g(h_1(x_1, \ldots, x_n), \ldots, h_m(x_1, \ldots, x_n))$

**原始递归 / Primitive Recursion:**

$f(0, x_2, \ldots, x_n) = g(x_2, \ldots, x_n)$
$f(x_1 + 1, x_2, \ldots, x_n) = h(x_1, f(x_1, x_2, \ldots, x_n), x_2, \ldots, x_n)$

### 5.2 一般递归函数 / General Recursive Functions

**$\mu$ 算子 / $\mu$ Operator:**

$\mu y[f(x_1, \ldots, x_n, y) = 0] = \text{least } y \text{ such that } f(x_1, \ldots, x_n, y) = 0$

**递归函数 / Recursive Functions:**

包含所有原始递归函数和$\mu$算子的最小函数类。

The smallest class of functions containing all primitive recursive functions and the $\mu$ operator.

### 5.3 递归可枚举集 / Recursively Enumerable Sets

**递归可枚举集 / Recursively Enumerable Set:**

集合 $A$ 是递归可枚举的，如果存在部分递归函数 $f$ 使得：

Set $A$ is recursively enumerable if there exists a partial recursive function $f$ such that:

$A = \text{range}(f) = \{f(x): x \in \mathbb{N}\}$

**递归集 / Recursive Set:**

集合 $A$ 是递归的，如果其特征函数是递归的。

Set $A$ is recursive if its characteristic function is recursive.

---

## 6. 算法分析 / Algorithm Analysis

### 6.1 渐近分析 / Asymptotic Analysis

**大O记号 / Big-O Notation:**

$f(n) = O(g(n))$ 如果存在常数 $c > 0$ 和 $n_0$ 使得：

$f(n) = O(g(n))$ if there exist constants $c > 0$ and $n_0$ such that:

$f(n) \leq c \cdot g(n)$ 对于所有 $n \geq n_0$

for all $n \geq n_0$

**常见复杂度 / Common Complexities:**

- $O(1)$: 常数时间 / constant time
- $O(\log n)$: 对数时间 / logarithmic time
- $O(n)$: 线性时间 / linear time
- $O(n \log n)$: 线性对数时间 / linearithmic time
- $O(n^2)$: 二次时间 / quadratic time
- $O(2^n)$: 指数时间 / exponential time

### 6.2 分治算法 / Divide and Conquer

**分治策略 / Divide and Conquer Strategy:**

1. 分解：将问题分解为子问题
2. 解决：递归解决子问题
3. 合并：将子问题的解合并

   1. Divide: decompose problem into subproblems
   2. Conquer: recursively solve subproblems
   3. Combine: merge solutions of subproblems

**主定理 / Master Theorem:**

对于递归式 $T(n) = aT(n/b) + f(n)$：

For recurrence $T(n) = aT(n/b) + f(n)$:

$$
T(n) = \begin{cases}
O(n^{\log_b a}) & \text{if } f(n) = O(n^{\log_b a - \epsilon}) \\
O(n^{\log_b a} \log n) & \text{if } f(n) = O(n^{\log_b a}) \\
O(f(n)) & \text{if } f(n) = \Omega(n^{\log_b a + \epsilon})
\end{cases}
$$

### 6.3 动态规划 / Dynamic Programming

**最优子结构 / Optimal Substructure:**

问题的最优解包含其子问题的最优解。

The optimal solution to a problem contains optimal solutions to its subproblems.

**重叠子问题 / Overlapping Subproblems:**

子问题被重复计算。

Subproblems are computed repeatedly.

**记忆化 / Memoization:**

存储已计算的子问题解。

Storing computed subproblem solutions.

---

## 7. 并行计算理论 / Parallel Computation Theory

### 7.1 PRAM模型 / PRAM Model

**PRAM / Parallel Random Access Machine:**

- 共享内存模型
- 多个处理器
- 同步执行

- Shared memory model
- Multiple processors
- Synchronous execution

**PRAM变体 / PRAM Variants:**

- EREW: 独占读独占写
- CREW: 并发读独占写
- CRCW: 并发读并发写

- EREW: Exclusive read exclusive write
- CREW: Concurrent read exclusive write
- CRCW: Concurrent read concurrent write

### 7.2 并行复杂性 / Parallel Complexity

**NC类 / NC Class:**

$NC = \bigcup_{i=1}^{\infty} NC^i$

其中 $NC^i$ 是使用 $O(\log^i n)$ 时间和多项式处理器的可解问题。

where $NC^i$ are problems solvable using $O(\log^i n)$ time and polynomial processors.

**P-完全性 / P-Completeness:**

问题 $A$ 是P-完全的，如果：

Problem $A$ is P-complete if:

1. $A \in P$
2. 对于所有 $B \in P$，$B \leq_{NC} A$

### 7.3 并行算法 / Parallel Algorithms

**并行排序 / Parallel Sorting:**

- 并行归并排序
- 并行快速排序
- 并行基数排序

- Parallel merge sort
- Parallel quicksort
- Parallel radix sort

**并行图算法 / Parallel Graph Algorithms:**

- 并行BFS
- 并行DFS
- 并行最短路径

- Parallel BFS
- Parallel DFS
- Parallel shortest path

---

## 8. 量子计算理论 / Quantum Computation Theory

### 8.1 量子比特 / Qubits

**量子比特 / Qubit:**

$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$

其中 $|\alpha|^2 + |\beta|^2 = 1$

where $|\alpha|^2 + |\beta|^2 = 1$

**量子门 / Quantum Gates:**

- Hadamard门：$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$
- CNOT门：$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$

### 8.2 量子算法 / Quantum Algorithms

**Deutsch-Jozsa算法 / Deutsch-Jozsa Algorithm:**

判断函数是否为常数或平衡。

Determine whether a function is constant or balanced.

**Grover算法 / Grover Algorithm:**

在无序数据库中搜索，复杂度 $O(\sqrt{N})$。

Search in unordered database with complexity $O(\sqrt{N})$.

**Shor算法 / Shor Algorithm:**

整数分解，量子多项式时间。

Integer factorization in quantum polynomial time.

### 8.3 量子复杂性 / Quantum Complexity

**BQP类 / BQP Class:**

有界错误量子多项式时间。

Bounded error quantum polynomial time.

**量子优势 / Quantum Advantage:**

某些问题在量子计算机上比经典计算机更高效。

Some problems are more efficient on quantum computers than classical computers.

---

## 9. 随机计算 / Randomized Computation

### 9.1 随机算法 / Randomized Algorithms

**Las Vegas算法 / Las Vegas Algorithm:**

总是返回正确答案，运行时间随机。

Always returns correct answer, running time is random.

**Monte Carlo算法 / Monte Carlo Algorithm:**

可能返回错误答案，运行时间确定。

May return incorrect answer, running time is deterministic.

### 9.2 随机复杂性类 / Randomized Complexity Classes

**RP类 / RP Class:**

随机多项式时间，单侧错误。

Randomized polynomial time with one-sided error.

**BPP类 / BPP Class:**

有界错误概率多项式时间。

Bounded error probability polynomial time.

**ZPP类 / ZPP类:**

零错误概率多项式时间。

Zero error probability polynomial time.

### 9.3 随机化技术 / Randomization Techniques

**随机采样 / Random Sampling:**

通过随机采样估计复杂函数的期望值。

Estimate expected value of complex functions through random sampling.

**随机游走 / Random Walks:**

在图或空间中随机移动。

Random movement in graphs or spaces.

**概率放大 / Probability Amplification:**

通过重复运行减少错误概率。

Reduce error probability through repeated runs.

---

## 10. 近似算法理论 / Approximation Algorithm Theory

### 10.1 近似比 / Approximation Ratio

**近似比 / Approximation Ratio:**

对于最小化问题：

For minimization problems:

$$\rho = \frac{C_{\text{approx}}}{C_{\text{opt}}}$$

对于最大化问题：

For maximization problems:

$$\rho = \frac{C_{\text{opt}}}{C_{\text{approx}}}$$

### 10.2 近似算法 / Approximation Algorithms

**贪心算法 / Greedy Algorithms:**

- 集合覆盖问题
- 顶点覆盖问题
- 旅行商问题

- Set cover problem
- Vertex cover problem
- Traveling salesman problem

**线性规划松弛 / Linear Programming Relaxation:**

将整数规划松弛为线性规划。

Relax integer programming to linear programming.

### 10.3 不可近似性 / Inapproximability

**PCP定理 / PCP Theorem:**

$\text{NP} = \text{PCP}(O(\log n), O(1))$

**不可近似结果 / Inapproximability Results:**

某些问题在P≠NP假设下无法有效近似。

Some problems cannot be efficiently approximated under P≠NP assumption.

---

## 代码示例 / Code Examples

### Rust实现：图灵机模拟器

```rust
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Direction {
    Left,
    Right,
    Stay,
}

#[derive(Debug, Clone)]
struct Transition {
    next_state: String,
    write_symbol: char,
    direction: Direction,
}

#[derive(Debug, Clone)]
struct TuringMachine {
    states: Vec<String>,
    alphabet: Vec<char>,
    tape_alphabet: Vec<char>,
    transitions: HashMap<(String, char), Transition>,
    initial_state: String,
    accept_states: Vec<String>,
    reject_states: Vec<String>,
}

impl TuringMachine {
    fn new() -> Self {
        TuringMachine {
            states: Vec::new(),
            alphabet: Vec::new(),
            tape_alphabet: Vec::new(),
            transitions: HashMap::new(),
            initial_state: String::new(),
            accept_states: Vec::new(),
            reject_states: Vec::new(),
        }
    }
    
    fn add_transition(&mut self, current_state: &str, read_symbol: char, 
                      next_state: &str, write_symbol: char, direction: Direction) {
        self.transitions.insert(
            (current_state.to_string(), read_symbol),
            Transition {
                next_state: next_state.to_string(),
                write_symbol,
                direction,
            }
        );
    }
    
    fn run(&self, input: &str) -> bool {
        let mut tape: Vec<char> = input.chars().collect();
        let mut head_position = 0;
        let mut current_state = self.initial_state.clone();
        
        // 扩展磁带
        while head_position >= tape.len() {
            tape.push('_');
        }
        while head_position < 0 {
            tape.insert(0, '_');
            head_position += 1;
        }
        
        let mut steps = 0;
        let max_steps = 10000; // 防止无限循环
        
        while steps < max_steps {
            let current_symbol = tape[head_position];
            
            // 检查是否在接受或拒绝状态
            if self.accept_states.contains(&current_state) {
                return true;
            }
            if self.reject_states.contains(&current_state) {
                return false;
            }
            
            // 查找转移
            let key = (current_state.clone(), current_symbol);
            if let Some(transition) = self.transitions.get(&key) {
                // 执行转移
                tape[head_position] = transition.write_symbol;
                current_state = transition.next_state.clone();
                
                match transition.direction {
                    Direction::Left => head_position -= 1,
                    Direction::Right => head_position += 1,
                    Direction::Stay => {},
                }
                
                // 扩展磁带
                while head_position >= tape.len() {
                    tape.push('_');
                }
                while head_position < 0 {
                    tape.insert(0, '_');
                    head_position += 1;
                }
            } else {
                // 没有转移，停机
                return false;
            }
            
            steps += 1;
        }
        
        false // 超过最大步数
    }
}

fn create_palindrome_checker() -> TuringMachine {
    let mut tm = TuringMachine::new();
    
    // 设置状态和字母表
    tm.states = vec!["q0".to_string(), "q1".to_string(), "q2".to_string(), 
                     "q3".to_string(), "q4".to_string(), "q_accept".to_string(), 
                     "q_reject".to_string()];
    tm.alphabet = vec!['0', '1'];
    tm.tape_alphabet = vec!['0', '1', '_', 'X', 'Y'];
    tm.initial_state = "q0".to_string();
    tm.accept_states = vec!["q_accept".to_string()];
    tm.reject_states = vec!["q_reject".to_string()];
    
    // 添加转移规则（简化的回文检查器）
    tm.add_transition("q0", '0', "q1", 'X', Direction::Right);
    tm.add_transition("q0", '1', "q2", 'Y', Direction::Right);
    tm.add_transition("q0", '_', "q_accept", '_', Direction::Stay);
    
    tm.add_transition("q1", '0', "q1", '0', Direction::Right);
    tm.add_transition("q1", '1', "q1", '1', Direction::Right);
    tm.add_transition("q1", '_', "q3", '_', Direction::Left);
    
    tm.add_transition("q2", '0', "q2", '0', Direction::Right);
    tm.add_transition("q2", '1', "q2", '1', Direction::Right);
    tm.add_transition("q2", '_', "q4", '_', Direction::Left);
    
    tm.add_transition("q3", '0', "q3", '0', Direction::Left);
    tm.add_transition("q3", '1', "q_reject", '1', Direction::Stay);
    tm.add_transition("q3", 'X', "q0", 'X', Direction::Right);
    
    tm.add_transition("q4", '0', "q_reject", '0', Direction::Stay);
    tm.add_transition("q4", '1', "q4", '1', Direction::Left);
    tm.add_transition("q4", 'Y', "q0", 'Y', Direction::Right);
    
    tm
}

fn main() {
    let tm = create_palindrome_checker();
    
    // 测试回文检查器
    let test_cases = vec!["", "0", "1", "00", "11", "01", "10", "000", "111", "010"];
    
    for test_case in test_cases {
        let result = tm.run(test_case);
        println!("输入: '{}', 结果: {}", test_case, result);
    }
}
```

### Haskell实现：计算复杂性分析

```haskell
import Data.List (sort, foldl')
import Data.Map (Map, fromList, (!))

-- 算法复杂度分析
data Complexity = O1 | OLogN | ON | ONLogN | ON2 | ON3 | O2N deriving (Show, Eq)

-- 算法分析结果
data AlgorithmAnalysis = AlgorithmAnalysis {
    name :: String,
    timeComplexity :: Complexity,
    spaceComplexity :: Complexity,
    isOptimal :: Bool
} deriving Show

-- 排序算法分析
sortingAlgorithms :: [AlgorithmAnalysis]
sortingAlgorithms = [
    AlgorithmAnalysis "Bubble Sort" ON2 ON False,
    AlgorithmAnalysis "Insertion Sort" ON2 O1 False,
    AlgorithmAnalysis "Selection Sort" ON2 O1 False,
    AlgorithmAnalysis "Merge Sort" ONLogN ON True,
    AlgorithmAnalysis "Quick Sort" ONLogN OLogN True,
    AlgorithmAnalysis "Heap Sort" ONLogN O1 True,
    AlgorithmAnalysis "Counting Sort" ON ON False,
    AlgorithmAnalysis "Radix Sort" ON ON False
]

-- 搜索算法分析
searchAlgorithms :: [AlgorithmAnalysis]
searchAlgorithms = [
    AlgorithmAnalysis "Linear Search" ON O1 False,
    AlgorithmAnalysis "Binary Search" OLogN O1 True,
    AlgorithmAnalysis "Hash Table Search" O1 ON False
]

-- 图算法分析
graphAlgorithms :: [AlgorithmAnalysis]
graphAlgorithms = [
    AlgorithmAnalysis "BFS" ON ON False,
    AlgorithmAnalysis "DFS" ON ON False,
    AlgorithmAnalysis "Dijkstra" ON2 ON False,
    AlgorithmAnalysis "Floyd-Warshall" ON3 ON False,
    AlgorithmAnalysis "Prim's MST" ON2 ON False,
    AlgorithmAnalysis "Kruskal's MST" ONLogN ON False
]

-- 复杂度比较
compareComplexity :: Complexity -> Complexity -> Ordering
compareComplexity a b = case (a, b) of
    (O1, _) -> LT
    (OLogN, O1) -> GT
    (OLogN, _) -> LT
    (ON, O1) -> GT
    (ON, OLogN) -> GT
    (ON, _) -> LT
    (ONLogN, O1) -> GT
    (ONLogN, OLogN) -> GT
    (ONLogN, ON) -> GT
    (ONLogN, _) -> LT
    (ON2, O1) -> GT
    (ON2, OLogN) -> GT
    (ON2, ON) -> GT
    (ON2, ONLogN) -> GT
    (ON2, _) -> LT
    (ON3, O2N) -> LT
    (ON3, _) -> GT
    (O2N, _) -> GT

-- 最优算法筛选
optimalAlgorithms :: [AlgorithmAnalysis] -> [AlgorithmAnalysis]
optimalAlgorithms = filter isOptimal

-- 按复杂度排序
sortByComplexity :: [AlgorithmAnalysis] -> [AlgorithmAnalysis]
sortByComplexity = sortBy (\a b -> compareComplexity (timeComplexity a) (timeComplexity b))

-- 复杂度统计
complexityStats :: [AlgorithmAnalysis] -> Map Complexity Int
complexityStats algorithms = foldl' (\acc alg -> 
    let comp = timeComplexity alg
        count = maybe 0 id (lookup comp acc) + 1
    in (comp, count) : delete comp acc) [] algorithms

-- 示例
main :: IO ()
main = do
    putStrLn "排序算法分析:"
    mapM_ print sortingAlgorithms
    
    putStrLn "\n最优排序算法:"
    mapM_ print (optimalAlgorithms sortingAlgorithms)
    
    putStrLn "\n按复杂度排序的搜索算法:"
    mapM_ print (sortByComplexity searchAlgorithms)
    
    putStrLn "\n图算法复杂度统计:"
    let stats = complexityStats graphAlgorithms
    mapM_ print (toList stats)
    
    putStrLn "\n计算理论总结:"
    putStrLn "- P类: 多项式时间可解问题"
    putStrLn "- NP类: 非确定性多项式时间可解问题"
    putStrLn "- NP完全问题: 最难的NP问题"
    putStrLn "- 可计算性: 图灵机可计算的问题"
    putStrLn "- 不可计算问题: 停机问题等"
```

---

## 参考文献 / References

1. Sipser, M. (2013). *Introduction to the Theory of Computation*. Cengage Learning.
2. Hopcroft, J. E., Motwani, R., & Ullman, J. D. (2006). *Introduction to Automata Theory, Languages, and Computation*. Pearson.
3. Arora, S., & Barak, B. (2009). *Computational Complexity: A Modern Approach*. Cambridge University Press.
4. Papadimitriou, C. H. (1994). *Computational Complexity*. Addison-Wesley.
5. Rogers, H. (1987). *Theory of Recursive Functions and Effective Computability*. MIT Press.
6. Cormen, T. H., et al. (2009). *Introduction to Algorithms*. MIT Press.
7. Nielsen, M. A., & Chuang, I. L. (2010). *Quantum Computation and Quantum Information*. Cambridge University Press.
8. Motwani, R., & Raghavan, P. (1995). *Randomized Algorithms*. Cambridge University Press.
9. Vazirani, V. V. (2001). *Approximation Algorithms*. Springer.

---

*本模块为FormalAI提供了坚实的计算理论基础，涵盖了从可计算性到复杂性的完整理论体系。*
