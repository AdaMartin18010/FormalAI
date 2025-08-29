# 1.1 形式逻辑 / Formal Logic / Formale Logik / Logique formelle

## 概述 / Overview

形式逻辑是研究推理形式和有效性的数学分支，为FormalAI提供严格的逻辑基础。本文档涵盖命题逻辑、谓词逻辑、模态逻辑等核心内容。

Formal logic is the mathematical study of reasoning forms and validity, providing rigorous logical foundations for FormalAI. This document covers propositional logic, predicate logic, modal logic, and other core content.

## 核心概念 / Core Concepts

### 命题逻辑 / Propositional Logic

- **命题 / Propositions**: 具有真值的陈述句
- **连接词 / Connectives**: ¬, ∧, ∨, →, ↔
- **真值表 / Truth Tables**: 显示所有可能真值组合
- **逻辑等价 / Logical Equivalence**: 在所有解释下具有相同真值

### 谓词逻辑 / Predicate Logic

- **谓词 / Predicates**: 描述对象性质的函数
- **量词 / Quantifiers**: ∀ (全称), ∃ (存在)
- **一阶逻辑 / First-Order Logic**: 包含量词的逻辑系统

### 模态逻辑 / Modal Logic

- **模态算子 / Modal Operators**: □ (必然), ◇ (可能)
- **可能世界语义 / Possible Worlds Semantics**: 克里普克模型
- **模态系统 / Modal Systems**: K, T, S4, S5

## 数学形式化 / Mathematical Formalization

### 命题逻辑形式化

$$\begin{align}
\text{Proposition} &= \text{Declarative\_Sentence} \land \text{Truth\_Value} \\
p \land q &= \text{AND}(p, q) \\
p \lor q &= \text{OR}(p, q) \\
p \rightarrow q &= \neg p \lor q \\
p \leftrightarrow q &= (p \rightarrow q) \land (q \rightarrow p)
\end{align}$$

### 谓词逻辑形式化
$$\begin{align}
\forall x P(x) &\equiv \text{For all } x, P(x) \\
\exists x P(x) &\equiv \text{There exists } x \text{ such that } P(x) \\
\neg \forall x P(x) &\equiv \exists x \neg P(x)
\end{align}$$

### 模态逻辑形式化
$$\begin{align}
\Box \phi &\equiv \text{Necessarily } \phi \\
\Diamond \phi &\equiv \text{Possibly } \phi \\
\Diamond \phi &\equiv \neg \Box \neg \phi
\end{align}$$

## 推理规则 / Inference Rules

### 基本推理规则
1. **假言推理 / Modus Ponens**: $\frac{p \rightarrow q \quad p}{q}$
2. **假言三段论 / Hypothetical Syllogism**: $\frac{p \rightarrow q \quad q \rightarrow r}{p \rightarrow r}$
3. **析取三段论 / Disjunctive Syllogism**: $\frac{p \lor q \quad \neg p}{q}$

### 等价律
- **德摩根律 / De Morgan's Laws**: $\neg(p \land q) \equiv \neg p \lor \neg q$
- **分配律 / Distributive Laws**: $p \land (q \lor r) \equiv (p \land q) \lor (p \land r)$
- **双重否定 / Double Negation**: $\neg \neg p \equiv p$

## 应用领域 / Applications

### AI中的应用
- **知识表示 / Knowledge Representation**: 使用逻辑表示知识
- **推理系统 / Reasoning Systems**: 基于逻辑的推理
- **形式化验证 / Formal Verification**: 验证系统正确性
- **自然语言处理 / Natural Language Processing**: 语义分析

### 计算机科学中的应用
- **程序验证 / Program Verification**: 证明程序正确性
- **数据库理论 / Database Theory**: 查询语言和约束
- **人工智能 / Artificial Intelligence**: 专家系统和推理

## 参考文献 / References

1. Enderton, H. B. (2001). *A Mathematical Introduction to Logic*. Academic Press.
2. Mendelson, E. (2015). *Introduction to Mathematical Logic*. CRC Press.
3. Hughes, G. E., & Cresswell, M. J. (1996). *A New Introduction to Modal Logic*. Routledge.
4. Boolos, G. S., Burgess, J. P., & Jeffrey, R. C. (2007). *Computability and Logic*. Cambridge University Press.

---

*本模块为FormalAI提供了完整的形式逻辑基础，为AI系统的逻辑推理能力提供了坚实的理论基础。*

## 目录 / Table of Contents

- [1.1 形式逻辑 / Formal Logic / Formale Logik / Logique formelle](#11-形式逻辑--formal-logic--formale-logik--logique-formelle)
  - [概述 / Overview](#概述--overview)
  - [核心概念 / Core Concepts](#核心概念--core-concepts)
    - [命题逻辑 / Propositional Logic](#命题逻辑--propositional-logic)
    - [谓词逻辑 / Predicate Logic](#谓词逻辑--predicate-logic)
    - [模态逻辑 / Modal Logic](#模态逻辑--modal-logic)
  - [数学形式化 / Mathematical Formalization](#数学形式化--mathematical-formalization)
    - [命题逻辑形式化](#命题逻辑形式化)
    - [谓词逻辑形式化](#谓词逻辑形式化)
    - [模态逻辑形式化](#模态逻辑形式化)
  - [推理规则 / Inference Rules](#推理规则--inference-rules)
    - [基本推理规则](#基本推理规则)
    - [等价律](#等价律)
  - [应用领域 / Applications](#应用领域--applications)
    - [AI中的应用](#ai中的应用)
    - [计算机科学中的应用](#计算机科学中的应用)
  - [参考文献 / References](#参考文献--references)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [相关章节 / Related Chapters](#相关章节--related-chapters)
  - [1. 命题逻辑 / Propositional Logic](#1-命题逻辑--propositional-logic)
    - [1.1 命题与连接词 / Propositions and Connectives](#11-命题与连接词--propositions-and-connectives)
    - [1.2 真值表 / Truth Tables](#12-真值表--truth-tables)
    - [1.3 逻辑等价 / Logical Equivalence](#13-逻辑等价--logical-equivalence)
    - [1.4 推理规则 / Inference Rules](#14-推理规则--inference-rules)
  - [2. 谓词逻辑 / Predicate Logic](#2-谓词逻辑--predicate-logic)
    - [2.1 谓词与量词 / Predicates and Quantifiers](#21-谓词与量词--predicates-and-quantifiers)
    - [2.2 一阶逻辑 / First-Order Logic](#22-一阶逻辑--first-order-logic)
    - [2.3 形式化理论 / Formal Theories](#23-形式化理论--formal-theories)
  - [3. 模态逻辑 / Modal Logic](#3-模态逻辑--modal-logic)
    - [3.1 模态算子 / Modal Operators](#31-模态算子--modal-operators)
    - [3.2 可能世界语义 / Possible Worlds Semantics](#32-可能世界语义--possible-worlds-semantics)
    - [3.3 模态系统 / Modal Systems](#33-模态系统--modal-systems)
  - [4. 证明理论 / Proof Theory](#4-证明理论--proof-theory)
    - [4.1 自然演绎 / Natural Deduction](#41-自然演绎--natural-deduction)
    - [4.2 序列演算 / Sequent Calculus](#42-序列演算--sequent-calculus)
    - [4.3 证明复杂性 / Proof Complexity](#43-证明复杂性--proof-complexity)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：逻辑推理引擎](#rust实现逻辑推理引擎)
    - [Haskell实现：类型化逻辑](#haskell实现类型化逻辑)
  - [参考文献 / References](#参考文献--references-1)

---

## 相关章节 / Related Chapters

**前置依赖 / Prerequisites / Voraussetzungen / Prérequis:**
- 无 / None / Keine / Aucune (基础模块 / Foundation module)

**后续应用 / Applications / Anwendungen / Applications:**
- [1.2 数学基础](02-mathematical-foundations/README.md) - 提供逻辑基础 / Provides logical foundation
- [3.1 形式化验证](../03-formal-methods/01-formal-verification/README.md) - 提供逻辑基础 / Provides logical foundation

---

## 1. 命题逻辑 / Propositional Logic

### 1.1 命题与连接词 / Propositions and Connectives

**命题定义 / Proposition Definition:**

命题是具有真值的陈述句：
A proposition is a declarative sentence with a truth value:

$$\text{Proposition} = \text{Declarative\_Sentence} \land \text{Truth\_Value}$$

**基本连接词 / Basic Connectives:**

1. **否定 / Negation:** $\neg p$ (NOT p)
2. **合取 / Conjunction:** $p \land q$ (p AND q)
3. **析取 / Disjunction:** $p \lor q$ (p OR q)
4. **蕴含 / Implication:** $p \rightarrow q$ (p IMPLIES q)
5. **等价 / Equivalence:** $p \leftrightarrow q$ (p IFF q)

**形式化定义 / Formal Definitions:**

$$\begin{align}
\neg p &= \text{NOT}(p) \\
p \land q &= \text{AND}(p, q) \\
p \lor q &= \text{OR}(p, q) \\
p \rightarrow q &= \neg p \lor q \\
p \leftrightarrow q &= (p \rightarrow q) \land (q \rightarrow p)
\end{align}$$

### 1.2 真值表 / Truth Tables

**真值表定义 / Truth Table Definition:**

真值表是显示命题在所有可能真值组合下真值的表格：
A truth table is a table showing the truth values of propositions under all possible truth value combinations:

| p | q | ¬p | p∧q | p∨q | p→q | p↔q |
|---|---|----|-----|-----|-----|-----|
| T | T | F  | T   | T   | T   | T   |
| T | F | F  | F   | T   | F   | F   |
| F | T | T  | F   | T   | T   | F   |
| F | F | T  | F   | F   | T   | T   |

**真值函数 / Truth Functions:**

$$\begin{align}
f_{\neg}(p) &= 1 - p \\
f_{\land}(p, q) &= \min(p, q) \\
f_{\lor}(p, q) &= \max(p, q) \\
f_{\rightarrow}(p, q) &= \max(1-p, q) \\
f_{\leftrightarrow}(p, q) &= 1 - |p - q|
\end{align}$$

### 1.3 逻辑等价 / Logical Equivalence

**逻辑等价定义 / Logical Equivalence Definition:**

两个命题在逻辑上等价，当且仅当它们在所有解释下具有相同的真值：
Two propositions are logically equivalent if and only if they have the same truth value under all interpretations:

$$p \equiv q \Leftrightarrow \forall I: I(p) = I(q)$$

**重要等价律 / Important Equivalences:**

1. **德摩根律 / De Morgan's Laws:**
   $$\neg(p \land q) \equiv \neg p \lor \neg q$$
   $$\neg(p \lor q) \equiv \neg p \land \neg q$$

2. **分配律 / Distributive Laws:**
   $$p \land (q \lor r) \equiv (p \land q) \lor (p \land r)$$
   $$p \lor (q \land r) \equiv (p \lor q) \land (p \lor r)$$

3. **双重否定 / Double Negation:**
   $$\neg \neg p \equiv p$$

4. **蕴含等价 / Implication Equivalence:**
   $$p \rightarrow q \equiv \neg p \lor q$$

### 1.4 推理规则 / Inference Rules

**推理规则定义 / Inference Rule Definition:**

推理规则是从前提推导结论的形式化规则：
An inference rule is a formal rule for deriving conclusions from premises:

$$\frac{\text{Premises}}{\text{Conclusion}}$$

**基本推理规则 / Basic Inference Rules:**

1. **假言推理 / Modus Ponens:**
   $$\frac{p \rightarrow q \quad p}{q}$$

2. **假言三段论 / Hypothetical Syllogism:**
   $$\frac{p \rightarrow q \quad q \rightarrow r}{p \rightarrow r}$$

3. **析取三段论 / Disjunctive Syllogism:**
   $$\frac{p \lor q \quad \neg p}{q}$$

4. **构造性二难 / Constructive Dilemma:**
   $$\frac{p \rightarrow q \quad r \rightarrow s \quad p \lor r}{q \lor s}$$

---

## 2. 谓词逻辑 / Predicate Logic

### 2.1 谓词与量词 / Predicates and Quantifiers

**谓词定义 / Predicate Definition:**

谓词是描述对象性质的函数：
A predicate is a function that describes properties of objects:

$$P(x_1, x_2, \ldots, x_n)$$

其中 $x_i$ 是变量，$P$ 是谓词符号。
where $x_i$ are variables and $P$ is a predicate symbol.

**量词定义 / Quantifier Definition:**

1. **全称量词 / Universal Quantifier:** $\forall x P(x)$ (For all x, P(x))
2. **存在量词 / Existential Quantifier:** $\exists x P(x)$ (There exists x such that P(x))

**量词等价律 / Quantifier Equivalences:**

$$\begin{align}
\neg \forall x P(x) &\equiv \exists x \neg P(x) \\
\neg \exists x P(x) &\equiv \forall x \neg P(x) \\
\forall x (P(x) \land Q(x)) &\equiv \forall x P(x) \land \forall x Q(x) \\
\exists x (P(x) \lor Q(x)) &\equiv \exists x P(x) \lor \exists x Q(x)
\end{align}$$

### 2.2 一阶逻辑 / First-Order Logic

**一阶逻辑语言 / First-Order Logic Language:**

$$\mathcal{L} = (\mathcal{C}, \mathcal{F}, \mathcal{P}, \mathcal{V})$$

其中：
- $\mathcal{C}$ 是常元集合 / set of constants
- $\mathcal{F}$ 是函数符号集合 / set of function symbols
- $\mathcal{P}$ 是谓词符号集合 / set of predicate symbols
- $\mathcal{V}$ 是变量集合 / set of variables

**项的定义 / Term Definition:**

1. 变量是项 / Variables are terms
2. 常元是项 / Constants are terms
3. 如果 $f$ 是 $n$ 元函数符号，$t_1, \ldots, t_n$ 是项，则 $f(t_1, \ldots, t_n)$ 是项

**公式的定义 / Formula Definition:**

1. 如果 $P$ 是 $n$ 元谓词符号，$t_1, \ldots, t_n$ 是项，则 $P(t_1, \ldots, t_n)$ 是原子公式
2. 如果 $\phi$ 和 $\psi$ 是公式，则 $\neg \phi$, $\phi \land \psi$, $\phi \lor \psi$, $\phi \rightarrow \psi$ 是公式
3. 如果 $\phi$ 是公式，$x$ 是变量，则 $\forall x \phi$ 和 $\exists x \phi$ 是公式

### 2.3 形式化理论 / Formal Theories

**理论定义 / Theory Definition:**

形式化理论是一组公理和推理规则：
A formal theory is a set of axioms and inference rules:

$$\mathcal{T} = (\mathcal{L}, \mathcal{A}, \mathcal{R})$$

其中：
- $\mathcal{L}$ 是语言 / language
- $\mathcal{A}$ 是公理集合 / set of axioms
- $\mathcal{R}$ 是推理规则集合 / set of inference rules

**常见理论 / Common Theories:**

1. **集合论 / Set Theory:**
   - 外延公理 / Axiom of Extensionality
   - 空集公理 / Axiom of Empty Set
   - 配对公理 / Axiom of Pairing
   - 并集公理 / Axiom of Union
   - 幂集公理 / Axiom of Power Set

2. **算术理论 / Arithmetic Theory:**
   - 皮亚诺公理 / Peano Axioms
   - 归纳原理 / Principle of Induction

---

## 3. 模态逻辑 / Modal Logic

### 3.1 模态算子 / Modal Operators

**模态算子定义 / Modal Operator Definition:**

模态逻辑扩展了经典逻辑，增加了模态算子：
Modal logic extends classical logic with modal operators:

1. **必然算子 / Necessity Operator:** $\Box \phi$ (necessarily $\phi$)
2. **可能算子 / Possibility Operator:** $\Diamond \phi$ (possibly $\phi$)

**模态等价 / Modal Equivalence:**

$$\Diamond \phi \equiv \neg \Box \neg \phi$$

**基本模态公理 / Basic Modal Axioms:**

1. **K公理 / K Axiom:** $\Box(\phi \rightarrow \psi) \rightarrow (\Box \phi \rightarrow \Box \psi)$
2. **T公理 / T Axiom:** $\Box \phi \rightarrow \phi$
3. **4公理 / 4 Axiom:** $\Box \phi \rightarrow \Box \Box \phi$
4. **5公理 / 5 Axiom:** $\Diamond \phi \rightarrow \Box \Diamond \phi$

### 3.2 可能世界语义 / Possible Worlds Semantics

**克里普克模型 / Kripke Model:**

$$\mathcal{M} = (W, R, V)$$

其中：
- $W$ 是可能世界集合 / set of possible worlds
- $R \subseteq W \times W$ 是可达关系 / accessibility relation
- $V: W \times \mathcal{P} \rightarrow \{0,1\}$ 是赋值函数 / valuation function

**真值定义 / Truth Definition:**

$$\begin{align}
\mathcal{M}, w &\models p \Leftrightarrow V(w, p) = 1 \\
\mathcal{M}, w &\models \neg \phi \Leftrightarrow \mathcal{M}, w \not\models \phi \\
\mathcal{M}, w &\models \phi \land \psi \Leftrightarrow \mathcal{M}, w \models \phi \text{ and } \mathcal{M}, w \models \psi \\
\mathcal{M}, w &\models \Box \phi \Leftrightarrow \forall v: wRv \Rightarrow \mathcal{M}, v \models \phi \\
\mathcal{M}, w &\models \Diamond \phi \Leftrightarrow \exists v: wRv \text{ and } \mathcal{M}, v \models \phi
\end{align}$$

### 3.3 模态系统 / Modal Systems

**常见模态系统 / Common Modal Systems:**

1. **K系统 / System K:** 基本模态逻辑
2. **T系统 / System T:** K + T公理
3. **S4系统 / System S4:** T + 4公理
4. **S5系统 / System S5:** T + 4 + 5公理

**对应关系 / Correspondence:**

- T公理 ↔ 自反性 / reflexivity
- 4公理 ↔ 传递性 / transitivity
- 5公理 ↔ 欧几里得性 / euclidean property

---

## 4. 证明理论 / Proof Theory

### 4.1 自然演绎 / Natural Deduction

**自然演绎系统 / Natural Deduction System:**

自然演绎是一种证明系统，使用引入和消除规则：
Natural deduction is a proof system using introduction and elimination rules.

**命题逻辑规则 / Propositional Logic Rules:**

**合取规则 / Conjunction Rules:**
$$\frac{\phi \quad \psi}{\phi \land \psi} \land I \quad \frac{\phi \land \psi}{\phi} \land E_1 \quad \frac{\phi \land \psi}{\psi} \land E_2$$

**析取规则 / Disjunction Rules:**
$$\frac{\phi}{\phi \lor \psi} \lor I_1 \quad \frac{\psi}{\phi \lor \psi} \lor I_2$$

**蕴含规则 / Implication Rules:**
$$\frac{[\phi] \quad \vdots \quad \psi}{\phi \rightarrow \psi} \rightarrow I \quad \frac{\phi \rightarrow \psi \quad \phi}{\psi} \rightarrow E$$

### 4.2 序列演算 / Sequent Calculus

**序列定义 / Sequent Definition:**

序列是形如 $\Gamma \vdash \Delta$ 的表达式：
A sequent is an expression of the form $\Gamma \vdash \Delta$:

$$\Gamma \vdash \Delta$$

其中 $\Gamma$ 和 $\Delta$ 是公式集合。
where $\Gamma$ and $\Delta$ are sets of formulas.

**基本规则 / Basic Rules:**

**左规则 / Left Rules:**
$$\frac{\Gamma, \phi \vdash \Delta}{\Gamma, \phi \land \psi \vdash \Delta} \land L_1 \quad \frac{\Gamma, \psi \vdash \Delta}{\Gamma, \phi \land \psi \vdash \Delta} \land L_2$$

**右规则 / Right Rules:**
$$\frac{\Gamma \vdash \phi, \Delta \quad \Gamma \vdash \psi, \Delta}{\Gamma \vdash \phi \land \psi, \Delta} \land R$$

### 4.3 证明复杂性 / Proof Complexity

**证明长度 / Proof Length:**

证明的长度是证明中步骤的数量：
The length of a proof is the number of steps in the proof.

**证明复杂性类 / Proof Complexity Classes:**

1. **多项式时间证明 / Polynomial-time proofs**
2. **指数时间证明 / Exponential-time proofs**
3. **不可证明性 / Unprovability**

---

## 代码示例 / Code Examples

### Rust实现：逻辑推理引擎

```rust
use std::collections::HashMap;

/// 命题逻辑推理引擎
pub struct PropositionalLogicEngine {
    variables: HashMap<String, bool>,
    rules: Vec<InferenceRule>,
}

/// 推理规则
pub struct InferenceRule {
    name: String,
    premises: Vec<Formula>,
    conclusion: Formula,
}

/// 逻辑公式
# [derive(Debug, Clone)]
pub enum Formula {
    Atom(String),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
    Or(Box<Formula>, Box<Formula>),
    Implies(Box<Formula>, Box<Formula>),
    Equiv(Box<Formula>, Box<Formula>),
}

impl PropositionalLogicEngine {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            rules: vec![
                InferenceRule::modus_ponens(),
                InferenceRule::modus_tollens(),
                InferenceRule::hypothetical_syllogism(),
                InferenceRule::disjunctive_syllogism(),
            ],
        }
    }

    /// 评估公式真值
    pub fn evaluate(&self, formula: &Formula) -> Option<bool> {
        match formula {
            Formula::Atom(name) => self.variables.get(name).copied(),
            Formula::Not(f) => self.evaluate(f).map(|v| !v),
            Formula::And(f1, f2) => {
                let v1 = self.evaluate(f1)?;
                let v2 = self.evaluate(f2)?;
                Some(v1 && v2)
            }
            Formula::Or(f1, f2) => {
                let v1 = self.evaluate(f1)?;
                let v2 = self.evaluate(f2)?;
                Some(v1 || v2)
            }
            Formula::Implies(f1, f2) => {
                let v1 = self.evaluate(f1)?;
                let v2 = self.evaluate(f2)?;
                Some(!v1 || v2)
            }
            Formula::Equiv(f1, f2) => {
                let v1 = self.evaluate(f1)?;
                let v2 = self.evaluate(f2)?;
                Some(v1 == v2)
            }
        }
    }

    /// 生成真值表
    pub fn generate_truth_table(&self, formula: &Formula) -> Vec<TruthTableRow> {
        let variables = self.extract_variables(formula);
        let mut table = Vec::new();

        for i in 0..(1 << variables.len()) {
            let mut row = TruthTableRow::new();

            for (j, var) in variables.iter().enumerate() {
                let value = (i >> j) & 1 == 1;
                row.assignments.insert(var.clone(), value);
            }

            // 临时设置变量值
            let original_vars = self.variables.clone();
            self.variables.extend(row.assignments.clone());

            row.result = self.evaluate(formula);

            // 恢复原始变量
            self.variables = original_vars;

            table.push(row);
        }

        table
    }

    /// 检查逻辑等价
    pub fn are_equivalent(&self, f1: &Formula, f2: &Formula) -> bool {
        let variables = self.extract_variables(f1);
        let variables2 = self.extract_variables(f2);
        let all_variables: Vec<_> = variables.union(&variables2).cloned().collect();

        for i in 0..(1 << all_variables.len()) {
            let mut assignments = HashMap::new();

            for (j, var) in all_variables.iter().enumerate() {
                let value = (i >> j) & 1 == 1;
                assignments.insert(var.clone(), value);
            }

            // 临时设置变量值
            let original_vars = self.variables.clone();
            self.variables.extend(assignments);

            let result1 = self.evaluate(f1);
            let result2 = self.evaluate(f2);

            // 恢复原始变量
            self.variables = original_vars;

            if result1 != result2 {
                return false;
            }
        }

        true
    }

    /// 应用推理规则
    pub fn apply_inference(&self, premises: &[Formula], rule: &InferenceRule) -> Option<Formula> {
        // 检查前提是否匹配
        if premises.len() != rule.premises.len() {
            return None;
        }

        // 检查前提是否逻辑等价
        for (premise, rule_premise) in premises.iter().zip(&rule.premises) {
            if !self.are_equivalent(premise, rule_premise) {
                return None;
            }
        }

        Some(rule.conclusion.clone())
    }

    /// 提取公式中的变量
    fn extract_variables(&self, formula: &Formula) -> std::collections::HashSet<String> {
        let mut variables = std::collections::HashSet::new();
        self.extract_variables_recursive(formula, &mut variables);
        variables
    }

    fn extract_variables_recursive(&self, formula: &Formula, variables: &mut std::collections::HashSet<String>) {
        match formula {
            Formula::Atom(name) => {
                variables.insert(name.clone());
            }
            Formula::Not(f) => {
                self.extract_variables_recursive(f, variables);
            }
            Formula::And(f1, f2) | Formula::Or(f1, f2) |
            Formula::Implies(f1, f2) | Formula::Equiv(f1, f2) => {
                self.extract_variables_recursive(f1, variables);
                self.extract_variables_recursive(f2, variables);
            }
        }
    }
}

impl InferenceRule {
    pub fn modus_ponens() -> Self {
        Self {
            name: "Modus Ponens".to_string(),
            premises: vec![
                Formula::Implies(Box::new(Formula::Atom("p".to_string())),
                               Box::new(Formula::Atom("q".to_string()))),
                Formula::Atom("p".to_string()),
            ],
            conclusion: Formula::Atom("q".to_string()),
        }
    }

    pub fn modus_tollens() -> Self {
        Self {
            name: "Modus Tollens".to_string(),
            premises: vec![
                Formula::Implies(Box::new(Formula::Atom("p".to_string())),
                               Box::new(Formula::Atom("q".to_string()))),
                Formula::Not(Box::new(Formula::Atom("q".to_string()))),
            ],
            conclusion: Formula::Not(Box::new(Formula::Atom("p".to_string()))),
        }
    }

    pub fn hypothetical_syllogism() -> Self {
        Self {
            name: "Hypothetical Syllogism".to_string(),
            premises: vec![
                Formula::Implies(Box::new(Formula::Atom("p".to_string())),
                               Box::new(Formula::Atom("q".to_string()))),
                Formula::Implies(Box::new(Formula::Atom("q".to_string())),
                               Box::new(Formula::Atom("r".to_string()))),
            ],
            conclusion: Formula::Implies(Box::new(Formula::Atom("p".to_string())),
                                       Box::new(Formula::Atom("r".to_string()))),
        }
    }

    pub fn disjunctive_syllogism() -> Self {
        Self {
            name: "Disjunctive Syllogism".to_string(),
            premises: vec![
                Formula::Or(Box::new(Formula::Atom("p".to_string())),
                           Box::new(Formula::Atom("q".to_string()))),
                Formula::Not(Box::new(Formula::Atom("p".to_string()))),
            ],
            conclusion: Formula::Atom("q".to_string()),
        }
    }
}

/// 真值表行
pub struct TruthTableRow {
    pub assignments: HashMap<String, bool>,
    pub result: Option<bool>,
}

impl TruthTableRow {
    pub fn new() -> Self {
        Self {
            assignments: HashMap::new(),
            result: None,
        }
    }
}

/// 谓词逻辑引擎
pub struct PredicateLogicEngine {
    domain: Vec<String>,
    interpretations: HashMap<String, Box<dyn Fn(&[String]) -> bool>>,
}

impl PredicateLogicEngine {
    pub fn new() -> Self {
        Self {
            domain: Vec::new(),
            interpretations: HashMap::new(),
        }
    }

    /// 添加谓词解释
    pub fn add_predicate_interpretation<P>(&mut self, name: String, interpretation: P)
    where
        P: Fn(&[String]) -> bool + 'static,
    {
        self.interpretations.insert(name, Box::new(interpretation));
    }

    /// 评估谓词公式
    pub fn evaluate_predicate(&self, predicate: &str, arguments: &[String]) -> Option<bool> {
        if let Some(interpretation) = self.interpretations.get(predicate) {
            Some(interpretation(arguments))
        } else {
            None
        }
    }
}

fn main() {
    println!("=== 形式逻辑推理引擎演示 ===");

    // 创建命题逻辑引擎
    let mut engine = PropositionalLogicEngine::new();

    // 设置变量值
    engine.variables.insert("p".to_string(), true);
    engine.variables.insert("q".to_string(), false);

    // 创建公式
    let formula = Formula::Implies(
        Box::new(Formula::Atom("p".to_string())),
        Box::new(Formula::Atom("q".to_string()))
    );

    // 评估公式
    let result = engine.evaluate(&formula);
    println!("p → q 的真值: {:?}", result);

    // 生成真值表
    let truth_table = engine.generate_truth_table(&formula);
    println!("真值表行数: {}", truth_table.len());

    // 测试逻辑等价
    let formula1 = Formula::Implies(
        Box::new(Formula::Atom("p".to_string())),
        Box::new(Formula::Atom("q".to_string()))
    );
    let formula2 = Formula::Or(
        Box::new(Formula::Not(Box::new(Formula::Atom("p".to_string())))),
        Box::new(Formula::Atom("q".to_string()))
    );

    let equivalent = engine.are_equivalent(&formula1, &formula2);
    println!("p → q 与 ¬p ∨ q 等价: {}", equivalent);

    // 应用推理规则
    let premises = vec![
        Formula::Implies(Box::new(Formula::Atom("p".to_string())),
                        Box::new(Formula::Atom("q".to_string()))),
        Formula::Atom("p".to_string()),
    ];

    let rule = InferenceRule::modus_ponens();
    let conclusion = engine.apply_inference(&premises, &rule);
    println!("推理结论: {:?}", conclusion);

    println!("形式逻辑演示完成！");
}
```

### Haskell实现：类型化逻辑

```haskell
-- 形式逻辑模块
module FormalLogic where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Maybe (fromMaybe)

-- 命题逻辑公式
data Formula = Atom String
             | Not Formula
             | And Formula Formula
             | Or Formula Formula
             | Implies Formula Formula
             | Equiv Formula Formula
             deriving (Show, Eq)

-- 谓词逻辑公式
data PredicateFormula = Predicate String [Term]
                      | ForAll String PredicateFormula
                      | Exists String PredicateFormula
                      | PredNot PredicateFormula
                      | PredAnd PredicateFormula PredicateFormula
                      | PredOr PredicateFormula PredicateFormula
                      | PredImplies PredicateFormula PredicateFormula
                      deriving (Show, Eq)

-- 项
data Term = Variable String
          | Constant String
          | Function String [Term]
          deriving (Show, Eq)

-- 解释
data Interpretation = Interpretation
    { domain :: [String]
    , predicateInterpretations :: Map String ([String] -> Bool)
    , functionInterpretations :: Map String ([String] -> String)
    }

-- 真值表行
data TruthTableRow = TruthTableRow
    { assignments :: Map String Bool
    , result :: Maybe Bool
    } deriving (Show)

-- 推理规则
data InferenceRule = InferenceRule
    { ruleName :: String
    , premises :: [Formula]
    , conclusion :: Formula
    } deriving (Show)

-- 命题逻辑引擎
data PropositionalLogicEngine = PropositionalLogicEngine
    { variables :: Map String Bool
    , rules :: [InferenceRule]
    }

-- 创建新的命题逻辑引擎
newPropositionalLogicEngine :: PropositionalLogicEngine
newPropositionalLogicEngine = PropositionalLogicEngine
    { variables = Map.empty
    , rules = [modusPonens, modusTollens, hypotheticalSyllogism, disjunctiveSyllogism]
    }

-- 评估公式
evaluate :: PropositionalLogicEngine -> Formula -> Maybe Bool
evaluate engine formula = case formula of
    Atom name -> Map.lookup name (variables engine)
    Not f -> not <$> evaluate engine f
    And f1 f2 -> do
        v1 <- evaluate engine f1
        v2 <- evaluate engine f2
        return (v1 && v2)
    Or f1 f2 -> do
        v1 <- evaluate engine f1
        v2 <- evaluate engine f2
        return (v1 || v2)
    Implies f1 f2 -> do
        v1 <- evaluate engine f1
        v2 <- evaluate engine f2
        return (not v1 || v2)
    Equiv f1 f2 -> do
        v1 <- evaluate engine f1
        v2 <- evaluate engine f2
        return (v1 == v2)

-- 生成真值表
generateTruthTable :: PropositionalLogicEngine -> Formula -> [TruthTableRow]
generateTruthTable engine formula =
    let vars = extractVariables formula
        assignments = generateAllAssignments vars
    in map (\assign -> TruthTableRow assign (evaluateWithAssignment engine formula assign)) assignments

-- 提取变量
extractVariables :: Formula -> Set String
extractVariables formula = case formula of
    Atom name -> Set.singleton name
    Not f -> extractVariables f
    And f1 f2 -> Set.union (extractVariables f1) (extractVariables f2)
    Or f1 f2 -> Set.union (extractVariables f1) (extractVariables f2)
    Implies f1 f2 -> Set.union (extractVariables f1) (extractVariables f2)
    Equiv f1 f2 -> Set.union (extractVariables f1) (extractVariables f2)

-- 生成所有赋值
generateAllAssignments :: Set String -> [Map String Bool]
generateAllAssignments vars =
    let varList = Set.toList vars
        n = length varList
    in map (\i -> Map.fromList (zip varList (bitsToBools i n))) [0..(2^n - 1)]

-- 将整数转换为布尔列表
bitsToBools :: Int -> Int -> [Bool]
bitsToBools num bits = [((num `div` (2^i)) `mod` 2) == 1 | i <- [0..bits-1]]

-- 使用赋值评估公式
evaluateWithAssignment :: PropositionalLogicEngine -> Formula -> Map String Bool -> Maybe Bool
evaluateWithAssignment engine formula assignment =
    let engineWithAssignment = engine { variables = assignment }
    in evaluate engineWithAssignment formula

-- 检查逻辑等价
areEquivalent :: PropositionalLogicEngine -> Formula -> Formula -> Bool
areEquivalent engine f1 f2 =
    let vars1 = extractVariables f1
        vars2 = extractVariables f2
        allVars = Set.union vars1 vars2
        assignments = generateAllAssignments allVars
    in all (\assign ->
        evaluateWithAssignment engine f1 assign == evaluateWithAssignment engine f2 assign) assignments

-- 应用推理规则
applyInference :: PropositionalLogicEngine -> [Formula] -> InferenceRule -> Maybe Formula
applyInference engine premises rule =
    if length premises == length (premises rule) &&
       all (\(p, rp) -> areEquivalent engine p rp) (zip premises (premises rule))
    then Just (conclusion rule)
    else Nothing

-- 推理规则定义
modusPonens :: InferenceRule
modusPonens = InferenceRule
    { ruleName = "Modus Ponens"
    , premises = [Implies (Atom "p") (Atom "q"), Atom "p"]
    , conclusion = Atom "q"
    }

modusTollens :: InferenceRule
modusTollens = InferenceRule
    { ruleName = "Modus Tollens"
    , premises = [Implies (Atom "p") (Atom "q"), Not (Atom "q")]
    , conclusion = Not (Atom "p")
    }

hypotheticalSyllogism :: InferenceRule
hypotheticalSyllogism = InferenceRule
    { ruleName = "Hypothetical Syllogism"
    , premises = [Implies (Atom "p") (Atom "q"), Implies (Atom "q") (Atom "r")]
    , conclusion = Implies (Atom "p") (Atom "r")
    }

disjunctiveSyllogism :: InferenceRule
disjunctiveSyllogism = InferenceRule
    { ruleName = "Disjunctive Syllogism"
    , premises = [Or (Atom "p") (Atom "q"), Not (Atom "p")]
    , conclusion = Atom "q"
    }

-- 谓词逻辑引擎
data PredicateLogicEngine = PredicateLogicEngine
    { domain :: [String]
    , predicateInterpretations :: Map String ([String] -> Bool)
    , functionInterpretations :: Map String ([String] -> String)
    }

-- 创建新的谓词逻辑引擎
newPredicateLogicEngine :: [String] -> PredicateLogicEngine
newPredicateLogicEngine domain = PredicateLogicEngine
    { domain = domain
    , predicateInterpretations = Map.empty
    , functionInterpretations = Map.empty
    }

-- 添加谓词解释
addPredicateInterpretation :: PredicateLogicEngine -> String -> ([String] -> Bool) -> PredicateLogicEngine
addPredicateInterpretation engine name interpretation =
    engine { predicateInterpretations = Map.insert name interpretation (predicateInterpretations engine) }

-- 评估谓词
evaluatePredicate :: PredicateLogicEngine -> String -> [String] -> Maybe Bool
evaluatePredicate engine predicateName arguments =
    Map.lookup predicateName (predicateInterpretations engine) >>= \interpretation ->
    Just (interpretation arguments)

-- 模态逻辑
data ModalFormula = ModalAtom String
                  | ModalNot ModalFormula
                  | ModalAnd ModalFormula ModalFormula
                  | ModalOr ModalFormula ModalFormula
                  | ModalImplies ModalFormula ModalFormula
                  | Necessity ModalFormula
                  | Possibility ModalFormula
                  deriving (Show, Eq)

-- 克里普克模型
data KripkeModel = KripkeModel
    { worlds :: [String]
    , accessibility :: Map String [String]
    , valuation :: Map (String, String) Bool  -- (world, atom) -> bool
    }

-- 评估模态公式
evaluateModal :: KripkeModel -> String -> ModalFormula -> Bool
evaluateModal model world formula = case formula of
    ModalAtom atom -> Map.findWithDefault False (world, atom) (valuation model)
    ModalNot f -> not (evaluateModal model world f)
    ModalAnd f1 f2 -> evaluateModal model world f1 && evaluateModal model world f2
    ModalOr f1 f2 -> evaluateModal model world f1 || evaluateModal model world f2
    ModalImplies f1 f2 -> not (evaluateModal model world f1) || evaluateModal model world f2
    Necessity f -> all (\w -> evaluateModal model w f) (accessibleWorlds model world)
    Possibility f -> any (\w -> evaluateModal model w f) (accessibleWorlds model world)

-- 获取可达世界
accessibleWorlds :: KripkeModel -> String -> [String]
accessibleWorlds model world = Map.findWithDefault [] world (accessibility model)

-- 示例使用
main :: IO ()
main = do
    putStrLn "=== 形式逻辑演示 ==="

    -- 创建命题逻辑引擎
    let engine = newPropositionalLogicEngine

    -- 设置变量
    let engineWithVars = engine { variables = Map.fromList [("p", True), ("q", False)] }

    -- 创建公式
    let formula = Implies (Atom "p") (Atom "q")

    -- 评估公式
    let result = evaluate engineWithVars formula
    putStrLn $ "p → q 的真值: " ++ show result

    -- 生成真值表
    let truthTable = generateTruthTable engine formula
    putStrLn $ "真值表行数: " ++ show (length truthTable)

    -- 测试逻辑等价
    let formula1 = Implies (Atom "p") (Atom "q")
    let formula2 = Or (Not (Atom "p")) (Atom "q")
    let equivalent = areEquivalent engine formula1 formula2
    putStrLn $ "p → q 与 ¬p ∨ q 等价: " ++ show equivalent

    -- 应用推理规则
    let premises = [Implies (Atom "p") (Atom "q"), Atom "p"]
    let conclusion = applyInference engine premises modusPonens
    putStrLn $ "推理结论: " ++ show conclusion

    -- 创建谓词逻辑引擎
    let predEngine = newPredicateLogicEngine ["a", "b", "c"]
    let predEngineWithInterp = addPredicateInterpretation predEngine "P" (\args -> head args == "a")

    -- 评估谓词
    let predResult = evaluatePredicate predEngineWithInterp "P" ["a"]
    putStrLn $ "谓词 P(a) 的真值: " ++ show predResult

    putStrLn "形式逻辑演示完成！"
```

---

## 参考文献 / References

1. **中文 / Chinese:**
   - 王宪钧 (1982). *数理逻辑引论*. 北京大学出版社.
   - 张清宇 (2003). *逻辑哲学九章*. 江苏人民出版社.
   - 李小五 (2005). *模态逻辑*. 中国社会科学出版社.

2. **English:**
   - Enderton, H. B. (2001). *A Mathematical Introduction to Logic*. Academic Press.
   - Mendelson, E. (2015). *Introduction to Mathematical Logic*. CRC Press.
   - Hughes, G. E., & Cresswell, M. J. (1996). *A New Introduction to Modal Logic*. Routledge.
   - Boolos, G. S., Burgess, J. P., & Jeffrey, R. C. (2007). *Computability and Logic*. Cambridge University Press.

3. **Deutsch / German:**
   - Ebbinghaus, H. D., Flum, J., & Thomas, W. (2018). *Einführung in die mathematische Logik*. Springer.
   - Rautenberg, W. (2008). *Einführung in die mathematische Logik*. Vieweg+Teubner.

4. **Français / French:**
   - Cori, R., & Lascar, D. (2003). *Logique mathématique*. Dunod.
   - David, R., Nour, K., & Raffalli, C. (2004). *Introduction à la logique*. Dunod.

---

*本模块为FormalAI提供了完整的形式逻辑基础，涵盖命题逻辑、谓词逻辑、模态逻辑等核心内容，为AI系统的逻辑推理能力提供了坚实的理论基础。*

*This module provides complete formal logic foundations for FormalAI, covering propositional logic, predicate logic, modal logic, and other core content, providing solid theoretical foundations for logical reasoning capabilities in AI systems.*
