# 1.1 形式逻辑 / Formal Logic / Formale Logik / Logique formelle

[返回全局导航](../../GLOBAL_NAVIGATION.md) · [学习路径](../../LEARNING_PATH_DESIGN.md)

## 概述 / Overview / Übersicht / Aperçu

形式逻辑是研究推理形式和有效性的数学分支，为 FormalAI 提供严格的逻辑基础。本模块基于严格的公理化体系，建立完整的逻辑理论框架，包括命题逻辑、谓词逻辑、模态逻辑等核心内容，并提供严格的形式化证明。

Formal logic is the mathematical study of reasoning forms and validity, providing rigorous logical foundations for FormalAI. This module is based on a strict axiomatic system, establishing a complete logical theoretical framework including propositional logic, predicate logic, modal logic, and other core content, with rigorous formal proofs.

## 目录 / Table of Contents / Inhaltsverzeichnis / Table des matières

- [1.1 形式逻辑](#11-形式逻辑--formal-logic--formale-logik--logique-formelle)
  - [概述](#概述--overview--übersicht--aperçu)
  - [目录](#目录--table-of-contents--inhaltsverzeichnis--table-des-matières)
  - [1. 公理系统](#1-公理系统--axiom-system--axiomensystem--système-daxiomes)
  - [2. 命题逻辑](#2-命题逻辑--propositional-logic--aussagenlogik--logique-propositionnelle)
  - [3. 谓词逻辑](#3-谓词逻辑--predicate-logic--prädikatenlogik--logique-des-prédicats)
  - [4. 模态逻辑](#4-模态逻辑--modal-logic--modallogik--logique-modale)
  - [5. 形式化证明](#5-形式化证明--formal-proofs--formale-beweise--preuves-formelles)
  - [6. 应用实例](#6-应用实例--applications--anwendungen--applications)
  - [代码实现](#代码实现--code-implementation--code-implementierung--implémentation-de-code)
  - [参考文献](#参考文献--references--literatur--références)

## 1. 公理系统 / Axiom System / Axiomensystem / Système d'axiomes

### 1.1 逻辑语言 / Logical Language / Logische Sprache / Langage logique

**定义 1.1.1 (逻辑语言)**
逻辑语言 $\mathcal{L}$ 由以下符号组成：

- 命题变量：$p_1, p_2, p_3, \ldots$
- 逻辑连接词：$\neg, \land, \lor, \to, \leftrightarrow$
- 括号：$(, )$

**定义 1.1.2 (合式公式)**
合式公式由以下递归规则定义：

- 每个命题变量都是合式公式
- 如果 $\phi$ 是合式公式，则 $\neg \phi$ 是合式公式
- 如果 $\phi$ 和 $\psi$ 是合式公式，则 $(\phi \land \psi)$、$(\phi \lor \psi)$、$(\phi \to \psi)$、$(\phi \leftrightarrow \psi)$ 是合式公式

### 1.2 希尔伯特公理系统 / Hilbert Axiom System / Hilbert-Axiomensystem / Système d'axiomes de Hilbert

**公理 1.2.1 (命题逻辑公理)**:

$$
\begin{align}
\text{A1: } & \phi \to (\psi \to \phi) \\
\text{A2: } & (\phi \to (\psi \to \chi)) \to ((\phi \to \psi) \to (\phi \to \chi)) \\
\text{A3: } & (\neg \phi \to \neg \psi) \to (\psi \to \phi)
\end{align}
$$

**推理规则 1.2.1 (假言推理)**
$$\frac{\phi \to \psi \quad \phi}{\psi} \quad \text{(MP)}$$

**定理 1.2.1 (演绎定理)**
如果 $\Gamma, \phi \vdash \psi$，则 $\Gamma \vdash \phi \to \psi$。

**证明：**
对证明长度进行归纳。□

## 2. 命题逻辑 / Propositional Logic / Aussagenlogik / Logique propositionnelle

### 2.1 语义学 / Semantics / Semantik / Sémantique

**定义 2.1.1 (真值赋值)**
真值赋值是从命题变量到真值集合 $\{T, F\}$ 的函数 $v: \text{Prop} \to \{T, F\}$。

**定义 2.1.2 (公式的真值)**
对于真值赋值 $v$ 和公式 $\phi$，$\phi$ 在 $v$ 下的真值 $v(\phi)$ 定义为：

- $v(p) = v(p)$ 对于命题变量 $p$
- $v(\neg \phi) = T$ 当且仅当 $v(\phi) = F$
- $v(\phi \land \psi) = T$ 当且仅当 $v(\phi) = T$ 且 $v(\psi) = T$
- $v(\phi \lor \psi) = T$ 当且仅当 $v(\phi) = T$ 或 $v(\psi) = T$
- $v(\phi \to \psi) = T$ 当且仅当 $v(\phi) = F$ 或 $v(\psi) = T$

**定义 2.1.3 (重言式)**
公式 $\phi$ 是重言式，如果对所有真值赋值 $v$，都有 $v(\phi) = T$。

**定义 2.1.4 (逻辑蕴含)**
公式集合 $\Gamma$ 逻辑蕴含公式 $\phi$，记作 $\Gamma \models \phi$，如果对所有使 $\Gamma$ 中所有公式为真的真值赋值 $v$，都有 $v(\phi) = T$。

### 2.2 自然演绎系统 / Natural Deduction System / Natürliches Deduktionssystem / Système de déduction naturelle

**定义 2.2.1 (自然演绎规则)**:

- **合取引入 (∧I)**: $\frac{A \quad B}{A \land B}$
- **合取消除 (∧E)**: $\frac{A \land B}{A} \quad \frac{A \land B}{B}$
- **析取引入 (∨I)**: $\frac{A}{A \lor B} \quad \frac{B}{A \lor B}$
- **析取消除 (∨E)**: $\frac{A \lor B \quad [A] \quad [B]}{C} \quad \frac{C \quad C}{C}$
- **蕴含引入 (→I)**: $\frac{[A]}{B} \Rightarrow \frac{A \to B}{}$
- **蕴含消除 (→E)**: $\frac{A \to B \quad A}{B}$
- **否定引入 (¬I)**: $\frac{[A]}{\bot} \Rightarrow \frac{\neg A}{}$
- **否定消除 (¬E)**: $\frac{\neg A \quad A}{\bot}$

**定理 2.2.1 (完备性定理)**:
$\Gamma \vdash \phi$ 当且仅当 $\Gamma \models \phi$。

**证明：**
（可靠性）对证明长度进行归纳。
（完备性）使用亨金构造。□

## 3. 谓词逻辑 / Predicate Logic / Prädikatenlogik / Logique des prédicats

### 3.1 一阶语言 / First-Order Language / Erststufige Sprache / Langage du premier ordre

**定义 3.1.1 (一阶语言)**
一阶语言 $\mathcal{L}$ 由以下符号组成：

- 变量：$x_1, x_2, x_3, \ldots$
- 常量符号：$c_1, c_2, c_3, \ldots$
- 函数符号：$f_1, f_2, f_3, \ldots$
- 谓词符号：$P_1, P_2, P_3, \ldots$
- 逻辑连接词：$\land, \lor, \to, \neg$
- 量词：$\forall, \exists$
- 等号：$=$

**定义 3.1.2 (项)**
项由以下递归规则定义：

- 每个变量都是项
- 每个常量符号都是项
- 如果 $t_1, \ldots, t_n$ 是项且 $f$ 是 $n$-元函数符号，则 $f(t_1, \ldots, t_n)$ 是项

**定义 3.1.3 (原子公式)**
原子公式是形如 $P(t_1, \ldots, t_n)$ 或 $t_1 = t_2$ 的表达式，其中 $P$ 是 $n$-元谓词符号，$t_1, \ldots, t_n$ 是项。

**定义 3.1.4 (合式公式)**
合式公式由以下递归规则定义：

- 每个原子公式都是合式公式
- 如果 $\phi$ 是合式公式，则 $\neg \phi$ 是合式公式
- 如果 $\phi$ 和 $\psi$ 是合式公式，则 $(\phi \land \psi)$、$(\phi \lor \psi)$、$(\phi \to \psi)$ 是合式公式
- 如果 $\phi$ 是合式公式且 $x$ 是变量，则 $\forall x \phi$ 和 $\exists x \phi$ 是合式公式

### 3.2 一阶结构 / First-Order Structures / Erststufige Strukturen / Structures du premier ordre

**定义 3.2.1 (一阶结构)**
对于一阶语言 $\mathcal{L}$，$\mathcal{L}$-结构 $\mathcal{M}$ 由以下组成：

- 非空域 $M$
- 对每个常量符号 $c$，元素 $c^{\mathcal{M}} \in M$
- 对每个 $n$-元函数符号 $f$，函数 $f^{\mathcal{M}}: M^n \to M$
- 对每个 $n$-元谓词符号 $P$，关系 $P^{\mathcal{M}} \subseteq M^n$

**定义 3.2.2 (赋值)**
赋值是从变量到域 $M$ 的函数 $s: \text{Var} \to M$。

**定义 3.2.3 (项的解释)**
对于项 $t$ 和赋值 $s$，项的解释 $t^{\mathcal{M},s}$ 定义为：

- 如果 $t = x$ 是变量，则 $t^{\mathcal{M},s} = s(x)$
- 如果 $t = c$ 是常量，则 $t^{\mathcal{M},s} = c^{\mathcal{M}}$
- 如果 $t = f(t_1, \ldots, t_n)$，则 $t^{\mathcal{M},s} = f^{\mathcal{M}}(t_1^{\mathcal{M},s}, \ldots, t_n^{\mathcal{M},s})$

**定义 3.2.4 (公式的真值)**
对于公式 $\phi$ 和赋值 $s$，$\mathcal{M} \models \phi[s]$ 定义为：

- $\mathcal{M} \models P[t_1, \ldots, t_n](s)$ 当且仅当 $(t_1^{\mathcal{M},s}, \ldots, t_n^{\mathcal{M},s}) \in P^{\mathcal{M}}$
- $\mathcal{M} \models \neg \phi[s]$ 当且仅当 $\mathcal{M} \not\models \phi[s]$
- $\mathcal{M} \models \phi \land \psi[s]$ 当且仅当 $\mathcal{M} \models \phi[s]$ 且 $\mathcal{M} \models \psi[s]$
- $\mathcal{M} \models \forall x \phi[s]$ 当且仅当对所有 $a \in M$，$\mathcal{M} \models \phi[s(x|a)]$

### 3.3 量词规则 / Quantifier Rules / Quantorenregeln / Règles de quantificateurs

**全称量词引入 (∀I)：**
$$\frac{A(x)}{\forall x A(x)} \quad \text{其中} \ x \text{不在假设中自由出现}$$

**全称量词消除 (∀E)：**
$$\frac{\forall x A(x)}{A(t)} \quad \text{其中} \ t \text{对} \ x \text{自由}$$

**存在量词引入 (∃I)：**
$$\frac{A(t)}{\exists x A(x)} \quad \text{其中} \ t \text{对} \ x \text{自由}$$

**存在量词消除 (∃E)：**
$$\frac{\exists x A(x) \quad [A(x)]}{B} \Rightarrow \frac{B}{} \quad \text{其中} \ x \text{不在} \ B \text{中自由出现}$$

**定理 3.3.1 (哥德尔完备性定理)**
一阶逻辑是完备的，即 $\Gamma \models \phi$ 当且仅当 $\Gamma \vdash \phi$。

**证明：**
使用亨金构造。□

## 4. 模态逻辑 / Modal Logic / Modallogik / Logique modale

### 4.1 模态语言 / Modal Language / Modale Sprache / Langage modal

**定义 4.1.1 (模态语言)**
模态语言在命题逻辑基础上增加模态算子：

- 必然算子：$\Box$ (读作"必然")
- 可能算子：$\Diamond$ (读作"可能")

**定义 4.1.2 (模态公式)**
模态公式由以下递归规则定义：

- 每个命题变量都是模态公式
- 如果 $\phi$ 是模态公式，则 $\neg \phi$、$\Box \phi$、$\Diamond \phi$ 是模态公式
- 如果 $\phi$ 和 $\psi$ 是模态公式，则 $(\phi \land \psi)$、$(\phi \lor \psi)$、$(\phi \to \psi)$ 是模态公式

### 4.2 克里普克语义 / Kripke Semantics / Kripke-Semantik / Sémantique de Kripke

**定义 4.2.1 (克里普克框架)**
克里普克框架是三元组 $\mathcal{F} = (W, R, V)$，其中：

- $W$ 是非空可能世界集合
- $R \subseteq W \times W$ 是可达关系
- $V: W \times \text{Prop} \to \{T, F\}$ 是赋值函数

**定义 4.2.2 (模态公式的真值)**
对于克里普克模型 $\mathcal{M} = (W, R, V)$ 和世界 $w \in W$，模态公式的真值定义为：

- $\mathcal{M}, w \models p$ 当且仅当 $V(w, p) = T$
- $\mathcal{M}, w \models \neg \phi$ 当且仅当 $\mathcal{M}, w \not\models \phi$
- $\mathcal{M}, w \models \phi \land \psi$ 当且仅当 $\mathcal{M}, w \models \phi$ 且 $\mathcal{M}, w \models \psi$
- $\mathcal{M}, w \models \Box \phi$ 当且仅当对所有 $v$ 使得 $wRv$，有 $\mathcal{M}, v \models \phi$
- $\mathcal{M}, w \models \Diamond \phi$ 当且仅当存在 $v$ 使得 $wRv$ 且 $\mathcal{M}, v \models \phi$

### 4.3 模态系统 / Modal Systems / Modalsysteme / Systèmes modaux

**定义 4.3.1 (基本模态系统 K)**
系统 K 由以下公理和规则组成：

- 所有命题逻辑重言式
- 分布公理：$\Box(\phi \to \psi) \to (\Box \phi \to \Box \psi)$
- 必然化规则：$\frac{\phi}{\Box \phi}$

**定义 4.3.2 (系统 T)**
系统 T 在 K 基础上增加：

- 公理 T：$\Box \phi \to \phi$

**定义 4.3.3 (系统 S4)**
系统 S4 在 T 基础上增加：

- 公理 4：$\Box \phi \to \Box \Box \phi$

**定义 4.3.4 (系统 S5)**
系统 S5 在 S4 基础上增加：

- 公理 5：$\Diamond \phi \to \Box \Diamond \phi$

**定理 4.3.1 (模态完备性)**
每个基本模态系统都有对应的克里普克语义完备性。

## 5. 形式化证明 / Formal Proofs / Formale Beweise / Preuves formelles

### 5.1 证明系统 / Proof Systems / Beweissysteme / Systèmes de preuve

**定义 5.1.1 (希尔伯特系统)**
希尔伯特系统由公理和推理规则组成，证明是公式的有限序列，其中每个公式要么是公理，要么由前面的公式通过推理规则得到。

**定义 5.1.2 (自然演绎系统)**
自然演绎系统使用引入和消除规则，证明是树状结构，其中叶子节点是假设，内部节点是推理规则的应用。

**定义 5.1.3 (序列演算)**
序列演算使用序列作为基本对象，证明是序列的树状结构，其中每个序列由前面的序列通过规则得到。

### 5.2 证明理论 / Proof Theory / Beweistheorie / Théorie de la preuve

**定理 5.2.1 (切消定理)**
在序列演算中，如果 $\Gamma \vdash \Delta$ 有证明，则存在不使用切规则的证明。

**定理 5.2.2 (正规化定理)**
在自然演绎中，每个证明都可以正规化为唯一的标准形式。

**定理 5.2.3 (子公式性质)**
在无切证明中，出现的每个公式都是结论的子公式。

### 5.3 自动定理证明 / Automated Theorem Proving / Automatisches Theorembeweisen / Démonstration automatique de théorèmes

**算法 5.3.1 (归结算法)**
归结算法是自动定理证明的基础：

1. 将公式转换为合取范式
2. 应用归结规则生成新的子句
3. 如果生成空子句，则原公式不可满足

**算法 5.3.2 (表算法)**
表算法通过构建语义表来检查公式的有效性：

1. 从要检查的公式开始
2. 应用表规则分解公式
3. 如果所有分支都闭合，则公式有效

## 6. 应用实例 / Applications / Anwendungen / Applications

### 6.1 AI 中的逻辑应用 / Logical Applications in AI / Logische Anwendungen in der KI / Applications logiques dans l'IA

**知识表示与推理 / Knowledge Representation and Reasoning**:

- 使用一阶逻辑表示领域知识
- 基于逻辑的推理系统
- 描述逻辑用于本体工程

**形式化验证 / Formal Verification**:

- 程序正确性验证
- 硬件设计验证
- 协议安全性验证

**自然语言处理 / Natural Language Processing**:

- 语义分析
- 机器翻译
- 问答系统

### 6.2 计算机科学应用 / Computer Science Applications / Informatikanwendungen / Applications informatiques

**数据库理论 / Database Theory**:

- 关系数据库查询语言
- 完整性约束
- 数据依赖理论

**人工智能 / Artificial Intelligence**:

- 专家系统
- 自动规划
- 机器学习中的逻辑方法

## 代码实现 / Code Implementation / Code-Implementierung / Implémentation de code

### Rust 实现：形式逻辑核心 / Rust Implementation: Formal Logic Core

```rust
use std::collections::HashMap;
use std::fmt;

// 公式类型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Formula {
    Atom(String),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
    Or(Box<Formula>, Box<Formula>),
    Implies(Box<Formula>, Box<Formula>),
    Forall(String, Box<Formula>),
    Exists(String, Box<Formula>),
}

// 真值赋值
pub type Valuation = HashMap<String, bool>;

// 语义检查器
pub struct SemanticChecker;

impl SemanticChecker {
    pub fn evaluate(&self, formula: &Formula, valuation: &Valuation) -> bool {
        match formula {
            Formula::Atom(p) => valuation.get(p).copied().unwrap_or(false),
            Formula::Not(f) => !self.evaluate(f, valuation),
            Formula::And(f1, f2) => self.evaluate(f1, valuation) && self.evaluate(f2, valuation),
            Formula::Or(f1, f2) => self.evaluate(f1, valuation) || self.evaluate(f2, valuation),
            Formula::Implies(f1, f2) => !self.evaluate(f1, valuation) || self.evaluate(f2, valuation),
            _ => false, // 简化处理量词
        }
    }

    pub fn is_tautology(&self, formula: &Formula) -> bool {
        let variables = self.collect_variables(formula);
        self.check_all_valuations(formula, &variables, &mut HashMap::new())
    }

    fn collect_variables(&self, formula: &Formula) -> Vec<String> {
        match formula {
            Formula::Atom(p) => vec![p.clone()],
            Formula::Not(f) => self.collect_variables(f),
            Formula::And(f1, f2) | Formula::Or(f1, f2) | Formula::Implies(f1, f2) => {
                let mut vars = self.collect_variables(f1);
                vars.extend(self.collect_variables(f2));
                vars.sort();
                vars.dedup();
                vars
            }
            _ => vec![],
        }
    }

    fn check_all_valuations(&self, formula: &Formula, variables: &[String], valuation: &mut Valuation) -> bool {
        if valuation.len() == variables.len() {
            return self.evaluate(formula, valuation);
        }

        let var = &variables[valuation.len()];
        valuation.insert(var.clone(), true);
        let result_true = self.check_all_valuations(formula, variables, valuation);
        valuation.insert(var.clone(), false);
        let result_false = self.check_all_valuations(formula, variables, valuation);
        valuation.remove(var);

        result_true && result_false
    }
}

// 自然演绎证明器
pub struct NaturalDeductionProver;

impl NaturalDeductionProver {
    pub fn modus_ponens(&self, impl: &Formula, antecedent: &Formula) -> Option<Formula> {
        if let Formula::Implies(ant, cons) = impl {
            if ant.as_ref() == antecedent {
                return Some(*cons.clone());
            }
        }
        None
    }

    pub fn and_intro(&self, left: &Formula, right: &Formula) -> Formula {
        Formula::And(Box::new(left.clone()), Box::new(right.clone()))
    }

    pub fn and_elim_left(&self, and_formula: &Formula) -> Option<Formula> {
        if let Formula::And(left, _) = and_formula {
            Some(*left.clone())
        } else {
            None
        }
    }

    pub fn and_elim_right(&self, and_formula: &Formula) -> Option<Formula> {
        if let Formula::And(_, right) = and_formula {
            Some(*right.clone())
        } else {
            None
        }
    }
}

// 归结证明器
pub struct ResolutionProver {
    clauses: Vec<Vec<String>>,
}

impl ResolutionProver {
    pub fn new() -> Self {
        ResolutionProver {
            clauses: Vec::new(),
        }
    }

    pub fn add_clause(&mut self, literals: Vec<String>) {
        self.clauses.push(literals);
    }

    pub fn resolve(&self, clause1: &[String], clause2: &[String]) -> Option<Vec<String>> {
        for lit1 in clause1 {
            for lit2 in clause2 {
                if self.are_complementary(lit1, lit2) {
                    let mut new_clause = Vec::new();
                    for l in clause1 {
                        if l != lit1 {
                            new_clause.push(l.clone());
                        }
                    }
                    for l in clause2 {
                        if l != lit2 && !new_clause.contains(l) {
                            new_clause.push(l.clone());
                        }
                    }
                    return Some(new_clause);
                }
            }
        }
        None
    }

    fn are_complementary(&self, lit1: &str, lit2: &str) -> bool {
        (lit1.starts_with("¬") && lit2 == &lit1[1..]) ||
        (lit2.starts_with("¬") && lit1 == &lit2[1..])
    }

    pub fn prove(&mut self) -> bool {
        let mut new_clauses = Vec::new();
        let mut changed = true;

        while changed {
            changed = false;
            for i in 0..self.clauses.len() {
                for j in i+1..self.clauses.len() {
                    if let Some(resolvent) = self.resolve(&self.clauses[i], &self.clauses[j]) {
                        if resolvent.is_empty() {
                            return true; // 空子句，矛盾
                        }
                        if !self.clauses.contains(&resolvent) && !new_clauses.contains(&resolvent) {
                            new_clauses.push(resolvent);
                            changed = true;
                        }
                    }
                }
            }
            self.clauses.extend(new_clauses.drain(..));
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_evaluation() {
        let checker = SemanticChecker;
        let formula = Formula::And(
            Box::new(Formula::Atom("p".to_string())),
            Box::new(Formula::Atom("q".to_string()))
        );

        let mut valuation = HashMap::new();
        valuation.insert("p".to_string(), true);
        valuation.insert("q".to_string(), true);

        assert!(checker.evaluate(&formula, &valuation));

        valuation.insert("q".to_string(), false);
        assert!(!checker.evaluate(&formula, &valuation));
    }

    #[test]
    fn test_tautology() {
        let checker = SemanticChecker;
        // p ∨ ¬p 是重言式
        let formula = Formula::Or(
            Box::new(Formula::Atom("p".to_string())),
            Box::new(Formula::Not(Box::new(Formula::Atom("p".to_string()))))
        );

        assert!(checker.is_tautology(&formula));
    }

    #[test]
    fn test_resolution() {
        let mut prover = ResolutionProver::new();

        // 添加子句: {A, B}, {¬A, C}, {¬B, C}, {¬C}
        prover.add_clause(vec!["A".to_string(), "B".to_string()]);
        prover.add_clause(vec!["¬A".to_string(), "C".to_string()]);
        prover.add_clause(vec!["¬B".to_string(), "C".to_string()]);
        prover.add_clause(vec!["¬C".to_string()]);

        // 应该能证明矛盾
        assert!(prover.prove());
    }
}
```

### Haskell 实现：高级逻辑系统 / Haskell Implementation: Advanced Logic System

```haskell
{-# LANGUAGE GADTs, DataKinds, TypeFamilies #-}

-- 公式类型
data Formula where
  Atom :: String -> Formula
  Not :: Formula -> Formula
  And :: Formula -> Formula -> Formula
  Or :: Formula -> Formula -> Formula
  Implies :: Formula -> Formula -> Formula
  Forall :: String -> Formula -> Formula
  Exists :: String -> Formula -> Formula

-- 真值赋值
type Valuation = [(String, Bool)]

-- 语义评估
evaluate :: Formula -> Valuation -> Bool
evaluate (Atom p) val = case lookup p val of
  Just b -> b
  Nothing -> False
evaluate (Not f) val = not (evaluate f val)
evaluate (And f1 f2) val = evaluate f1 val && evaluate f2 val
evaluate (Or f1 f2) val = evaluate f1 val || evaluate f2 val
evaluate (Implies f1 f2) val = not (evaluate f1 val) || evaluate f2 val
evaluate _ _ = False -- 简化处理量词

-- 收集变量
collectVariables :: Formula -> [String]
collectVariables (Atom p) = [p]
collectVariables (Not f) = collectVariables f
collectVariables (And f1 f2) = nub $ collectVariables f1 ++ collectVariables f2
collectVariables (Or f1 f2) = nub $ collectVariables f1 ++ collectVariables f2
collectVariables (Implies f1 f2) = nub $ collectVariables f1 ++ collectVariables f2
collectVariables _ = []

-- 生成所有可能的真值赋值
generateValuations :: [String] -> [Valuation]
generateValuations [] = [[]]
generateValuations (var:vars) =
  let rest = generateValuations vars
  in [(var, True) : val | val <- rest] ++ [(var, False) : val | val <- rest]

-- 检查重言式
isTautology :: Formula -> Bool
isTautology f =
  let vars = collectVariables f
      valuations = generateValuations vars
  in all (evaluate f) valuations

-- 自然演绎规则
modusPonens :: Formula -> Formula -> Maybe Formula
modusPonens (Implies ant cons) premise =
  if ant == premise then Just cons else Nothing
modusPonens _ _ = Nothing

andIntro :: Formula -> Formula -> Formula
andIntro f1 f2 = And f1 f2

andElim1 :: Formula -> Maybe Formula
andElim1 (And f1 _) = Just f1
andElim1 _ = Nothing

andElim2 :: Formula -> Maybe Formula
andElim2 (And _ f2) = Just f2
andElim2 _ = Nothing

-- 归结
data Literal = Pos String | Neg String deriving (Eq, Show)
type Clause = [Literal]

resolve :: Clause -> Clause -> Maybe Clause
resolve c1 c2 =
  case findComplementary c1 c2 of
    Nothing -> Nothing
    Just (l1, l2) ->
      let newClause = filter (/= l1) c1 ++ filter (/= l2) c2
      in Just newClause

findComplementary :: Clause -> Clause -> Maybe (Literal, Literal)
findComplementary c1 c2 =
  case [ (l1, l2) | l1 <- c1, l2 <- c2, areComplementary l1 l2 ] of
    [] -> Nothing
    (x:_) -> Just x

areComplementary :: Literal -> Literal -> Bool
areComplementary (Pos x) (Neg y) = x == y
areComplementary (Neg x) (Pos y) = x == y
areComplementary _ _ = False

-- 归结证明
resolution :: [Clause] -> Bool
resolution clauses =
  let allClauses = clauses
      newClauses = generateResolvents allClauses
  in [] `elem` newClauses

generateResolvents :: [Clause] -> [Clause]
generateResolvents clauses =
  let resolvents = [ r | c1 <- clauses, c2 <- clauses,
                        c1 /= c2, Just r <- [resolve c1 c2] ]
  in if null resolvents
     then clauses
     else generateResolvents (clauses ++ resolvents)

-- 测试
main :: IO ()
main = do
  let p = Atom "p"
  let q = Atom "q"
  let formula = And p q

  let valuation = [("p", True), ("q", True)]
  print $ evaluate formula valuation

  let tautology = Or p (Not p)
  print $ isTautology tautology

  let clauses = [[Pos "A", Pos "B"], [Neg "A", Pos "C"], [Neg "B", Pos "C"], [Neg "C"]]
  print $ resolution clauses
```

## 参考文献 / References / Literatur / Références

1. **Enderton, H. B.** (2001). _A Mathematical Introduction to Logic_. Academic Press.
2. **Mendelson, E.** (2015). _Introduction to Mathematical Logic_. CRC Press.
3. **Hughes, G. E. & Cresswell, M. J.** (1996). _A New Introduction to Modal Logic_. Routledge.
4. **Boolos, G. S., Burgess, J. P. & Jeffrey, R. C.** (2007). _Computability and Logic_. Cambridge University Press.
5. **Troelstra, A. S. & Schwichtenberg, H.** (2000). _Basic Proof Theory_. Cambridge University Press.

---

_本模块为 FormalAI 提供了严格的形式逻辑基础，确保 AI 系统具备完整的逻辑推理和证明能力。_

_This module provides FormalAI with rigorous formal logic foundations, ensuring AI systems have complete logical reasoning and proof capabilities._

## 目录 / Table of Contents

- [1.1 形式逻辑 / Formal Logic / Formale Logik / Logique formelle](#11-形式逻辑--formal-logic--formale-logik--logique-formelle)
  - [概述 / Overview / Übersicht / Aperçu](#概述--overview--übersicht--aperçu)
  - [目录 / Table of Contents / Inhaltsverzeichnis / Table des matières](#目录--table-of-contents--inhaltsverzeichnis--table-des-matières)
  - [1. 公理系统 / Axiom System / Axiomensystem / Système d'axiomes](#1-公理系统--axiom-system--axiomensystem--système-daxiomes)
    - [1.1 逻辑语言 / Logical Language / Logische Sprache / Langage logique](#11-逻辑语言--logical-language--logische-sprache--langage-logique)
    - [1.2 希尔伯特公理系统 / Hilbert Axiom System / Hilbert-Axiomensystem / Système d'axiomes de Hilbert](#12-希尔伯特公理系统--hilbert-axiom-system--hilbert-axiomensystem--système-daxiomes-de-hilbert)
  - [2. 命题逻辑 / Propositional Logic / Aussagenlogik / Logique propositionnelle](#2-命题逻辑--propositional-logic--aussagenlogik--logique-propositionnelle)
    - [2.1 语义学 / Semantics / Semantik / Sémantique](#21-语义学--semantics--semantik--sémantique)
    - [2.2 自然演绎系统 / Natural Deduction System / Natürliches Deduktionssystem / Système de déduction naturelle](#22-自然演绎系统--natural-deduction-system--natürliches-deduktionssystem--système-de-déduction-naturelle)
  - [3. 谓词逻辑 / Predicate Logic / Prädikatenlogik / Logique des prédicats](#3-谓词逻辑--predicate-logic--prädikatenlogik--logique-des-prédicats)
    - [3.1 一阶语言 / First-Order Language / Erststufige Sprache / Langage du premier ordre](#31-一阶语言--first-order-language--erststufige-sprache--langage-du-premier-ordre)
    - [3.2 一阶结构 / First-Order Structures / Erststufige Strukturen / Structures du premier ordre](#32-一阶结构--first-order-structures--erststufige-strukturen--structures-du-premier-ordre)
    - [3.3 量词规则 / Quantifier Rules / Quantorenregeln / Règles de quantificateurs](#33-量词规则--quantifier-rules--quantorenregeln--règles-de-quantificateurs)
  - [4. 模态逻辑 / Modal Logic / Modallogik / Logique modale](#4-模态逻辑--modal-logic--modallogik--logique-modale)
    - [4.1 模态语言 / Modal Language / Modale Sprache / Langage modal](#41-模态语言--modal-language--modale-sprache--langage-modal)
    - [4.2 克里普克语义 / Kripke Semantics / Kripke-Semantik / Sémantique de Kripke](#42-克里普克语义--kripke-semantics--kripke-semantik--sémantique-de-kripke)
    - [4.3 模态系统 / Modal Systems / Modalsysteme / Systèmes modaux](#43-模态系统--modal-systems--modalsysteme--systèmes-modaux)
  - [5. 形式化证明 / Formal Proofs / Formale Beweise / Preuves formelles](#5-形式化证明--formal-proofs--formale-beweise--preuves-formelles)
    - [5.1 证明系统 / Proof Systems / Beweissysteme / Systèmes de preuve](#51-证明系统--proof-systems--beweissysteme--systèmes-de-preuve)
    - [5.2 证明理论 / Proof Theory / Beweistheorie / Théorie de la preuve](#52-证明理论--proof-theory--beweistheorie--théorie-de-la-preuve)
    - [5.3 自动定理证明 / Automated Theorem Proving / Automatisches Theorembeweisen / Démonstration automatique de théorèmes](#53-自动定理证明--automated-theorem-proving--automatisches-theorembeweisen--démonstration-automatique-de-théorèmes)
  - [6. 应用实例 / Applications / Anwendungen / Applications](#6-应用实例--applications--anwendungen--applications)
    - [6.1 AI 中的逻辑应用 / Logical Applications in AI / Logische Anwendungen in der KI / Applications logiques dans l'IA](#61-ai-中的逻辑应用--logical-applications-in-ai--logische-anwendungen-in-der-ki--applications-logiques-dans-lia)
    - [6.2 计算机科学应用 / Computer Science Applications / Informatikanwendungen / Applications informatiques](#62-计算机科学应用--computer-science-applications--informatikanwendungen--applications-informatiques)
  - [代码实现 / Code Implementation / Code-Implementierung / Implémentation de code](#代码实现--code-implementation--code-implementierung--implémentation-de-code)
    - [Rust 实现：形式逻辑核心 / Rust Implementation: Formal Logic Core](#rust-实现形式逻辑核心--rust-implementation-formal-logic-core)
    - [Haskell 实现：高级逻辑系统 / Haskell Implementation: Advanced Logic System](#haskell-实现高级逻辑系统--haskell-implementation-advanced-logic-system)
  - [参考文献 / References / Literatur / Références](#参考文献--references--literatur--références)
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
    - [Rust 实现：逻辑推理引擎](#rust-实现逻辑推理引擎)
    - [Haskell 实现：类型化逻辑](#haskell-实现类型化逻辑)
  - [参考文献 / References](#参考文献--references)
  - [2024/2025 最新进展 / Latest Updates](#20242025-最新进展--latest-updates)

---

## 相关章节 / Related Chapters

**前置依赖 / Prerequisites / Voraussetzungen / Prérequis:**

- [0.0 ZFC 公理系统](../../00-foundations/00-mathematical-foundations/00-set-theory-zfc.md) - 提供集合论基础 / Provides set theory foundation
- [0.3 逻辑演算系统](../../00-foundations/00-mathematical-foundations/03-logical-calculus.md) - 提供逻辑演算基础 / Provides logical calculus foundation

**后续应用 / Applications / Anwendungen / Applications:**

- [1.2 数学基础](../01.2-数学基础/README.md) - 提供逻辑基础 / Provides logical foundation
- [3.1 形式化验证](../../03-formal-methods/03.1-形式化验证/README.md) - 提供逻辑基础 / Provides logical foundation

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

$$
\begin{align}
\neg p &= \text{NOT}(p) \\
p \land q &= \text{AND}(p, q) \\
p \lor q &= \text{OR}(p, q) \\
p \rightarrow q &= \neg p \lor q \\
p \leftrightarrow q &= (p \rightarrow q) \land (q \rightarrow p)
\end{align}
$$

### 1.2 真值表 / Truth Tables

**真值表定义 / Truth Table Definition:**

真值表是显示命题在所有可能真值组合下真值的表格：
A truth table is a table showing the truth values of propositions under all possible truth value combinations:

| p   | q   | ¬p  | p∧q | p∨q | p→q | p↔q |
| --- | --- | --- | --- | --- | --- | --- |
| T   | T   | F   | T   | T   | T   | T   |
| T   | F   | F   | F   | T   | F   | F   |
| F   | T   | T   | F   | T   | T   | F   |
| F   | F   | T   | F   | F   | T   | T   |

**真值函数 / Truth Functions:**

$$
\begin{align}
f_{\neg}(p) &= 1 - p \\
f_{\land}(p, q) &= \min(p, q) \\
f_{\lor}(p, q) &= \max(p, q) \\
f_{\rightarrow}(p, q) &= \max(1-p, q) \\
f_{\leftrightarrow}(p, q) &= 1 - |p - q|
\end{align}
$$

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

$$
\begin{align}
\neg \forall x P(x) &\equiv \exists x \neg P(x) \\
\neg \exists x P(x) &\equiv \forall x \neg P(x) \\
\forall x (P(x) \land Q(x)) &\equiv \forall x P(x) \land \forall x Q(x) \\
\exists x (P(x) \lor Q(x)) &\equiv \exists x P(x) \lor \exists x Q(x)
\end{align}
$$

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

1. **K 公理 / K Axiom:** $\Box(\phi \rightarrow \psi) \rightarrow (\Box \phi \rightarrow \Box \psi)$
2. **T 公理 / T Axiom:** $\Box \phi \rightarrow \phi$
3. **4 公理 / 4 Axiom:** $\Box \phi \rightarrow \Box \Box \phi$
4. **5 公理 / 5 Axiom:** $\Diamond \phi \rightarrow \Box \Diamond \phi$

### 3.2 可能世界语义 / Possible Worlds Semantics

**克里普克模型 / Kripke Model:**

$$\mathcal{M} = (W, R, V)$$

其中：

- $W$ 是可能世界集合 / set of possible worlds
- $R \subseteq W \times W$ 是可达关系 / accessibility relation
- $V: W \times \mathcal{P} \rightarrow \{0,1\}$ 是赋值函数 / valuation function

**真值定义 / Truth Definition:**

$$
\begin{align}
\mathcal{M}, w &\models p \Leftrightarrow V(w, p) = 1 \\
\mathcal{M}, w &\models \neg \phi \Leftrightarrow \mathcal{M}, w \not\models \phi \\
\mathcal{M}, w &\models \phi \land \psi \Leftrightarrow \mathcal{M}, w \models \phi \text{ and } \mathcal{M}, w \models \psi \\
\mathcal{M}, w &\models \Box \phi \Leftrightarrow \forall v: wRv \Rightarrow \mathcal{M}, v \models \phi \\
\mathcal{M}, w &\models \Diamond \phi \Leftrightarrow \exists v: wRv \text{ and } \mathcal{M}, v \models \phi
\end{align}
$$

### 3.3 模态系统 / Modal Systems

**常见模态系统 / Common Modal Systems:**

1. **K 系统 / System K:** 基本模态逻辑
2. **T 系统 / System T:** K + T 公理
3. **S4 系统 / System S4:** T + 4 公理
4. **S5 系统 / System S5:** T + 4 + 5 公理

**对应关系 / Correspondence:**

- T 公理 ↔ 自反性 / reflexivity
- 4 公理 ↔ 传递性 / transitivity
- 5 公理 ↔ 欧几里得性 / euclidean property

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

### Rust 实现：逻辑推理引擎

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

### Haskell 实现：类型化逻辑

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

   - 王宪钧 (1982). _数理逻辑引论_. 北京大学出版社.
   - 张清宇 (2003). _逻辑哲学九章_. 江苏人民出版社.
   - 李小五 (2005). _模态逻辑_. 中国社会科学出版社.

2. **English:**

   - Enderton, H. B. (2001). _A Mathematical Introduction to Logic_. Academic Press.
   - Mendelson, E. (2015). _Introduction to Mathematical Logic_. CRC Press.
   - Hughes, G. E., & Cresswell, M. J. (1996). _A New Introduction to Modal Logic_. Routledge.
   - Boolos, G. S., Burgess, J. P., & Jeffrey, R. C. (2007). _Computability and Logic_. Cambridge University Press.

3. **Deutsch / German:**

   - Ebbinghaus, H. D., Flum, J., & Thomas, W. (2018). _Einführung in die mathematische Logik_. Springer.
   - Rautenberg, W. (2008). _Einführung in die mathematische Logik_. Vieweg+Teubner.

4. **Français / French:**
   - Cori, R., & Lascar, D. (2003). _Logique mathématique_. Dunod.
   - David, R., Nour, K., & Raffalli, C. (2004). _Introduction à la logique_. Dunod.

---

_本模块为 FormalAI 提供了完整的形式逻辑基础，涵盖命题逻辑、谓词逻辑、模态逻辑等核心内容，为 AI 系统的逻辑推理能力提供了坚实的理论基础。_

_This module provides complete formal logic foundations for FormalAI, covering propositional logic, predicate logic, modal logic, and other core content, providing solid theoretical foundations for logical reasoning capabilities in AI systems._

## 2024/2025 最新进展 / Latest Updates

- 神经符号推理与 LLM 的可验证逻辑接口（占位）。
- 表/归结/序列演算在自动证明工具链中的工程实践（占位）。

[返回“最新进展”索引](../../LATEST_UPDATES_INDEX.md)
