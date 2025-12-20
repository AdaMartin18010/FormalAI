# 0.3 逻辑演算系统 / Logical Calculus System / Logisches Kalkülsystem / Système de calcul logique

[返回全局导航](../../GLOBAL_NAVIGATION.md) · [学习路径](../../LEARNING_PATH_DESIGN.md)

## 概述 / Overview / Übersicht / Aperçu

逻辑演算系统是形式化推理的基础，为 FormalAI 提供严格的证明理论和推理机制。本模块建立完整的逻辑演算体系，包括自然演绎、序列演算和模型论。

Logical calculus system is the foundation of formal reasoning, providing FormalAI with rigorous proof theory and inference mechanisms. This module establishes a complete logical calculus system, including natural deduction, sequent calculus, and model theory.

## 目录 / Table of Contents / Inhaltsverzeichnis / Table des matières

- [0.3 逻辑演算系统](#03-逻辑演算系统--logical-calculus-system--logisches-kalkülsystem--système-de-calcul-logique)
  - [概述](#概述--overview--übersicht--aperçu)
  - [目录](#目录--table-of-contents--inhaltsverzeichnis--table-des-matières)
  - [1. 自然演绎系统](#1-自然演绎系统--natural-deduction-system--natürliches-deduktionssystem--système-de-déduction-naturelle)
  - [2. 序列演算系统](#2-序列演算系统--sequent-calculus-system--sequenzenkalkül--système-de-calcul-des-séquents)
  - [3. 模型论基础](#3-模型论基础--model-theory-foundations--modelltheoretische-grundlagen--fondements-de-la-théorie-des-modèles)
  - [4. 证明论](#4-证明论--proof-theory--beweistheorie--théorie-de-la-preuve)
  - [5. 完备性定理](#5-完备性定理--completeness-theorems--vollständigkeitssätze--théorèmes-de-complétude)
  - [6. AI 中的逻辑演算应用](#6-ai中的逻辑演算应用--logical-calculus-applications-in-ai--anwendungen-des-logischen-kalküls-in-der-ki--applications-du-calcul-logique-dans-lia)
  - [代码实现](#代码实现--code-implementation--code-implementierung--implémentation-de-code)
  - [参考文献](#参考文献--references--literatur--références)

## 1. 自然演绎系统 / Natural Deduction System / Natürliches Deduktionssystem / Système de déduction naturelle

### 1.1 命题逻辑自然演绎 / Propositional Logic Natural Deduction / Aussagenlogik natürliche Deduktion / Déduction naturelle de la logique propositionnelle

**定义 1.1.1 (自然演绎规则)**
自然演绎系统由以下推理规则组成：

**合取引入 (∧I)：**
$$\frac{A \quad B}{A \land B}$$

**合取消除 (∧E)：**
$$\frac{A \land B}{A} \quad \frac{A \land B}{B}$$

**析取引入 (∨I)：**
$$\frac{A}{A \lor B} \quad \frac{B}{A \lor B}$$

**析取消除 (∨E)：**
$$\frac{A \lor B \quad [A] \quad [B]}{C} \quad \frac{C \quad C}{C}$$

**蕴含引入 (→I)：**
$$\frac{[A]}{B} \Rightarrow \frac{A \to B}{}$$

**蕴含消除 (→E)：**
$$\frac{A \to B \quad A}{B}$$

**否定引入 (¬I)：**
$$\frac{[A]}{\bot} \Rightarrow \frac{\neg A}{}$$

**否定消除 (¬E)：**
$$\frac{\neg A \quad A}{\bot}$$

**爆炸原理 (⊥E)：**
$$\frac{\bot}{A}$$

### 1.2 谓词逻辑自然演绎 / Predicate Logic Natural Deduction / Prädikatenlogik natürliche Deduktion / Déduction naturelle de la logique des prédicats

**全称量词引入 (∀I)：**
$$\frac{A(x)}{\forall x A(x)} \quad \text{其中} \ x \text{不在假设中自由出现}$$

**全称量词消除 (∀E)：**
$$\frac{\forall x A(x)}{A(t)} \quad \text{其中} \ t \text{对} \ x \text{自由}$$

**存在量词引入 (∃I)：**
$$\frac{A(t)}{\exists x A(x)} \quad \text{其中} \ t \text{对} \ x \text{自由}$$

**存在量词消除 (∃E)：**
$$\frac{\exists x A(x) \quad [A(x)]}{B} \Rightarrow \frac{B}{} \quad \text{其中} \ x \text{不在} \ B \text{中自由出现}$$

### 1.3 模态逻辑自然演绎 / Modal Logic Natural Deduction / Modallogik natürliche Deduktion / Déduction naturelle de la logique modale

**必然性引入 (□I)：**
$$\frac{A}{\Box A} \quad \text{其中} \ A \text{不依赖于任何假设}$$

**必然性消除 (□E)：**
$$\frac{\Box A}{A}$$

**可能性引入 (◇I)：**
$$\frac{A}{\Diamond A}$$

**可能性消除 (◇E)：**
$$\frac{\Diamond A \quad [A]}{B} \Rightarrow \frac{B}{} \quad \text{其中} \ A \text{不依赖于其他假设}$$

## 2. 序列演算系统 / Sequent Calculus System / Sequenzenkalkül / Système de calcul des séquents

### 2.1 序列的定义 / Definition of Sequents / Definition von Sequenzen / Définition des séquents

**定义 2.1.1 (序列)**
序列是形如 $\Gamma \vdash \Delta$ 的表达式，其中：

- $\Gamma$ 是前提公式的有限序列
- $\Delta$ 是结论公式的有限序列

**定义 2.1.2 (序列的有效性)**
序列 $\Gamma \vdash \Delta$ 是有效的，如果 $\bigwedge \Gamma \to \bigvee \Delta$ 是重言式。

### 2.2 结构规则 / Structural Rules / Strukturregeln / Règles structurelles

**弱化规则 (W)：**
$$\frac{\Gamma \vdash \Delta}{\Gamma, A \vdash \Delta} \quad \frac{\Gamma \vdash \Delta}{\Gamma \vdash \Delta, A}$$

**收缩规则 (C)：**
$$\frac{\Gamma, A, A \vdash \Delta}{\Gamma, A \vdash \Delta} \quad \frac{\Gamma \vdash \Delta, A, A}{\Gamma \vdash \Delta, A}$$

**交换规则 (X)：**
$$\frac{\Gamma, A, B, \Gamma' \vdash \Delta}{\Gamma, B, A, \Gamma' \vdash \Delta} \quad \frac{\Gamma \vdash \Delta, A, B, \Delta'}{\Gamma \vdash \Delta, B, A, \Delta'}$$

### 2.3 逻辑规则 / Logical Rules / Logische Regeln / Règles logiques

**合取规则：**
$$\frac{\Gamma, A, B \vdash \Delta}{\Gamma, A \land B \vdash \Delta} \quad \frac{\Gamma \vdash \Delta, A \quad \Gamma \vdash \Delta, B}{\Gamma \vdash \Delta, A \land B}$$

**析取规则：**
$$\frac{\Gamma, A \vdash \Delta \quad \Gamma, B \vdash \Delta}{\Gamma, A \lor B \vdash \Delta} \quad \frac{\Gamma \vdash \Delta, A, B}{\Gamma \vdash \Delta, A \lor B}$$

**蕴含规则：**
$$\frac{\Gamma \vdash \Delta, A \quad \Gamma, B \vdash \Delta}{\Gamma, A \to B \vdash \Delta} \quad \frac{\Gamma, A \vdash \Delta, B}{\Gamma \vdash \Delta, A \to B}$$

**否定规则：**
$$\frac{\Gamma \vdash \Delta, A}{\Gamma, \neg A \vdash \Delta} \quad \frac{\Gamma, A \vdash \Delta}{\Gamma \vdash \Delta, \neg A}$$

### 2.4 量词规则 / Quantifier Rules / Quantorenregeln / Règles de quantificateurs

**全称量词规则：**
$$\frac{\Gamma, A(t) \vdash \Delta}{\Gamma, \forall x A(x) \vdash \Delta} \quad \frac{\Gamma \vdash \Delta, A(x)}{\Gamma \vdash \Delta, \forall x A(x)}$$

**存在量词规则：**
$$\frac{\Gamma, A(x) \vdash \Delta}{\Gamma, \exists x A(x) \vdash \Delta} \quad \frac{\Gamma \vdash \Delta, A(t)}{\Gamma \vdash \Delta, \exists x A(x)}$$

## 3. 模型论基础 / Model Theory Foundations / Modelltheoretische Grundlagen / Fondements de la théorie des modèles

### 3.1 一阶结构 / First-Order Structures / Erststufige Strukturen / Structures du premier ordre

**定义 3.1.1 (一阶语言)**
一阶语言 $\mathcal{L}$ 由以下符号组成：

- 变量：$x_1, x_2, x_3, \ldots$
- 常量符号：$c_1, c_2, c_3, \ldots$
- 函数符号：$f_1, f_2, f_3, \ldots$
- 谓词符号：$P_1, P_2, P_3, \ldots$
- 逻辑连接词：$\land, \lor, \to, \neg$
- 量词：$\forall, \exists$
- 等号：$=$

**定义 3.1.2 (一阶结构)**
对于一阶语言 $\mathcal{L}$，$\mathcal{L}$-结构 $\mathcal{M}$ 由以下组成：

- 非空域 $M$
- 对每个常量符号 $c$，元素 $c^{\mathcal{M}} \in M$
- 对每个 $n$-元函数符号 $f$，函数 $f^{\mathcal{M}}: M^n \to M$
- 对每个 $n$-元谓词符号 $P$，关系 $P^{\mathcal{M}} \subseteq M^n$

### 3.2 真值定义 / Truth Definition / Wahrheitsdefinition / Définition de vérité

**定义 3.2.1 (赋值)**
赋值是从变量到域 $M$ 的函数 $s: \text{Var} \to M$。

**定义 3.2.2 (项的解释)**
对于项 $t$ 和赋值 $s$，项的解释 $t^{\mathcal{M},s}$ 定义为：

- 如果 $t = x$ 是变量，则 $t^{\mathcal{M},s} = s(x)$
- 如果 $t = c$ 是常量，则 $t^{\mathcal{M},s} = c^{\mathcal{M}}$
- 如果 $t = f(t_1, \ldots, t_n)$，则 $t^{\mathcal{M},s} = f^{\mathcal{M}}(t_1^{\mathcal{M},s}, \ldots, t_n^{\mathcal{M},s})$

**定义 3.2.3 (公式的真值)**
对于公式 $\phi$ 和赋值 $s$，$\mathcal{M} \models \phi[s]$ 定义为：

- $\mathcal{M} \models P[t_1, \ldots, t_n](s)$ 当且仅当 $(t_1^{\mathcal{M},s}, \ldots, t_n^{\mathcal{M},s}) \in P^{\mathcal{M}}$
- $\mathcal{M} \models \neg \phi[s]$ 当且仅当 $\mathcal{M} \not\models \phi[s]$
- $\mathcal{M} \models \phi \land \psi[s]$ 当且仅当 $\mathcal{M} \models \phi[s]$ 且 $\mathcal{M} \models \psi[s]$
- $\mathcal{M} \models \forall x \phi[s]$ 当且仅当对所有 $a \in M$，$\mathcal{M} \models \phi[s(x|a)]$

### 3.3 紧致性定理 / Compactness Theorem / Kompaktheitssatz / Théorème de compacité

**定理 3.3.1 (紧致性定理)**
一阶理论 $T$ 有模型当且仅当 $T$ 的每个有限子集都有模型。

**证明：**
（必要性）显然。
（充分性）使用超积构造。设 $I$ 是 $T$ 的所有有限子集的集合，对每个 $i \in I$，设 $\mathcal{M}_i$ 是 $i$ 的模型。定义超滤子 $F$ 使得对每个 $i \in I$，$\{j \in I : i \subseteq j\} \in F$。则超积 $\prod_{i \in I} \mathcal{M}_i / F$ 是 $T$ 的模型。□

## 4. 证明论 / Proof Theory / Beweistheorie / Théorie de la preuve

### 4.1 证明的语法 / Syntax of Proofs / Syntax von Beweisen / Syntaxe des preuves

**定义 4.1.1 (证明树)**
证明树是满足以下条件的有限树：

- 每个节点标记为公式
- 每个叶子节点是公理或假设
- 每个内部节点由其子节点通过推理规则得到

**定义 4.1.2 (证明)**
从假设 $\Gamma$ 到结论 $A$ 的证明是根节点为 $A$ 的证明树，其叶子节点要么是 $\Gamma$ 中的公式，要么是公理。

### 4.2 切消定理 / Cut Elimination Theorem / Schnitteliminationssatz / Théorème d'élimination des coupures

**定理 4.2.1 (切消定理)**
在序列演算中，如果 $\Gamma \vdash \Delta$ 有证明，则存在不使用切规则的证明。

**证明：**
使用归纳法，按切公式的复杂度进行归纳。对每个切，要么可以消除，要么可以转换为更简单的切。□

**推论 4.2.1 (子公式性质)**
在无切证明中，出现的每个公式都是结论的子公式。

### 4.3 正规化定理 / Normalization Theorem / Normalisierungssatz / Théorème de normalisation

**定理 4.3.1 (强正规化)**
在自然演绎中，每个证明都可以正规化为唯一的标准形式。

**证明：**
定义归约关系，证明它是强正规化的（即没有无限归约序列）。□

## 5. 完备性定理 / Completeness Theorems / Vollständigkeitssätze / Théorèmes de complétude

### 5.1 哥德尔完备性定理 / Gödel's Completeness Theorem / Gödelscher Vollständigkeitssatz / Théorème de complétude de Gödel

**定理 5.1.1 (哥德尔完备性定理)**
一阶逻辑是完备的，即：
$$\Gamma \models \phi \Rightarrow \Gamma \vdash \phi$$

**证明：**
使用亨金构造。如果 $\Gamma \not\vdash \phi$，则 $\Gamma \cup \{\neg \phi\}$ 是一致的。通过添加新常量和亨金公理，可以构造一个模型使得 $\Gamma \cup \{\neg \phi\}$ 在其中为真。□

### 5.2 可靠性定理 / Soundness Theorem / Korrektheitssatz / Théorème de correction

**定理 5.2.1 (可靠性定理)**
一阶逻辑是可靠的，即：
$$\Gamma \vdash \phi \Rightarrow \Gamma \models \phi$$

**证明：**
对证明长度进行归纳，验证每个推理规则保持有效性。□

### 5.3 勒文海姆-斯科伦定理 / Löwenheim-Skolem Theorem / Löwenheim-Skolem-Satz / Théorème de Löwenheim-Skolem

**定理 5.3.1 (向下勒文海姆-斯科伦定理)**
如果一阶理论有无限模型，则它有任意大基数的模型。

**定理 5.3.2 (向上勒文海姆-斯科伦定理)**
如果一阶理论有无限模型，则它有任意大基数的模型。

## 6. AI 中的逻辑演算应用 / Logical Calculus Applications in AI / Anwendungen des logischen Kalküls in der KI / Applications du calcul logique dans l'IA

### 6.1 自动定理证明 / Automated Theorem Proving / Automatisches Theorembeweisen / Démonstration automatique de théorèmes

**定义 6.1.1 (归结原理)**
归结原理是自动定理证明的基础：
$$\frac{C_1 \lor A \quad C_2 \lor \neg A}{C_1 \lor C_2}$$

**算法 6.1.1 (归结算法)**:

```haskell
data Clause = Clause [Literal] deriving (Eq, Show)
data Literal = Pos String | Neg String deriving (Eq, Show)

resolve :: Clause -> Clause -> Maybe Clause
resolve (Clause lits1) (Clause lits2) =
  case findComplementary lits1 lits2 of
    Nothing -> Nothing
    Just (lit1, lit2) ->
      let newLits = filter (/= lit1) lits1 ++ filter (/= lit2) lits2
      in Just (Clause newLits)

resolution :: [Clause] -> [Clause] -> Bool
resolution clauses goal =
  let allClauses = clauses ++ map negateClause goal
      newClauses = generateResolvents allClauses
  in emptyClause `elem` newClauses
```

### 6.2 知识表示与推理 / Knowledge Representation and Reasoning / Wissensrepräsentation und Schlussfolgerung / Représentation des connaissances et raisonnement

**定义 6.2.1 (描述逻辑)**
描述逻辑是一阶逻辑的可判定子集，用于知识表示：

**概念构造子：**

- 原子概念：$A, B, C, \ldots$
- 概念合取：$C \sqcap D$
- 概念析取：$C \sqcup D$
- 概念否定：$\neg C$
- 存在限制：$\exists R.C$
- 全称限制：$\forall R.C$

**公理：**

- 概念包含：$C \sqsubseteq D$
- 概念等价：$C \equiv D$
- 实例断言：$C(a)$
- 角色断言：$R(a, b)$

### 6.3 形式化验证 / Formal Verification / Formale Verifikation / Vérification formelle

**定义 6.3.1 (霍尔逻辑)**
霍尔逻辑用于程序验证：

**霍尔三元组：**
$$\{P\} \ S \ \{Q\}$$

其中 $P$ 是前置条件，$S$ 是程序语句，$Q$ 是后置条件。

**推理规则：**

- 赋值规则：$\{Q[E/x]\} \ x := E \ \{Q\}$
- 序列规则：$\frac{\{P\} \ S_1 \ \{R\} \quad \{R\} \ S_2 \ \{Q\}}{\{P\} \ S_1; S_2 \ \{Q\}}$
- 条件规则：$\frac{\{P \land B\} \ S_1 \ \{Q\} \quad \{P \land \neg B\} \ S_2 \ \{Q\}}{\{P\} \ \text{if } B \text{ then } S_1 \text{ else } S_2 \ \{Q\}}$
- 循环规则：$\frac{\{P \land B\} \ S \ \{P\}}{\{P\} \ \text{while } B \text{ do } S \ \{P \land \neg B\}}$

## 代码实现 / Code Implementation / Code-Implementierung / Implémentation de code

### Rust 实现：逻辑演算核心 / Rust Implementation: Logical Calculus Core

```rust
use std::collections::HashSet;
use std::fmt;

// 公式定义
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

// 证明树节点
#[derive(Debug, Clone)]
pub struct ProofNode {
    pub formula: Formula,
    pub rule: String,
    pub premises: Vec<ProofNode>,
}

// 自然演绎证明器
pub struct NaturalDeductionProver {
    assumptions: Vec<Formula>,
}

impl NaturalDeductionProver {
    pub fn new() -> Self {
        NaturalDeductionProver {
            assumptions: Vec::new(),
        }
    }

    // 合取引入
    pub fn and_intro(&self, left: &Formula, right: &Formula) -> ProofNode {
        ProofNode {
            formula: Formula::And(Box::new(left.clone()), Box::new(right.clone())),
            rule: "∧I".to_string(),
            premises: vec![
                ProofNode {
                    formula: left.clone(),
                    rule: "assumption".to_string(),
                    premises: vec![],
                },
                ProofNode {
                    formula: right.clone(),
                    rule: "assumption".to_string(),
                    premises: vec![],
                },
            ],
        }
    }

    // 合取消除
    pub fn and_elim_left(&self, and_formula: &Formula) -> Option<ProofNode> {
        if let Formula::And(left, _) = and_formula {
            Some(ProofNode {
                formula: *left.clone(),
                rule: "∧E".to_string(),
                premises: vec![ProofNode {
                    formula: and_formula.clone(),
                    rule: "assumption".to_string(),
                    premises: vec![],
                }],
            })
        } else {
            None
        }
    }

    // 蕴含引入
    pub fn impl_intro(&self, assumption: &Formula, conclusion: &Formula) -> ProofNode {
        ProofNode {
            formula: Formula::Implies(Box::new(assumption.clone()), Box::new(conclusion.clone())),
            rule: "→I".to_string(),
            premises: vec![ProofNode {
                formula: conclusion.clone(),
                rule: "assumption".to_string(),
                premises: vec![],
            }],
        }
    }

    // 蕴含消除 (Modus Ponens)
    pub fn impl_elim(&self, impl: &Formula, antecedent: &Formula) -> Option<ProofNode> {
        if let Formula::Implies(ant, cons) = impl {
            if ant.as_ref() == antecedent {
                Some(ProofNode {
                    formula: *cons.clone(),
                    rule: "→E".to_string(),
                    premises: vec![
                        ProofNode {
                            formula: impl.clone(),
                            rule: "assumption".to_string(),
                            premises: vec![],
                        },
                        ProofNode {
                            formula: antecedent.clone(),
                            rule: "assumption".to_string(),
                            premises: vec![],
                        },
                    ],
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    // 全称量词引入
    pub fn forall_intro(&self, var: &str, formula: &Formula) -> ProofNode {
        ProofNode {
            formula: Formula::Forall(var.to_string(), Box::new(formula.clone())),
            rule: "∀I".to_string(),
            premises: vec![ProofNode {
                formula: formula.clone(),
                rule: "assumption".to_string(),
                premises: vec![],
            }],
        }
    }

    // 全称量词消除
    pub fn forall_elim(&self, forall_formula: &Formula, term: &str) -> Option<ProofNode> {
        if let Formula::Forall(var, body) = forall_formula {
            let substituted = self.substitute(body, var, term);
            Some(ProofNode {
                formula: substituted,
                rule: "∀E".to_string(),
                premises: vec![ProofNode {
                    formula: forall_formula.clone(),
                    rule: "assumption".to_string(),
                    premises: vec![],
                }],
            })
        } else {
            None
        }
    }

    // 变量替换
    fn substitute(&self, formula: &Formula, var: &str, term: &str) -> Formula {
        match formula {
            Formula::Atom(name) => {
                if name == var {
                    Formula::Atom(term.to_string())
                } else {
                    formula.clone()
                }
            }
            Formula::Not(f) => Formula::Not(Box::new(self.substitute(f, var, term))),
            Formula::And(f1, f2) => Formula::And(
                Box::new(self.substitute(f1, var, term)),
                Box::new(self.substitute(f2, var, term))
            ),
            Formula::Or(f1, f2) => Formula::Or(
                Box::new(self.substitute(f1, var, term)),
                Box::new(self.substitute(f2, var, term))
            ),
            Formula::Implies(f1, f2) => Formula::Implies(
                Box::new(self.substitute(f1, var, term)),
                Box::new(self.substitute(f2, var, term))
            ),
            Formula::Forall(v, f) => {
                if v == var {
                    formula.clone()
                } else {
                    Formula::Forall(v.clone(), Box::new(self.substitute(f, var, term)))
                }
            }
            Formula::Exists(v, f) => {
                if v == var {
                    formula.clone()
                } else {
                    Formula::Exists(v.clone(), Box::new(self.substitute(f, var, term)))
                }
            }
        }
    }
}

// 序列演算证明器
pub struct SequentProver {
    sequents: Vec<(Vec<Formula>, Vec<Formula>)>,
}

impl SequentProver {
    pub fn new() -> Self {
        SequentProver {
            sequents: Vec::new(),
        }
    }

    // 合取左规则
    pub fn and_left(&self, gamma: &[Formula], delta: &[Formula], a: &Formula, b: &Formula) -> (Vec<Formula>, Vec<Formula>) {
        let mut new_gamma = gamma.to_vec();
        new_gamma.push(Formula::And(Box::new(a.clone()), Box::new(b.clone())));
        (new_gamma, delta.to_vec())
    }

    // 合取右规则
    pub fn and_right(&self, gamma: &[Formula], delta: &[Formula], a: &Formula, b: &Formula) -> Vec<(Vec<Formula>, Vec<Formula>)> {
        vec![
            (gamma.to_vec(), {
                let mut new_delta = delta.to_vec();
                new_delta.push(a.clone());
                new_delta
            }),
            (gamma.to_vec(), {
                let mut new_delta = delta.to_vec();
                new_delta.push(b.clone());
                new_delta
            }),
        ]
    }

    // 蕴含左规则
    pub fn impl_left(&self, gamma: &[Formula], delta: &[Formula], a: &Formula, b: &Formula) -> Vec<(Vec<Formula>, Vec<Formula>)> {
        vec![
            (gamma.to_vec(), {
                let mut new_delta = delta.to_vec();
                new_delta.push(a.clone());
                new_delta
            }),
            ({
                let mut new_gamma = gamma.to_vec();
                new_gamma.push(b.clone());
                new_gamma
            }, delta.to_vec()),
        ]
    }

    // 蕴含右规则
    pub fn impl_right(&self, gamma: &[Formula], delta: &[Formula], a: &Formula, b: &Formula) -> (Vec<Formula>, Vec<Formula>) {
        let mut new_gamma = gamma.to_vec();
        new_gamma.push(a.clone());
        let mut new_delta = delta.to_vec();
        new_delta.push(Formula::Implies(Box::new(a.clone()), Box::new(b.clone())));
        (new_gamma, new_delta)
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

    // 添加子句
    pub fn add_clause(&mut self, literals: Vec<String>) {
        self.clauses.push(literals);
    }

    // 归结
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

    // 检查互补文字
    fn are_complementary(&self, lit1: &str, lit2: &str) -> bool {
        (lit1.starts_with("¬") && lit2 == &lit1[1..]) ||
        (lit2.starts_with("¬") && lit1 == &lit2[1..])
    }

    // 归结证明
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
    fn test_natural_deduction() {
        let prover = NaturalDeductionProver::new();

        // 测试合取引入
        let a = Formula::Atom("A".to_string());
        let b = Formula::Atom("B".to_string());
        let proof = prover.and_intro(&a, &b);

        assert_eq!(proof.rule, "∧I");
        assert_eq!(proof.formula, Formula::And(Box::new(a), Box::new(b)));
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

### Haskell 实现：高级逻辑演算 / Haskell Implementation: Advanced Logical Calculus

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

-- 证明树
data ProofTree where
  Assumption :: Formula -> ProofTree
  AndIntro :: ProofTree -> ProofTree -> ProofTree
  AndElim1 :: ProofTree -> ProofTree
  AndElim2 :: ProofTree -> ProofTree
  OrIntro1 :: ProofTree -> ProofTree
  OrIntro2 :: ProofTree -> ProofTree
  OrElim :: ProofTree -> ProofTree -> ProofTree -> ProofTree
  ImplIntro :: Formula -> ProofTree -> ProofTree
  ImplElim :: ProofTree -> ProofTree -> ProofTree
  NotIntro :: Formula -> ProofTree -> ProofTree
  NotElim :: ProofTree -> ProofTree -> ProofTree
  ForallIntro :: String -> ProofTree -> ProofTree
  ForallElim :: ProofTree -> String -> ProofTree
  ExistsIntro :: ProofTree -> String -> ProofTree
  ExistsElim :: ProofTree -> String -> ProofTree -> ProofTree

-- 自然演绎证明器
class NaturalDeduction a where
  andIntro :: a -> a -> ProofTree
  andElim1 :: a -> ProofTree
  andElim2 :: a -> ProofTree
  implIntro :: Formula -> a -> ProofTree
  implElim :: a -> a -> ProofTree

-- 序列演算
data Sequent = Sequent [Formula] [Formula]

-- 序列演算规则
andLeft :: Sequent -> Formula -> Formula -> Sequent
andLeft (Sequent gamma delta) a b =
  Sequent (Formula::And a b : gamma) delta

andRight :: Sequent -> Formula -> Formula -> [Sequent]
andRight (Sequent gamma delta) a b =
  [Sequent gamma (a : delta), Sequent gamma (b : delta)]

implLeft :: Sequent -> Formula -> Formula -> [Sequent]
implLeft (Sequent gamma delta) a b =
  [Sequent gamma (a : delta), Sequent (b : gamma) delta]

implRight :: Sequent -> Formula -> Formula -> Sequent
implRight (Sequent gamma delta) a b =
  Sequent (a : gamma) (Formula::Implies a b : delta)

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

-- 模型检查
data Model = Model
  { domain :: [String]
  , interpretations :: [(String, [String])]
  }

satisfies :: Model -> Formula -> Bool
satisfies model (Atom p) =
  case lookup p (interpretations model) of
    Just [] -> True
    _ -> False
satisfies model (Not f) = not (satisfies model f)
satisfies model (And f1 f2) = satisfies model f1 && satisfies model f2
satisfies model (Or f1 f2) = satisfies model f1 || satisfies model f2
satisfies model (Implies f1 f2) = not (satisfies model f1) || satisfies model f2
satisfies model (Forall x f) = all (\a -> satisfies (substitute model x a) f) (domain model)
satisfies model (Exists x f) = any (\a -> satisfies (substitute model x a) f) (domain model)

substitute :: Model -> String -> String -> Model
substitute model x a = model -- 简化实现

-- 测试
main :: IO ()
main = do
  let a = Atom "A"
  let b = Atom "B"
  let proof = AndIntro (Assumption a) (Assumption b)

  print "Natural deduction proof:"
  print proof

  let clauses = [[Pos "A", Pos "B"], [Neg "A", Pos "C"], [Neg "B", Pos "C"], [Neg "C"]]
  let result = resolution clauses
  print $ "Resolution result: " ++ show result
```

## 参考文献 / References / Literatur / Références

1. **Gentzen, G.** (1935). _Untersuchungen über das logische Schließen_. Mathematische Zeitschrift.
2. **Prawitz, D.** (1965). _Natural Deduction: A Proof-Theoretical Study_. Almqvist & Wiksell.
3. **Troelstra, A. S. & Schwichtenberg, H.** (2000). _Basic Proof Theory_. Cambridge University Press.
4. **Chang, C. L. & Lee, R. C. T.** (1973). _Symbolic Logic and Mechanical Theorem Proving_. Academic Press.
5. **Robinson, J. A.** (1965). _A Machine-Oriented Logic Based on the Resolution Principle_. Journal of the ACM.

---

_本模块为 FormalAI 提供了严格的逻辑演算基础，确保 AI 系统的形式化推理和证明能力。_

_This module provides FormalAI with rigorous logical calculus foundations, ensuring formal reasoning and proof capabilities of AI systems._

## 相关章节 / Related Chapters

**前置依赖 / Prerequisites:**

- [0.0 ZFC 公理系统](00-set-theory-zfc.md)
- [0.2 类型理论](02-type-theory.md)

**后续依赖 / Follow-ups:**

- [0.5 形式化证明](05-formal-proofs.md)
- [docs/01-foundations/01.1-形式逻辑/README.md](../../01-foundations/01.1-形式逻辑/README.md)

## 2024/2025 最新进展 / Latest Updates

- 表算法与神经符号混合推理的接口（占位）。
- 一阶模型检验在 LLM 事实校验中的应用（占位）。

## Lean 占位模板 / Lean Placeholder

```lean
-- 占位：自然演绎与序列演算的最小规则集
-- 已实现：公式语法与若干推理规则，可靠性草案

-- 公式语法定义
inductive Formula where
  | atom (name : String) : Formula  -- 原子公式
  | not (φ : Formula) : Formula     -- 否定
  | and (φ ψ : Formula) : Formula  -- 合取
  | or (φ ψ : Formula) : Formula   -- 析取
  | impl (φ ψ : Formula) : Formula -- 蕴含
  | forall (var : String) (φ : Formula) : Formula  -- 全称量词
  | exists (var : String) (φ : Formula) : Formula  -- 存在量词

-- 自然演绎规则
inductive ND_Rule where
  | assumption : Formula → ND_Rule  -- 假设规则
  | modus_ponens : Formula → Formula → Formula → ND_Rule  -- 肯定前件
  | and_intro : Formula → Formula → Formula → ND_Rule    -- 合取引入
  | and_elim_left : Formula → Formula → ND_Rule          -- 合取消去（左）
  | and_elim_right : Formula → Formula → ND_Rule         -- 合取消去（右）
  | or_intro_left : Formula → Formula → ND_Rule          -- 析取引入（左）
  | or_intro_right : Formula → Formula → ND_Rule         -- 析取引入（右）
  | not_intro : Formula → Formula → ND_Rule              -- 否定引入
  | not_elim : Formula → Formula → ND_Rule               -- 否定消去
  | forall_intro : String → Formula → Formula → ND_Rule -- 全称引入
  | forall_elim : String → Formula → Formula → Formula → ND_Rule  -- 全称消去

-- 推导关系定义
-- 表示从前提集Γ可以推导出公式φ
inductive Derives : List Formula → Formula → List ND_Rule → Prop where
  | assumption (φ : Formula) : Derives [φ] φ [ND_Rule.assumption φ]
  | modus_ponens (Γ : List Formula) (A B : Formula) (r1 r2 : List ND_Rule) :
    Derives Γ (Formula.impl A B) r1 →
    Derives Γ A r2 →
    Derives Γ B (r1 ++ r2 ++ [ND_Rule.modus_ponens A B (Formula.impl A B)])
  | and_intro (Γ : List Formula) (A B : Formula) (r1 r2 : List ND_Rule) :
    Derives Γ A r1 →
    Derives Γ B r2 →
    Derives Γ (Formula.and A B) (r1 ++ r2 ++ [ND_Rule.and_intro A B (Formula.and A B)])
  | and_elim_left (Γ : List Formula) (A B : Formula) (r : List ND_Rule) :
    Derives Γ (Formula.and A B) r →
    Derives Γ A (r ++ [ND_Rule.and_elim_left (Formula.and A B) A])
  | and_elim_right (Γ : List Formula) (A B : Formula) (r : List ND_Rule) :
    Derives Γ (Formula.and A B) r →
    Derives Γ B (r ++ [ND_Rule.and_elim_right (Formula.and A B) B])

-- 模型定义（简化版）
structure Model where
  domain : Type
  interpretation : String → domain → Bool  -- 谓词解释
  atom_truth : String → Bool  -- 原子公式真值

-- 满足关系定义
inductive Satisfies : Model → Formula → Prop where
  | atom (M : Model) (name : String) :
    M.atom_truth name → Satisfies M (Formula.atom name)
  | not (M : Model) (φ : Formula) :
    ¬ Satisfies M φ → Satisfies M (Formula.not φ)
  | and (M : Model) (φ ψ : Formula) :
    Satisfies M φ → Satisfies M ψ → Satisfies M (Formula.and φ ψ)
  | or_left (M : Model) (φ ψ : Formula) :
    Satisfies M φ → Satisfies M (Formula.or φ ψ)
  | or_right (M : Model) (φ ψ : Formula) :
    Satisfies M ψ → Satisfies M (Formula.or φ ψ)
  | impl (M : Model) (φ ψ : Formula) :
    (Satisfies M φ → Satisfies M ψ) → Satisfies M (Formula.impl φ ψ)

-- 可靠性定理（Soundness）草案
-- 定理：如果公式φ可以从前提集Γ推导出，且模型M满足Γ中的所有公式，则M也满足φ
theorem soundness (Γ : List Formula) (φ : Formula) (derivation : List ND_Rule)
  (M : Model) (h_derives : Derives Γ φ derivation)
  (h_models : ∀ ψ ∈ Γ, Satisfies M ψ) :
  Satisfies M φ := by
  -- 证明思路：
  -- 1. 对推导结构进行归纳
  -- 2. 对每个推理规则，证明如果前提在模型中为真，则结论也为真
  -- 3. 基础情况：假设规则 - 如果φ在Γ中，则由h_models直接得到
  -- 4. 归纳步骤：对每个规则类型进行分情况讨论
  induction h_derives with
  | assumption φ' =>
    -- 基础情况：φ是假设
    have h : φ' ∈ [φ'] := by simp
    have h_satisfies : Satisfies M φ' := h_models φ' h
    exact h_satisfies
  | modus_ponens Γ' A B r1 r2 h1 h2 ih1 ih2 =>
    -- 肯定前件：如果 A → B 和 A 都为真，则 B 为真
    have h_impl : Satisfies M (Formula.impl A B) := ih1 h_models
    have h_A : Satisfies M A := ih2 h_models
    cases h_impl with
    | impl _ _ h_impl_prop =>
      exact h_impl_prop h_A
  | and_intro Γ' A B r1 r2 h1 h2 ih1 ih2 =>
    -- 合取引入：如果 A 和 B 都为真，则 A ∧ B 为真
    have h_A : Satisfies M A := ih1 h_models
    have h_B : Satisfies M B := ih2 h_models
    exact Satisfies.and M A B h_A h_B
  | and_elim_left Γ' A B r h ih =>
    -- 合取消去（左）：如果 A ∧ B 为真，则 A 为真
    have h_and : Satisfies M (Formula.and A B) := ih h_models
    cases h_and with
    | and _ _ h_A _ =>
      exact h_A
  | and_elim_right Γ' A B r h ih =>
    -- 合取消去（右）：如果 A ∧ B 为真，则 B 为真
    have h_and : Satisfies M (Formula.and A B) := ih h_models
    cases h_and with
    | and _ _ _ h_B =>
      exact h_B
```
