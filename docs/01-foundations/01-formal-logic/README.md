# 1.1 形式化逻辑基础 / Formal Logic Foundations / Grundlagen der formalen Logik / Fondements de la logique formelle

## 概述 / Overview / Übersicht / Aperçu

形式化逻辑是FormalAI的理论基石，为人工智能的推理、验证和形式化方法提供数学基础。

Formal logic serves as the theoretical foundation of FormalAI, providing mathematical basis for reasoning, verification, and formal methods in artificial intelligence.

Die formale Logik dient als theoretische Grundlage von FormalAI und liefert die mathematische Basis für Schlussfolgerungen, Verifikation und formale Methoden in der künstlichen Intelligenz.

La logique formelle sert de fondement théorique à FormalAI, fournissant la base mathématique pour le raisonnement, la vérification et les méthodes formelles en intelligence artificielle.

## 核心概念定义 / Core Concept Definitions / Kernbegriffsdefinitionen / Définitions des concepts fondamentaux

### 形式化逻辑 / Formal Logic / Formale Logik / Logique formelle

**定义 / Definition / Definition / Définition:**

形式化逻辑是研究有效推理形式的数学学科，通过符号系统来精确表达和验证逻辑关系。

Formal logic is the mathematical discipline that studies valid forms of reasoning, using symbolic systems to precisely express and verify logical relationships.

Die formale Logik ist die mathematische Disziplin, die gültige Formen des Schließens untersucht und symbolische Systeme verwendet, um logische Beziehungen präzise auszudrücken und zu verifizieren.

La logique formelle est la discipline mathématique qui étudie les formes valides de raisonnement, utilisant des systèmes symboliques pour exprimer et vérifier précisément les relations logiques.

**内涵 / Intension / Intension / Intension:**

- 符号化表达 / Symbolic expression / Symbolische Darstellung / Expression symbolique
- 形式化推理 / Formal reasoning / Formales Schließen / Raisonnement formel
- 有效性验证 / Validity verification / Gültigkeitsverifikation / Vérification de validité

**外延 / Extension / Extension / Extension:**

- 命题逻辑 / Propositional logic / Aussagenlogik / Logique propositionnelle
- 谓词逻辑 / Predicate logic / Prädikatenlogik / Logique des prédicats
- 模态逻辑 / Modal logic / Modallogik / Logique modale
- 直觉逻辑 / Intuitionistic logic / Intuitionistische Logik / Logique intuitionniste

**属性 / Properties / Eigenschaften / Propriétés:**

- 一致性 / Consistency / Konsistenz / Cohérence
- 完备性 / Completeness / Vollständigkeit / Complétude
- 可靠性 / Soundness / Korrektheit / Correction
- 可判定性 / Decidability / Entscheidbarkeit / Décidabilité

## 相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes

**前置依赖 / Prerequisites / Voraussetzungen / Prérequis:**

- 无 / None / Keine / Aucun

**后续应用 / Applications / Anwendungen / Applications:**

- [3.1 形式化验证](../03-formal-methods/01-formal-verification/README.md) - 提供逻辑基础 / Provides logical foundation
- [3.4 证明系统](../03-formal-methods/04-proof-systems/README.md) - 提供推理基础 / Provides reasoning foundation
- [4.2 形式化语义](../04-language-models/02-formal-semantics/README.md) - 提供语义基础 / Provides semantic foundation
- [6.1 可解释性理论](../06-interpretable-ai/01-interpretability-theory/README.md) - 提供解释基础 / Provides interpretability foundation

## 目录 / Table of Contents / Inhaltsverzeichnis / Table des matières

- [1.1 形式化逻辑基础 / Formal Logic Foundations / Grundlagen der formalen Logik / Fondements de la logique formelle](#11-形式化逻辑基础--formal-logic-foundations--grundlagen-der-formalen-logik--fondements-de-la-logique-formelle)
  - [概述 / Overview / Übersicht / Aperçu](#概述--overview--übersicht--aperçu)
  - [核心概念定义 / Core Concept Definitions / Kernbegriffsdefinitionen / Définitions des concepts fondamentaux](#核心概念定义--core-concept-definitions--kernbegriffsdefinitionen--définitions-des-concepts-fondamentaux)
    - [形式化逻辑 / Formal Logic / Formale Logik / Logique formelle](#形式化逻辑--formal-logic--formale-logik--logique-formelle)
  - [相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes](#相关章节--related-chapters--verwandte-kapitel--chapitres-connexes)
  - [目录 / Table of Contents / Inhaltsverzeichnis / Table des matières](#目录--table-of-contents--inhaltsverzeichnis--table-des-matières)
  - [1. 命题逻辑 / Propositional Logic / Aussagenlogik / Logique propositionnelle](#1-命题逻辑--propositional-logic--aussagenlogik--logique-propositionnelle)
    - [1.1 基本概念 / Basic Concepts / Grundbegriffe / Concepts de base](#11-基本概念--basic-concepts--grundbegriffe--concepts-de-base)
      - [形式化定义 / Formal Definition / Formale Definition / Définition formelle](#形式化定义--formal-definition--formale-definition--définition-formelle)
      - [语义 / Semantics / Semantik / Sémantique](#语义--semantics--semantik--sémantique)
      - [语义论证 / Semantic Arguments / Semantische Argumente / Arguments sémantiques](#语义论证--semantic-arguments--semantische-argumente--arguments-sémantiques)
    - [1.2 推理系统 / Inference Systems / Schlusssysteme / Systèmes d'inférence](#12-推理系统--inference-systems--schlusssysteme--systèmes-dinférence)
      - [自然演绎 / Natural Deduction / Natürliches Schließen / Déduction naturelle](#自然演绎--natural-deduction--natürliches-schließen--déduction-naturelle)
      - [希尔伯特系统 / Hilbert System / Hilbert-System / Système de Hilbert](#希尔伯特系统--hilbert-system--hilbert-system--système-de-hilbert)
      - [证明过程 / Proof Processes / Beweisprozesse / Processus de preuve](#证明过程--proof-processes--beweisprozesse--processus-de-preuve)
    - [1.3 完备性定理 / Completeness Theorem / Vollständigkeitssatz / Théorème de complétude](#13-完备性定理--completeness-theorem--vollständigkeitssatz--théorème-de-complétude)
      - [哥德尔完备性定理 / Gödel's Completeness Theorem / Gödelscher Vollständigkeitssatz / Théorème de complétude de Gödel](#哥德尔完备性定理--gödels-completeness-theorem--gödelscher-vollständigkeitssatz--théorème-de-complétude-de-gödel)
  - [2. 一阶逻辑 / First-Order Logic / Prädikatenlogik / Logique du premier ordre](#2-一阶逻辑--first-order-logic--prädikatenlogik--logique-du-premier-ordre)
    - [2.1 语言结构 / Language Structure / Sprachstruktur / Structure du langage](#21-语言结构--language-structure--sprachstruktur--structure-du-langage)
    - [2.2 项和公式 / Terms and Formulas / Terme und Formeln / Termes et formules](#22-项和公式--terms-and-formulas--terme-und-formeln--termes-et-formules)
    - [2.3 语义 / Semantics / Semantik / Sémantique](#23-语义--semantics--semantik--sémantique)
    - [2.4 推理规则 / Inference Rules / Schlussregeln / Règles d'inférence](#24-推理规则--inference-rules--schlussregeln--règles-dinférence)
  - [代码示例 / Code Examples / Codebeispiele / Exemples de code](#代码示例--code-examples--codebeispiele--exemples-de-code)
    - [Rust实现：命题逻辑求解器](#rust实现命题逻辑求解器)
    - [Haskell实现：自然演绎系统](#haskell实现自然演绎系统)
  - [参考文献 / References / Literatur / Références](#参考文献--references--literatur--références)

---

## 1. 命题逻辑 / Propositional Logic / Aussagenlogik / Logique propositionnelle

### 1.1 基本概念 / Basic Concepts / Grundbegriffe / Concepts de base

**命题逻辑**是研究简单命题之间逻辑关系的数学理论。

**Propositional logic** is the mathematical theory studying logical relationships between simple propositions.

**Aussagenlogik** ist die mathematische Theorie, die logische Beziehungen zwischen einfachen Aussagen untersucht.

**La logique propositionnelle** est la théorie mathématique qui étudie les relations logiques entre propositions simples.

#### 形式化定义 / Formal Definition / Formale Definition / Définition formelle

设 $\mathcal{P}$ 为命题变元集合，命题逻辑的语言 $\mathcal{L}$ 定义为：

Let $\mathcal{P}$ be a set of propositional variables, the language $\mathcal{L}$ of propositional logic is defined as:

Sei $\mathcal{P}$ eine Menge von Aussagenvariablen, die Sprache $\mathcal{L}$ der Aussagenlogik ist definiert als:

Soit $\mathcal{P}$ un ensemble de variables propositionnelles, le langage $\mathcal{L}$ de la logique propositionnelle est défini comme:

$$\mathcal{L} ::= \mathcal{P} \mid \bot \mid \top \mid \neg \phi \mid \phi \land \psi \mid \phi \lor \psi \mid \phi \rightarrow \psi \mid \phi \leftrightarrow \psi$$

#### 语义 / Semantics / Semantik / Sémantique

**真值函数** $\mathcal{V}: \mathcal{P} \rightarrow \{0,1\}$ 的扩展：

**Truth function** extension of $\mathcal{V}: \mathcal{P} \rightarrow \{0,1\}$:

**Wahrheitsfunktion** Erweiterung von $\mathcal{V}: \mathcal{P} \rightarrow \{0,1\}$:

**Fonction de vérité** extension de $\mathcal{V}: \mathcal{P} \rightarrow \{0,1\}$:

$$
\begin{align}
\mathcal{V}(\bot) &= 0 \\
\mathcal{V}(\top) &= 1 \\
\mathcal{V}(\neg \phi) &= 1 - \mathcal{V}(\phi) \\
\mathcal{V}(\phi \land \psi) &= \min(\mathcal{V}(\phi), \mathcal{V}(\psi)) \\
\mathcal{V}(\phi \lor \psi) &= \max(\mathcal{V}(\phi), \mathcal{V}(\psi)) \\
\mathcal{V}(\phi \rightarrow \psi) &= \max(1 - \mathcal{V}(\phi), \mathcal{V}(\psi))
\end{align}
$$

#### 语义论证 / Semantic Arguments / Semantische Argumente / Arguments sémantiques

**语义有效性论证 / Semantic Validity Arguments:**

$$\text{Validity}(\phi) = \forall \mathcal{V}: \mathcal{V}(\phi) = 1$$

**语义一致性论证 / Semantic Consistency Arguments:**

$$\text{Consistency}(\Gamma) = \exists \mathcal{V}: \forall \psi \in \Gamma, \mathcal{V}(\psi) = 1$$

**语义蕴涵论证 / Semantic Entailment Arguments:**

$$\Gamma \models \phi \Leftrightarrow \forall \mathcal{V}: (\forall \psi \in \Gamma, \mathcal{V}(\psi) = 1) \Rightarrow \mathcal{V}(\phi) = 1$$

### 1.2 推理系统 / Inference Systems / Schlusssysteme / Systèmes d'inférence

#### 自然演绎 / Natural Deduction / Natürliches Schließen / Déduction naturelle

**自然演绎规则 / Natural Deduction Rules:**

```rust
#[derive(Debug, Clone)]
enum NaturalDeductionRule {
    // 引入规则 / Introduction Rules / Einführungsregeln / Règles d'introduction
    AndIntro(Box<Proof>, Box<Proof>),
    OrIntroLeft(Box<Proof>),
    OrIntroRight(Box<Proof>),
    ImplicationIntro(Box<Proof>),
    
    // 消去规则 / Elimination Rules / Beseitigungsregeln / Règles d'élimination
    AndElimLeft(Box<Proof>),
    AndElimRight(Box<Proof>),
    OrElim(Box<Proof>, Box<Proof>, Box<Proof>),
    ImplicationElim(Box<Proof>, Box<Proof>),
    
    // 假设规则 / Assumption Rules / Annahmeregeln / Règles d'hypothèse
    Assumption(Formula),
    
    // 矛盾规则 / Contradiction Rules / Widerspruchsregeln / Règles de contradiction
    ContradictionIntro(Box<Proof>, Box<Proof>),
    ContradictionElim(Box<Proof>),
}

impl NaturalDeductionRule {
    fn is_valid(&self) -> bool {
        match self {
            NaturalDeductionRule::AndIntro(p1, p2) => {
                p1.is_valid() && p2.is_valid()
            },
            NaturalDeductionRule::ImplicationIntro(proof) => {
                proof.is_valid()
            },
            NaturalDeductionRule::ImplicationElim(p1, p2) => {
                p1.is_valid() && p2.is_valid()
            },
            _ => true
        }
    }
}
```

#### 希尔伯特系统 / Hilbert System / Hilbert-System / Système de Hilbert

**希尔伯特公理 / Hilbert Axioms:**

$$\text{Ax1}: \phi \rightarrow (\psi \rightarrow \phi)$$
$$\text{Ax2}: (\phi \rightarrow (\psi \rightarrow \chi)) \rightarrow ((\phi \rightarrow \psi) \rightarrow (\phi \rightarrow \chi))$$
$$\text{Ax3}: (\neg \phi \rightarrow \neg \psi) \rightarrow (\psi \rightarrow \phi)$$

**推理规则 / Inference Rule:**

$$\frac{\phi \quad \phi \rightarrow \psi}{\psi} \text{ (MP)}$$

#### 证明过程 / Proof Processes / Beweisprozesse / Processus de preuve

**证明构造算法 / Proof Construction Algorithm:**

```haskell
-- 证明构造 / Proof Construction / Beweiskonstruktion / Construction de preuve
data ProofStep = 
    Axiom Formula
  | ModusPonens ProofStep ProofStep
  | Assumption Formula
  | Discharge Formula ProofStep
  | ConjunctionIntro ProofStep ProofStep
  | ConjunctionElim1 ProofStep
  | ConjunctionElim2 ProofStep
  | DisjunctionIntro1 Formula ProofStep
  | DisjunctionIntro2 Formula ProofStep
  | DisjunctionElim ProofStep ProofStep ProofStep
  | ImplicationIntro Formula ProofStep
  | ImplicationElim ProofStep ProofStep
  | NegationIntro Formula ProofStep
  | NegationElim ProofStep ProofStep
  | ExFalso Formula ProofStep

-- 证明验证 / Proof Verification / Beweisverifikation / Vérification de preuve
verifyProof :: [ProofStep] -> Formula -> Bool
verifyProof steps conclusion = 
    let validSteps = map verifyStep steps
        dischargedAssumptions = collectDischarges steps
        conclusionMatches = checkConclusion steps conclusion
    in all id validSteps && null dischargedAssumptions && conclusionMatches

-- 证明搜索 / Proof Search / Beweissuche / Recherche de preuve
searchProof :: [Formula] -> Formula -> Maybe [ProofStep]
searchProof assumptions goal = 
    case findDirectProof assumptions goal of
        Just proof -> Just proof
        Nothing -> case findContradictionProof assumptions goal of
            Just proof -> Just proof
            Nothing -> findIndirectProof assumptions goal

-- 直接证明搜索 / Direct Proof Search / Direkte Beweissuche / Recherche de preuve directe
findDirectProof :: [Formula] -> Formula -> Maybe [ProofStep]
findDirectProof assumptions goal = 
    let candidates = generateProofCandidates assumptions goal
        validProofs = filter (\p -> verifyProof p goal) candidates
    in case validProofs of
        (proof:_) -> Just proof
        [] -> Nothing

-- 矛盾证明搜索 / Contradiction Proof Search / Widerspruchsbeweissuche / Recherche de preuve par contradiction
findContradictionProof :: [Formula] -> Formula -> Maybe [ProofStep]
findContradictionProof assumptions goal = 
    let negatedGoal = Negation goal
        extendedAssumptions = assumptions ++ [negatedGoal]
        contradiction = findContradiction extendedAssumptions
    in case contradiction of
        Just _ -> Just (constructContradictionProof assumptions goal)
        Nothing -> Nothing

-- 间接证明搜索 / Indirect Proof Search / Indirekte Beweissuche / Recherche de preuve indirecte
findIndirectProof :: [Formula] -> Formula -> Maybe [ProofStep]
findIndirectProof assumptions goal = 
    let equivalentForms = findEquivalentForms goal
        alternativeProofs = map (\form -> findDirectProof assumptions form) equivalentForms
    in case catMaybes alternativeProofs of
        (proof:_) -> Just (convertToOriginalGoal proof goal)
        [] -> Nothing
```

### 1.3 完备性定理 / Completeness Theorem / Vollständigkeitssatz / Théorème de complétude

#### 哥德尔完备性定理 / Gödel's Completeness Theorem / Gödelscher Vollständigkeitssatz / Théorème de complétude de Gödel

**定理陈述 / Theorem Statement:**

对于一阶逻辑，如果 $\Gamma \models \phi$，那么 $\Gamma \vdash \phi$。

For first-order logic, if $\Gamma \models \phi$, then $\Gamma \vdash \phi$.

Für die Prädikatenlogik erster Stufe gilt: Wenn $\Gamma \models \phi$, dann $\Gamma \vdash \phi$.

Pour la logique du premier ordre, si $\Gamma \models \phi$, alors $\Gamma \vdash \phi$.

**证明构造 / Proof Construction:**

```rust
#[derive(Debug, Clone)]
struct CompletenessProof {
    consistent_theory: Theory,
    maximal_consistent_extension: Theory,
    canonical_model: Model,
}

impl CompletenessProof {
    fn construct_canonical_model(&self) -> Model {
        // 1. 构建极大一致理论 / Build maximal consistent theory
        let maximal_theory = self.extend_to_maximal_consistent();
        
        // 2. 构造典范模型 / Construct canonical model
        let domain = self.construct_domain();
        let interpretation = self.construct_interpretation(&maximal_theory);
        
        Model {
            domain,
            interpretation,
        }
    }
    
    fn extend_to_maximal_consistent(&self) -> Theory {
        // 使用Zorn引理构造极大一致理论
        // Use Zorn's lemma to construct maximal consistent theory
        // Verwende das Lemma von Zorn zur Konstruktion maximal konsistenter Theorie
        // Utiliser le lemme de Zorn pour construire une théorie maximale cohérente
        let mut extended_theory = self.consistent_theory.clone();
        let all_formulas = self.enumerate_all_formulas();
        
        for formula in all_formulas {
            if !extended_theory.add_formula(&formula).is_inconsistent() {
                extended_theory.add_formula(&formula);
            }
        }
        
        extended_theory
    }
    
    fn construct_domain(&self) -> Domain {
        // 构造Herbrand域 / Construct Herbrand domain
        // Konstruiere Herbrand-Domäne / Construire le domaine de Herbrand
        let constants = self.collect_constants();
        let functions = self.collect_functions();
        
        Domain::new(constants, functions)
    }
    
    fn construct_interpretation(&self, theory: &Theory) -> Interpretation {
        // 构造解释函数 / Construct interpretation function
        // Konstruiere Interpretationsfunktion / Construire la fonction d'interprétation
        let mut interpretation = Interpretation::new();
        
        for formula in theory.get_formulas() {
            match formula {
                Formula::Atomic(predicate, terms) => {
                    interpretation.add_predicate(predicate.clone(), terms.clone());
                }
                Formula::Equality(term1, term2) => {
                    interpretation.add_equality(term1.clone(), term2.clone());
                }
                _ => {}
            }
        }
        
        interpretation
    }
    
    fn verify_completeness(&self) -> bool {
        // 验证完备性 / Verify completeness
        // Verifiziere Vollständigkeit / Vérifier la complétude
        let canonical_model = self.construct_canonical_model();
        
        for formula in &self.consistent_theory.formulas {
            if !canonical_model.satisfies(formula) {
                return false;
            }
        }
        
        true
    }
}

#[derive(Debug, Clone)]
struct Theory {
    formulas: Vec<Formula>,
    signature: Signature,
}

impl Theory {
    fn add_formula(&mut self, formula: &Formula) -> &mut Self {
        self.formulas.push(formula.clone());
        self
    }
    
    fn is_inconsistent(&self) -> bool {
        // 检查理论是否不一致 / Check if theory is inconsistent
        // Prüfe ob Theorie inkonsistent ist / Vérifier si la théorie est incohérente
        self.formulas.iter().any(|f| matches!(f, Formula::Contradiction))
    }
    
    fn get_formulas(&self) -> &[Formula] {
        &self.formulas
    }
}

#[derive(Debug, Clone)]
struct Model {
    domain: Domain,
    interpretation: Interpretation,
}

impl Model {
    fn satisfies(&self, formula: &Formula) -> bool {
        // 模型满足公式 / Model satisfies formula
        // Modell erfüllt Formel / Modèle satisfait la formule
        match formula {
            Formula::Atomic(predicate, terms) => {
                self.interpretation.evaluate_predicate(predicate, terms)
            }
            Formula::Equality(term1, term2) => {
                self.interpretation.evaluate_equality(term1, term2)
            }
            Formula::Negation(inner) => !self.satisfies(inner),
            Formula::Conjunction(left, right) => {
                self.satisfies(left) && self.satisfies(right)
            }
            Formula::Disjunction(left, right) => {
                self.satisfies(left) || self.satisfies(right)
            }
            Formula::Implication(antecedent, consequent) => {
                !self.satisfies(antecedent) || self.satisfies(consequent)
            }
            Formula::Universal(variable, body) => {
                self.satisfies_universal(variable, body)
            }
            Formula::Existential(variable, body) => {
                self.satisfies_existential(variable, body)
            }
            Formula::Contradiction => false,
            Formula::Tautology => true,
        }
    }
}
```

---

## 2. 一阶逻辑 / First-Order Logic / Prädikatenlogik / Logique du premier ordre

### 2.1 语言结构 / Language Structure / Sprachstruktur / Structure du langage

**一阶语言 / First-Order Language:**

$$\mathcal{L} = \langle \mathcal{C}, \mathcal{F}, \mathcal{P}, \mathcal{V} \rangle$$

其中：

- $\mathcal{C}$ 是常元集合
- $\mathcal{F}$ 是函数符号集合
- $\mathcal{P}$ 是谓词符号集合
- $\mathcal{V}$ 是变元集合

where:

- $\mathcal{C}$ is the set of constants
- $\mathcal{F}$ is the set of function symbols
- $\mathcal{P}$ is the set of predicate symbols
- $\mathcal{V}$ is the set of variables

wobei:

- $\mathcal{C}$ ist die Menge der Konstanten
- $\mathcal{F}$ ist die Menge der Funktionssymbole
- $\mathcal{P}$ ist die Menge der Prädikatsymbole
- $\mathcal{V}$ ist die Menge der Variablen

où:

- $\mathcal{C}$ est l'ensemble des constantes
- $\mathcal{F}$ est l'ensemble des symboles de fonction
- $\mathcal{P}$ est l'ensemble des symboles de prédicat
- $\mathcal{V}$ est l'ensemble des variables

### 2.2 项和公式 / Terms and Formulas / Terme und Formeln / Termes et formules

**项的定义 / Term Definition:**

$$t ::= x \mid c \mid f(t_1, ..., t_n)$$

其中 $x \in \mathcal{V}$, $c \in \mathcal{C}$, $f \in \mathcal{F}$。

**公式的定义 / Formula Definition:**

$$\phi ::= P(t_1, ..., t_n) \mid t_1 = t_2 \mid \neg \phi \mid \phi \land \psi \mid \phi \lor \psi \mid \phi \rightarrow \psi \mid \forall x \phi \mid \exists x \phi$$

### 2.3 语义 / Semantics / Semantik / Sémantique

**结构 / Structure:**

$$\mathcal{A} = \langle A, I \rangle$$

其中 $A$ 是论域，$I$ 是解释函数。

**赋值 / Assignment:**

$$\sigma: \mathcal{V} \rightarrow A$$

**项的解释 / Term Interpretation:**

$$
\sigma^*(t) = \begin{cases}
\sigma(x) & \text{if } t = x \\
I(c) & \text{if } t = c \\
I(f)(\sigma^*(t_1), ..., \sigma^*(t_n)) & \text{if } t = f(t_1, ..., t_n)
\end{cases}
$$

**公式的满足关系 / Satisfaction Relation:**

$$\mathcal{A} \models_\sigma \phi$$

### 2.4 推理规则 / Inference Rules / Schlussregeln / Règles d'inférence

**全称消去 / Universal Elimination:**

$$\frac{\forall x \phi}{\phi[t/x]}$$

**全称引入 / Universal Introduction:**

$$\frac{\phi}{\forall x \phi} \text{ (if } x \text{ not free in assumptions)}$$

**存在引入 / Existential Introduction:**

$$\frac{\phi[t/x]}{\exists x \phi}$$

**存在消去 / Existential Elimination:**

$$\frac{\exists x \phi \quad \phi \vdash \psi}{\psi} \text{ (if } x \text{ not free in } \psi\text{)}$$

---

## 代码示例 / Code Examples / Codebeispiele / Exemples de code

### Rust实现：命题逻辑求解器

```rust
use std::collections::HashMap;

# [derive(Debug, Clone, PartialEq)]
enum Formula {
    Atom(String),
    Not(Box<Formula>),
    And(Box<Formula>, Box<Formula>),
    Or(Box<Formula>, Box<Formula>),
    Implies(Box<Formula>, Box<Formula>),
    Iff(Box<Formula>, Box<Formula>),
}

# [derive(Debug)]
struct PropositionalSolver {
    variables: HashMap<String, bool>,
}

impl PropositionalSolver {
    fn new() -> Self {
        PropositionalSolver {
            variables: HashMap::new(),
        }
    }

    fn evaluate(&self, formula: &Formula) -> bool {
        match formula {
            Formula::Atom(name) => *self.variables.get(name).unwrap_or(&false),
            Formula::Not(f) => !self.evaluate(f),
            Formula::And(f1, f2) => self.evaluate(f1) && self.evaluate(f2),
            Formula::Or(f1, f2) => self.evaluate(f1) || self.evaluate(f2),
            Formula::Implies(f1, f2) => !self.evaluate(f1) || self.evaluate(f2),
            Formula::Iff(f1, f2) => self.evaluate(f1) == self.evaluate(f2),
        }
    }

    fn is_tautology(&self, formula: &Formula) -> bool {
        let variables = self.collect_variables(formula);
        self.check_all_assignments(formula, &variables)
    }

    fn collect_variables(&self, formula: &Formula) -> Vec<String> {
        match formula {
            Formula::Atom(name) => vec![name.clone()],
            Formula::Not(f) => self.collect_variables(f),
            Formula::And(f1, f2) | Formula::Or(f1, f2) |
            Formula::Implies(f1, f2) | Formula::Iff(f1, f2) => {
                let mut vars = self.collect_variables(f1);
                vars.extend(self.collect_variables(f2));
                vars.sort();
                vars.dedup();
                vars
            }
        }
    }

    fn check_all_assignments(&mut self, formula: &Formula, variables: &[String]) -> bool {
        if variables.is_empty() {
            return self.evaluate(formula);
        }

        let var = &variables[0];
        let remaining_vars = &variables[1..];

        // 尝试真值 / Try true value / Versuche Wahrheitswert / Essayer valeur vraie
        self.variables.insert(var.clone(), true);
        let result_true = self.check_all_assignments(formula, remaining_vars);

        // 尝试假值 / Try false value / Versuche Falschwert / Essayer valeur fausse
        self.variables.insert(var.clone(), false);
        let result_false = self.check_all_assignments(formula, remaining_vars);

        result_true && result_false
    }

    fn find_satisfying_assignment(&mut self, formula: &Formula) -> Option<HashMap<String, bool>> {
        let variables = self.collect_variables(formula);
        self.find_satisfying_assignment_recursive(formula, &variables)
    }

    fn find_satisfying_assignment_recursive(&mut self, formula: &Formula, variables: &[String]) -> Option<HashMap<String, bool>> {
        if variables.is_empty() {
            return if self.evaluate(formula) {
                Some(self.variables.clone())
            } else {
                None
            };
        }

        let var = &variables[0];
        let remaining_vars = &variables[1..];

        // 尝试真值 / Try true value / Versuche Wahrheitswert / Essayer valeur vraie
        self.variables.insert(var.clone(), true);
        if let Some(assignment) = self.find_satisfying_assignment_recursive(formula, remaining_vars) {
            return Some(assignment);
        }

        // 尝试假值 / Try false value / Versuche Falschwert / Essayer valeur fausse
        self.variables.insert(var.clone(), false);
        self.find_satisfying_assignment_recursive(formula, remaining_vars)
    }
}

fn main() {
    let mut solver = PropositionalSolver::new();

    // 示例公式 / Example formula / Beispielformel / Formule d'exemple
    let formula = Formula::Implies(
        Box::new(Formula::Atom("p".to_string())),
        Box::new(Formula::Atom("q".to_string()))
    );

    println!("公式: {:?}", formula);
    println!("是重言式: {}", solver.is_tautology(&formula));

    if let Some(assignment) = solver.find_satisfying_assignment(&formula) {
        println!("满足赋值: {:?}", assignment);
    }
}
```

### Haskell实现：自然演绎系统

```haskell
-- 自然演绎系统 / Natural Deduction System / Natürliches Deduktionssystem / Système de déduction naturelle
data Formula =
    Atom String
  | Not Formula
  | And Formula Formula
  | Or Formula Formula
  | Implies Formula Formula
  | Iff Formula Formula
  deriving (Show, Eq)

data Proof =
    Assumption Formula
  | AndIntro Proof Proof
  | AndElimLeft Proof
  | AndElimRight Proof
  | OrIntroLeft Proof
  | OrIntroRight Proof
  | OrElim Proof Proof Proof
  | ImplicationIntro Proof
  | ImplicationElim Proof Proof
  | NotIntro Proof
  | NotElim Proof Proof
  | IffIntro Proof Proof
  | IffElimLeft Proof
  | IffElimRight Proof
  deriving (Show)

-- 证明验证 / Proof Verification / Beweisverifikation / Vérification de preuve
verifyProof :: Proof -> [Formula] -> Formula -> Bool
verifyProof proof assumptions conclusion =
    case proof of
        Assumption f -> f `elem` assumptions

        AndIntro p1 p2 ->
            let (f1, f2) = extractAndFormulas conclusion
            in verifyProof p1 assumptions f1 && verifyProof p2 assumptions f2

        AndElimLeft p ->
            let f = extractLeftFormula conclusion
            in verifyProof p assumptions (And f conclusion)

        AndElimRight p ->
            let f = extractRightFormula conclusion
            in verifyProof p assumptions (And conclusion f)

        OrIntroLeft p ->
            let f = extractLeftFormula conclusion
            in verifyProof p assumptions f

        OrIntroRight p ->
            let f = extractRightFormula conclusion
            in verifyProof p assumptions f

        OrElim p1 p2 p3 ->
            let (f1, f2) = extractOrFormulas conclusion
            in verifyProof p1 assumptions (Or f1 f2) &&
               verifyProof p2 (f1:assumptions) conclusion &&
               verifyProof p3 (f2:assumptions) conclusion

        ImplicationIntro p ->
            let (f1, f2) = extractImpliesFormulas conclusion
            in verifyProof p (f1:assumptions) f2

        ImplicationElim p1 p2 ->
            let (f1, f2) = extractImpliesFormulas conclusion
            in verifyProof p1 assumptions (Implies f1 f2) &&
               verifyProof p2 assumptions f1

-- 证明搜索 / Proof Search / Beweissuche / Recherche de preuve
searchProof :: [Formula] -> Formula -> Maybe Proof
searchProof assumptions goal =
    case goal of
        And f1 f2 ->
            case (searchProof assumptions f1, searchProof assumptions f2) of
                (Just p1, Just p2) -> Just (AndIntro p1 p2)
                _ -> Nothing

        Or f1 f2 ->
            case searchProof assumptions f1 of
                Just p -> Just (OrIntroLeft p)
                Nothing ->
                    case searchProof assumptions f2 of
                        Just p -> Just (OrIntroRight p)
                        Nothing -> Nothing

        Implies f1 f2 ->
            case searchProof (f1:assumptions) f2 of
                Just p -> Just (ImplicationIntro p)
                Nothing -> Nothing

        _ -> searchDirectProof assumptions goal

-- 直接证明搜索 / Direct Proof Search / Direkte Beweissuche / Recherche de preuve directe
searchDirectProof :: [Formula] -> Formula -> Maybe Proof
searchDirectProof assumptions goal =
    case find (\f -> f == goal) assumptions of
        Just _ -> Just (Assumption goal)
        Nothing -> searchBackwardProof assumptions goal

-- 反向证明搜索 / Backward Proof Search / Rückwärtsbeweissuche / Recherche de preuve arrière
searchBackwardProof :: [Formula] -> Formula -> Maybe Proof
searchBackwardProof assumptions goal =
    case goal of
        And f1 f2 ->
            case (searchProof assumptions f1, searchProof assumptions f2) of
                (Just p1, Just p2) -> Just (AndIntro p1 p2)
                _ -> Nothing

        Or f1 f2 ->
            case searchProof assumptions f1 of
                Just p -> Just (OrIntroLeft p)
                Nothing ->
                    case searchProof assumptions f2 of
                        Just p -> Just (OrIntroRight p)
                        Nothing -> Nothing

        Implies f1 f2 ->
            case searchProof (f1:assumptions) f2 of
                Just p -> Just (ImplicationIntro p)
                Nothing -> Nothing

        _ -> Nothing

-- 辅助函数 / Helper Functions / Hilfsfunktionen / Fonctions auxiliaires
extractAndFormulas :: Formula -> (Formula, Formula)
extractAndFormulas (And f1 f2) = (f1, f2)
extractAndFormulas _ = error "Not an And formula"

extractOrFormulas :: Formula -> (Formula, Formula)
extractOrFormulas (Or f1 f2) = (f1, f2)
extractOrFormulas _ = error "Not an Or formula"

extractImpliesFormulas :: Formula -> (Formula, Formula)
extractImpliesFormulas (Implies f1 f2) = (f1, f2)
extractImpliesFormulas _ = error "Not an Implies formula"

extractLeftFormula :: Formula -> Formula
extractLeftFormula (And f1 _) = f1
extractLeftFormula (Or f1 _) = f1
extractLeftFormula _ = error "Not a binary formula"

extractRightFormula :: Formula -> Formula
extractRightFormula (And _ f2) = f2
extractRightFormula (Or _ f2) = f2
extractRightFormula _ = error "Not a binary formula"

-- 主函数 / Main Function / Hauptfunktion / Fonction principale
main :: IO ()
main = do
    let assumptions = [Atom "p", Atom "q"]
    let goal = And (Atom "p") (Atom "q")

    case searchProof assumptions goal of
        Just proof -> do
            putStrLn "找到证明:"
            print proof
            putStrLn $ "证明有效: " ++ show (verifyProof proof assumptions goal)
        Nothing -> putStrLn "未找到证明"
```

---

## 参考文献 / References / Literatur / Références

1. **中文 / Chinese:**
   - 王浩 (1981). *数理逻辑*. 科学出版社.
   - 张锦文 (1997). *集合论与连续统假设*. 科学出版社.

2. **English:**
   - Gödel, K. (1930). Die Vollständigkeit der Axiome des logischen Funktionenkalküls. *Monatshefte für Mathematik und Physik*, 37(1), 349-360.
   - Church, A. (1936). An unsolvable problem of elementary number theory. *American Journal of Mathematics*, 58(2), 345-363.
   - Tarski, A. (1936). Der Wahrheitsbegriff in den formalisierten Sprachen. *Studia Philosophica*, 1, 261-405.

3. **Deutsch / German:**
   - Hilbert, D., & Ackermann, W. (1928). *Grundzüge der theoretischen Logik*. Springer.
   - Gentzen, G. (1935). Untersuchungen über das logische Schließen. *Mathematische Zeitschrift*, 39(1), 176-210.
   - Gödel, K. (1931). Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme I. *Monatshefte für Mathematik und Physik*, 38(1), 173-198.

4. **Français / French:**
   - Bourbaki, N. (1970). *Éléments de mathématique: Théorie des ensembles*. Hermann.
   - Girard, J.-Y. (1987). *Proof Theory and Logical Complexity*. Bibliopolis.
   - Coquand, T., & Huet, G. (1988). The calculus of constructions. *Information and Computation*, 76(2-3), 95-120.

---

*本模块为FormalAI提供了形式化逻辑的完整理论基础，结合国际标准Wiki的概念定义，使用中英德法四语言诠释核心概念，为AI系统的形式化推理和验证提供了严格的数学基础。*
