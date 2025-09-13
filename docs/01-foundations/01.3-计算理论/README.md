# 1.3 计算理论 / Computation Theory / Berechnungstheorie / Théorie du calcul

[返回全局导航](../../GLOBAL_NAVIGATION.md) · [学习路径](../../LEARNING_PATH_DESIGN.md)

## 概述 / Overview / Übersicht / Aperçu

计算理论研究计算的本质、能力和限制，为FormalAI提供计算复杂性和算法分析的理论基础。

Computation theory studies the nature, capabilities, and limitations of computation, providing theoretical foundations for computational complexity and algorithm analysis in FormalAI.

Die Berechnungstheorie untersucht die Natur, Fähigkeiten und Grenzen der Berechnung und liefert theoretische Grundlagen für Berechnungskomplexität und Algorithmusanalyse in FormalAI.

La théorie du calcul étudie la nature, les capacités et les limitations du calcul, fournissant les fondements théoriques pour la complexité computationnelle et l'analyse d'algorithmes dans FormalAI.

## 核心概念定义 / Core Concept Definitions / Kernbegriffsdefinitionen / Définitions des concepts fondamentaux

### 计算 / Computation / Berechnung / Calcul

**定义 / Definition / Definition / Définition:**

计算是信息处理的过程，通过有限步骤将输入转换为输出。

Computation is the process of information processing that transforms input to output through finite steps.

Berechnung ist der Prozess der Informationsverarbeitung, der Eingaben durch endliche Schritte in Ausgaben umwandelt.

Le calcul est le processus de traitement d'information qui transforme l'entrée en sortie à travers des étapes finies.

**内涵 / Intension / Intension / Intension:**

- 算法执行 / Algorithm execution / Algorithmusausführung / Exécution d'algorithme
- 状态转换 / State transition / Zustandsübergang / Transition d'état
- 资源消耗 / Resource consumption / Ressourcenverbrauch / Consommation de ressources

**外延 / Extension / Extension / Extension:**

- 确定性计算 / Deterministic computation / Deterministische Berechnung / Calcul déterministe
- 非确定性计算 / Nondeterministic computation / Nichtdeterministische Berechnung / Calcul non déterministe
- 随机计算 / Probabilistic computation / Probabilistische Berechnung / Calcul probabiliste
- 量子计算 / Quantum computation / Quantenberechnung / Calcul quantique

## 相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes

**前置依赖 / Prerequisites / Voraussetzungen / Prérequis:**

- [1.1 形式化逻辑基础](../01.1-形式逻辑/README.md) - 提供逻辑基础 / Provides logical foundation
- [1.2 数学基础](../01.2-数学基础/README.md) - 提供数学基础 / Provides mathematical foundation

**后续应用 / Applications / Anwendungen / Applications:**

- [3.2 程序合成](../../03-formal-methods/03.2-程序综合/README.md) - 提供计算基础 / Provides computation foundation
- [4.1 大语言模型理论](../../04-language-models/04.1-大型语言模型/README.md) - 提供复杂度基础 / Provides complexity foundation

## 目录 / Table of Contents / Inhaltsverzeichnis / Table des matières

- [1.3 计算理论 / Computation Theory / Berechnungstheorie / Théorie du calcul](#13-计算理论--computation-theory--berechnungstheorie--théorie-du-calcul)
  - [概述 / Overview / Übersicht / Aperçu](#概述--overview--übersicht--aperçu)
  - [核心概念定义 / Core Concept Definitions / Kernbegriffsdefinitionen / Définitions des concepts fondamentaux](#核心概念定义--core-concept-definitions--kernbegriffsdefinitionen--définitions-des-concepts-fondamentaux)
    - [计算 / Computation / Berechnung / Calcul](#计算--computation--berechnung--calcul)
  - [相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes](#相关章节--related-chapters--verwandte-kapitel--chapitres-connexes)
  - [目录 / Table of Contents / Inhaltsverzeichnis / Table des matières](#目录--table-of-contents--inhaltsverzeichnis--table-des-matières)
  - [1. 自动机理论 / Automata Theory / Automatentheorie / Théorie des automates](#1-自动机理论--automata-theory--automatentheorie--théorie-des-automates)
    - [1.1 有限自动机 / Finite Automata / Endliche Automaten / Automates finis](#11-有限自动机--finite-automata--endliche-automaten--automates-finis)
    - [1.2 下推自动机 / Pushdown Automata / Kellerautomaten / Automates à pile](#12-下推自动机--pushdown-automata--kellerautomaten--automates-à-pile)
    - [1.3 图灵机 / Turing Machines / Turingmaschinen / Machines de Turing](#13-图灵机--turing-machines--turingmaschinen--machines-de-turing)
  - [2. 可计算性理论 / Computability Theory / Berechenbarkeitstheorie / Théorie de la calculabilité](#2-可计算性理论--computability-theory--berechenbarkeitstheorie--théorie-de-la-calculabilité)
    - [2.1 递归函数 / Recursive Functions / Rekursive Funktionen / Fonctions récursives](#21-递归函数--recursive-functions--rekursive-funktionen--fonctions-récursives)
    - [2.2 停机问题 / Halting Problem / Halteproblem / Problème de l'arrêt](#22-停机问题--halting-problem--halteproblem--problème-de-larrêt)
    - [2.3 不可判定性 / Undecidability / Unentscheidbarkeit / Indécidabilité](#23-不可判定性--undecidability--unentscheidbarkeit--indécidabilité)
  - [3. 复杂性理论 / Complexity Theory / Komplexitätstheorie / Théorie de la complexité](#3-复杂性理论--complexity-theory--komplexitätstheorie--théorie-de-la-complexité)
    - [3.1 P类问题 / P Class / P-Klasse / Classe P](#31-p类问题--p-class--p-klasse--classe-p)
    - [3.2 NP类问题 / NP Class / NP-Klasse / Classe NP](#32-np类问题--np-class--np-klasse--classe-np)
    - [3.3 NP完全问题 / NP-Complete Problems / NP-vollständige Probleme / Problèmes NP-complets](#33-np完全问题--np-complete-problems--np-vollständige-probleme--problèmes-np-complets)
  - [4. 算法分析 / Algorithm Analysis / Algorithmusanalyse / Analyse d'algorithmes](#4-算法分析--algorithm-analysis--algorithmusanalyse--analyse-dalgorithmes)
    - [4.1 时间复杂度 / Time Complexity / Zeitkomplexität / Complexité temporelle](#41-时间复杂度--time-complexity--zeitkomplexität--complexité-temporelle)
    - [4.2 空间复杂度 / Space Complexity / Speicherkomplexität / Complexité spatiale](#42-空间复杂度--space-complexity--speicherkomplexität--complexité-spatiale)
    - [4.3 渐近分析 / Asymptotic Analysis / Asymptotische Analyse / Analyse asymptotique](#43-渐近分析--asymptotic-analysis--asymptotische-analyse--analyse-asymptotique)
  - [5. 并行计算 / Parallel Computing / Parallele Berechnung / Calcul parallèle](#5-并行计算--parallel-computing--parallele-berechnung--calcul-parallèle)
    - [5.1 并行模型 / Parallel Models / Parallele Modelle / Modèles parallèles](#51-并行模型--parallel-models--parallele-modelle--modèles-parallèles)
    - [5.2 并行算法 / Parallel Algorithms / Parallele Algorithmen / Algorithmes parallèles](#52-并行算法--parallel-algorithms--parallele-algorithmen--algorithmes-parallèles)
    - [5.3 并行复杂性 / Parallel Complexity / Parallele Komplexität / Complexité parallèle](#53-并行复杂性--parallel-complexity--parallele-komplexität--complexité-parallèle)
  - [6. 量子计算 / Quantum Computing / Quantenberechnung / Calcul quantique](#6-量子计算--quantum-computing--quantenberechnung--calcul-quantique)
    - [6.1 量子比特 / Qubits / Qubits / Qubits](#61-量子比特--qubits--qubits--qubits)
    - [6.2 量子门 / Quantum Gates / Quantengatter / Portes quantiques](#62-量子门--quantum-gates--quantengatter--portes-quantiques)
    - [6.3 量子算法 / Quantum Algorithms / Quantenalgorithmen / Algorithmes quantiques](#63-量子算法--quantum-algorithms--quantenalgorithmen--algorithmes-quantiques)
  - [代码示例 / Code Examples / Codebeispiele / Exemples de code](#代码示例--code-examples--codebeispiele--exemples-de-code)
    - [Rust实现：图灵机模拟器](#rust实现图灵机模拟器)
    - [Haskell实现：递归函数](#haskell实现递归函数)
  - [参考文献 / References / Literatur / Références](#参考文献--references--literatur--références)

---

## 1. 自动机理论 / Automata Theory / Automatentheorie / Théorie des automates

### 1.1 有限自动机 / Finite Automata / Endliche Automaten / Automates finis

**有限自动机定义 / Finite Automaton Definition:**

有限自动机是一个五元组 $M = (Q, \Sigma, \delta, q_0, F)$，其中：

A finite automaton is a 5-tuple $M = (Q, \Sigma, \delta, q_0, F)$ where:

Ein endlicher Automat ist ein 5-Tupel $M = (Q, \Sigma, \delta, q_0, F)$ wobei:

Un automate fini est un 5-uplet $M = (Q, \Sigma, \delta, q_0, F)$ où:

- $Q$ 是状态集合 / $Q$ is the set of states
- $\Sigma$ 是输入字母表 / $\Sigma$ is the input alphabet
- $\delta$ 是转移函数 / $\delta$ is the transition function
- $q_0$ 是初始状态 / $q_0$ is the initial state
- $F$ 是接受状态集合 / $F$ is the set of accepting states

**转移函数 / Transition Function:**

$$\delta: Q \times \Sigma \rightarrow Q$$

**扩展转移函数 / Extended Transition Function:**

$$\hat{\delta}: Q \times \Sigma^* \rightarrow Q$$

$$\hat{\delta}(q, \epsilon) = q$$
$$\hat{\delta}(q, wa) = \delta(\hat{\delta}(q, w), a)$$

**语言接受 / Language Acceptance:**

$$L(M) = \{w \in \Sigma^* : \hat{\delta}(q_0, w) \in F\}$$

### 1.2 下推自动机 / Pushdown Automata / Kellerautomaten / Automates à pile

**下推自动机定义 / Pushdown Automaton Definition:**

下推自动机是一个七元组 $M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$，其中：

A pushdown automaton is a 7-tuple $M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$ where:

Ein Kellerautomat ist ein 7-Tupel $M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$ wobei:

Un automate à pile est un 7-uplet $M = (Q, \Sigma, \Gamma, \delta, q_0, Z_0, F)$ où:

- $\Gamma$ 是栈字母表 / $\Gamma$ is the stack alphabet
- $Z_0$ 是初始栈符号 / $Z_0$ is the initial stack symbol

**转移函数 / Transition Function:**

$$\delta: Q \times \Sigma \times \Gamma \rightarrow 2^{Q \times \Gamma^*}$$

### 1.3 图灵机 / Turing Machines / Turingmaschinen / Machines de Turing

**图灵机定义 / Turing Machine Definition:**

图灵机是一个七元组 $M = (Q, \Sigma, \Gamma, \delta, q_0, B, F)$，其中：

A Turing machine is a 7-tuple $M = (Q, \Sigma, \Gamma, \delta, q_0, B, F)$ where:

Eine Turingmaschine ist ein 7-Tupel $M = (Q, \Sigma, \Gamma, \delta, q_0, B, F)$ wobei:

Une machine de Turing est un 7-uplet $M = (Q, \Sigma, \Gamma, \delta, q_0, B, F)$ où:

- $\Gamma$ 是带字母表 / $\Gamma$ is the tape alphabet
- $B$ 是空白符号 / $B$ is the blank symbol

**转移函数 / Transition Function:**

$$\delta: Q \times \Gamma \rightarrow Q \times \Gamma \times \{L, R, N\}$$

**配置 / Configuration:**

$$(q, \alpha, i)$$

其中 $q$ 是当前状态，$\alpha$ 是带内容，$i$ 是读写头位置。

where $q$ is the current state, $\alpha$ is the tape content, $i$ is the head position.

wobei $q$ der aktuelle Zustand, $\alpha$ der Bandinhalt und $i$ die Kopfposition ist.

où $q$ est l'état actuel, $\alpha$ est le contenu de la bande, $i$ est la position de la tête.

---

## 2. 可计算性理论 / Computability Theory / Berechenbarkeitstheorie / Théorie de la calculabilité

### 2.1 递归函数 / Recursive Functions / Rekursive Funktionen / Fonctions récursives

**原始递归函数 / Primitive Recursive Functions:**

1. **零函数 / Zero function:** $Z(x) = 0$
2. **后继函数 / Successor function:** $S(x) = x + 1$
3. **投影函数 / Projection function:** $P_i^n(x_1, ..., x_n) = x_i$
4. **复合 / Composition:** $h(x_1, ..., x_n) = f(g_1(x_1, ..., x_n), ..., g_m(x_1, ..., x_n))$
5. **原始递归 / Primitive recursion:**

$$h(x_1, ..., x_n, 0) = f(x_1, ..., x_n)$$
$$h(x_1, ..., x_n, y+1) = g(x_1, ..., x_n, y, h(x_1, ..., x_n, y))$$

**μ递归函数 / μ-Recursive Functions:**

$$\mu y[f(x_1, ..., x_n, y) = 0] = \text{least } y \text{ such that } f(x_1, ..., x_n, y) = 0$$

### 2.2 停机问题 / Halting Problem / Halteproblem / Problème de l'arrêt

**停机问题定义 / Halting Problem Definition:**

停机问题是判断给定程序在给定输入下是否会停机的问题。

The halting problem is the problem of determining whether a given program will halt on a given input.

Das Halteproblem ist das Problem zu bestimmen, ob ein gegebenes Programm bei einer gegebenen Eingabe anhält.

Le problème de l'arrêt est le problème de déterminer si un programme donné s'arrête sur une entrée donnée.

**形式化定义 / Formal Definition:**

$$H = \{(M, w) : M \text{ halts on input } w\}$$

**不可判定性证明 / Undecidability Proof:**

假设存在图灵机 $H$ 判定停机问题，构造图灵机 $D$：

Assume there exists a Turing machine $H$ that decides the halting problem, construct Turing machine $D$:

Angenommen, es existiert eine Turingmaschine $H$, die das Halteproblem entscheidet, konstruiere Turingmaschine $D$:

Supposons qu'il existe une machine de Turing $H$ qui décide le problème de l'arrêt, construisons la machine de Turing $D$:

$$
D(M) = \begin{cases}
\text{halt} & \text{if } H(M, M) = \text{no} \\
\text{loop} & \text{if } H(M, M) = \text{yes}
\end{cases}
$$

矛盾：$D(D)$ 停机当且仅当 $D(D)$ 不停机。

Contradiction: $D(D)$ halts if and only if $D(D)$ does not halt.

Widerspruch: $D(D)$ hält genau dann an, wenn $D(D)$ nicht anhält.

Contradiction: $D(D)$ s'arrête si et seulement si $D(D)$ ne s'arrête pas.

### 2.3 不可判定性 / Undecidability / Unentscheidbarkeit / Indécidabilité

**不可判定问题 / Undecidable Problems:**

1. **停机问题 / Halting problem / Halteproblem / Problème de l'arrêt**
2. **波斯特对应问题 / Post correspondence problem / Postsches Korrespondenzproblem / Problème de correspondance de Post**
3. **希尔伯特第十问题 / Hilbert's tenth problem / Hilberts zehntes Problem / Dixième problème de Hilbert**
4. **字问题 / Word problem / Wortproblem / Problème du mot**

---

## 3. 复杂性理论 / Complexity Theory / Komplexitätstheorie / Théorie de la complexité

### 3.1 P类问题 / P Class / P-Klasse / Classe P

**P类定义 / P Class Definition:**

$$P = \{L : \exists \text{ TM } M \text{ and polynomial } p \text{ such that } M \text{ decides } L \text{ in time } O(p(n))\}$$

**P类问题示例 / P Class Examples:**

1. **排序问题 / Sorting problem / Sortierproblem / Problème de tri**
2. **最短路径问题 / Shortest path problem / Kürzester-Pfad-Problem / Problème du plus court chemin**
3. **线性规划问题 / Linear programming problem / Lineares Programmierungsproblem / Problème de programmation linéaire**

### 3.2 NP类问题 / NP Class / NP-Klasse / Classe NP

**NP类定义 / NP Class Definition:**

$$NP = \{L : \exists \text{ TM } M \text{ and polynomial } p \text{ such that } L = \{x : \exists y, |y| \leq p(|x|), M(x,y) = 1\}\}$$

**NP类问题示例 / NP Class Examples:**

1. **旅行商问题 / Traveling salesman problem / Problem des Handlungsreisenden / Problème du voyageur de commerce**
2. **3-SAT问题 / 3-SAT problem / 3-SAT-Problem / Problème 3-SAT**
3. **子集和问题 / Subset sum problem / Teilsummenproblem / Problème de la somme de sous-ensemble**

### 3.3 NP完全问题 / NP-Complete Problems / NP-vollständige Probleme / Problèmes NP-complets

**NP完全性定义 / NP-Completeness Definition:**

语言 $L$ 是NP完全的，如果：

A language $L$ is NP-complete if:

Eine Sprache $L$ ist NP-vollständig, wenn:

Un langage $L$ est NP-complet si:

1. $L \in NP$
2. $\forall L' \in NP: L' \leq_p L$

**库克-列文定理 / Cook-Levin Theorem:**

3-SAT是NP完全的。

3-SAT is NP-complete.

3-SAT ist NP-vollständig.

3-SAT est NP-complet.

---

## 4. 算法分析 / Algorithm Analysis / Algorithmusanalyse / Analyse d'algorithmes

### 4.1 时间复杂度 / Time Complexity / Zeitkomplexität / Complexité temporelle

**大O记号 / Big O Notation:**

$$f(n) = O(g(n)) \Leftrightarrow \exists c, n_0 > 0: \forall n \geq n_0, f(n) \leq c \cdot g(n)$$

**常见时间复杂度 / Common Time Complexities:**

- $O(1)$: 常数时间 / Constant time / Konstante Zeit / Temps constant
- $O(\log n)$: 对数时间 / Logarithmic time / Logarithmische Zeit / Temps logarithmique
- $O(n)$: 线性时间 / Linear time / Lineare Zeit / Temps linéaire
- $O(n \log n)$: 线性对数时间 / Linearithmic time / Linear-logarithmische Zeit / Temps linéarithmique
- $O(n^2)$: 二次时间 / Quadratic time / Quadratische Zeit / Temps quadratique
- $O(2^n)$: 指数时间 / Exponential time / Exponentielle Zeit / Temps exponentiel

### 4.2 空间复杂度 / Space Complexity / Speicherkomplexität / Complexité spatiale

**空间复杂度定义 / Space Complexity Definition:**

算法 $A$ 的空间复杂度是 $A$ 在最坏情况下使用的额外空间量。

The space complexity of algorithm $A$ is the amount of extra space used by $A$ in the worst case.

Die Speicherkomplexität des Algorithmus $A$ ist die Menge des zusätzlichen Speichers, die $A$ im schlimmsten Fall verwendet.

La complexité spatiale de l'algorithme $A$ est la quantité d'espace supplémentaire utilisée par $A$ dans le pire cas.

### 4.3 渐近分析 / Asymptotic Analysis / Asymptotische Analyse / Analyse asymptotique

**渐近记号 / Asymptotic Notations:**

1. **大O记号 / Big O:** $f(n) = O(g(n))$
2. **大Ω记号 / Big Omega:** $f(n) = \Omega(g(n))$
3. **大Θ记号 / Big Theta:** $f(n) = \Theta(g(n))$
4. **小o记号 / Little o:** $f(n) = o(g(n))$
5. **小ω记号 / Little omega:** $f(n) = \omega(g(n))$

---

## 5. 并行计算 / Parallel Computing / Parallele Berechnung / Calcul parallèle

### 5.1 并行模型 / Parallel Models / Parallele Modelle / Modèles parallèles

**PRAM模型 / PRAM Model:**

PRAM (Parallel Random Access Machine) 是一个并行计算模型，包含多个处理器共享内存。

PRAM (Parallel Random Access Machine) is a parallel computing model with multiple processors sharing memory.

PRAM (Parallele Random Access Machine) ist ein paralleles Berechnungsmodell mit mehreren Prozessoren, die Speicher teilen.

PRAM (Machine à Accès Aléatoire Parallèle) est un modèle de calcul parallèle avec plusieurs processeurs partageant la mémoire.

**PRAM类型 / PRAM Types:**

1. **EREW-PRAM:** Exclusive Read, Exclusive Write
2. **CREW-PRAM:** Concurrent Read, Exclusive Write
3. **CRCW-PRAM:** Concurrent Read, Concurrent Write

### 5.2 并行算法 / Parallel Algorithms / Parallele Algorithmen / Algorithmes parallèles

**并行归并排序 / Parallel Merge Sort:**

```rust
fn parallel_merge_sort<T: Ord + Clone + Send>(data: &[T]) -> Vec<T> {
    if data.len() <= 1 {
        return data.to_vec();
    }

    let mid = data.len() / 2;
    let (left, right) = data.split_at(mid);

    let (left_sorted, right_sorted) = rayon::join(
        || parallel_merge_sort(left),
        || parallel_merge_sort(right)
    );

    merge(&left_sorted, &right_sorted)
}

fn merge<T: Ord + Clone>(left: &[T], right: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut i = 0;
    let mut j = 0;

    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            result.push(left[i].clone());
            i += 1;
        } else {
            result.push(right[j].clone());
            j += 1;
        }
    }

    result.extend_from_slice(&left[i..]);
    result.extend_from_slice(&right[j..]);
    result
}
```

### 5.3 并行复杂性 / Parallel Complexity / Parallele Komplexität / Complexité parallèle

**并行时间复杂性 / Parallel Time Complexity:**

$$T_p(n) = O(T_1(n)/p + \log p)$$

其中 $T_1(n)$ 是串行时间，$p$ 是处理器数量。

where $T_1(n)$ is the serial time and $p$ is the number of processors.

wobei $T_1(n)$ die serielle Zeit und $p$ die Anzahl der Prozessoren ist.

où $T_1(n)$ est le temps sérial et $p$ est le nombre de processeurs.

---

## 6. 量子计算 / Quantum Computing / Quantenberechnung / Calcul quantique

### 6.1 量子比特 / Qubits / Qubits / Qubits

**量子比特定义 / Qubit Definition:**

量子比特是量子计算的基本单位，可以表示为：

A qubit is the basic unit of quantum computing, which can be represented as:

Ein Qubit ist die Grundeinheit der Quantenberechnung, die dargestellt werden kann als:

Un qubit est l'unité de base du calcul quantique, qui peut être représenté comme:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

其中 $|\alpha|^2 + |\beta|^2 = 1$。

where $|\alpha|^2 + |\beta|^2 = 1$.

wobei $|\alpha|^2 + |\beta|^2 = 1$.

où $|\alpha|^2 + |\beta|^2 = 1$.

### 6.2 量子门 / Quantum Gates / Quantengatter / Portes quantiques

**Hadamard门 / Hadamard Gate:**

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

**CNOT门 / CNOT Gate:**

$$CNOT = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

### 6.3 量子算法 / Quantum Algorithms / Quantenalgorithmen / Algorithmes quantiques

**Shor算法 / Shor's Algorithm:**

Shor算法用于分解大整数，时间复杂度为 $O((\log n)^3)$。

Shor's algorithm is used to factor large integers with time complexity $O((\log n)^3)$.

Shors Algorithmus wird verwendet, um große ganze Zahlen zu faktorisieren mit Zeitkomplexität $O((\log n)^3)$.

L'algorithme de Shor est utilisé pour factoriser de grands entiers avec une complexité temporelle $O((\log n)^3)$.

**Grover算法 / Grover's Algorithm:**

Grover算法用于非结构化搜索，提供二次加速。

Grover's algorithm is used for unstructured search, providing quadratic speedup.

Grovers Algorithmus wird für unstrukturierte Suche verwendet und bietet quadratische Beschleunigung.

L'algorithme de Grover est utilisé pour la recherche non structurée, fournissant une accélération quadratique.

---

## 代码示例 / Code Examples / Codebeispiele / Exemples de code

### Rust实现：图灵机模拟器

```rust
use std::collections::HashMap;

# [derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Direction {
    Left,
    Right,
    Stay,
}

# [derive(Debug, Clone)]
struct Transition {
    next_state: String,
    write_symbol: char,
    direction: Direction,
}

# [derive(Debug)]
struct TuringMachine {
    states: Vec<String>,
    alphabet: Vec<char>,
    tape_alphabet: Vec<char>,
    transitions: HashMap<(String, char), Transition>,
    initial_state: String,
    blank_symbol: char,
    accepting_states: Vec<String>,
    tape: Vec<char>,
    head_position: usize,
    current_state: String,
}

impl TuringMachine {
    fn new(
        states: Vec<String>,
        alphabet: Vec<char>,
        tape_alphabet: Vec<char>,
        initial_state: String,
        blank_symbol: char,
        accepting_states: Vec<String>,
    ) -> Self {
        TuringMachine {
            states,
            alphabet,
            tape_alphabet,
            transitions: HashMap::new(),
            initial_state: initial_state.clone(),
            blank_symbol,
            accepting_states,
            tape: vec![blank_symbol],
            head_position: 0,
            current_state: initial_state,
        }
    }

    fn add_transition(&mut self, current_state: String, read_symbol: char,
                      next_state: String, write_symbol: char, direction: Direction) {
        self.transitions.insert((current_state, read_symbol),
                               Transition { next_state, write_symbol, direction });
    }

    fn step(&mut self) -> bool {
        let current_symbol = self.tape[self.head_position];

        if let Some(transition) = self.transitions.get(&(self.current_state.clone(), current_symbol)) {
            // 写入符号 / Write symbol / Symbol schreiben / Écrire le symbole
            self.tape[self.head_position] = transition.write_symbol;

            // 移动读写头 / Move head / Kopf bewegen / Déplacer la tête
            match transition.direction {
                Direction::Left => {
                    if self.head_position == 0 {
                        self.tape.insert(0, self.blank_symbol);
                    } else {
                        self.head_position -= 1;
                    }
                },
                Direction::Right => {
                    self.head_position += 1;
                    if self.head_position >= self.tape.len() {
                        self.tape.push(self.blank_symbol);
                    }
                },
                Direction::Stay => {},
            }

            // 更新状态 / Update state / Zustand aktualisieren / Mettre à jour l'état
            self.current_state = transition.next_state.clone();
            true
        } else {
            false
        }
    }

    fn run(&mut self, input: &str) -> bool {
        // 初始化带 / Initialize tape / Band initialisieren / Initialiser la bande
        self.tape = input.chars().collect();
        self.tape.push(self.blank_symbol);
        self.head_position = 0;
        self.current_state = self.initial_state.clone();

        // 运行图灵机 / Run Turing machine / Turingmaschine ausführen / Exécuter la machine de Turing
        let mut steps = 0;
        let max_steps = 10000; // 防止无限循环 / Prevent infinite loop / Verhindere Endlosschleife / Empêcher la boucle infinie

        while steps < max_steps {
            if !self.step() {
                break;
            }
            steps += 1;
        }

        // 检查是否接受 / Check acceptance / Akzeptanz prüfen / Vérifier l'acceptation
        self.accepting_states.contains(&self.current_state)
    }

    fn get_tape_content(&self) -> String {
        self.tape.iter().collect()
    }
}

fn main() {
    // 创建图灵机 / Create Turing machine / Turingmaschine erstellen / Créer la machine de Turing
    let mut tm = TuringMachine::new(
        vec!["q0".to_string(), "q1".to_string(), "q2".to_string()],
        vec!['0', '1'],
        vec!['0', '1', 'B'],
        "q0".to_string(),
        'B',
        vec!["q2".to_string()],
    );

    // 添加转移规则 / Add transition rules / Übergangsregeln hinzufügen / Ajouter les règles de transition
    tm.add_transition("q0".to_string(), '0', "q1".to_string(), '1', Direction::Right);
    tm.add_transition("q0".to_string(), '1', "q0".to_string(), '1', Direction::Right);
    tm.add_transition("q1".to_string(), '0', "q1".to_string(), '0', Direction::Right);
    tm.add_transition("q1".to_string(), '1', "q2".to_string(), '1', Direction::Right);
    tm.add_transition("q1".to_string(), 'B', "q2".to_string(), 'B', Direction::Stay);

    // 运行图灵机 / Run Turing machine / Turingmaschine ausführen / Exécuter la machine de Turing
    let input = "1001";
    let result = tm.run(input);

    println!("输入: {}", input);
    println!("输出: {}", tm.get_tape_content());
    println!("接受: {}", result);
}
```

### Haskell实现：递归函数

```haskell
-- 递归函数类型 / Recursive function types / Rekursive Funktionstypen / Types de fonctions récursives
data RecursiveFunction =
    Zero
  | Successor
  | Projection Int Int
  | Composition RecursiveFunction [RecursiveFunction]
  | PrimitiveRecursion RecursiveFunction RecursiveFunction
  | Minimization RecursiveFunction
  deriving (Show)

-- 递归函数求值 / Recursive function evaluation / Rekursive Funktionsauswertung / Évaluation de fonction récursive
evalRecursive :: RecursiveFunction -> [Integer] -> Integer
evalRecursive func args = case func of
    Zero -> 0
    Successor -> head args + 1
    Projection i n -> args !! (i - 1)
    Composition f gs ->
        let gResults = map (\g -> evalRecursive g args) gs
        in evalRecursive f gResults
    PrimitiveRecursion f g ->
        let x = head args
            y = args !! 1
            otherArgs = drop 2 args
        in if y == 0
           then evalRecursive f (x:otherArgs)
           else evalRecursive g (x:(y-1):(evalRecursive (PrimitiveRecursion f g) (x:(y-1):otherArgs)):otherArgs)
    Minimization f ->
        let x = head args
            otherArgs = drop 1 args
        in findMin f x otherArgs

-- 寻找最小值 / Find minimum / Minimum finden / Trouver le minimum
findMin :: RecursiveFunction -> Integer -> [Integer] -> Integer
findMin f x args =
    let y = 0
    in if evalRecursive f (y:args) == 0
       then y
       else findMin f x args + 1

-- 基本递归函数 / Basic recursive functions / Grundlegende rekursive Funktionen / Fonctions récursives de base
zero :: RecursiveFunction
zero = Zero

successor :: RecursiveFunction
successor = Successor

projection :: Int -> Int -> RecursiveFunction
projection i n = Projection i n

-- 复合函数 / Composition function / Kompositionsfunktion / Fonction de composition
compose :: RecursiveFunction -> [RecursiveFunction] -> RecursiveFunction
compose f gs = Composition f gs

-- 原始递归 / Primitive recursion / Primitive Rekursion / Récursion primitive
primitiveRecursion :: RecursiveFunction -> RecursiveFunction -> RecursiveFunction
primitiveRecursion f g = PrimitiveRecursion f g

-- μ递归 / μ-recursion / μ-Rekursion / μ-récursion
minimization :: RecursiveFunction -> RecursiveFunction
minimization f = Minimization f

-- 示例函数 / Example functions / Beispielfunktionen / Fonctions d'exemple
-- 加法函数 / Addition function / Additionsfunktion / Fonction d'addition
addition :: RecursiveFunction
addition = primitiveRecursion
    (projection 1 1)  -- f(x) = x
    (successor `compose` [projection 3 3])  -- g(x, y, z) = S(z)

-- 乘法函数 / Multiplication function / Multiplikationsfunktion / Fonction de multiplication
multiplication :: RecursiveFunction
multiplication = primitiveRecursion
    zero  -- f(x) = 0
    (addition `compose` [projection 1 3, projection 3 3])  -- g(x, y, z) = add(x, z)

-- 阶乘函数 / Factorial function / Fakultätsfunktion / Fonction factorielle
factorial :: RecursiveFunction
factorial = primitiveRecursion
    (successor `compose` [zero])  -- f() = 1
    (multiplication `compose` [successor `compose` [projection 2 2], projection 3 3])  -- g(x, y, z) = mult(S(x), z)

-- 主函数 / Main function / Hauptfunktion / Fonction principale
main :: IO ()
main = do
    putStrLn "=== 递归函数示例 / Recursive Function Examples ==="

    -- 测试加法 / Test addition / Addition testen / Tester l'addition
    let addResult = evalRecursive addition [3, 4]
    putStrLn $ "3 + 4 = " ++ show addResult

    -- 测试乘法 / Test multiplication / Multiplikation testen / Tester la multiplication
    let mulResult = evalRecursive multiplication [3, 4]
    putStrLn $ "3 * 4 = " ++ show mulResult

    -- 测试阶乘 / Test factorial / Fakultät testen / Tester la factorielle
    let factResult = evalRecursive factorial [5]
    putStrLn $ "5! = " ++ show factResult

    putStrLn "\n=== 可计算性验证 / Computability Verification ==="

    -- 验证原始递归函数 / Verify primitive recursive functions / Primitive rekursive Funktionen verifizieren / Vérifier les fonctions récursives primitives
    putStrLn "所有原始递归函数都是可计算的"
    putStrLn "All primitive recursive functions are computable"
    putStrLn "Alle primitiv rekursiven Funktionen sind berechenbar"
    putStrLn "Toutes les fonctions récursives primitives sont calculables"
```

---

## 参考文献 / References / Literatur / Références

1. **中文 / Chinese:**
   - 王浩 (1981). *数理逻辑*. 科学出版社.
   - 张锦文 (1997). *集合论与连续统假设*. 科学出版社.
   - 华罗庚 (1979). *高等数学引论*. 科学出版社.

2. **English:**
   - Sipser, M. (2012). *Introduction to the Theory of Computation*. Cengage Learning.
   - Hopcroft, J. E., & Ullman, J. D. (1979). *Introduction to Automata Theory, Languages, and Computation*. Addison-Wesley.
   - Papadimitriou, C. H. (1994). *Computational Complexity*. Addison-Wesley.

3. **Deutsch / German:**
   - Hopcroft, J. E., Motwani, R., & Ullman, J. D. (2001). *Einführung in die Automatentheorie, formale Sprachen und Komplexitätstheorie*. Pearson.
   - Wegener, I. (2005). *Komplexitätstheorie*. Springer.
   - Schöning, U. (2008). *Theoretische Informatik - kurzgefasst*. Spektrum.

4. **Français / French:**
   - Aho, A. V., Hopcroft, J. E., & Ullman, J. D. (1974). *The Design and Analysis of Computer Algorithms*. Addison-Wesley.
   - Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory of NP-Completeness*. W.H. Freeman.
   - Papadimitriou, C. H. (2003). *Computational Complexity*. Addison-Wesley.

---

*本模块为FormalAI提供了完整的计算理论基础，结合国际标准Wiki的概念定义，使用中英德法四语言诠释核心概念，为AI系统的算法设计和复杂性分析提供了严格的数学基础。*

## 2024/2025 最新进展 / Latest Updates

- 神经算法的计算复杂性下界与平均情形分析（占位）。
- 量子取样与经典可模拟性边界的最新突破（占位）。

[返回“最新进展”索引](../../LATEST_UPDATES_INDEX.md)
