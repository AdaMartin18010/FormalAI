# 4.2 形式化语义 / Formal Semantics / Formale Semantik / Sémantique formelle

[返回全局导航](../../GLOBAL_NAVIGATION.md) · [学习路径](../../LEARNING_PATH_DESIGN.md)

## 概述 / Overview

### 1. 语义学基本记号 / Semantic Notation / Semantische Notation / Notation sémantique

- 语义赋值 / Denotation: 设表达式为 $e$，环境为 $\rho$，模型为 $\mathcal{M}$，则

$$\llbracket e \rrbracket_{\rho}^{\mathcal{M}} \in D$$

- 满足关系 / Satisfaction:

$$(\mathcal{M}, g) \vDash \varphi \iff \text{公式 } \varphi \text{ 在模型 } \mathcal{M} \text{ 下由赋值 } g \text{ 满足}$$

- 组合性原理 / Compositionality:

$$\llbracket f(e_1, \ldots, e_n) \rrbracket_{\rho}^{\mathcal{M}} = f^{\mathcal{M}}\big(\llbracket e_1 \rrbracket_{\rho}^{\mathcal{M}}, \ldots, \llbracket e_n \rrbracket_{\rho}^{\mathcal{M}}\big)$$

### 2. Haskell示例：极简Lambda演算求值 / Minimal Lambda Calculus Evaluator / Minimaler Lambda-Kalkül-Auswerter / Évaluateur du lambda-calcul minimal

```haskell
-- 表达式 / Expressions
data Expr = Var String | Lam String Expr | App Expr Expr | Lit Int | Add Expr Expr
  deriving (Eq, Show)

type Env = [(String, Value)]

data Value = VInt Int | VFun (Value -> Value)

-- 求值 / Evaluation (call-by-value)
eval :: Env -> Expr -> Value
eval env (Var x)        = maybe (error ("unbound: " ++ x)) id (lookup x env)
eval env (Lam x body)   = VFun (\v -> eval ((x, v):env) body)
eval env (App e1 e2)    = case eval env e1 of
  VFun f -> f (eval env e2)
  _      -> error "apply non-function"
eval _   (Lit n)        = VInt n
eval env (Add e1 e2)    = case (eval env e1, eval env e2) of
  (VInt a, VInt b) -> VInt (a + b)
  _                -> error "type error"

-- 例 / Example: (\x. x + 1) 41
example :: Value
example = eval [] (App (Lam "x" (Add (Var "x") (Lit 1))) (Lit 41))
```

形式化语义研究自然语言的形式化表示和语义解释，为语言模型提供理论基础。

Formal semantics studies the formal representation and semantic interpretation of natural language, providing theoretical foundations for language models.

## 目录 / Table of Contents

- [4.2 形式化语义 / Formal Semantics / Formale Semantik / Sémantique formelle](#42-形式化语义--formal-semantics--formale-semantik--sémantique-formelle)
  - [概述 / Overview](#概述--overview)
    - [1. 语义学基本记号 / Semantic Notation / Semantische Notation / Notation sémantique](#1-语义学基本记号--semantic-notation--semantische-notation--notation-sémantique)
    - [2. Haskell示例：极简Lambda演算求值 / Minimal Lambda Calculus Evaluator / Minimaler Lambda-Kalkül-Auswerter / Évaluateur du lambda-calcul minimal](#2-haskell示例极简lambda演算求值--minimal-lambda-calculus-evaluator--minimaler-lambda-kalkül-auswerter--évaluateur-du-lambda-calcul-minimal)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes](#相关章节--related-chapters--verwandte-kapitel--chapitres-connexes)
  - [1. 蒙塔古语法 / Montague Grammar](#1-蒙塔古语法--montague-grammar)
    - [1.1 类型驱动语义 / Type-Driven Semantics](#11-类型驱动语义--type-driven-semantics)
    - [1.2 组合性原则 / Principle of Compositionality](#12-组合性原则--principle-of-compositionality)
    - [1.3 内涵语义 / Intensional Semantics](#13-内涵语义--intensional-semantics)
  - [2. 动态语义 / Dynamic Semantics](#2-动态语义--dynamic-semantics)
    - [2.1 话语表示理论 / Discourse Representation Theory](#21-话语表示理论--discourse-representation-theory)
    - [2.2 动态谓词逻辑 / Dynamic Predicate Logic](#22-动态谓词逻辑--dynamic-predicate-logic)
    - [2.3 更新语义 / Update Semantics](#23-更新语义--update-semantics)
  - [3. 分布语义 / Distributional Semantics](#3-分布语义--distributional-semantics)
    - [3.1 向量空间模型 / Vector Space Models](#31-向量空间模型--vector-space-models)
    - [3.2 词嵌入 / Word Embeddings](#32-词嵌入--word-embeddings)
    - [3.3 语义相似度 / Semantic Similarity](#33-语义相似度--semantic-similarity)
  - [4. 神经语义 / Neural Semantics](#4-神经语义--neural-semantics)
    - [4.1 神经语言模型 / Neural Language Models](#41-神经语言模型--neural-language-models)
    - [4.2 语义组合 / Semantic Composition](#42-语义组合--semantic-composition)
    - [4.3 语义表示学习 / Semantic Representation Learning](#43-语义表示学习--semantic-representation-learning)
  - [5. 多模态语义 / Multimodal Semantics](#5-多模态语义--multimodal-semantics)
    - [5.1 视觉-语言对齐 / Vision-Language Alignment](#51-视觉-语言对齐--vision-language-alignment)
    - [5.2 跨模态推理 / Cross-Modal Reasoning](#52-跨模态推理--cross-modal-reasoning)
    - [5.3 多模态融合 / Multimodal Fusion](#53-多模态融合--multimodal-fusion)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：语义解析器](#rust实现语义解析器)
    - [Haskell实现：分布语义模型](#haskell实现分布语义模型)
  - [参考文献 / References](#参考文献--references)
  - [2024/2025 最新进展 / Latest Updates / Neueste Entwicklungen / Derniers développements](#20242025-最新进展--latest-updates--neueste-entwicklungen--derniers-développements)
    - [大模型语义理论 / Large Model Semantic Theory](#大模型语义理论--large-model-semantic-theory)
    - [神经语义前沿 / Neural Semantic Frontiers](#神经语义前沿--neural-semantic-frontiers)
    - [多模态语义理论 / Multimodal Semantic Theory](#多模态语义理论--multimodal-semantic-theory)
    - [语义计算理论 / Semantic Computing Theory](#语义计算理论--semantic-computing-theory)
    - [语义评估理论 / Semantic Evaluation Theory](#语义评估理论--semantic-evaluation-theory)
    - [实用工具链 / Practical Toolchain](#实用工具链--practical-toolchain)
  - [进一步阅读（2025 持续滚动） / Further Reading (Rolling 2025)](#进一步阅读2025-持续滚动--further-reading-rolling-2025)

---

## 相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes

**前置依赖 / Prerequisites / Voraussetzungen / Prérequis:**

- [1.1 形式化逻辑基础](../01-foundations/01-formal-logic/README.md) - 提供逻辑基础 / Provides logical foundation
- [3.3 类型理论](../03-formal-methods/03-type-theory/README.md) - 提供类型基础 / Provides type foundation

**后续应用 / Applications / Anwendungen / Applications:**

- [4.3 知识表示](03-knowledge-representation/README.md) - 提供语义基础 / Provides semantic foundation
- [4.4 推理机制](04-reasoning-mechanisms/README.md) - 提供语义基础 / Provides semantic foundation

---

## 1. 蒙塔古语法 / Montague Grammar

### 1.1 类型驱动语义 / Type-Driven Semantics

**语义类型 / Semantic Types:**

$$\tau ::= e \mid t \mid \langle s, \tau \rangle \mid \langle \tau_1, \tau_2 \rangle$$

其中：

- $e$ 是个体类型 / individual type
- $t$ 是命题类型 / proposition type
- $\langle s, \tau \rangle$ 是内涵类型 / intensional type
- $\langle \tau_1, \tau_2 \rangle$ 是函数类型 / function type

**类型提升 / Type Raising:**

$$\text{lift}(A) = \langle \langle A, t \rangle, t \rangle$$

### 1.2 组合性原则 / Principle of Compositionality

**组合性原则 / Principle of Compositionality:**

复杂表达式的意义是其组成部分意义的函数：

The meaning of a complex expression is a function of the meanings of its parts:

$$\text{meaning}(AB) = f(\text{meaning}(A), \text{meaning}(B))$$

**函数应用 / Function Application:**

$$\frac{A : \langle \alpha, \beta \rangle \quad B : \alpha}{AB : \beta}$$

### 1.3 内涵语义 / Intensional Semantics

**内涵 / Intension:**

$$\text{intension}(A) = \lambda w. \text{extension}(A)(w)$$

其中 $w$ 是可能世界。

where $w$ is a possible world.

**外延 / Extension:**

$$\text{extension}(A)(w) = \text{denotation}(A) \text{ in } w$$

---

## 2. 动态语义 / Dynamic Semantics

### 2.1 话语表示理论 / Discourse Representation Theory

**话语表示结构 / Discourse Representation Structure:**

$$\text{DRS} = \langle U, C \rangle$$

其中：

- $U$ 是话语指称集合 / set of discourse referents
- $C$ 是条件集合 / set of conditions

**DRS构建规则 / DRS Construction Rules:**

$$\frac{\text{NP} : x \quad \text{VP} : P}{\text{S} : \langle \{x\}, \{P(x)\} \rangle}$$

### 2.2 动态谓词逻辑 / Dynamic Predicate Logic

**动态合取 / Dynamic Conjunction:**

$$\phi \land \psi = \lambda g. \exists h (\phi(g, h) \land \psi(h))$$

其中 $g, h$ 是赋值函数。

where $g, h$ are assignment functions.

**动态存在量词 / Dynamic Existential:**

$$\exists x \phi = \lambda g. \exists h \exists a (g[x]h \land \phi(h, a))$$

### 2.3 更新语义 / Update Semantics

**更新函数 / Update Function:**

$$[\phi] : \text{Info} \rightarrow \text{Info}$$

其中 $\text{Info}$ 是信息状态集合。

where $\text{Info}$ is the set of information states.

**信息更新 / Information Update:**

$$s[\phi] = \{w \in s : w \models \phi\}$$

---

## 3. 分布语义 / Distributional Semantics

### 3.1 向量空间模型 / Vector Space Models

**词向量 / Word Vector:**

$$\mathbf{v}_w \in \mathbb{R}^d$$

其中 $d$ 是向量维度。

where $d$ is the vector dimension.

**共现矩阵 / Co-occurrence Matrix:**

$$M_{ij} = \text{count}(w_i, w_j)$$

### 3.2 词嵌入 / Word Embeddings

**Skip-gram目标 / Skip-gram Objective:**

$$\mathcal{L} = \sum_{(w, c) \in D} \log P(c|w)$$

其中：

where:

$$P(c|w) = \frac{\exp(\mathbf{v}_c^T \mathbf{v}_w)}{\sum_{c' \in V} \exp(\mathbf{v}_{c'}^T \mathbf{v}_w)}$$

**负采样 / Negative Sampling:**

$$\mathcal{L} = \sum_{(w, c) \in D} \left[\log \sigma(\mathbf{v}_c^T \mathbf{v}_w) + \sum_{c' \in N} \log \sigma(-\mathbf{v}_{c'}^T \mathbf{v}_w)\right]$$

### 3.3 语义相似度 / Semantic Similarity

**余弦相似度 / Cosine Similarity:**

$$\text{sim}(w_1, w_2) = \frac{\mathbf{v}_{w_1} \cdot \mathbf{v}_{w_2}}{\|\mathbf{v}_{w_1}\| \|\mathbf{v}_{w_2}\|}$$

**语义类比 / Semantic Analogy:**

$$\mathbf{v}_{king} - \mathbf{v}_{man} + \mathbf{v}_{woman} \approx \mathbf{v}_{queen}$$

---

## 4. 神经语义 / Neural Semantics

### 4.1 神经语言模型 / Neural Language Models

**循环神经网络 / Recurrent Neural Network:**

$$h_t = \text{tanh}(W_h h_{t-1} + W_x x_t + b)$$

**注意力机制 / Attention Mechanism:**

$$\alpha_t = \text{softmax}(\text{score}(h_t, h_i))$$

$$c_t = \sum_i \alpha_{t,i} h_i$$

### 4.2 语义组合 / Semantic Composition

**递归神经网络 / Recursive Neural Network:**

$$h = f(W_1 h_l + W_2 h_r + b)$$

其中 $h_l, h_r$ 是左右子节点的表示。

where $h_l, h_r$ are representations of left and right children.

**树LSTM / Tree LSTM:**

$$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$$
$$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$$
$$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \text{tanh}(W_c x_t + U_c h_{t-1} + b_c)$$
$$h_t = o_t \odot \text{tanh}(c_t)$$

### 4.3 语义表示学习 / Semantic Representation Learning

**BERT表示 / BERT Representation:**

$$\text{BERT}(x) = \text{Transformer}(\text{Embed}(x) + \text{PE}(x))$$

其中 $\text{PE}$ 是位置编码。

where $\text{PE}$ is positional encoding.

**语义角色标注 / Semantic Role Labeling:**

$$P(\text{role}_i | \text{context}) = \text{softmax}(W \text{BERT}(\text{context})_i)$$

---

## 5. 多模态语义 / Multimodal Semantics

### 5.1 视觉-语言对齐 / Vision-Language Alignment

**跨模态注意力 / Cross-Modal Attention:**

$$\alpha_{ij} = \frac{\exp(\text{score}(v_i, l_j))}{\sum_k \exp(\text{score}(v_i, l_k))}$$

其中 $v_i$ 是视觉特征，$l_j$ 是语言特征。

where $v_i$ are visual features and $l_j$ are language features.

**对比学习 / Contrastive Learning:**

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(v, l)/\tau)}{\sum_{l' \in N} \exp(\text{sim}(v, l')/\tau)}$$

### 5.2 跨模态推理 / Cross-Modal Reasoning

**视觉问答 / Visual Question Answering:**

$$P(a|q, v) = \text{softmax}(W[\text{BERT}(q); \text{Vision}(v)])$$

**图像描述生成 / Image Captioning:**

$$P(c|v) = \prod_{i=1}^n P(w_i|w_{<i}, v)$$

### 5.3 多模态融合 / Multimodal Fusion

**早期融合 / Early Fusion:**

$$f_{\text{early}} = \text{MLP}([\text{vision}; \text{language}])$$

**晚期融合 / Late Fusion:**

$$f_{\text{late}} = \alpha \cdot f_{\text{vision}} + (1-\alpha) \cdot f_{\text{language}}$$

**注意力融合 / Attention Fusion:**

$$f_{\text{attention}} = \sum_i \alpha_i \cdot f_i$$

---

## 代码示例 / Code Examples

### Rust实现：语义解析器

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
enum SemanticType {
    Individual,
    Proposition,
    Function(Box<SemanticType>, Box<SemanticType>),
    Intensional(Box<SemanticType>),
}

#[derive(Debug, Clone)]
enum SemanticValue {
    Individual(String),
    Proposition(bool),
    Function(Box<dyn Fn(SemanticValue) -> SemanticValue>),
    Intensional(Box<dyn Fn(String) -> SemanticValue>),
}

#[derive(Debug)]
struct SemanticParser {
    lexicon: HashMap<String, SemanticType>,
    composition_rules: Vec<CompositionRule>,
}

#[derive(Debug)]
struct CompositionRule {
    name: String,
    input_types: Vec<SemanticType>,
    output_type: SemanticType,
    function: Box<dyn Fn(Vec<SemanticValue>) -> SemanticValue>,
}

impl SemanticParser {
    fn new() -> Self {
        let mut parser = SemanticParser {
            lexicon: HashMap::new(),
            composition_rules: Vec::new(),
        };
        
        // 初始化词汇表
        parser.initialize_lexicon();
        parser.initialize_composition_rules();
        
        parser
    }
    
    fn initialize_lexicon(&mut self) {
        // 名词
        self.lexicon.insert("John".to_string(), SemanticType::Individual);
        self.lexicon.insert("Mary".to_string(), SemanticType::Individual);
        
        // 不及物动词
        self.lexicon.insert("sleeps".to_string(), 
            SemanticType::Function(Box::new(SemanticType::Individual), Box::new(SemanticType::Proposition)));
        
        // 及物动词
        self.lexicon.insert("loves".to_string(), 
            SemanticType::Function(
                Box::new(SemanticType::Individual),
                Box::new(SemanticType::Function(Box::new(SemanticType::Individual), Box::new(SemanticType::Proposition)))
            ));
        
        // 形容词
        self.lexicon.insert("tall".to_string(), 
            SemanticType::Function(Box::new(SemanticType::Individual), Box::new(SemanticType::Proposition)));
    }
    
    fn initialize_composition_rules(&mut self) {
        // 函数应用规则
        let function_application = CompositionRule {
            name: "Function Application".to_string(),
            input_types: vec![
                SemanticType::Function(Box::new(SemanticType::Individual), Box::new(SemanticType::Proposition)),
                SemanticType::Individual
            ],
            output_type: SemanticType::Proposition,
            function: Box::new(|args| {
                if args.len() == 2 {
                    // 简化的语义计算
                    SemanticValue::Proposition(true)
                } else {
                    SemanticValue::Proposition(false)
                }
            }),
        };
        
        self.composition_rules.push(function_application);
    }
    
    fn parse(&self, tokens: &[String]) -> Option<SemanticValue> {
        if tokens.len() == 0 {
            return None;
        }
        
        if tokens.len() == 1 {
            // 单个词的情况
            return self.lexicon.get(&tokens[0]).map(|_| {
                SemanticValue::Individual(tokens[0].clone())
            });
        }
        
        // 尝试不同的组合方式
        for i in 1..tokens.len() {
            let left_tokens = &tokens[..i];
            let right_tokens = &tokens[i..];
            
            if let (Some(left_sem), Some(right_sem)) = (self.parse(left_tokens), self.parse(right_tokens)) {
                // 尝试应用组合规则
                if let Some(result) = self.apply_composition_rules(&left_sem, &right_sem) {
                    return Some(result);
                }
            }
        }
        
        None
    }
    
    fn apply_composition_rules(&self, left: &SemanticValue, right: &SemanticValue) -> Option<SemanticValue> {
        for rule in &self.composition_rules {
            if rule.input_types.len() == 2 {
                // 简化的类型检查
                if self.type_check(left, &rule.input_types[0]) && 
                   self.type_check(right, &rule.input_types[1]) {
                    return Some((rule.function)(vec![left.clone(), right.clone()]));
                }
            }
        }
        None
    }
    
    fn type_check(&self, value: &SemanticValue, expected_type: &SemanticType) -> bool {
        match (value, expected_type) {
            (SemanticValue::Individual(_), SemanticType::Individual) => true,
            (SemanticValue::Proposition(_), SemanticType::Proposition) => true,
            _ => false, // 简化版本
        }
    }
    
    fn evaluate_sentence(&self, sentence: &str) -> Option<bool> {
        let tokens: Vec<String> = sentence.split_whitespace().map(|s| s.to_string()).collect();
        
        if let Some(semantic_value) = self.parse(&tokens) {
            match semantic_value {
                SemanticValue::Proposition(b) => Some(b),
                _ => None,
            }
        } else {
            None
        }
    }
}

// 分布语义模型
#[derive(Debug)]
struct DistributionalSemantics {
    word_vectors: HashMap<String, Vec<f64>>,
    vocabulary: Vec<String>,
    vector_dim: usize,
}

impl DistributionalSemantics {
    fn new(dim: usize) -> Self {
        DistributionalSemantics {
            word_vectors: HashMap::new(),
            vocabulary: Vec::new(),
            vector_dim: dim,
        }
    }
    
    fn add_word(&mut self, word: &str, vector: Vec<f64>) {
        if vector.len() == self.vector_dim {
            self.word_vectors.insert(word.to_string(), vector);
            if !self.vocabulary.contains(&word.to_string()) {
                self.vocabulary.push(word.to_string());
            }
        }
    }
    
    fn get_vector(&self, word: &str) -> Option<&Vec<f64>> {
        self.word_vectors.get(word)
    }
    
    fn cosine_similarity(&self, word1: &str, word2: &str) -> Option<f64> {
        if let (Some(v1), Some(v2)) = (self.get_vector(word1), self.get_vector(word2)) {
            let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
            let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
            
            if norm1 > 0.0 && norm2 > 0.0 {
                Some(dot_product / (norm1 * norm2))
            } else {
                Some(0.0)
            }
        } else {
            None
        }
    }
    
    fn find_similar_words(&self, word: &str, top_k: usize) -> Vec<(String, f64)> {
        let mut similarities = Vec::new();
        
        for vocab_word in &self.vocabulary {
            if vocab_word != word {
                if let Some(sim) = self.cosine_similarity(word, vocab_word) {
                    similarities.push((vocab_word.clone(), sim));
                }
            }
        }
        
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(top_k);
        similarities
    }
    
    fn analogy(&self, a: &str, b: &str, c: &str) -> Option<String> {
        if let (Some(va), Some(vb), Some(vc)) = (self.get_vector(a), self.get_vector(b), self.get_vector(c)) {
            let target_vector: Vec<f64> = va.iter().zip(vb.iter()).zip(vc.iter())
                .map(|((a_val, b_val), c_val)| c_val + (b_val - a_val))
                .collect();
            
            let mut best_word = None;
            let mut best_similarity = -1.0;
            
            for word in &self.vocabulary {
                if word != a && word != b && word != c {
                    if let Some(word_vec) = self.get_vector(word) {
                        let similarity = self.cosine_similarity_vectors(&target_vector, word_vec);
                        if similarity > best_similarity {
                            best_similarity = similarity;
                            best_word = Some(word.clone());
                        }
                    }
                }
            }
            
            best_word
        } else {
            None
        }
    }
    
    fn cosine_similarity_vectors(&self, v1: &[f64], v2: &[f64]) -> f64 {
        let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

fn main() {
    // 测试语义解析器
    let mut parser = SemanticParser::new();
    
    let sentences = vec![
        "John sleeps".to_string(),
        "Mary loves John".to_string(),
    ];
    
    println!("语义解析结果:");
    for sentence in sentences {
        match parser.evaluate_sentence(&sentence) {
            Some(result) => println!("'{}' -> {}", sentence, result),
            None => println!("'{}' -> 无法解析", sentence),
        }
    }
    
    // 测试分布语义模型
    let mut ds = DistributionalSemantics::new(3);
    
    // 添加一些示例词向量
    ds.add_word("king", vec![1.0, 0.0, 0.0]);
    ds.add_word("queen", vec![0.0, 1.0, 0.0]);
    ds.add_word("man", vec![0.5, 0.0, 0.5]);
    ds.add_word("woman", vec![0.0, 0.5, 0.5]);
    
    println!("\n分布语义结果:");
    
    // 计算相似度
    if let Some(sim) = ds.cosine_similarity("king", "queen") {
        println!("king 和 queen 的相似度: {:.3}", sim);
    }
    
    // 查找相似词
    let similar = ds.find_similar_words("king", 3);
    println!("与 king 最相似的词:");
    for (word, sim) in similar {
        println!("  {}: {:.3}", word, sim);
    }
    
    // 类比推理
    if let Some(result) = ds.analogy("man", "king", "woman") {
        println!("man : king :: woman : {}", result);
    }
}
```

### Haskell实现：分布语义模型

```haskell
import Data.List (foldl', sortBy)
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Vector (Vector, fromList, (!), length)
import qualified Data.Vector as V

-- 词向量类型
type WordVector = Vector Double
type Vocabulary = [String]
type WordVectors = Map String WordVector

-- 分布语义模型
data DistributionalModel = DistributionalModel {
    wordVectors :: WordVectors,
    vocabulary :: Vocabulary,
    vectorDim :: Int
} deriving Show

-- 创建模型
createModel :: Int -> DistributionalModel
createModel dim = DistributionalModel {
    wordVectors = Map.empty,
    vocabulary = [],
    vectorDim = dim
}

-- 添加词向量
addWordVector :: DistributionalModel -> String -> WordVector -> DistributionalModel
addWordVector model word vector
    | V.length vector == vectorDim model = model {
        wordVectors = Map.insert word vector (wordVectors model),
        vocabulary = if word `elem` vocabulary model 
                    then vocabulary model 
                    else word : vocabulary model
    }
    | otherwise = model

-- 获取词向量
getWordVector :: DistributionalModel -> String -> Maybe WordVector
getWordVector model word = Map.lookup word (wordVectors model)

-- 向量运算
dotProduct :: WordVector -> WordVector -> Double
dotProduct v1 v2 = V.sum $ V.zipWith (*) v1 v2

vectorNorm :: WordVector -> Double
vectorNorm v = sqrt $ V.sum $ V.map (^2) v

cosineSimilarity :: WordVector -> WordVector -> Double
cosineSimilarity v1 v2 =
    let dot = dotProduct v1 v2
        norm1 = vectorNorm v1
        norm2 = vectorNorm v2
    in if norm1 > 0 && norm2 > 0 
       then dot / (norm1 * norm2)
       else 0.0

-- 计算两个词的相似度
wordSimilarity :: DistributionalModel -> String -> String -> Maybe Double
wordSimilarity model word1 word2 = do
    v1 <- getWordVector model word1
    v2 <- getWordVector model word2
    return $ cosineSimilarity v1 v2

-- 查找相似词
findSimilarWords :: DistributionalModel -> String -> Int -> [(String, Double)]
findSimilarWords model word topK =
    let similarities = [(w, sim) | w <- vocabulary model, w /= word,
                                  Just sim <- [wordSimilarity model word w]]
        sorted = sortBy (\a b -> compare (snd b) (snd a)) similarities
    in take topK sorted

-- 向量加减
vectorAdd :: WordVector -> WordVector -> WordVector
vectorAdd v1 v2 = V.zipWith (+) v1 v2

vectorSubtract :: WordVector -> WordVector -> WordVector
vectorSubtract v1 v2 = V.zipWith (-) v1 v2

-- 类比推理
analogy :: DistributionalModel -> String -> String -> String -> Maybe String
analogy model a b c = do
    va <- getWordVector model a
    vb <- getWordVector model b
    vc <- getWordVector model c
    
    let targetVector = vectorAdd vc (vectorSubtract vb va)
        candidates = [(w, sim) | w <- vocabulary model, w /= a, w /= b, w /= c,
                                Just vw <- [getWordVector model w],
                                let sim = cosineSimilarity targetVector vw]
        sorted = sortBy (\x y -> compare (snd y) (snd x)) candidates
    
    case sorted of
        ((word, _):_) -> Just word
        [] -> Nothing

-- 语义聚类
semanticClustering :: DistributionalModel -> [String] -> [[String]]
semanticClustering model words =
    let similarities = [(w1, w2, sim) | w1 <- words, w2 <- words, w1 < w2,
                                       Just sim <- [wordSimilarity model w1 w2]]
        threshold = 0.7  -- 相似度阈值
        similarPairs = [(w1, w2) | (w1, w2, sim) <- similarities, sim > threshold]
    in clusterWords words similarPairs
  where
    clusterWords :: [String] -> [(String, String)] -> [[String]]
    clusterWords [] _ = []
    clusterWords (w:ws) pairs =
        let cluster = w : [w2 | (w1, w2) <- pairs, w1 == w]
            remaining = [w' | w' <- ws, w' `notElem` cluster]
        in cluster : clusterWords remaining pairs

-- 语义空间可视化（简化版）
semanticSpace :: DistributionalModel -> [(String, Double, Double)]
semanticSpace model =
    let referenceWords = take 2 (vocabulary model)
        (ref1, ref2) = case referenceWords of
            (w1:w2:_) -> (w1, w2)
            _ -> ("", "")
    in [(word, x, y) | word <- vocabulary model,
                       Just v <- [getWordVector model word],
                       Just v1 <- [getWordVector model ref1],
                       Just v2 <- [getWordVector model ref2],
                       let x = cosineSimilarity v v1,
                       let y = cosineSimilarity v v2]

-- 语义组合
semanticComposition :: DistributionalModel -> String -> String -> Maybe WordVector
semanticComposition model word1 word2 = do
    v1 <- getWordVector model word1
    v2 <- getWordVector model word2
    -- 简单的加法组合
    return $ vectorAdd v1 v2

-- 语义相似度矩阵
similarityMatrix :: DistributionalModel -> [String] -> [[Double]]
similarityMatrix model words =
    [[fromMaybe 0.0 (wordSimilarity model w1 w2) | w2 <- words] | w1 <- words]

-- 主函数
main :: IO ()
main = do
    let model = createModel 3
        
        -- 添加示例词向量
        model1 = addWordVector model "king" (fromList [1.0, 0.0, 0.0])
        model2 = addWordVector model1 "queen" (fromList [0.0, 1.0, 0.0])
        model3 = addWordVector model2 "man" (fromList [0.5, 0.0, 0.5])
        model4 = addWordVector model3 "woman" (fromList [0.0, 0.5, 0.5])
        model5 = addWordVector model4 "prince" (fromList [0.8, 0.2, 0.0])
        model6 = addWordVector model5 "princess" (fromList [0.2, 0.8, 0.0])
        
        finalModel = model6
    
    putStrLn "分布语义模型示例:"
    
    -- 计算相似度
    putStrLn "\n词相似度:"
    mapM_ (\pair -> do
        let (w1, w2) = pair
        case wordSimilarity finalModel w1 w2 of
            Just sim -> putStrLn $ w1 ++ " 和 " ++ w2 ++ " 的相似度: " ++ show sim
            Nothing -> putStrLn $ "无法计算 " ++ w1 ++ " 和 " ++ w2 ++ " 的相似度"
    ) [("king", "queen"), ("man", "woman"), ("king", "man")]
    
    -- 查找相似词
    putStrLn "\n与 'king' 最相似的词:"
    let similar = findSimilarWords finalModel "king" 3
    mapM_ (\(word, sim) -> putStrLn $ "  " ++ word ++ ": " ++ show sim) similar
    
    -- 类比推理
    putStrLn "\n类比推理:"
    case analogy finalModel "man" "king" "woman" of
        Just result -> putStrLn $ "man : king :: woman : " ++ result
        Nothing -> putStrLn "无法进行类比推理"
    
    case analogy finalModel "prince" "king" "princess" of
        Just result -> putStrLn $ "prince : king :: princess : " ++ result
        Nothing -> putStrLn "无法进行类比推理"
    
    -- 语义聚类
    putStrLn "\n语义聚类:"
    let clusters = semanticClustering finalModel (vocabulary finalModel)
    mapM_ (\cluster -> putStrLn $ "聚类: " ++ show cluster) clusters
    
    putStrLn "\n分布语义模型演示完成！"
```

---

## 参考文献 / References

1. Montague, R. (1973). The proper treatment of quantification in ordinary English. *Approaches to Natural Language*.
2. Kamp, H., & Reyle, U. (1993). *From Discourse to Logic*. Kluwer.
3. Heim, I., & Kratzer, A. (1998). *Semantics in Generative Grammar*. Blackwell.
4. Mikolov, T., et al. (2013). Efficient estimation of word representations in vector space. *ICLR*.
5. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL*.
6. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.

---

## 2024/2025 最新进展 / Latest Updates / Neueste Entwicklungen / Derniers développements

### 大模型语义理论 / Large Model Semantic Theory

**2024年重大进展**:

- **上下文语义**: 研究大语言模型如何利用上下文信息进行语义理解
- **动态语义更新**: 探索模型在对话过程中动态更新语义表示的理论
- **多语言语义对齐**: 研究不同语言之间语义空间的映射和对齐机制

**理论创新**:

- **语义涌现理论**: 探索大规模预训练中语义能力的涌现机制
- **语义组合理论**: 研究复杂语义结构的组合和分解原理
- **语义泛化理论**: 探索模型语义知识的泛化和迁移机制

### 神经语义前沿 / Neural Semantic Frontiers

**前沿发展**:

- **Transformer语义理论**: 深入研究Transformer架构的语义处理机制
- **注意力语义**: 探索注意力机制如何捕获和表示语义关系
- **位置编码语义**: 研究位置信息在语义理解中的作用

**理论突破**:

- **语义表示学习**: 开发更有效的语义表示学习方法
- **语义相似度计算**: 研究更准确的语义相似度计算理论
- **语义推理**: 探索基于神经网络的语义推理机制

### 多模态语义理论 / Multimodal Semantic Theory

**2024年发展**:

- **视觉-语言语义**: 研究视觉和语言模态的语义对齐理论
- **跨模态语义映射**: 探索不同模态间语义空间的映射关系
- **多模态语义融合**: 研究多模态信息的语义融合策略

**理论创新**:

- **统一语义空间**: 建立跨模态的统一语义表示理论
- **语义注意力机制**: 设计能够跨模态关注语义信息的注意力
- **多模态语义生成**: 研究同时处理多种模态的语义生成理论

### 语义计算理论 / Semantic Computing Theory

**计算效率发展**:

- **语义计算优化**: 研究语义计算的算法优化和加速技术
- **分布式语义处理**: 探索大规模语义计算的分布式处理理论
- **实时语义理解**: 研究低延迟语义理解的理论和方法

**理论突破**:

- **语义压缩**: 开发语义表示的压缩和量化理论
- **语义缓存**: 研究语义计算结果的缓存和复用机制
- **语义并行化**: 探索语义计算的并行化策略

### 语义评估理论 / Semantic Evaluation Theory

**评估方法创新**:

- **语义质量评估**: 建立评估语义表示质量的标准化指标
- **语义一致性检验**: 研究检验语义表示一致性的理论方法
- **语义鲁棒性测试**: 探索语义系统的鲁棒性评估框架

**理论发展**:

- **语义基准测试**: 开发标准化的语义理解基准测试集
- **语义错误分析**: 研究语义理解错误的分类和分析理论
- **语义改进策略**: 探索基于评估结果的语义系统改进方法

### 实用工具链 / Practical Toolchain

**2024年工具发展**:

- **语义解析工具**: 提供多种语义解析算法的实现和比较
- **语义可视化**: 开发语义空间和语义关系的可视化工具
- **语义调试**: 提供语义理解过程的调试和分析工具
- **语义API**: 建立标准化的语义计算API接口

---

*本模块为FormalAI提供了形式化语义的理论基础，涵盖了从蒙塔古语法到神经语义的各个方面，为语言模型的语义理解提供了数学工具。*

---

## 进一步阅读（2025 持续滚动） / Further Reading (Rolling 2025)

- 年度权威索引：见 `docs/LATEST_UPDATES_INDEX.md` 的“权威索引（2025 持续滚动）”
- 来源类别锚点：
  - 顶尖大学课程：MIT/Stanford/CMU/Berkeley/Harvard（形式语义、蒙塔古语法、动态语义、分布语义）
  - A类会议/期刊：ACL/EMNLP/NAACL/COLING/NeurIPS/ICLR 等
  - 标准与基准：NIST、ISO/IEC、W3C；术语/评测/可复现协议
  - 长期综述：Survey/Blueprint/Position（以期刊或arXiv正式版为准）

注：二手资料以一手论文与标准为准；在引用处标注版本/日期。
