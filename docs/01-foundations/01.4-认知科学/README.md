# 1.4 认知科学 / Cognitive Science / Kognitionswissenschaft / Sciences cognitives

[返回全局导航](../../GLOBAL_NAVIGATION.md) · [学习路径](../../LEARNING_PATH_DESIGN.md)

## 概述 / Overview / Übersicht / Aperçu

认知科学研究人类智能的机制和过程，为 FormalAI 提供认知建模和智能系统设计的理论基础。

Cognitive science studies the mechanisms and processes of human intelligence, providing theoretical foundations for cognitive modeling and intelligent system design in FormalAI.

Die Kognitionswissenschaft untersucht die Mechanismen und Prozesse der menschlichen Intelligenz und liefert theoretische Grundlagen für kognitive Modellierung und intelligente Systemgestaltung in FormalAI.

Les sciences cognitives étudient les mécanismes et processus de l'intelligence humaine, fournissant les fondements théoriques pour la modélisation cognitive et la conception de systèmes intelligents dans FormalAI.

**权威来源**：[AUTHORITY_REFERENCE_INDEX](../../AUTHORITY_REFERENCE_INDEX.md) §2.10 — [COG-01~05] Schema 理论、认知负荷理论、间隔重复与检索练习。

**前置 Schema**（[COGNITIVE_LEARNING_PATH_OPTIMIZED](../../COGNITIVE_LEARNING_PATH_OPTIMIZED.md)）：01.1 形式逻辑、01.2 数学基础。

**后续 Schema**：02.1 统计学习理论（学习理论）、09.1 AI 哲学、09.2 意识理论。

## 核心概念定义 / Core Concept Definitions / Kernbegriffsdefinitionen / Définitions des concepts fondamentaux

### 认知 / Cognition / Kognition / Cognition

**定义 / Definition / Definition / Définition:**

认知是信息处理、知识获取和智能行为的心智过程。

Cognition is the mental process of information processing, knowledge acquisition, and intelligent behavior.

Kognition ist der mentale Prozess der Informationsverarbeitung, Wissenserwerb und intelligenten Verhaltens.

La cognition est le processus mental de traitement d'information, d'acquisition de connaissances et de comportement intelligent.

**内涵 / Intension / Intension / Intension:**

- 感知处理 / Perceptual processing / Wahrnehmungsverarbeitung / Traitement perceptuel
- 记忆存储 / Memory storage / Gedächtnisspeicherung / Stockage mémoriel
- 推理决策 / Reasoning and decision-making / Schlussfolgerung und Entscheidungsfindung / Raisonnement et prise de décision
- 学习适应 / Learning and adaptation / Lernen und Anpassung / Apprentissage et adaptation

**外延 / Extension / Extension / Extension:**

- 感知认知 / Perceptual cognition / Wahrnehmungskognition / Cognition perceptuelle
- 记忆认知 / Memory cognition / Gedächtniskognition / Cognition mémorielle
- 语言认知 / Linguistic cognition / Sprachkognition / Cognition linguistique
- 空间认知 / Spatial cognition / Raumkognition / Cognition spatiale

**属性 / Properties / Eigenschaften / Propriétés:**

- 并行处理 / Parallel processing / Parallele Verarbeitung / Traitement parallèle
- 层次结构 / Hierarchical structure / Hierarchische Struktur / Structure hiérarchique
- 适应性 / Adaptability / Anpassungsfähigkeit / Adaptabilité
- 涌现性 / Emergence / Emergenz / Émergence

**与本主题相关的 concepts/Philosophy 文档**：跨模块映射见 [PROJECT_CROSS_MODULE_MAPPING](../../../PROJECT_CROSS_MODULE_MAPPING.md)；概念判断树/决策树见 [CONCEPT_DECISION_TREES](../../CONCEPT_DECISION_TREES.md)、[TECHNICAL_SELECTION_DECISION_TREES](../../TECHNICAL_SELECTION_DECISION_TREES.md)；公理-定理推理见 [AXIOM_THEOREM_INFERENCE_TREE](../../AXIOM_THEOREM_INFERENCE_TREE.md)；认知负荷与间隔重复见 [COGNITIVE_LEARNING_PATH_OPTIMIZED](../../COGNITIVE_LEARNING_PATH_OPTIMIZED.md)、[AUTHORITY_REFERENCE_INDEX](../../AUTHORITY_REFERENCE_INDEX.md) COG-01~05。

## 相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes

**前置依赖 / Prerequisites / Voraussetzungen / Prérequis:**

- [1.1 形式化逻辑基础](../01.1-形式逻辑/README.md) - 提供逻辑基础 / Provides logical foundation

**后续应用 / Applications / Anwendungen / Applications:**

- [9.1 AI 哲学](../../09-philosophy-ethics/09.1-AI哲学/README.md) - 提供认知基础 / Provides cognitive foundation
- [9.2 意识理论](../../09-philosophy-ethics/09.2-意识理论/README.md) - 提供意识基础 / Provides consciousness foundation

## 目录 / Table of Contents / Inhaltsverzeichnis / Table des matières

---

## 1. 认知架构 / Cognitive Architecture / Kognitive Architektur / Architecture cognitive

### 1.1 ACT-R 架构 / ACT-R Architecture / ACT-R-Architektur / Architecture ACT-R

**ACT-R 定义 / ACT-R Definition:**:

ACT-R 是一个认知架构，将认知建模为产生式规则系统。

ACT-R is a cognitive architecture that models cognition as a production rule system.

ACT-R ist eine kognitive Architektur, die Kognition als Produktionsregelsystem modelliert.

ACT-R est une architecture cognitive qui modélise la cognition comme un système de règles de production.

**形式化定义 / Formal Definition:**

$$ACT\text{-}R = (P, D, G, M)$$

其中 / where / wobei / où:

- $P$ 是产生式规则集合 / $P$ is the set of production rules
- $D$ 是声明性记忆 / $D$ is declarative memory
- $G$ 是目标栈 / $G$ is the goal stack
- $M$ 是模块集合 / $M$ is the set of modules

**产生式规则 / Production Rules:**

$$p: \text{IF } \text{condition} \text{ THEN } \text{action}$$

**匹配过程 / Matching Process:**

$$
\text{match}(p, \text{state}) = \begin{cases}
\text{true} & \text{if condition matches state} \\
\text{false} & \text{otherwise}
\end{cases}
$$

### 1.2 SOAR 架构 / SOAR Architecture / SOAR-Architektur / Architecture SOAR

**SOAR 定义 / SOAR Definition:**

SOAR 是一个基于问题空间的认知架构。

SOAR is a problem-space based cognitive architecture.

SOAR ist eine problemraum-basierte kognitive Architektur.

SOAR est une architecture cognitive basée sur l'espace de problèmes.

**问题空间 / Problem Space:**

$$\text{ProblemSpace} = (S, O, G)$$

其中 / where / wobei / où:

- $S$ 是状态集合 / $S$ is the set of states
- $O$ 是操作集合 / $O$ is the set of operators
- $G$ 是目标状态 / $G$ is the goal state

**决策过程 / Decision Process:**

$$\text{decision}(s) = \arg\max_{o \in O} \text{utility}(o, s)$$

### 1.3 连接主义架构 / Connectionist Architecture / Konnektionistische Architektur / Architecture connexionniste

**神经网络模型 / Neural Network Model:**

$$\text{NN} = (L, W, \sigma)$$

其中 / where / wobei / où:

- $L$ 是层集合 / $L$ is the set of layers
- $W$ 是权重矩阵 / $W$ is the weight matrix
- $\sigma$ 是激活函数 / $\sigma$ is the activation function

**前向传播 / Forward Propagation:**

$$a^{(l+1)} = \sigma(W^{(l)}a^{(l)} + b^{(l)})$$

---

## 2. 记忆模型 / Memory Models / Gedächtnismodelle / Modèles de mémoire

### 2.1 工作记忆 / Working Memory / Arbeitsgedächtnis / Mémoire de travail

**工作记忆定义 / Working Memory Definition:**

工作记忆是用于临时存储和处理信息的认知系统。

Working memory is the cognitive system for temporary storage and processing of information.

Das Arbeitsgedächtnis ist das kognitive System für temporäre Speicherung und Verarbeitung von Informationen.

La mémoire de travail est le système cognitif pour le stockage temporaire et le traitement d'informations.

**Baddeley 模型 / Baddeley Model:**

$$\text{WM} = (\text{CE}, \text{PL}, \text{VSSP}, \text{EB})$$

其中 / where / wobei / où:

- $\text{CE}$ 是中央执行器 / $\text{CE}$ is the central executive
- $\text{PL}$ 是语音环路 / $\text{PL}$ is the phonological loop
- $\text{VSSP}$ 是视觉空间画板 / $\text{VSSP}$ is the visuospatial sketchpad
- $\text{EB}$ 是情景缓冲器 / $\text{EB}$ is the episodic buffer

**容量限制 / Capacity Limit:**

$$\text{capacity} = 7 \pm 2 \text{ chunks}$$

### 2.2 长期记忆 / Long-term Memory / Langzeitgedächtnis / Mémoire à long terme

**长期记忆分类 / Long-term Memory Classification:**

$$\text{LTM} = \{\text{Declarative}, \text{Procedural}\}$$

**声明性记忆 / Declarative Memory:**

$$\text{Declarative} = \{\text{Semantic}, \text{Episodic}\}$$

**程序性记忆 / Procedural Memory:**

$$\text{Procedural} = \{\text{Skills}, \text{Habits}\}$$

### 2.3 记忆巩固 / Memory Consolidation / Gedächtniskonsolidierung / Consolidation mémorielle

**巩固过程 / Consolidation Process:**

$$\text{consolidate}(m, t) = \text{strengthen}(m) \cdot e^{-\lambda t}$$

其中 / where / wobei / où:

- $m$ 是记忆强度 / $m$ is memory strength
- $t$ 是时间 / $t$ is time
- $\lambda$ 是衰减常数 / $\lambda$ is decay constant

---

## 3. 注意力机制 / Attention Mechanisms / Aufmerksamkeitsmechanismen / Mécanismes d'attention

### 3.1 选择性注意力 / Selective Attention / Selektive Aufmerksamkeit / Attention sélective

**注意力函数 / Attention Function:**

$$\text{attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 / where / wobei / où:

- $Q$ 是查询矩阵 / $Q$ is the query matrix
- $K$ 是键矩阵 / $K$ is the key matrix
- $V$ 是值矩阵 / $V$ is the value matrix
- $d_k$ 是键维度 / $d_k$ is the key dimension

**注意力权重 / Attention Weights:**

$$w_{ij} = \frac{\exp(s_{ij})}{\sum_k \exp(s_{ik})}$$

其中 / where / wobei / où:

$$s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}$$

### 3.2 分配性注意力 / Divided Attention / Geteilte Aufmerksamkeit / Attention divisée

**多任务处理 / Multitasking:**

$$\text{performance}(T_1, T_2) = f(\text{resource\_allocation}(T_1, T_2))$$

**资源竞争 / Resource Competition:**

$$\text{competition}(R_1, R_2) = \frac{R_1 \cdot R_2}{R_1 + R_2}$$

### 3.3 执行注意力 / Executive Attention / Exekutive Aufmerksamkeit / Attention exécutive

**执行控制 / Executive Control:**

$$\text{control}(S) = \text{inhibit}(S) \oplus \text{activate}(S) \oplus \text{shift}(S)$$

---

## 4. 学习理论 / Learning Theory / Lerntheorie / Théorie de l'apprentissage

### 4.1 经典条件反射 / Classical Conditioning / Klassische Konditionierung / Conditionnement classique

**巴甫洛夫条件反射 / Pavlovian Conditioning:**

$$\text{CS} \rightarrow \text{CR}$$

其中 / where / wobei / où:

- $\text{CS}$ 是条件刺激 / $\text{CS}$ is the conditioned stimulus
- $\text{CR}$ 是条件反应 / $\text{CR}$ is the conditioned response

**学习曲线 / Learning Curve:**

$$P(\text{CR}|\text{CS}) = 1 - e^{-\lambda n}$$

其中 / where / wobei / où:

- $n$ 是试验次数 / $n$ is the number of trials
- $\lambda$ 是学习率 / $\lambda$ is the learning rate

### 4.2 操作条件反射 / Operant Conditioning / Operante Konditionierung / Conditionnement opérant

**强化学习 / Reinforcement Learning:**

$$Q(s, a) = Q(s, a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

其中 / where / wobei / où:

- $Q(s, a)$ 是状态-动作值函数 / $Q(s, a)$ is the state-action value function
- $\alpha$ 是学习率 / $\alpha$ is the learning rate
- $\gamma$ 是折扣因子 / $\gamma$ is the discount factor

### 4.3 观察学习 / Observational Learning / Beobachtungslernen / Apprentissage par observation

**Bandura 模型 / Bandura Model:**

$$\text{learning} = f(\text{attention}, \text{retention}, \text{reproduction}, \text{motivation})$$

---

## 5. 决策理论 / Decision Theory / Entscheidungstheorie / Théorie de la décision

### 5.1 期望效用理论 / Expected Utility Theory / Erwartungsnutzentheorie / Théorie de l'utilité espérée

**期望效用 / Expected Utility:**

$$EU(A) = \sum_i p_i \cdot u(x_i)$$

其中 / where / wobei / où:

- $A$ 是行动 / $A$ is the action
- $p_i$ 是概率 / $p_i$ is the probability
- $u(x_i)$ 是效用函数 / $u(x_i)$ is the utility function

**决策规则 / Decision Rule:**

$$A^* = \arg\max_A EU(A)$$

### 5.2 前景理论 / Prospect Theory / Prospekt-Theorie / Théorie des perspectives

**价值函数 / Value Function:**

$$
v(x) = \begin{cases}
x^\alpha & \text{if } x \geq 0 \\
-\lambda(-x)^\beta & \text{if } x < 0
\end{cases}
$$

其中 / where / wobei / où:

- $\alpha, \beta$ 是风险态度参数 / $\alpha, \beta$ are risk attitude parameters
- $\lambda$ 是损失厌恶参数 / $\lambda$ is the loss aversion parameter

**权重函数 / Weighting Function:**

$$\pi(p) = \frac{p^\gamma}{(p^\gamma + (1-p)^\gamma)^{1/\gamma}}$$

### 5.3 启发式决策 / Heuristic Decision Making / Heuristische Entscheidungsfindung / Prise de décision heuristique

**可用性启发式 / Availability Heuristic:**

$$P(A) \propto \text{ease\_of\_retrieval}(A)$$

**代表性启发式 / Representativeness Heuristic:**

$$P(A|B) \approx \text{similarity}(A, B)$$

---

## 代码示例 / Code Examples / Codebeispiele / Exemples de code

### Rust 实现：工作记忆模型

```rust
use std::collections::HashMap;
use std::collections::VecDeque;

# [derive(Debug, Clone)]
struct WorkingMemory {
    central_executive: CentralExecutive,
    phonological_loop: PhonologicalLoop,
    visuospatial_sketchpad: VisuospatialSketchpad,
    episodic_buffer: EpisodicBuffer,
    capacity: usize,
}

# [derive(Debug, Clone)]
struct CentralExecutive {
    current_goals: VecDeque<String>,
    attention_focus: String,
    inhibition_control: f64,
}

# [derive(Debug, Clone)]
struct PhonologicalLoop {
    phonological_store: VecDeque<String>,
    articulatory_control: VecDeque<String>,
    decay_rate: f64,
}

# [derive(Debug, Clone)]
struct VisuospatialSketchpad {
    visual_cache: HashMap<String, VisualObject>,
    inner_scribe: VecDeque<SpatialRelation>,
}

# [derive(Debug, Clone)]
struct EpisodicBuffer {
    integrated_info: VecDeque<IntegratedChunk>,
    binding_capacity: usize,
}

# [derive(Debug, Clone)]
struct VisualObject {
    id: String,
    properties: HashMap<String, String>,
    spatial_position: (f64, f64),
}

# [derive(Debug, Clone)]
struct SpatialRelation {
    object1: String,
    object2: String,
    relation: String,
}

# [derive(Debug, Clone)]
struct IntegratedChunk {
    semantic_content: String,
    episodic_context: String,
    binding_strength: f64,
}

impl WorkingMemory {
    fn new(capacity: usize) -> Self {
        WorkingMemory {
            central_executive: CentralExecutive {
                current_goals: VecDeque::new(),
                attention_focus: String::new(),
                inhibition_control: 1.0,
            },
            phonological_loop: PhonologicalLoop {
                phonological_store: VecDeque::new(),
                articulatory_control: VecDeque::new(),
                decay_rate: 0.1,
            },
            visuospatial_sketchpad: VisuospatialSketchpad {
                visual_cache: HashMap::new(),
                inner_scribe: VecDeque::new(),
            },
            episodic_buffer: EpisodicBuffer {
                integrated_info: VecDeque::new(),
                binding_capacity: capacity / 2,
            },
            capacity,
        }
    }

    fn add_phonological_item(&mut self, item: String) -> bool {
        if self.phonological_loop.phonological_store.len() < self.capacity {
            self.phonological_loop.phonological_store.push_back(item.clone());
            self.phonological_loop.articulatory_control.push_back(item);
            true
        } else {
            false
        }
    }

    fn add_visual_object(&mut self, object: VisualObject) -> bool {
        if self.visuospatial_sketchpad.visual_cache.len() < self.capacity {
            self.visuospatial_sketchpad.visual_cache.insert(object.id.clone(), object);
            true
        } else {
            false
        }
    }

    fn integrate_information(&mut self, semantic: String, episodic: String) -> bool {
        if self.episodic_buffer.integrated_info.len() < self.episodic_buffer.binding_capacity {
            let chunk = IntegratedChunk {
                semantic_content: semantic,
                episodic_context: episodic,
                binding_strength: 1.0,
            };
            self.episodic_buffer.integrated_info.push_back(chunk);
            true
        } else {
            false
        }
    }

    fn decay_phonological_items(&mut self) {
        let decay_factor = 1.0 - self.phonological_loop.decay_rate;
        let new_length = (self.phonological_loop.phonological_store.len() as f64 * decay_factor) as usize;

        while self.phonological_loop.phonological_store.len() > new_length {
            self.phonological_loop.phonological_store.pop_back();
        }

        while self.phonological_loop.articulatory_control.len() > new_length {
            self.phonological_loop.articulatory_control.pop_back();
        }
    }

    fn focus_attention(&mut self, target: String) {
        self.central_executive.attention_focus = target;
    }

    fn add_goal(&mut self, goal: String) {
        self.central_executive.current_goals.push_back(goal);
    }

    fn get_current_goals(&self) -> Vec<String> {
        self.central_executive.current_goals.iter().cloned().collect()
    }

    fn get_phonological_items(&self) -> Vec<String> {
        self.phonological_loop.phonological_store.iter().cloned().collect()
    }

    fn get_visual_objects(&self) -> Vec<VisualObject> {
        self.visuospatial_sketchpad.visual_cache.values().cloned().collect()
    }

    fn get_integrated_info(&self) -> Vec<IntegratedChunk> {
        self.episodic_buffer.integrated_info.iter().cloned().collect()
    }

    fn is_at_capacity(&self) -> bool {
        let total_items = self.phonological_loop.phonological_store.len() +
                         self.visuospatial_sketchpad.visual_cache.len() +
                         self.episodic_buffer.integrated_info.len();
        total_items >= self.capacity
    }
}

// 注意力机制实现 / Attention Mechanism Implementation
# [derive(Debug, Clone)]
struct AttentionMechanism {
    query: Vec<f64>,
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    attention_weights: Vec<f64>,
}

impl AttentionMechanism {
    fn new(query: Vec<f64>, keys: Vec<Vec<f64>>, values: Vec<Vec<f64>>) -> Self {
        AttentionMechanism {
            query,
            keys,
            values,
            attention_weights: Vec::new(),
        }
    }

    fn compute_attention(&mut self) -> Vec<f64> {
        let d_k = self.keys[0].len() as f64;

        // 计算注意力分数 / Compute attention scores / Berechne Aufmerksamkeitsbewertungen / Calculer les scores d'attention
        let mut scores = Vec::new();
        for key in &self.keys {
            let score = self.dot_product(&self.query, key) / d_k.sqrt();
            scores.push(score);
        }

        // 应用softmax / Apply softmax / Wende Softmax an / Appliquer softmax
        self.attention_weights = self.softmax(&scores);

        // 计算加权和 / Compute weighted sum / Berechne gewichtete Summe / Calculer la somme pondérée
        let mut output = vec![0.0; self.values[0].len()];
        for (i, value) in self.values.iter().enumerate() {
            for (j, &val) in value.iter().enumerate() {
                output[j] += self.attention_weights[i] * val;
            }
        }

        output
    }

    fn dot_product(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn softmax(&self, scores: &[f64]) -> Vec<f64> {
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f64> = scores.iter().map(|&x| (x - max_score).exp()).collect();
        let sum_exp = exp_scores.iter().sum::<f64>();
        exp_scores.iter().map(|&x| x / sum_exp).collect()
    }

    fn get_attention_weights(&self) -> &[f64] {
        &self.attention_weights
    }
}

fn main() {
    // 工作记忆示例 / Working memory example / Arbeitsgedächtnis Beispiel / Exemple de mémoire de travail
    let mut wm = WorkingMemory::new(7);

    println!("=== 工作记忆模型示例 / Working Memory Model Example ===");

    // 添加语音信息 / Add phonological information / Füge phonologische Informationen hinzu / Ajouter des informations phonologiques
    wm.add_phonological_item("apple".to_string());
    wm.add_phonological_item("banana".to_string());
    wm.add_phonological_item("cherry".to_string());

    println!("语音项目: {:?}", wm.get_phonological_items());

    // 添加视觉对象 / Add visual objects / Füge visuelle Objekte hinzu / Ajouter des objets visuels
    let apple_visual = VisualObject {
        id: "apple_visual".to_string(),
        properties: HashMap::from([
            ("color".to_string(), "red".to_string()),
            ("shape".to_string(), "round".to_string()),
        ]),
        spatial_position: (10.0, 20.0),
    };

    wm.add_visual_object(apple_visual);

    // 集成信息 / Integrate information / Integriere Informationen / Intégrer des informations
    wm.integrate_information(
        "fruit".to_string(),
        "kitchen_table".to_string()
    );

    println!("集成信息: {:?}", wm.get_integrated_info());
    println!("是否达到容量: {}", wm.is_at_capacity());

    // 注意力机制示例 / Attention mechanism example / Aufmerksamkeitsmechanismus Beispiel / Exemple de mécanisme d'attention
    println!("\n=== 注意力机制示例 / Attention Mechanism Example ===");

    let query = vec![1.0, 2.0, 3.0];
    let keys = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let values = vec![
        vec![0.1, 0.2],
        vec![0.3, 0.4],
        vec![0.5, 0.6],
    ];

    let mut attention = AttentionMechanism::new(query, keys, values);
    let output = attention.compute_attention();

    println!("注意力权重: {:?}", attention.get_attention_weights());
    println!("输出: {:?}", output);
}
```

### Haskell 实现：注意力机制

```haskell
-- 工作记忆类型 / Working memory type / Arbeitsgedächtnistyp / Type mémoire de travail
data WorkingMemory = WorkingMemory {
    centralExecutive :: CentralExecutive,
    phonologicalLoop :: PhonologicalLoop,
    visuospatialSketchpad :: VisuospatialSketchpad,
    episodicBuffer :: EpisodicBuffer,
    capacity :: Int
} deriving (Show)

data CentralExecutive = CentralExecutive {
    currentGoals :: [String],
    attentionFocus :: String,
    inhibitionControl :: Double
} deriving (Show)

data PhonologicalLoop = PhonologicalLoop {
    phonologicalStore :: [String],
    articulatoryControl :: [String],
    decayRate :: Double
} deriving (Show)

data VisuospatialSketchpad = VisuospatialSketchpad {
    visualCache :: [(String, VisualObject)],
    innerScribe :: [SpatialRelation]
} deriving (Show)

data EpisodicBuffer = EpisodicBuffer {
    integratedInfo :: [IntegratedChunk],
    bindingCapacity :: Int
} deriving (Show)

data VisualObject = VisualObject {
    objectId :: String,
    properties :: [(String, String)],
    spatialPosition :: (Double, Double)
} deriving (Show)

data SpatialRelation = SpatialRelation {
    object1 :: String,
    object2 :: String,
    relation :: String
} deriving (Show)

data IntegratedChunk = IntegratedChunk {
    semanticContent :: String,
    episodicContext :: String,
    bindingStrength :: Double
} deriving (Show)

-- 工作记忆操作 / Working memory operations / Arbeitsgedächtnisoperationen / Opérations de mémoire de travail
newWorkingMemory :: Int -> WorkingMemory
newWorkingMemory cap = WorkingMemory {
    centralExecutive = CentralExecutive [] "" 1.0,
    phonologicalLoop = PhonologicalLoop [] [] 0.1,
    visuospatialSketchpad = VisuospatialSketchpad [] [],
    episodicBuffer = EpisodicBuffer [] (cap `div` 2),
    capacity = cap
}

addPhonologicalItem :: WorkingMemory -> String -> WorkingMemory
addPhonologicalItem wm item
    | length (phonologicalStore (phonologicalLoop wm)) < capacity wm =
        wm { phonologicalLoop = (phonologicalLoop wm) {
            phonologicalStore = item : phonologicalStore (phonologicalLoop wm),
            articulatoryControl = item : articulatoryControl (phonologicalLoop wm)
        }}
    | otherwise = wm

addVisualObject :: WorkingMemory -> VisualObject -> WorkingMemory
addVisualObject wm obj
    | length (visualCache (visuospatialSketchpad wm)) < capacity wm =
        wm { visuospatialSketchpad = (visuospatialSketchpad wm) {
            visualCache = (objectId obj, obj) : visualCache (visuospatialSketchpad wm)
        }}
    | otherwise = wm

integrateInformation :: WorkingMemory -> String -> String -> WorkingMemory
integrateInformation wm semantic episodic
    | length (integratedInfo (episodicBuffer wm)) < bindingCapacity (episodicBuffer wm) =
        let chunk = IntegratedChunk semantic episodic 1.0
        in wm { episodicBuffer = (episodicBuffer wm) {
            integratedInfo = chunk : integratedInfo (episodicBuffer wm)
        }}
    | otherwise = wm

decayPhonologicalItems :: WorkingMemory -> WorkingMemory
decayPhonologicalItems wm =
    let decayFactor = 1.0 - decayRate (phonologicalLoop wm)
        newLength = floor (fromIntegral (length (phonologicalStore (phonologicalLoop wm))) * decayFactor)
        newStore = take newLength (phonologicalStore (phonologicalLoop wm))
        newControl = take newLength (articulatoryControl (phonologicalLoop wm))
    in wm { phonologicalLoop = (phonologicalLoop wm) {
        phonologicalStore = newStore,
        articulatoryControl = newControl
    }}

focusAttention :: WorkingMemory -> String -> WorkingMemory
focusAttention wm target = wm { centralExecutive = (centralExecutive wm) {
    attentionFocus = target
}}

addGoal :: WorkingMemory -> String -> WorkingMemory
addGoal wm goal = wm { centralExecutive = (centralExecutive wm) {
    currentGoals = goal : currentGoals (centralExecutive wm)
}}

getCurrentGoals :: WorkingMemory -> [String]
getCurrentGoals wm = currentGoals (centralExecutive wm)

getPhonologicalItems :: WorkingMemory -> [String]
getPhonologicalItems wm = phonologicalStore (phonologicalLoop wm)

getVisualObjects :: WorkingMemory -> [VisualObject]
getVisualObjects wm = map snd (visualCache (visuospatialSketchpad wm))

getIntegratedInfo :: WorkingMemory -> [IntegratedChunk]
getIntegratedInfo wm = integratedInfo (episodicBuffer wm)

isAtCapacity :: WorkingMemory -> Bool
isAtCapacity wm =
    let totalItems = length (phonologicalStore (phonologicalLoop wm)) +
                     length (visualCache (visuospatialSketchpad wm)) +
                     length (integratedInfo (episodicBuffer wm))
    in totalItems >= capacity wm

-- 注意力机制类型 / Attention mechanism type / Aufmerksamkeitsmechanismustyp / Type mécanisme d'attention
data AttentionMechanism = AttentionMechanism {
    query :: [Double],
    keys :: [[Double]],
    values :: [[Double]],
    attentionWeights :: [Double]
} deriving (Show)

-- 注意力机制操作 / Attention mechanism operations / Aufmerksamkeitsmechanismusoperationen / Opérations de mécanisme d'attention
newAttentionMechanism :: [Double] -> [[Double]] -> [[Double]] -> AttentionMechanism
newAttentionMechanism q k v = AttentionMechanism q k v []

computeAttention :: AttentionMechanism -> AttentionMechanism
computeAttention am =
    let d_k = fromIntegral (length (head (keys am)))
        scores = map (\key -> dotProduct (query am) key / sqrt d_k) (keys am)
        weights = softmax scores
        output = weightedSum weights (values am)
    in am { attentionWeights = weights }

dotProduct :: [Double] -> [Double] -> Double
dotProduct a b = sum (zipWith (*) a b)

softmax :: [Double] -> [Double]
softmax scores =
    let maxScore = maximum scores
        expScores = map (\x -> exp (x - maxScore)) scores
        sumExp = sum expScores
    in map (/ sumExp) expScores

weightedSum :: [Double] -> [[Double]] -> [Double]
weightedSum weights values =
    let valueLength = length (head values)
    in [sum [weights !! i * (values !! i) !! j | i <- [0..length weights - 1]] | j <- [0..valueLength - 1]]

getAttentionWeights :: AttentionMechanism -> [Double]
getAttentionWeights am = attentionWeights am

-- 学习理论实现 / Learning theory implementation / Lerntheorieimplementierung / Implémentation de théorie d'apprentissage
data ClassicalConditioning = ClassicalConditioning {
    conditionedStimulus :: String,
    unconditionedStimulus :: String,
    conditionedResponse :: String,
    learningRate :: Double,
    trials :: Int
} deriving (Show)

classicalConditioning :: ClassicalConditioning -> Double
classicalConditioning cc =
    let n = trials cc
        lambda = learningRate cc
    in 1 - exp (-lambda * fromIntegral n)

data OperantConditioning = OperantConditioning {
    state :: String,
    action :: String,
    reward :: Double,
    learningRate :: Double,
    discountFactor :: Double
} deriving (Show)

qLearning :: OperantConditioning -> Double -> Double
qLearning oc currentQ =
    let alpha = learningRate oc
        gamma = discountFactor oc
        r = reward oc
        maxNextQ = currentQ  -- 简化版本 / Simplified version / Vereinfachte Version / Version simplifiée
    in currentQ + alpha * (r + gamma * maxNextQ - currentQ)

-- 决策理论实现 / Decision theory implementation / Entscheidungstheorieimplementierung / Implémentation de théorie de décision
data Decision = Decision {
    action :: String,
    outcomes :: [(String, Double, Double)]  -- (outcome, probability, utility)
} deriving (Show)

expectedUtility :: Decision -> Double
expectedUtility decision = sum [p * u | (_, p, u) <- outcomes decision]

prospectTheory :: [Double] -> [Double] -> Double
prospectTheory gains losses =
    let alpha = 0.88  -- 风险态度参数 / Risk attitude parameter / Risikoeinstellungsparameter / Paramètre d'attitude au risque
        beta = 0.88
        lambda = 2.25  -- 损失厌恶参数 / Loss aversion parameter / Verlustaversionparameter / Paramètre d'aversion aux pertes
        valueGains = sum (map (\x -> x ** alpha) gains)
        valueLosses = sum (map (\x -> -lambda * ((-x) ** beta)) losses)
    in valueGains + valueLosses

-- 主函数 / Main function / Hauptfunktion / Fonction principale
main :: IO ()
main = do
    putStrLn "=== 认知科学模型示例 / Cognitive Science Model Example ==="

    -- 工作记忆示例 / Working memory example / Arbeitsgedächtnis Beispiel / Exemple de mémoire de travail
    let wm = newWorkingMemory 7
    let wm1 = addPhonologicalItem wm "apple"
    let wm2 = addPhonologicalItem wm1 "banana"
    let wm3 = addPhonologicalItem wm2 "cherry"

    putStrLn $ "语音项目: " ++ show (getPhonologicalItems wm3)

    let visualObj = VisualObject "apple_visual" [("color", "red"), ("shape", "round")] (10.0, 20.0)
    let wm4 = addVisualObject wm3 visualObj

    let wm5 = integrateInformation wm4 "fruit" "kitchen_table"

    putStrLn $ "集成信息: " ++ show (getIntegratedInfo wm5)
    putStrLn $ "是否达到容量: " ++ show (isAtCapacity wm5)

    -- 注意力机制示例 / Attention mechanism example / Aufmerksamkeitsmechanismus Beispiel / Exemple de mécanisme d'attention
    putStrLn "\n=== 注意力机制示例 / Attention Mechanism Example ==="

    let query = [1.0, 2.0, 3.0]
    let keys = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    let values = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    let attention = newAttentionMechanism query keys values
    let attentionResult = computeAttention attention

    putStrLn $ "注意力权重: " ++ show (getAttentionWeights attentionResult)

    -- 学习理论示例 / Learning theory example / Lerntheorie Beispiel / Exemple de théorie d'apprentissage
    putStrLn "\n=== 学习理论示例 / Learning Theory Example ==="

    let cc = ClassicalConditioning "bell" "food" "salivation" 0.1 10
    putStrLn $ "经典条件反射概率: " ++ show (classicalConditioning cc)

    let oc = OperantConditioning "state1" "action1" 1.0 0.1 0.9
    putStrLn $ "Q学习值: " ++ show (qLearning oc 0.5)

    -- 决策理论示例 / Decision theory example / Entscheidungstheorie Beispiel / Exemple de théorie de décision
    putStrLn "\n=== 决策理论示例 / Decision Theory Example ==="

    let decision = Decision "invest" [("gain", 0.6, 100), ("loss", 0.4, -50)]
    putStrLn $ "期望效用: " ++ show (expectedUtility decision)

    let gains = [100, 50]
    let losses = [-30, -20]
    putStrLn $ "前景理论值: " ++ show (prospectTheory gains losses)
```

---



---

## 2025年最新发展 / Latest Developments 2025

### 认知科学的最新发展

**2025年关键突破**：

1. **认知科学与推理架构**
   - **o1/o3系列**：新的推理架构在认知建模方面表现出色，认知科学为推理架构提供了人类认知的理论基础
   - **DeepSeek-R1**：纯RL驱动架构在元认知能力方面取得突破，展示了认知科学在AI系统中的重要性
   - **技术影响**：认知科学为推理架构提供了人类认知的理论基础，推动了AI系统在认知能力上的提升

2. **认知架构与AI系统**
   - **认知模型**：认知科学在认知模型设计中的应用持续优化，为AI系统的认知架构提供了理论基础
   - **记忆系统**：认知科学在记忆系统设计中的应用持续深入，为AI系统的记忆机制提供了理论基础
   - **技术影响**：认知科学为AI系统的认知架构和记忆系统提供了理论基础

3. **认知科学与意识研究**
   - **意识理论**：认知科学在意识理论研究中的应用持续优化，为AI系统的意识研究提供了理论基础
   - **自我意识**：认知科学在自我意识研究中的应用持续深入，为AI系统的自我意识提供了理论基础
   - **技术影响**：认知科学为AI系统的意识研究和自我意识提供了理论基础

**详细内容**：参见 [2024-2025年最新AI技术发展总结](../../LATEST_AI_DEVELOPMENTS_2025.md)

---

**最后更新**：2026-01-11

## 参考文献 / References / Literatur / Références

1. **中文 / Chinese:**

   - 安德森, J. R. (2004). _认知心理学及其启示_. 人民邮电出版社.
   - 巴德利, A. (2007). _工作记忆_. 北京大学出版社.
   - 卡尼曼, D. (2012). _思考，快与慢_. 中信出版社.

2. **English:**

   - Anderson, J. R. (2007). _How Can the Human Mind Occur in the Physical Universe?_ Oxford University Press.
   - Baddeley, A. (2012). _Working Memory: Theories, Models, and Controversies_. Annual Review of Psychology.
   - Kahneman, D. (2011). _Thinking, Fast and Slow_. Farrar, Straus and Giroux.

3. **Deutsch / German:**

   - Anderson, J. R. (2007). _Wie kann der menschliche Geist im physischen Universum existieren?_ Oxford University Press.
   - Baddeley, A. (2012). _Arbeitsgedächtnis: Theorien, Modelle und Kontroversen_. Annual Review of Psychology.
   - Kahneman, D. (2012). _Schnelles Denken, langsames Denken_. Siedler Verlag.

4. **Français / French:**
   - Anderson, J. R. (2007). _Comment l'esprit humain peut-il exister dans l'univers physique?_ Oxford University Press.
   - Baddeley, A. (2012). _Mémoire de travail: Théories, modèles et controverses_. Annual Review of Psychology.
   - Kahneman, D. (2012). _Système 1, système 2: Les deux vitesses de la pensée_. Flammarion.

---

_本模块为 FormalAI 提供了完整的认知科学理论基础，结合国际标准 Wiki 的概念定义，使用中英德法四语言诠释核心概念，为 AI 系统的认知建模和智能行为设计提供了科学的理论基础。_

## 2024/2025 最新进展 / Latest Updates

### 认知科学理论的最新突破

**2025年关键进展**：

1. **认知架构与AI系统**
   - **COGENT3涌现认知架构**（2025）：集体增长和熵调制三元系统，模式形成网络与群体影响动态结合
   - **神经符号概念架构**（Mao et al., 2025）：以概念为中心的持续学习和灵活推理范式
   - **认知形态学架构**（2025）：基于形态学原理的认知结构设计
   - **元认知LLM架构**（2025）：大语言模型的元认知能力增强

2. **具身智能与世界模型**
   - **具身智能**：具身智能与世界模型在任务泛化中的作用持续深入研究
   - **世界模型**：世界模型在认知科学中的应用持续优化
   - **技术影响**：为AI系统的具身认知提供了理论基础

3. **注意力机制的认知神经学证据**
   - **注意力机制**：注意力机制的认知神经学证据与AI对照实验持续深入
   - **认知对照**：AI系统与人类认知的对照研究持续优化
   - **技术影响**：为AI系统的注意力机制提供了认知科学基础

4. **认知科学与推理架构**
   - **推理架构**：认知科学为推理架构提供了人类推理的理论基础
   - **元认知能力**：认知科学在元认知能力研究中的应用持续深入
   - **技术影响**：认知科学为AI系统的推理和元认知提供了理论基础

**详细内容**：参见 [2024-2025年最新AI技术发展总结](../../LATEST_AI_DEVELOPMENTS_2025.md)

[返回“最新进展”索引](../../LATEST_UPDATES_INDEX.md)
