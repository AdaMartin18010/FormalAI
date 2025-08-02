# 1.4 认知科学 / Cognitive Science

## 概述 / Overview

认知科学研究人类智能的认知过程，为FormalAI提供认知建模和智能系统设计的理论基础。

Cognitive science studies human cognitive processes, providing theoretical foundations for cognitive modeling and intelligent system design in FormalAI.

## 目录 / Table of Contents

- [1.4 认知科学 / Cognitive Science](#14-认知科学--cognitive-science)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [1. 认知架构 / Cognitive Architecture](#1-认知架构--cognitive-architecture)
    - [1.1 ACT-R架构 / ACT-R Architecture](#11-act-r架构--act-r-architecture)
    - [1.2 SOAR架构 / SOAR Architecture](#12-soar架构--soar-architecture)
    - [1.3 CLARION架构 / CLARION Architecture](#13-clarion架构--clarion-architecture)
  - [2. 记忆模型 / Memory Models](#2-记忆模型--memory-models)
    - [2.1 工作记忆 / Working Memory](#21-工作记忆--working-memory)
    - [2.2 长期记忆 / Long-Term Memory](#22-长期记忆--long-term-memory)
    - [2.3 记忆编码 / Memory Encoding](#23-记忆编码--memory-encoding)
  - [3. 注意力机制 / Attention Mechanisms](#3-注意力机制--attention-mechanisms)
    - [3.1 选择性注意力 / Selective Attention](#31-选择性注意力--selective-attention)
    - [3.2 分配性注意力 / Divided Attention](#32-分配性注意力--divided-attention)
    - [3.3 注意力控制 / Attention Control](#33-注意力控制--attention-control)
  - [4. 学习理论 / Learning Theory](#4-学习理论--learning-theory)
    - [4.1 经典条件反射 / Classical Conditioning](#41-经典条件反射--classical-conditioning)
    - [4.2 操作条件反射 / Operant Conditioning](#42-操作条件反射--operant-conditioning)
    - [4.3 观察学习 / Observational Learning](#43-观察学习--observational-learning)
  - [5. 决策理论 / Decision Theory](#5-决策理论--decision-theory)
    - [5.1 期望效用理论 / Expected Utility Theory](#51-期望效用理论--expected-utility-theory)
    - [5.2 前景理论 / Prospect Theory](#52-前景理论--prospect-theory)
    - [5.3 启发式决策 / Heuristic Decision Making](#53-启发式决策--heuristic-decision-making)
  - [6. 语言认知 / Language Cognition](#6-语言认知--language-cognition)
    - [6.1 语言习得 / Language Acquisition](#61-语言习得--language-acquisition)
    - [6.2 语言理解 / Language Comprehension](#62-语言理解--language-comprehension)
    - [6.3 语言产生 / Language Production](#63-语言产生--language-production)
  - [7. 视觉认知 / Visual Cognition](#7-视觉认知--visual-cognition)
    - [7.1 视觉感知 / Visual Perception](#71-视觉感知--visual-perception)
    - [7.2 物体识别 / Object Recognition](#72-物体识别--object-recognition)
    - [7.3 空间认知 / Spatial Cognition](#73-空间认知--spatial-cognition)
  - [8. 情感认知 / Emotional Cognition](#8-情感认知--emotional-cognition)
    - [8.1 情感理论 / Emotion Theory](#81-情感理论--emotion-theory)
    - [8.2 情感调节 / Emotion Regulation](#82-情感调节--emotion-regulation)
    - [8.3 情感与决策 / Emotion and Decision Making](#83-情感与决策--emotion-and-decision-making)
  - [9. 元认知 / Metacognition](#9-元认知--metacognition)
    - [9.1 元认知监控 / Metacognitive Monitoring](#91-元认知监控--metacognitive-monitoring)
    - [9.2 元认知控制 / Metacognitive Control](#92-元认知控制--metacognitive-control)
    - [9.3 自我调节学习 / Self-Regulated Learning](#93-自我调节学习--self-regulated-learning)
  - [10. 社会认知 / Social Cognition](#10-社会认知--social-cognition)
    - [10.1 心理理论 / Theory of Mind](#101-心理理论--theory-of-mind)
    - [10.2 社会学习 / Social Learning](#102-社会学习--social-learning)
    - [10.3 群体认知 / Group Cognition](#103-群体认知--group-cognition)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：认知架构模拟器](#rust实现认知架构模拟器)
    - [Haskell实现：记忆模型](#haskell实现记忆模型)
  - [参考文献 / References](#参考文献--references)

---

## 1. 认知架构 / Cognitive Architecture

### 1.1 ACT-R架构 / ACT-R Architecture

**ACT-R定义 / ACT-R Definition:**

ACT-R (Adaptive Control of Thought-Rational) 是一个认知架构，包含：

ACT-R (Adaptive Control of Thought-Rational) is a cognitive architecture containing:

- **声明性记忆 / Declarative Memory:** 存储事实和知识
- **程序性记忆 / Procedural Memory:** 存储产生式规则
- **工作记忆 / Working Memory:** 当前激活的信息
- **感知-运动模块 / Perceptual-Motor Modules:** 与外部环境交互

**产生式规则 / Production Rules:**

$$IF \text{ condition } THEN \text{ action}$$

**激活计算 / Activation Calculation:**

$$A_i = B_i + \sum_j W_j S_{ji} + \sum_k W_k S_{ki}$$

其中：

- $A_i$ 是节点 $i$ 的激活
- $B_i$ 是基础激活
- $W_j$ 是上下文权重
- $S_{ji}$ 是关联强度

### 1.2 SOAR架构 / SOAR Architecture

**SOAR组件 / SOAR Components:**

1. **长期记忆 / Long-Term Memory:**
   - 语义记忆 / Semantic memory
   - 程序性记忆 / Procedural memory
   - 情节记忆 / Episodic memory

2. **工作记忆 / Working Memory:**
   - 当前目标 / Current goals
   - 问题空间 / Problem spaces
   - 操作符 / Operators

3. **决策周期 / Decision Cycle:**
   - 输入阶段 / Input phase
   - 提议阶段 / Propose phase
   - 应用阶段 / Apply phase
   - 输出阶段 / Output phase

### 1.3 CLARION架构 / CLARION Architecture

**CLARION层次 / CLARION Levels:**

1. **显式层 / Explicit Layer:**
   - 符号表示 / Symbolic representations
   - 规则学习 / Rule learning

2. **隐式层 / Implicit Layer:**
   - 分布式表示 / Distributed representations
   - 模式学习 / Pattern learning

**学习机制 / Learning Mechanisms:**

$$\Delta w_{ij} = \alpha \delta_i x_j$$

其中 $\alpha$ 是学习率，$\delta_i$ 是误差信号。

## 2. 记忆模型 / Memory Models

### 2.1 工作记忆 / Working Memory

**工作记忆容量 / Working Memory Capacity:**

根据Cowan的理论，工作记忆容量约为4±1个项目：

According to Cowan's theory, working memory capacity is approximately 4±1 items:

$$C = 4 \pm 1$$

**注意力焦点 / Focus of Attention:**

$$F = \{i_1, i_2, ..., i_k\}$$

其中 $k \leq C$。

### 2.2 长期记忆 / Long-Term Memory

**记忆强度 / Memory Strength:**

$$S(t) = S_0 e^{-\lambda t} + \sum_i S_i e^{-\lambda_i (t-t_i)}$$

其中：

- $S_0$ 是初始强度
- $\lambda$ 是衰减率
- $S_i$ 是复习强度
- $t_i$ 是复习时间

**检索概率 / Retrieval Probability:**

$$P(R|S) = \frac{S}{S + \theta}$$

其中 $\theta$ 是阈值参数。

### 2.3 记忆编码 / Memory Encoding

**编码深度 / Encoding Depth:**

$$D = \sum_i w_i f_i$$

其中 $w_i$ 是权重，$f_i$ 是编码特征。

## 3. 注意力机制 / Attention Mechanisms

### 3.1 选择性注意力 / Selective Attention

**注意力权重 / Attention Weights:**

$$w_i = \frac{\exp(e_i)}{\sum_j \exp(e_j)}$$

其中 $e_i$ 是注意力分数：

$$e_i = f(Q, K_i)$$

**注意力计算 / Attention Computation:**

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 3.2 分配性注意力 / Divided Attention

**注意力分配 / Attention Allocation:**

$$\sum_i w_i = 1$$

**性能函数 / Performance Function:**

$$P = \sum_i w_i P_i$$

其中 $P_i$ 是任务 $i$ 的性能。

### 3.3 注意力控制 / Attention Control

**执行控制 / Executive Control:**

$$C = \alpha \cdot \text{Top-down} + \beta \cdot \text{Bottom-up}$$

其中 $\alpha$ 和 $\beta$ 是控制权重。

## 4. 学习理论 / Learning Theory

### 4.1 经典条件反射 / Classical Conditioning

**巴甫洛夫条件反射 / Pavlovian Conditioning:**

$$CS \rightarrow US \rightarrow UR$$

**条件反射强度 / Conditioning Strength:**

$$V_{CS} = \alpha \beta (\lambda - V_{CS})$$

其中：

- $\alpha$ 是CS的显著性
- $\beta$ 是US的显著性
- $\lambda$ 是最大强度

### 4.2 操作条件反射 / Operant Conditioning

**强化学习 / Reinforcement Learning:**

$$Q(s,a) = Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

**策略梯度 / Policy Gradient:**

$$\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi(a|s) Q(s,a)]$$

### 4.3 观察学习 / Observational Learning

**模仿学习 / Imitation Learning:**

$$\pi^*(a|s) = \arg\max_\pi \mathbb{E}_{s,a \sim \pi_E}[\log \pi(a|s)]$$

## 5. 决策理论 / Decision Theory

### 5.1 期望效用理论 / Expected Utility Theory

**期望效用 / Expected Utility:**

$$EU(A) = \sum_i p_i u(x_i)$$

**决策规则 / Decision Rule:**

$$A^* = \arg\max_A EU(A)$$

### 5.2 前景理论 / Prospect Theory

**价值函数 / Value Function:**

$$
v(x) = \begin{cases}
x^\alpha & \text{if } x \geq 0 \\
-\lambda(-x)^\beta & \text{if } x < 0
\end{cases}
$$

**权重函数 / Weighting Function:**

$$\pi(p) = \frac{p^\gamma}{(p^\gamma + (1-p)^\gamma)^{1/\gamma}}$$

### 5.3 启发式决策 / Heuristic Decision Making

**可用性启发式 / Availability Heuristic:**

$$P(A) \propto \text{availability of A}$$

**代表性启发式 / Representativeness Heuristic:**

$$P(A|B) \propto \text{similarity}(A, B)$$

## 6. 语言认知 / Language Cognition

### 6.1 语言习得 / Language Acquisition

**语言习得模型 / Language Acquisition Model:**

$$L(t) = L_0 + \alpha \int_0^t I(\tau) d\tau$$

其中 $I(t)$ 是输入强度。

### 6.2 语言理解 / Language Comprehension

**句法分析 / Syntactic Parsing:**

$$P(T|S) = \frac{P(S|T)P(T)}{P(S)}$$

**语义理解 / Semantic Understanding:**

$$M = \text{Composition}(w_1, w_2, ..., w_n)$$

### 6.3 语言产生 / Language Production

**语言产生模型 / Language Production Model:**

1. **概念化 / Conceptualization**
2. **公式化 / Formulation**
3. **发音 / Articulation**

## 7. 视觉认知 / Visual Cognition

### 7.1 视觉感知 / Visual Perception

**视觉处理层次 / Visual Processing Hierarchy:**

1. **视网膜 / Retina**
2. **外侧膝状体 / LGN**
3. **初级视觉皮层 / V1**
4. **高级视觉区域 / Higher Visual Areas**

**感受野 / Receptive Field:**

$$RF(x,y) = \sum_i w_i I(x_i, y_i)$$

### 7.2 物体识别 / Object Recognition

**模板匹配 / Template Matching:**

$$S = \sum_{i,j} T(i,j) \cdot I(i,j)$$

**特征检测 / Feature Detection:**

$$F = \sigma(W \cdot I + b)$$

### 7.3 空间认知 / Spatial Cognition

**空间表示 / Spatial Representation:**

$$\mathbf{x} = [x, y, z, \theta]^T$$

**空间变换 / Spatial Transformation:**

$$
T = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
$$

## 8. 情感认知 / Emotional Cognition

### 8.1 情感理论 / Emotion Theory

**情感维度 / Emotional Dimensions:**

1. **效价 / Valence:** 正面-负面
2. **唤醒度 / Arousal:** 高-低
3. **支配度 / Dominance:** 控制-被控制

**情感强度 / Emotional Intensity:**

$$I = \sqrt{\sum_i e_i^2}$$

### 8.2 情感调节 / Emotion Regulation

**认知重评 / Cognitive Reappraisal:**

$$E' = f(E, C)$$

其中 $C$ 是认知策略。

**表达抑制 / Expressive Suppression:**

$$E_{suppressed} = \alpha E_{original}$$

### 8.3 情感与决策 / Emotion and Decision Making

**情感影响决策 / Emotional Influence on Decision:**

$$D = D_{rational} + \beta E$$

其中 $\beta$ 是情感权重。

## 9. 元认知 / Metacognition

### 9.1 元认知监控 / Metacognitive Monitoring

**信心判断 / Confidence Judgment:**

$$C = f(\text{performance}, \text{effort}, \text{difficulty})$$

**学习判断 / Judgments of Learning:**

$$JOL = \alpha \text{familiarity} + \beta \text{processing fluency}$$

### 9.2 元认知控制 / Metacognitive Control

**学习时间分配 / Study Time Allocation:**

$$T_i = f(\text{difficulty}_i, \text{importance}_i, \text{time available})$$

**策略选择 / Strategy Selection:**

$$S^* = \arg\max_S U(S)$$

### 9.3 自我调节学习 / Self-Regulated Learning

**自我调节循环 / Self-Regulation Cycle:**

1. **计划 / Planning**
2. **监控 / Monitoring**
3. **控制 / Control**
4. **反思 / Reflection**

## 10. 社会认知 / Social Cognition

### 10.1 心理理论 / Theory of Mind

**心理状态推理 / Mental State Inference:**

$$P(M|B) = \frac{P(B|M)P(M)}{P(B)}$$

**信念-欲望-意图模型 / Belief-Desire-Intention Model:**

$$A = f(B, D, I)$$

### 10.2 社会学习 / Social Learning

**观察学习 / Observational Learning:**

$$L_{social} = \alpha L_{direct} + \beta L_{observational}$$

**模仿学习 / Imitation Learning:**

$$\pi_{imitation} = \arg\min_\pi \mathbb{E}[d(\pi, \pi_{expert})]$$

### 10.3 群体认知 / Group Cognition

**集体智能 / Collective Intelligence:**

$$I_{group} = f(I_1, I_2, ..., I_n, \text{interaction})$$

**群体决策 / Group Decision Making:**

$$D_{group} = \sum_i w_i D_i$$

## 代码示例 / Code Examples

### Rust实现：认知架构模拟器

```rust
use std::collections::HashMap;

// 认知架构模拟器
struct CognitiveArchitecture {
    declarative_memory: HashMap<String, f64>,
    procedural_memory: Vec<ProductionRule>,
    working_memory: Vec<String>,
    attention_focus: Vec<String>,
}

struct ProductionRule {
    condition: String,
    action: String,
    strength: f64,
}

impl CognitiveArchitecture {
    fn new() -> Self {
        Self {
            declarative_memory: HashMap::new(),
            procedural_memory: Vec::new(),
            working_memory: Vec::new(),
            attention_focus: Vec::new(),
        }
    }
    
    // 激活计算
    fn calculate_activation(&self, node: &str) -> f64 {
        let base_activation = self.declarative_memory.get(node).unwrap_or(&0.0);
        let contextual_activation = self.calculate_contextual_activation(node);
        base_activation + contextual_activation
    }
    
    // 注意力分配
    fn allocate_attention(&mut self, items: Vec<String>) {
        let capacity = 4.0; // 工作记忆容量
        let weights = self.calculate_attention_weights(&items);
        
        self.attention_focus = items
            .into_iter()
            .zip(weights)
            .filter(|(_, w)| *w > 0.1)
            .take(capacity as usize)
            .map(|(item, _)| item)
            .collect();
    }
    
    // 决策过程
    fn make_decision(&self, options: Vec<String>) -> String {
        let utilities: Vec<f64> = options
            .iter()
            .map(|option| self.calculate_utility(option))
            .collect();
        
        let max_index = utilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        options[max_index].clone()
    }
}

// 记忆模型
struct MemoryModel {
    working_memory: Vec<MemoryItem>,
    long_term_memory: HashMap<String, MemoryTrace>,
}

struct MemoryItem {
    content: String,
    activation: f64,
    timestamp: f64,
}

struct MemoryTrace {
    strength: f64,
    last_accessed: f64,
    access_count: u32,
}

impl MemoryModel {
    fn new() -> Self {
        Self {
            working_memory: Vec::new(),
            long_term_memory: HashMap::new(),
        }
    }
    
    // 记忆编码
    fn encode(&mut self, content: String, depth: f64) {
        let activation = depth * 0.8 + 0.2; // 基础激活 + 深度激活
        let item = MemoryItem {
            content: content.clone(),
            activation,
            timestamp: 0.0, // 当前时间
        };
        
        self.working_memory.push(item);
        
        // 存储到长期记忆
        let trace = MemoryTrace {
            strength: activation,
            last_accessed: 0.0,
            access_count: 1,
        };
        self.long_term_memory.insert(content, trace);
    }
    
    // 记忆检索
    fn retrieve(&mut self, cue: &str) -> Option<String> {
        let mut best_match = None;
        let mut best_score = 0.0;
        
        for (key, trace) in &mut self.long_term_memory {
            if key.contains(cue) || cue.contains(key) {
                let score = self.calculate_retrieval_score(trace);
                if score > best_score {
                    best_score = score;
                    best_match = Some(key.clone());
                }
                trace.last_accessed = 0.0; // 当前时间
                trace.access_count += 1;
            }
        }
        
        best_match
    }
    
    // 计算检索分数
    fn calculate_retrieval_score(&self, trace: &MemoryTrace) -> f64 {
        let time_decay = (-0.1 * (0.0 - trace.last_accessed)).exp();
        let frequency_boost = (trace.access_count as f64).ln() + 1.0;
        trace.strength * time_decay * frequency_boost
    }
}

// 注意力机制
struct AttentionMechanism {
    focus: Vec<String>,
    capacity: usize,
}

impl AttentionMechanism {
    fn new(capacity: usize) -> Self {
        Self {
            focus: Vec::new(),
            capacity,
        }
    }
    
    // 选择性注意力
    fn selective_attention(&mut self, stimuli: Vec<String>, relevance: Vec<f64>) {
        let mut scored_items: Vec<(String, f64)> = stimuli
            .into_iter()
            .zip(relevance)
            .collect();
        
        scored_items.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        
        self.focus = scored_items
            .into_iter()
            .take(self.capacity)
            .map(|(item, _)| item)
            .collect();
    }
    
    // 分配性注意力
    fn divided_attention(&self, tasks: Vec<String>) -> Vec<f64> {
        let n = tasks.len();
        if n == 0 {
            return Vec::new();
        }
        
        // 平均分配注意力
        let weight_per_task = 1.0 / n as f64;
        vec![weight_per_task; n]
    }
}

fn main() {
    // 创建认知架构
    let mut cognitive_arch = CognitiveArchitecture::new();
    
    // 创建记忆模型
    let mut memory_model = MemoryModel::new();
    
    // 创建注意力机制
    let mut attention = AttentionMechanism::new(4);
    
    // 模拟认知过程
    println!("=== 认知科学模拟 ===");
    
    // 1. 记忆编码
    memory_model.encode("认知科学很重要".to_string(), 0.9);
    memory_model.encode("注意力机制".to_string(), 0.8);
    memory_model.encode("工作记忆".to_string(), 0.7);
    
    // 2. 注意力分配
    let stimuli = vec![
        "视觉信息".to_string(),
        "听觉信息".to_string(),
        "触觉信息".to_string(),
        "嗅觉信息".to_string(),
    ];
    let relevance = vec![0.9, 0.7, 0.5, 0.3];
    attention.selective_attention(stimuli, relevance);
    
    println!("注意力焦点: {:?}", attention.focus);
    
    // 3. 记忆检索
    if let Some(retrieved) = memory_model.retrieve("认知") {
        println!("检索到: {}", retrieved);
    }
    
    // 4. 决策过程
    let options = vec![
        "继续学习".to_string(),
        "休息一下".to_string(),
        "换个主题".to_string(),
    ];
    let decision = cognitive_arch.make_decision(options);
    println!("决策结果: {}", decision);
}
```

### Haskell实现：记忆模型

```haskell
-- 认知科学模块
module CognitiveScience where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.List (sortBy)
import Data.Ord (comparing)

-- 记忆项
data MemoryItem = MemoryItem
    { content :: String
    , activation :: Double
    , timestamp :: Double
    } deriving (Show, Eq)

-- 记忆痕迹
data MemoryTrace = MemoryTrace
    { strength :: Double
    , lastAccessed :: Double
    , accessCount :: Int
    } deriving (Show, Eq)

-- 工作记忆
data WorkingMemory = WorkingMemory
    { items :: [MemoryItem]
    , capacity :: Int
    } deriving (Show, Eq)

-- 长期记忆
type LongTermMemory = Map String MemoryTrace

-- 记忆模型
data MemoryModel = MemoryModel
    { workingMemory :: WorkingMemory
    , longTermMemory :: LongTermMemory
    } deriving (Show, Eq)

-- 注意力机制
data AttentionMechanism = AttentionMechanism
    { focus :: [String]
    , capacity :: Int
    } deriving (Show, Eq)

-- 认知架构
data CognitiveArchitecture = CognitiveArchitecture
    { declarativeMemory :: Map String Double
    , proceduralMemory :: [ProductionRule]
    , workingMemory :: [String]
    , attentionFocus :: [String]
    } deriving (Show, Eq)

-- 产生式规则
data ProductionRule = ProductionRule
    { condition :: String
    , action :: String
    , ruleStrength :: Double
    } deriving (Show, Eq)

-- 创建新的记忆模型
newMemoryModel :: MemoryModel
newMemoryModel = MemoryModel
    { workingMemory = WorkingMemory [] 7
    , longTermMemory = Map.empty
    }

-- 创建新的注意力机制
newAttentionMechanism :: Int -> AttentionMechanism
newAttentionMechanism cap = AttentionMechanism
    { focus = []
    , capacity = cap
    }

-- 创建新的认知架构
newCognitiveArchitecture :: CognitiveArchitecture
newCognitiveArchitecture = CognitiveArchitecture
    { declarativeMemory = Map.empty
    , proceduralMemory = []
    , workingMemory = []
    , attentionFocus = []
    }

-- 记忆编码
encodeMemory :: MemoryModel -> String -> Double -> MemoryModel
encodeMemory model content depth = model
    { workingMemory = updatedWorkingMemory
    , longTermMemory = updatedLongTermMemory
    }
  where
    activation = depth * 0.8 + 0.2
    newItem = MemoryItem content activation 0.0
    updatedWorkingMemory = (workingMemory model)
        { items = newItem : items (workingMemory model)
        }
    newTrace = MemoryTrace activation 0.0 1
    updatedLongTermMemory = Map.insert content newTrace (longTermMemory model)

-- 记忆检索
retrieveMemory :: MemoryModel -> String -> Maybe String
retrieveMemory model cue = 
    case bestMatch of
        Just (key, _) -> Just key
        Nothing -> Nothing
  where
    matches = Map.toList $ Map.filterWithKey (\k _ -> 
        cue `isInfixOf` k || k `isInfixOf` cue) (longTermMemory model)
    scoredMatches = map (\(k, trace) -> (k, calculateRetrievalScore trace)) matches
    bestMatch = if null scoredMatches 
        then Nothing 
        else Just $ maximumBy (comparing snd) scoredMatches

-- 计算检索分数
calculateRetrievalScore :: MemoryTrace -> Double
calculateRetrievalScore trace = 
    strength trace * timeDecay * frequencyBoost
  where
    timeDecay = exp (-0.1 * (0.0 - lastAccessed trace))
    frequencyBoost = log (fromIntegral (accessCount trace) + 1.0)

-- 选择性注意力
selectiveAttention :: AttentionMechanism -> [String] -> [Double] -> AttentionMechanism
selectiveAttention mechanism stimuli relevance = mechanism
    { focus = take (capacity mechanism) selectedItems
    }
  where
    scoredItems = zip stimuli relevance
    sortedItems = sortBy (comparing (negate . snd)) scoredItems
    selectedItems = map fst sortedItems

-- 分配性注意力
dividedAttention :: AttentionMechanism -> [String] -> [Double]
dividedAttention mechanism tasks
    | null tasks = []
    | otherwise = replicate (length tasks) weightPerTask
  where
    weightPerTask = 1.0 / fromIntegral (length tasks)

-- 激活计算
calculateActivation :: CognitiveArchitecture -> String -> Double
calculateActivation arch node = 
    baseActivation + contextualActivation
  where
    baseActivation = Map.findWithDefault 0.0 node (declarativeMemory arch)
    contextualActivation = calculateContextualActivation arch node

-- 计算上下文激活
calculateContextualActivation :: CognitiveArchitecture -> String -> Double
calculateContextualActivation arch node = 
    sum [weight * strength | (relatedNode, weight) <- contextItems]
  where
    contextItems = Map.toList $ Map.filterWithKey (\k _ -> k /= node) (declarativeMemory arch)
    strength = Map.findWithDefault 0.0 node (declarativeMemory arch)

-- 决策过程
makeDecision :: CognitiveArchitecture -> [String] -> String
makeDecision arch options = 
    options !! maxIndex
  where
    utilities = map (calculateUtility arch) options
    maxIndex = snd $ maximum $ zip utilities [0..]

-- 计算效用
calculateUtility :: CognitiveArchitecture -> String -> Double
calculateUtility arch option = 
    Map.findWithDefault 0.0 option (declarativeMemory arch)

-- 示例使用
main :: IO ()
main = do
    putStrLn "=== 认知科学模拟 ==="
    
    -- 创建记忆模型
    let initialModel = newMemoryModel
    
    -- 记忆编码
    let model1 = encodeMemory initialModel "认知科学很重要" 0.9
    let model2 = encodeMemory model1 "注意力机制" 0.8
    let model3 = encodeMemory model2 "工作记忆" 0.7
    
    -- 记忆检索
    case retrieveMemory model3 "认知" of
        Just retrieved -> putStrLn $ "检索到: " ++ retrieved
        Nothing -> putStrLn "未找到相关记忆"
    
    -- 注意力机制
    let attention = newAttentionMechanism 4
    let stimuli = ["视觉信息", "听觉信息", "触觉信息", "嗅觉信息"]
    let relevance = [0.9, 0.7, 0.5, 0.3]
    let focusedAttention = selectiveAttention attention stimuli relevance
    
    putStrLn $ "注意力焦点: " ++ show (focus focusedAttention)
    
    -- 认知架构
    let arch = newCognitiveArchitecture
    let options = ["继续学习", "休息一下", "换个主题"]
    let decision = makeDecision arch options
    
    putStrLn $ "决策结果: " ++ decision
```

## 参考文献 / References

1. Anderson, J. R. (2007). How can the human mind occur in the physical universe? Oxford University Press.
2. Baddeley, A. (2012). Working memory: Theories, models, and controversies. Annual Review of Psychology, 63, 1-29.
3. Kahneman, D. (2011). Thinking, fast and slow. Farrar, Straus and Giroux.
4. Laird, J. E. (2012). The SOAR cognitive architecture. MIT Press.
5. Sun, R. (2006). The CLARION cognitive architecture: Extending cognitive modeling. In Proceedings of the 28th Annual Conference of the Cognitive Science Society.
6. Posner, M. I., & Petersen, S. E. (1990). The attention system of the human brain. Annual Review of Neuroscience, 13, 25-42.
7. Tulving, E. (2002). Episodic memory: From mind to brain. Annual Review of Psychology, 53, 1-25.
8. Bandura, A. (1977). Social learning theory. Prentice-Hall.
9. Damasio, A. (1994). Descartes' error: Emotion, reason, and the human brain. Putnam.
10. Flavell, J. H. (1979). Metacognition and cognitive monitoring: A new area of cognitive-developmental inquiry. American Psychologist, 34(10), 906-911.

---

*认知科学为FormalAI提供了理解人类智能认知过程的理论基础，为构建更智能的AI系统提供了重要指导。*

*Cognitive science provides theoretical foundations for understanding human cognitive processes in FormalAI, offering important guidance for building more intelligent AI systems.*
