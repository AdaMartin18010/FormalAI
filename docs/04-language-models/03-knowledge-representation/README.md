# 4.3 知识表示 / Knowledge Representation

## 概述 / Overview

知识表示研究如何在计算机中表示和组织知识，为FormalAI提供结构化和语义化的知识管理理论基础。

Knowledge representation studies how to represent and organize knowledge in computers, providing theoretical foundations for structured and semantic knowledge management in FormalAI.

## 目录 / Table of Contents

- [4.3 知识表示 / Knowledge Representation](#43-知识表示--knowledge-representation)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [1. 语义网络 / Semantic Networks](#1-语义网络--semantic-networks)
  - [2. 框架理论 / Frame Theory](#2-框架理论--frame-theory)
  - [3. 描述逻辑 / Description Logic](#3-描述逻辑--description-logic)
  - [4. 本体论 / Ontology](#4-本体论--ontology)
  - [5. 知识图谱 / Knowledge Graph](#5-知识图谱--knowledge-graph)
  - [6. 神经知识表示 / Neural Knowledge Representation](#6-神经知识表示--neural-knowledge-representation)
  - [代码示例 / Code Examples](#代码示例--code-examples)
  - [参考文献 / References](#参考文献--references)

---

## 1. 语义网络 / Semantic Networks

### 1.1 语义网络定义 / Semantic Network Definition

**语义网络结构 / Semantic Network Structure:**

$$G = (V, E, L)$$

其中：
- $V$ 是节点集合（概念）
- $E$ 是边集合（关系）
- $L$ 是标签函数

**节点类型 / Node Types:**

- **概念节点 / Concept Nodes:** 表示实体或概念
- **实例节点 / Instance Nodes:** 表示具体实例
- **属性节点 / Property Nodes:** 表示属性或特征

### 1.2 关系类型 / Relation Types

**层次关系 / Hierarchical Relations:**

- **is-a:** 类属关系
- **part-of:** 部分关系
- **instance-of:** 实例关系

**语义关系 / Semantic Relations:**

- **synonym:** 同义关系
- **antonym:** 反义关系
- **meronym:** 整体-部分关系
- **holonym:** 部分-整体关系

### 1.3 语义网络推理 / Semantic Network Reasoning

**继承推理 / Inheritance Reasoning:**

$$\frac{A \text{ is-a } B \quad B \text{ has-property } P}{A \text{ has-property } P}$$

**传递推理 / Transitive Reasoning:**

$$\frac{A \text{ R } B \quad B \text{ R } C}{A \text{ R } C}$$

**相似性推理 / Similarity Reasoning:**

$$\text{sim}(A, B) = \frac{|\text{common\_properties}(A, B)|}{|\text{all\_properties}(A, B)|}$$

## 2. 框架理论 / Frame Theory

### 2.1 框架定义 / Frame Definition

**框架结构 / Frame Structure:**

$$\text{Frame} = \begin{pmatrix}
\text{name} & \text{FrameName} \\
\text{slots} & \text{Slot}_1, \text{Slot}_2, ..., \text{Slot}_n \\
\text{defaults} & \text{Default}_1, \text{Default}_2, ..., \text{Default}_n \\
\text{constraints} & \text{Constraint}_1, \text{Constraint}_2, ..., \text{Constraint}_n
\end{pmatrix}$$

**槽位定义 / Slot Definition:**

$$\text{Slot} = \begin{pmatrix}
\text{name} & \text{SlotName} \\
\text{type} & \text{DataType} \\
\text{value} & \text{Value} \\
\text{facet} & \text{Facet}_1, \text{Facet}_2, ..., \text{Facet}_n
\end{pmatrix}$$

### 2.2 框架层次 / Frame Hierarchy

**框架继承 / Frame Inheritance:**

$$\text{ChildFrame} \text{ is-a } \text{ParentFrame}$$

**槽位继承 / Slot Inheritance:**

$$\frac{\text{ChildFrame} \text{ is-a } \text{ParentFrame} \quad \text{ParentFrame.slot} = v}{\text{ChildFrame.slot} = v}$$

### 2.3 框架匹配 / Frame Matching

**匹配度计算 / Match Degree Calculation:**

$$\text{match}(F_1, F_2) = \sum_{i} w_i \cdot \text{sim}(\text{slot}_i(F_1), \text{slot}_i(F_2))$$

**激活传播 / Activation Spreading:**

$$A_i(t+1) = \alpha A_i(t) + \beta \sum_{j} w_{ij} A_j(t)$$

## 3. 描述逻辑 / Description Logic

### 3.1 描述逻辑语法 / Description Logic Syntax

**概念构造器 / Concept Constructors:**

- **原子概念 / Atomic Concepts:** $A, B, C$
- **顶概念 / Top Concept:** $\top$
- **底概念 / Bottom Concept:** $\bot$
- **否定 / Negation:** $\neg C$
- **合取 / Conjunction:** $C \sqcap D$
- **析取 / Disjunction:** $C \sqcup D$
- **存在量词 / Existential Quantifier:** $\exists R.C$
- **全称量词 / Universal Quantifier:** $\forall R.C$

### 3.2 描述逻辑语义 / Description Logic Semantics

**解释函数 / Interpretation Function:**

$$\mathcal{I} = (\Delta^\mathcal{I}, \cdot^\mathcal{I})$$

其中：
- $\Delta^\mathcal{I}$ 是解释域
- $\cdot^\mathcal{I}$ 是解释函数

**语义定义 / Semantic Definition:**

- $A^\mathcal{I} \subseteq \Delta^\mathcal{I}$
- $(\neg C)^\mathcal{I} = \Delta^\mathcal{I} \setminus C^\mathcal{I}$
- $(C \sqcap D)^\mathcal{I} = C^\mathcal{I} \cap D^\mathcal{I}$
- $(C \sqcup D)^\mathcal{I} = C^\mathcal{I} \cup D^\mathcal{I}$
- $(\exists R.C)^\mathcal{I} = \{x \mid \exists y. (x,y) \in R^\mathcal{I} \land y \in C^\mathcal{I}\}$

### 3.3 推理服务 / Reasoning Services

**概念包含 / Concept Subsumption:**

$$C \sqsubseteq D \text{ iff } C^\mathcal{I} \subseteq D^\mathcal{I}$$

**概念等价 / Concept Equivalence:**

$$C \equiv D \text{ iff } C^\mathcal{I} = D^\mathcal{I}$$

**概念满足 / Concept Satisfiability:**

$$\text{SAT}(C) \text{ iff } C^\mathcal{I} \neq \emptyset$$

## 4. 本体论 / Ontology

### 4.1 本体定义 / Ontology Definition

**本体结构 / Ontology Structure:**

$$\mathcal{O} = (C, R, I, A)$$

其中：
- $C$ 是概念集合
- $R$ 是关系集合
- $I$ 是实例集合
- $A$ 是公理集合

**本体层次 / Ontology Hierarchy:**

$$\text{Upper Ontology} \rightarrow \text{Domain Ontology} \rightarrow \text{Application Ontology}$$

### 4.2 本体语言 / Ontology Languages

**OWL语言 / OWL Language:**

$$\text{Class} \sqsubseteq \text{Thing}$$
$$\text{ObjectProperty} \sqsubseteq \text{Property}$$
$$\text{DataProperty} \sqsubseteq \text{Property}$$

**RDF语言 / RDF Language:**

$$(\text{subject}, \text{predicate}, \text{object})$$

### 4.3 本体工程 / Ontology Engineering

**本体构建 / Ontology Construction:**

1. **需求分析 / Requirements Analysis**
2. **概念提取 / Concept Extraction**
3. **关系定义 / Relation Definition**
4. **公理构建 / Axiom Construction**
5. **验证测试 / Validation Testing**

**本体评估 / Ontology Evaluation:**

$$\text{Quality}(O) = \alpha \cdot \text{Completeness}(O) + \beta \cdot \text{Consistency}(O) + \gamma \cdot \text{Clarity}(O)$$

## 5. 知识图谱 / Knowledge Graph

### 5.1 知识图谱定义 / Knowledge Graph Definition

**图谱结构 / Graph Structure:**

$$KG = (E, R, T)$$

其中：
- $E$ 是实体集合
- $R$ 是关系集合
- $T$ 是三元组集合

**三元组表示 / Triple Representation:**

$$(h, r, t) \in E \times R \times E$$

其中 $h$ 是头实体，$r$ 是关系，$t$ 是尾实体。

### 5.2 知识图谱嵌入 / Knowledge Graph Embedding

**TransE模型 / TransE Model:**

$$\mathbf{h} + \mathbf{r} \approx \mathbf{t}$$

**DistMult模型 / DistMult Model:**

$$\text{score}(h, r, t) = \sum_i \mathbf{h}_i \cdot \mathbf{r}_i \cdot \mathbf{t}_i$$

**ComplEx模型 / ComplEx Model:**

$$\text{score}(h, r, t) = \text{Re}(\sum_i \mathbf{h}_i \cdot \mathbf{r}_i \cdot \bar{\mathbf{t}}_i)$$

### 5.3 知识图谱推理 / Knowledge Graph Reasoning

**链接预测 / Link Prediction:**

$$P(t|h, r) = \frac{\exp(\text{score}(h, r, t))}{\sum_{t'} \exp(\text{score}(h, r, t'))}$$

**路径推理 / Path Reasoning:**

$$\text{path}(e_1, e_n) = e_1 \xrightarrow{r_1} e_2 \xrightarrow{r_2} ... \xrightarrow{r_{n-1}} e_n$$

**规则学习 / Rule Learning:**

$$\text{Rule}: h \xrightarrow{r_1} ? \xrightarrow{r_2} t \Rightarrow h \xrightarrow{r_3} t$$

## 6. 神经知识表示 / Neural Knowledge Representation

### 6.1 神经嵌入 / Neural Embeddings

**实体嵌入 / Entity Embeddings:**

$$\mathbf{e}_i = f_\theta(x_i)$$

其中 $f_\theta$ 是神经网络，$x_i$ 是实体特征。

**关系嵌入 / Relation Embeddings:**

$$\mathbf{r}_i = g_\phi(y_i)$$

其中 $g_\phi$ 是神经网络，$y_i$ 是关系特征。

### 6.2 神经知识图谱 / Neural Knowledge Graph

**图神经网络 / Graph Neural Networks:**

$$\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(l)}\right)$$

**注意力机制 / Attention Mechanism:**

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_k]))}$$

### 6.3 神经推理 / Neural Reasoning

**神经逻辑编程 / Neural Logic Programming:**

$$P(y|x) = \sum_{z} P(y|z) P(z|x)$$

**神经符号推理 / Neural Symbolic Reasoning:**

$$\text{Reasoning}(x) = \text{Symbolic}(\text{Neural}(x))$$

## 代码示例 / Code Examples

### Rust实现：知识表示系统

```rust
use std::collections::{HashMap, HashSet};
use std::fmt;

// 实体定义
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Entity {
    id: String,
    name: String,
    entity_type: String,
    properties: HashMap<String, String>,
}

// 关系定义
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Relation {
    id: String,
    name: String,
    domain: String,
    range: String,
    properties: HashMap<String, String>,
}

// 三元组定义
#[derive(Clone, Debug)]
struct Triple {
    head: Entity,
    relation: Relation,
    tail: Entity,
}

// 语义网络
struct SemanticNetwork {
    entities: HashMap<String, Entity>,
    relations: HashMap<String, Relation>,
    triples: Vec<Triple>,
    graph: HashMap<String, HashMap<String, Vec<String>>>,
}

impl SemanticNetwork {
    fn new() -> Self {
        Self {
            entities: HashMap::new(),
            relations: HashMap::new(),
            triples: Vec::new(),
            graph: HashMap::new(),
        }
    }
    
    // 添加实体
    fn add_entity(&mut self, entity: Entity) {
        self.entities.insert(entity.id.clone(), entity);
    }
    
    // 添加关系
    fn add_relation(&mut self, relation: Relation) {
        self.relations.insert(relation.id.clone(), relation);
    }
    
    // 添加三元组
    fn add_triple(&mut self, triple: Triple) {
        self.triples.push(triple.clone());
        
        // 更新图结构
        let head_id = triple.head.id.clone();
        let tail_id = triple.tail.id.clone();
        let relation_id = triple.relation.id.clone();
        
        self.graph.entry(head_id.clone())
            .or_insert_with(HashMap::new)
            .entry(relation_id.clone())
            .or_insert_with(Vec::new)
            .push(tail_id.clone());
    }
    
    // 继承推理
    fn inheritance_reasoning(&self, entity_id: &str, property: &str) -> Option<String> {
        if let Some(entity) = self.entities.get(entity_id) {
            // 检查直接属性
            if let Some(value) = entity.properties.get(property) {
                return Some(value.clone());
            }
            
            // 检查继承属性
            if let Some(is_a_relations) = self.graph.get(entity_id) {
                for (relation_id, target_ids) in is_a_relations {
                    if relation_id == "is-a" {
                        for target_id in target_ids {
                            if let Some(inherited_value) = self.inheritance_reasoning(target_id, property) {
                                return Some(inherited_value);
                            }
                        }
                    }
                }
            }
        }
        None
    }
    
    // 相似性推理
    fn similarity_reasoning(&self, entity1_id: &str, entity2_id: &str) -> f64 {
        if let (Some(entity1), Some(entity2)) = (self.entities.get(entity1_id), self.entities.get(entity2_id)) {
            let common_properties: HashSet<_> = entity1.properties.keys()
                .intersection(&entity2.properties.keys())
                .collect();
            
            let all_properties: HashSet<_> = entity1.properties.keys()
                .union(&entity2.properties.keys())
                .collect();
            
            if all_properties.is_empty() {
                return 0.0;
            }
            
            common_properties.len() as f64 / all_properties.len() as f64
        } else {
            0.0
        }
    }
    
    // 路径推理
    fn path_reasoning(&self, start_id: &str, end_id: &str, max_depth: usize) -> Vec<Vec<String>> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        
        self.dfs_path(start_id, end_id, &mut Vec::new(), &mut paths, &mut visited, max_depth);
        paths
    }
    
    fn dfs_path(&self, current_id: &str, end_id: &str, current_path: &mut Vec<String>, 
                paths: &mut Vec<Vec<String>>, visited: &mut HashSet<String>, max_depth: usize) {
        if current_path.len() > max_depth {
            return;
        }
        
        if current_id == end_id {
            paths.push(current_path.clone());
            return;
        }
        
        if visited.contains(current_id) {
            return;
        }
        
        visited.insert(current_id.to_string());
        current_path.push(current_id.to_string());
        
        if let Some(neighbors) = self.graph.get(current_id) {
            for (relation_id, target_ids) in neighbors {
                for target_id in target_ids {
                    self.dfs_path(target_id, end_id, current_path, paths, visited, max_depth);
                }
            }
        }
        
        current_path.pop();
        visited.remove(current_id);
    }
}

// 框架系统
struct Frame {
    name: String,
    slots: HashMap<String, Slot>,
    parent: Option<String>,
    children: Vec<String>,
}

struct Slot {
    name: String,
    slot_type: String,
    value: Option<String>,
    default_value: Option<String>,
    constraints: Vec<String>,
}

struct FrameSystem {
    frames: HashMap<String, Frame>,
}

impl FrameSystem {
    fn new() -> Self {
        Self {
            frames: HashMap::new(),
        }
    }
    
    // 添加框架
    fn add_frame(&mut self, frame: Frame) {
        self.frames.insert(frame.name.clone(), frame);
    }
    
    // 框架匹配
    fn frame_matching(&self, frame1_name: &str, frame2_name: &str) -> f64 {
        if let (Some(frame1), Some(frame2)) = (self.frames.get(frame1_name), self.frames.get(frame2_name)) {
            let mut total_similarity = 0.0;
            let mut slot_count = 0;
            
            for (slot_name, slot1) in &frame1.slots {
                if let Some(slot2) = frame2.slots.get(slot_name) {
                    let similarity = self.slot_similarity(slot1, slot2);
                    total_similarity += similarity;
                    slot_count += 1;
                }
            }
            
            if slot_count > 0 {
                total_similarity / slot_count as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    // 槽位相似性
    fn slot_similarity(&self, slot1: &Slot, slot2: &Slot) -> f64 {
        let mut similarity = 0.0;
        
        // 类型相似性
        if slot1.slot_type == slot2.slot_type {
            similarity += 0.5;
        }
        
        // 值相似性
        if let (Some(val1), Some(val2)) = (&slot1.value, &slot2.value) {
            if val1 == val2 {
                similarity += 0.5;
            }
        }
        
        similarity
    }
    
    // 框架继承
    fn frame_inheritance(&self, child_name: &str, property: &str) -> Option<String> {
        if let Some(child_frame) = self.frames.get(child_name) {
            // 检查直接属性
            if let Some(slot) = child_frame.slots.get(property) {
                if let Some(value) = &slot.value {
                    return Some(value.clone());
                }
            }
            
            // 检查父框架
            if let Some(parent_name) = &child_frame.parent {
                return self.frame_inheritance(parent_name, property);
            }
        }
        None
    }
}

// 知识图谱嵌入
struct KnowledgeGraphEmbedding {
    entity_embeddings: HashMap<String, Vec<f64>>,
    relation_embeddings: HashMap<String, Vec<f64>>,
    embedding_dim: usize,
}

impl KnowledgeGraphEmbedding {
    fn new(embedding_dim: usize) -> Self {
        Self {
            entity_embeddings: HashMap::new(),
            relation_embeddings: HashMap::new(),
            embedding_dim,
        }
    }
    
    // TransE模型
    fn transe_score(&self, head_id: &str, relation_id: &str, tail_id: &str) -> f64 {
        if let (Some(head_emb), Some(rel_emb), Some(tail_emb)) = 
            (self.entity_embeddings.get(head_id), 
             self.relation_embeddings.get(relation_id),
             self.entity_embeddings.get(tail_id)) {
            
            let mut score = 0.0;
            for i in 0..self.embedding_dim {
                score += (head_emb[i] + rel_emb[i] - tail_emb[i]).powi(2);
            }
            -score.sqrt()
        } else {
            f64::NEG_INFINITY
        }
    }
    
    // DistMult模型
    fn distmult_score(&self, head_id: &str, relation_id: &str, tail_id: &str) -> f64 {
        if let (Some(head_emb), Some(rel_emb), Some(tail_emb)) = 
            (self.entity_embeddings.get(head_id), 
             self.relation_embeddings.get(relation_id),
             self.entity_embeddings.get(tail_id)) {
            
            let mut score = 0.0;
            for i in 0..self.embedding_dim {
                score += head_emb[i] * rel_emb[i] * tail_emb[i];
            }
            score
        } else {
            f64::NEG_INFINITY
        }
    }
    
    // 链接预测
    fn link_prediction(&self, head_id: &str, relation_id: &str, candidates: &[String]) -> Vec<(String, f64)> {
        let mut scores = Vec::new();
        
        for candidate in candidates {
            let score = self.transe_score(head_id, relation_id, candidate);
            scores.push((candidate.clone(), score));
        }
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores
    }
}

fn main() {
    println!("=== 知识表示系统示例 ===");
    
    // 1. 语义网络
    let mut semantic_net = SemanticNetwork::new();
    
    // 添加实体
    let person_entity = Entity {
        id: "person".to_string(),
        name: "Person".to_string(),
        entity_type: "Class".to_string(),
        properties: HashMap::new(),
    };
    
    let student_entity = Entity {
        id: "student".to_string(),
        name: "Student".to_string(),
        entity_type: "Class".to_string(),
        properties: HashMap::new(),
    };
    
    let john_entity = Entity {
        id: "john".to_string(),
        name: "John".to_string(),
        entity_type: "Instance".to_string(),
        properties: HashMap::new(),
    };
    
    semantic_net.add_entity(person_entity);
    semantic_net.add_entity(student_entity);
    semantic_net.add_entity(john_entity);
    
    // 添加关系
    let is_a_relation = Relation {
        id: "is-a".to_string(),
        name: "is-a".to_string(),
        domain: "Instance".to_string(),
        range: "Class".to_string(),
        properties: HashMap::new(),
    };
    
    semantic_net.add_relation(is_a_relation.clone());
    
    // 添加三元组
    let triple = Triple {
        head: student_entity.clone(),
        relation: is_a_relation.clone(),
        tail: person_entity.clone(),
    };
    
    semantic_net.add_triple(triple);
    
    // 推理
    if let Some(inherited) = semantic_net.inheritance_reasoning("student", "has-property") {
        println!("继承推理结果: {}", inherited);
    }
    
    let similarity = semantic_net.similarity_reasoning("person", "student");
    println!("相似性推理结果: {:.2}", similarity);
    
    // 2. 框架系统
    let mut frame_system = FrameSystem::new();
    
    let vehicle_frame = Frame {
        name: "Vehicle".to_string(),
        slots: HashMap::new(),
        parent: None,
        children: vec!["Car".to_string()],
    };
    
    let car_frame = Frame {
        name: "Car".to_string(),
        slots: HashMap::new(),
        parent: Some("Vehicle".to_string()),
        children: Vec::new(),
    };
    
    frame_system.add_frame(vehicle_frame);
    frame_system.add_frame(car_frame);
    
    let matching_score = frame_system.frame_matching("Car", "Vehicle");
    println!("框架匹配分数: {:.2}", matching_score);
    
    // 3. 知识图谱嵌入
    let mut kg_embedding = KnowledgeGraphEmbedding::new(10);
    
    // 初始化嵌入（实际应用中需要训练）
    kg_embedding.entity_embeddings.insert("head".to_string(), vec![0.1; 10]);
    kg_embedding.relation_embeddings.insert("relation".to_string(), vec![0.2; 10]);
    kg_embedding.entity_embeddings.insert("tail".to_string(), vec![0.3; 10]);
    
    let transe_score = kg_embedding.transe_score("head", "relation", "tail");
    println!("TransE分数: {:.4}", transe_score);
    
    let distmult_score = kg_embedding.distmult_score("head", "relation", "tail");
    println!("DistMult分数: {:.4}", distmult_score);
}
```

### Haskell实现：知识表示

```haskell
-- 知识表示模块
module KnowledgeRepresentation where

import Data.Map (Map)
import qualified Data.Map as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Maybe (fromMaybe)

-- 实体定义
data Entity = Entity
    { entityId :: String
    , entityName :: String
    , entityType :: String
    , properties :: Map String String
    } deriving (Show, Eq, Ord)

-- 关系定义
data Relation = Relation
    { relationId :: String
    , relationName :: String
    , domain :: String
    , range :: String
    , relationProperties :: Map String String
    } deriving (Show, Eq, Ord)

-- 三元组定义
data Triple = Triple
    { head :: Entity
    , relation :: Relation
    , tail :: Entity
    } deriving (Show, Eq)

-- 语义网络
data SemanticNetwork = SemanticNetwork
    { entities :: Map String Entity
    , relations :: Map String Relation
    , triples :: [Triple]
    , graph :: Map String (Map String [String])
    } deriving (Show)

-- 框架定义
data Frame = Frame
    { frameName :: String
    , slots :: Map String Slot
    , parent :: Maybe String
    , children :: [String]
    } deriving (Show)

data Slot = Slot
    { slotName :: String
    , slotType :: String
    , slotValue :: Maybe String
    , defaultValue :: Maybe String
    , constraints :: [String]
    } deriving (Show)

-- 框架系统
data FrameSystem = FrameSystem
    { frames :: Map String Frame
    } deriving (Show)

-- 知识图谱嵌入
data KnowledgeGraphEmbedding = KnowledgeGraphEmbedding
    { entityEmbeddings :: Map String [Double]
    , relationEmbeddings :: Map String [Double]
    , embeddingDim :: Int
    } deriving (Show)

-- 创建新的语义网络
newSemanticNetwork :: SemanticNetwork
newSemanticNetwork = SemanticNetwork Map.empty Map.empty [] Map.empty

-- 添加实体
addEntity :: SemanticNetwork -> Entity -> SemanticNetwork
addEntity network entity = network
    { entities = Map.insert (entityId entity) entity (entities network)
    }

-- 添加关系
addRelation :: SemanticNetwork -> Relation -> SemanticNetwork
addRelation network relation = network
    { relations = Map.insert (relationId relation) relation (relations network)
    }

-- 添加三元组
addTriple :: SemanticNetwork -> Triple -> SemanticNetwork
addTriple network triple = network
    { triples = triple : triples network
    , graph = updateGraph (graph network) triple
    }
  where
    updateGraph g t = 
        let headId = entityId (head t)
            tailId = entityId (tail t)
            relationId = relationId (relation t)
            currentNeighbors = Map.findWithDefault Map.empty headId g
            currentRelations = Map.findWithDefault [] relationId currentNeighbors
            newRelations = Map.insert relationId (tailId : currentRelations) currentNeighbors
        in Map.insert headId newRelations g

-- 继承推理
inheritanceReasoning :: SemanticNetwork -> String -> String -> Maybe String
inheritanceReasoning network entityId property = 
    case Map.lookup entityId (entities network) of
        Just entity -> 
            case Map.lookup property (properties entity) of
                Just value -> Just value
                Nothing -> checkInheritance network entityId property
        Nothing -> Nothing
  where
    checkInheritance net eId prop = 
        case Map.lookup eId (graph net) of
            Just neighbors -> 
                case Map.lookup "is-a" neighbors of
                    Just targetIds -> 
                        foldr (\targetId acc -> 
                            case acc of
                                Just _ -> acc
                                Nothing -> inheritanceReasoning net targetId prop) 
                            Nothing targetIds
                    Nothing -> Nothing
            Nothing -> Nothing

-- 相似性推理
similarityReasoning :: SemanticNetwork -> String -> String -> Double
similarityReasoning network entity1Id entity2Id = 
    case (Map.lookup entity1Id (entities network), Map.lookup entity2Id (entities network)) of
        (Just entity1, Just entity2) -> 
            let commonProps = Set.intersection 
                    (Set.fromList $ Map.keys $ properties entity1)
                    (Set.fromList $ Map.keys $ properties entity2)
                allProps = Set.union 
                    (Set.fromList $ Map.keys $ properties entity1)
                    (Set.fromList $ Map.keys $ properties entity2)
            in if Set.null allProps 
                then 0.0 
                else fromIntegral (Set.size commonProps) / fromIntegral (Set.size allProps)
        _ -> 0.0

-- 路径推理
pathReasoning :: SemanticNetwork -> String -> String -> Int -> [[String]]
pathReasoning network startId endId maxDepth = 
    dfsPath network startId endId [] [] Set.empty maxDepth
  where
    dfsPath net current end currentPath paths visited depth
        | length currentPath > depth = paths
        | current == end = paths ++ [reverse currentPath]
        | Set.member current visited = paths
        | otherwise = 
            let newVisited = Set.insert current visited
                newPath = current : currentPath
                neighbors = getNeighbors net current
                newPaths = foldr (\neighbor acc -> 
                    dfsPath net neighbor end newPath acc newVisited depth) 
                    paths neighbors
            in dfsPath net current end currentPath newPaths visited depth
    
    getNeighbors net nodeId = 
        case Map.lookup nodeId (graph net) of
            Just neighbors -> concat $ Map.elems neighbors
            Nothing -> []

-- 创建新的框架系统
newFrameSystem :: FrameSystem
newFrameSystem = FrameSystem Map.empty

-- 添加框架
addFrame :: FrameSystem -> Frame -> FrameSystem
addFrame system frame = system
    { frames = Map.insert (frameName frame) frame (frames system)
    }

-- 框架匹配
frameMatching :: FrameSystem -> String -> String -> Double
frameMatching system frame1Name frame2Name = 
    case (Map.lookup frame1Name (frames system), Map.lookup frame2Name (frames system)) of
        (Just frame1, Just frame2) -> 
            let slotPairs = Map.intersectionWith (,) (slots frame1) (slots frame2)
                similarities = map (uncurry slotSimilarity) $ Map.toList slotPairs
            in if null similarities 
                then 0.0 
                else sum similarities / fromIntegral (length similarities)
        _ -> 0.0
  where
    slotSimilarity slot1 slot2 = 
        let typeSim = if slotType slot1 == slotType slot2 then 0.5 else 0.0
            valueSim = case (slotValue slot1, slotValue slot2) of
                (Just v1, Just v2) -> if v1 == v2 then 0.5 else 0.0
                _ -> 0.0
        in typeSim + valueSim

-- 框架继承
frameInheritance :: FrameSystem -> String -> String -> Maybe String
frameInheritance system childName property = 
    case Map.lookup childName (frames system) of
        Just childFrame -> 
            case Map.lookup property (slots childFrame) of
                Just slot -> slotValue slot
                Nothing -> 
                    case parent childFrame of
                        Just parentName -> frameInheritance system parentName property
                        Nothing -> Nothing
        Nothing -> Nothing

-- 创建新的知识图谱嵌入
newKnowledgeGraphEmbedding :: Int -> KnowledgeGraphEmbedding
newKnowledgeGraphEmbedding dim = KnowledgeGraphEmbedding Map.empty Map.empty dim

-- TransE模型
transeScore :: KnowledgeGraphEmbedding -> String -> String -> String -> Double
transeScore embedding headId relationId tailId = 
    case (Map.lookup headId (entityEmbeddings embedding), 
          Map.lookup relationId (relationEmbeddings embedding),
          Map.lookup tailId (entityEmbeddings embedding)) of
        (Just headEmb, Just relEmb, Just tailEmb) -> 
            let score = sum $ zipWith3 (\h r t -> (h + r - t)^2) headEmb relEmb tailEmb
            in -sqrt score
        _ -> negate infinity

-- DistMult模型
distmultScore :: KnowledgeGraphEmbedding -> String -> String -> String -> Double
distmultScore embedding headId relationId tailId = 
    case (Map.lookup headId (entityEmbeddings embedding), 
          Map.lookup relationId (relationEmbeddings embedding),
          Map.lookup tailId (entityEmbeddings embedding)) of
        (Just headEmb, Just relEmb, Just tailEmb) -> 
            sum $ zipWith3 (\h r t -> h * r * t) headEmb relEmb tailEmb
        _ -> 0.0

-- 示例使用
main :: IO ()
main = do
    putStrLn "=== 知识表示系统示例 ==="
    
    -- 1. 语义网络
    let initialNetwork = newSemanticNetwork
    
    let personEntity = Entity "person" "Person" "Class" Map.empty
    let studentEntity = Entity "student" "Student" "Class" Map.empty
    let johnEntity = Entity "john" "John" "Instance" Map.empty
    
    let network1 = addEntity initialNetwork personEntity
    let network2 = addEntity network1 studentEntity
    let network3 = addEntity network2 johnEntity
    
    let isARelation = Relation "is-a" "is-a" "Instance" "Class" Map.empty
    let network4 = addRelation network3 isARelation
    
    let triple = Triple studentEntity isARelation personEntity
    let finalNetwork = addTriple network4 triple
    
    -- 推理
    case inheritanceReasoning finalNetwork "student" "has-property" of
        Just inherited -> putStrLn $ "继承推理结果: " ++ inherited
        Nothing -> putStrLn "无继承属性"
    
    let similarity = similarityReasoning finalNetwork "person" "student"
    putStrLn $ "相似性推理结果: " ++ show similarity
    
    -- 2. 框架系统
    let initialSystem = newFrameSystem
    
    let vehicleFrame = Frame "Vehicle" Map.empty Nothing ["Car"]
    let carFrame = Frame "Car" Map.empty (Just "Vehicle") []
    
    let system1 = addFrame initialSystem vehicleFrame
    let system2 = addFrame system1 carFrame
    
    let matchingScore = frameMatching system2 "Car" "Vehicle"
    putStrLn $ "框架匹配分数: " ++ show matchingScore
    
    -- 3. 知识图谱嵌入
    let embedding = newKnowledgeGraphEmbedding 10
    
    let transeScore = transeScore embedding "head" "relation" "tail"
    putStrLn $ "TransE分数: " ++ show transeScore
    
    let distmultScore = distmultScore embedding "head" "relation" "tail"
    putStrLn $ "DistMult分数: " ++ show distmultScore
```

## 参考文献 / References

1. Sowa, J. F. (1991). Principles of semantic networks: Explorations in the representation of knowledge. Morgan Kaufmann.
2. Minsky, M. (1975). A framework for representing knowledge. The Psychology of Computer Vision.
3. Baader, F., et al. (2003). The description logic handbook: Theory, implementation, and applications. Cambridge University Press.
4. Gruber, T. R. (1993). A translation approach to portable ontology specifications. Knowledge Acquisition.
5. Bollacker, K., et al. (2008). Freebase: A collaboratively created graph database for structuring human knowledge. SIGMOD.
6. Bordes, A., et al. (2013). Translating embeddings for modeling multi-relational data. NeurIPS.
7. Yang, B., et al. (2015). Embedding entities and relations for learning and inference in knowledge bases. ICLR.
8. Trouillon, T., et al. (2016). Complex embeddings for simple link prediction. ICML.
9. Dettmers, T., et al. (2018). Convolutional 2D knowledge graph embeddings. AAAI.
10. Vashishth, S., et al. (2020). Composition-based multi-relational graph convolutional networks. ICLR.

---

*知识表示为FormalAI提供了结构化和语义化的知识管理能力，是实现智能推理和知识发现的重要理论基础。*

*Knowledge representation provides structured and semantic knowledge management capabilities for FormalAI, serving as important theoretical foundations for intelligent reasoning and knowledge discovery.*
