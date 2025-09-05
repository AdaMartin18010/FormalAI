# 0.1 范畴论 / Category Theory / Kategorientheorie / Théorie des catégories

[返回全局导航](../../GLOBAL_NAVIGATION.md) · [学习路径](../../LEARNING_PATH_DESIGN.md)

## 概述 / Overview / Übersicht / Aperçu

范畴论是现代数学的统一语言，为FormalAI提供抽象的数学框架。本模块建立完整的范畴论基础，将AI理论统一在范畴论的框架下。

Category theory is the unified language of modern mathematics, providing FormalAI with an abstract mathematical framework. This module establishes a complete foundation of category theory, unifying AI theories under the categorical framework.

## 目录 / Table of Contents / Inhaltsverzeichnis / Table des matières

- [0.1 范畴论](#01-范畴论--category-theory--kategorientheorie--théorie-des-catégories)
  - [概述](#概述--overview--übersicht--aperçu)
  - [目录](#目录--table-of-contents--inhaltsverzeichnis--table-des-matières)
  - [1. 基本概念](#1-基本概念--basic-concepts--grundbegriffe--concepts-de-base)
  - [2. 范畴的公理化定义](#2-范畴的公理化定义--axiomatic-definition-of-categories--axiomatische-definition-von-kategorien--définition-axiomatique-des-catégories)
  - [3. 重要范畴](#3-重要范畴--important-categories--wichtige-kategorien--catégories-importantes)
  - [4. 函子与自然变换](#4-函子与自然变换--functors-and-natural-transformations--funktoren-und-natürliche-transformationen--foncteurs-et-transformations-naturelles)
  - [5. 极限与余极限](#5-极限与余极限--limits-and-colimits--limites-und-kolimites--limites-et-colimites)
  - [6. 伴随函子](#6-伴随函子--adjoint-functors--adjungierte-funktoren--foncteurs-adjoints)
  - [7. AI理论中的范畴论应用](#7-ai理论中的范畴论应用--categorical-applications-in-ai-theory--kategorientheoretische-anwendungen-in-der-ki-theorie--applications-catégorielles-dans-la-théorie-ia)
  - [代码实现](#代码实现--code-implementation--code-implementierung--implémentation-de-code)
  - [参考文献](#参考文献--references--literatur--références)

## 1. 基本概念 / Basic Concepts / Grundbegriffe / Concepts de base

### 1.1 范畴的定义 / Definition of Category / Definition der Kategorie / Définition de catégorie

**定义 1.1.1 (范畴)**
范畴 $\mathcal{C}$ 由以下数据组成：

1. **对象类** $\text{Ob}(\mathcal{C})$：范畴中的对象
2. **态射类** $\text{Mor}(\mathcal{C})$：对象之间的态射
3. **复合运算** $\circ$：态射的复合
4. **恒等态射** $\text{id}_A$：每个对象的恒等态射

满足以下公理：

**公理 1.1.1 (结合律)**
对于态射 $f: A \to B$，$g: B \to C$，$h: C \to D$：
$$h \circ (g \circ f) = (h \circ g) \circ f$$

**公理 1.1.2 (恒等律)**
对于态射 $f: A \to B$：
$$f \circ \text{id}_A = f = \text{id}_B \circ f$$

**公理 1.1.3 (复合的良定义性)**
态射 $f: A \to B$ 和 $g: C \to D$ 可以复合当且仅当 $B = C$。

### 1.2 态射的性质 / Properties of Morphisms / Eigenschaften von Morphismen / Propriétés des morphismes

**定义 1.2.1 (单态射)**
态射 $f: A \to B$ 是单态射，如果对于任意态射 $g, h: C \to A$：
$$f \circ g = f \circ h \Rightarrow g = h$$

**定义 1.2.2 (满态射)**
态射 $f: A \to B$ 是满态射，如果对于任意态射 $g, h: B \to C$：
$$g \circ f = h \circ f \Rightarrow g = h$$

**定义 1.2.3 (同构)**
态射 $f: A \to B$ 是同构，如果存在态射 $g: B \to A$ 使得：
$$g \circ f = \text{id}_A \quad \text{且} \quad f \circ g = \text{id}_B$$

## 2. 范畴的公理化定义 / Axiomatic Definition of Categories / Axiomatische Definition von Kategorien / Définition axiomatique des catégories

### 2.1 小范畴与大范畴 / Small and Large Categories / Kleine und große Kategorien / Petites et grandes catégories

**定义 2.1.1 (小范畴)**
如果 $\text{Ob}(\mathcal{C})$ 和 $\text{Mor}(\mathcal{C})$ 都是集合，则称 $\mathcal{C}$ 为小范畴。

**定义 2.1.2 (局部小范畴)**
如果对于任意对象 $A, B$，态射集合 $\text{Hom}(A, B)$ 是集合，则称 $\mathcal{C}$ 为局部小范畴。

### 2.2 范畴的构造 / Construction of Categories / Konstruktion von Kategorien / Construction de catégories

**定理 2.2.1 (对偶范畴)**
对于范畴 $\mathcal{C}$，存在对偶范畴 $\mathcal{C}^{\text{op}}$，其中：

- $\text{Ob}(\mathcal{C}^{\text{op}}) = \text{Ob}(\mathcal{C})$
- $\text{Hom}_{\mathcal{C}^{\text{op}}}(A, B) = \text{Hom}_{\mathcal{C}}(B, A)$
- 复合运算反向：$(f \circ g)^{\text{op}} = g^{\text{op}} \circ f^{\text{op}}$

**证明：**
直接验证范畴公理即可。□

## 3. 重要范畴 / Important Categories / Wichtige Kategorien / Catégories importantes

### 3.1 集合范畴 / Category of Sets / Kategorie der Mengen / Catégorie des ensembles

**定义 3.1.1 (Set)**
集合范畴 $\mathbf{Set}$ 定义为：

- 对象：所有集合
- 态射：集合之间的函数
- 复合：函数的复合
- 恒等：恒等函数

**定理 3.1.1**
$\mathbf{Set}$ 是局部小范畴。

**证明：**
对于集合 $A$ 和 $B$，$\text{Hom}(A, B) = \{f: A \to B\}$ 是集合。□

### 3.2 群范畴 / Category of Groups / Kategorie der Gruppen / Catégorie des groupes

**定义 3.2.1 (Grp)**
群范畴 $\mathbf{Grp}$ 定义为：

- 对象：所有群
- 态射：群同态
- 复合：群同态的复合
- 恒等：恒等群同态

### 3.3 拓扑空间范畴 / Category of Topological Spaces / Kategorie der topologischen Räume / Catégorie des espaces topologiques

**定义 3.3.1 (Top)**
拓扑空间范畴 $\mathbf{Top}$ 定义为：

- 对象：所有拓扑空间
- 态射：连续函数
- 复合：连续函数的复合
- 恒等：恒等连续函数

### 3.4 向量空间范畴 / Category of Vector Spaces / Kategorie der Vektorräume / Catégorie des espaces vectoriels

**定义 3.4.1 (Vect)**
向量空间范畴 $\mathbf{Vect}_k$（$k$ 是域）定义为：

- 对象：$k$ 上的向量空间
- 态射：线性映射
- 复合：线性映射的复合
- 恒等：恒等线性映射

## 4. 函子与自然变换 / Functors and Natural Transformations / Funktoren und natürliche Transformationen / Foncteurs et transformations naturelles

### 4.1 函子的定义 / Definition of Functor / Definition des Funktors / Définition du foncteur

**定义 4.1.1 (协变函子)**
协变函子 $F: \mathcal{C} \to \mathcal{D}$ 由以下数据组成：

1. 对象映射：$F: \text{Ob}(\mathcal{C}) \to \text{Ob}(\mathcal{D})$
2. 态射映射：$F: \text{Hom}(A, B) \to \text{Hom}(F(A), F(B))$

满足：

- $F(\text{id}_A) = \text{id}_{F(A)}$
- $F(g \circ f) = F(g) \circ F(f)$

**定义 4.1.2 (反变函子)**
反变函子 $F: \mathcal{C} \to \mathcal{D}$ 是协变函子 $F: \mathcal{C}^{\text{op}} \to \mathcal{D}$。

### 4.2 自然变换 / Natural Transformations / Natürliche Transformationen / Transformations naturelles

**定义 4.2.1 (自然变换)**
对于函子 $F, G: \mathcal{C} \to \mathcal{D}$，自然变换 $\eta: F \Rightarrow G$ 是态射族 $\{\eta_A: F(A) \to G(A)\}_{A \in \text{Ob}(\mathcal{C})}$，使得对于任意态射 $f: A \to B$：

$$\eta_B \circ F(f) = G(f) \circ \eta_A$$

即下图交换：

```text
F(A) --F(f)--> F(B)
 |              |
η_A             η_B
 |              |
 v              v
G(A) --G(f)--> G(B)
```

### 4.3 重要函子 / Important Functors / Wichtige Funktoren / Foncteurs importants

**定义 4.3.1 (遗忘函子)**
遗忘函子 $U: \mathbf{Grp} \to \mathbf{Set}$ 将群映射到其底层集合，群同态映射到函数。

**定义 4.3.2 (自由函子)**
自由函子 $F: \mathbf{Set} \to \mathbf{Grp}$ 将集合映射到其生成的自由群。

**定理 4.3.1 (伴随关系)**
$F \dashv U$，即 $F$ 是 $U$ 的左伴随。

**证明：**
需要证明自然同构：
$$\text{Hom}_{\mathbf{Grp}}(F(X), G) \cong \text{Hom}_{\mathbf{Set}}(X, U(G))$$

对于集合 $X$ 和群 $G$，这个同构由自由群的泛性质给出。□

## 5. 极限与余极限 / Limits and Colimits / Limites und Kolimites / Limites et colimites

### 5.1 锥与余锥 / Cones and Cocones / Kegel und Kokegel / Cônes et cocônes

**定义 5.1.1 (锥)**
对于函子 $F: \mathcal{J} \to \mathcal{C}$，锥 $(C, \psi)$ 由对象 $C$ 和态射族 $\{\psi_j: C \to F(j)\}_{j \in \mathcal{J}}$ 组成，使得对于 $\mathcal{J}$ 中的任意态射 $f: j \to j'$：

$$F(f) \circ \psi_j = \psi_{j'}$$

**定义 5.1.2 (极限)**
锥 $(L, \phi)$ 是 $F$ 的极限，如果对于任意锥 $(C, \psi)$，存在唯一的态射 $u: C \to L$ 使得：

$$\phi_j \circ u = \psi_j \quad \forall j \in \mathcal{J}$$

### 5.2 特殊极限 / Special Limits / Spezielle Limites / Limites spéciales

**定义 5.2.1 (积)**
两个对象 $A$ 和 $B$ 的积是对象 $A \times B$ 和投影态射 $\pi_1: A \times B \to A$，$\pi_2: A \times B \to B$，满足泛性质。

**定义 5.2.2 (等化子)**
态射 $f, g: A \to B$ 的等化子是对象 $E$ 和态射 $e: E \to A$，使得 $f \circ e = g \circ e$，且满足泛性质。

**定义 5.2.3 (拉回)**
态射 $f: A \to C$ 和 $g: B \to C$ 的拉回是对象 $P$ 和态射 $p_1: P \to A$，$p_2: P \to B$，使得 $f \circ p_1 = g \circ p_2$，且满足泛性质。

### 5.3 余极限 / Colimits / Kolimites / Colimites

**定义 5.3.1 (余锥)**
对于函子 $F: \mathcal{J} \to \mathcal{C}$，余锥 $(C, \psi)$ 由对象 $C$ 和态射族 $\{\psi_j: F(j) \to C\}_{j \in \mathcal{J}}$ 组成。

**定义 5.3.2 (余极限)**
余锥 $(L, \phi)$ 是 $F$ 的余极限，如果对于任意余锥 $(C, \psi)$，存在唯一的态射 $u: L \to C$ 使得：

$$u \circ \phi_j = \psi_j \quad \forall j \in \mathcal{J}$$

## 6. 伴随函子 / Adjoint Functors / Adjungierte Funktoren / Foncteurs adjoints

### 6.1 伴随的定义 / Definition of Adjoint / Definition der Adjunktion / Définition de l'adjoint

**定义 6.1.1 (伴随)**
函子 $F: \mathcal{C} \to \mathcal{D}$ 和 $G: \mathcal{D} \to \mathcal{C}$ 是伴随的，记作 $F \dashv G$，如果存在自然同构：

$$\text{Hom}_{\mathcal{D}}(F(C), D) \cong \text{Hom}_{\mathcal{C}}(C, G(D))$$

### 6.2 伴随的等价定义 / Equivalent Definitions of Adjoint / Äquivalente Definitionen der Adjunktion / Définitions équivalentes de l'adjoint

**定理 6.2.1**
$F \dashv G$ 当且仅当存在自然变换：

- $\eta: \text{id}_{\mathcal{C}} \Rightarrow G \circ F$（单位）
- $\varepsilon: F \circ G \Rightarrow \text{id}_{\mathcal{D}}$（余单位）

满足三角恒等式：

- $(G\varepsilon) \circ (\eta G) = \text{id}_G$
- $(\varepsilon F) \circ (F\eta) = \text{id}_F$

**证明：**
（必要性）设 $F \dashv G$，定义：

- $\eta_C = \phi_{F(C), F(C)}(\text{id}_{F(C)})$
- $\varepsilon_D = \phi_{G(D), D}^{-1}(\text{id}_{G(D)})$

（充分性）定义 $\phi_{C,D}(f) = G(f) \circ \eta_C$。□

## 7. AI理论中的范畴论应用 / Categorical Applications in AI Theory / Kategorientheoretische Anwendungen in der KI-Theorie / Applications catégorielles dans la théorie IA

### 7.1 机器学习范畴 / Machine Learning Category / Maschinelles Lernen Kategorie / Catégorie d'apprentissage automatique

**定义 7.1.1 (ML范畴)**
机器学习范畴 $\mathbf{ML}$ 定义为：

- 对象：$(X, Y, \mathcal{H}, \ell)$，其中 $X$ 是输入空间，$Y$ 是输出空间，$\mathcal{H}$ 是假设空间，$\ell$ 是损失函数
- 态射：$(f, g): (X_1, Y_1, \mathcal{H}_1, \ell_1) \to (X_2, Y_2, \mathcal{H}_2, \ell_2)$ 是数据变换 $f: X_1 \to X_2$ 和模型变换 $g: \mathcal{H}_1 \to \mathcal{H_2}$

**定理 7.1.1**
$\mathbf{ML}$ 是范畴。

**证明：**
验证范畴公理：

1. 恒等态射：$(\text{id}_X, \text{id}_{\mathcal{H}})$
2. 复合：$(f_2, g_2) \circ (f_1, g_1) = (f_2 \circ f_1, g_2 \circ g_1)$
3. 结合律和恒等律显然成立。□

### 7.2 神经网络范畴 / Neural Network Category / Neuronales Netzwerk Kategorie / Catégorie de réseau neuronal

**定义 7.2.1 (NN范畴)**
神经网络范畴 $\mathbf{NN}$ 定义为：

- 对象：$(V, E, \sigma, W)$，其中 $V$ 是节点集，$E$ 是边集，$\sigma$ 是激活函数，$W$ 是权重函数
- 态射：网络同态，保持网络结构

### 7.3 强化学习范畴 / Reinforcement Learning Category / Verstärkungslernen Kategorie / Catégorie d'apprentissage par renforcement

**定义 7.3.1 (RL范畴)**
强化学习范畴 $\mathbf{RL}$ 定义为：

- 对象：$(S, A, P, R, \gamma)$，其中 $S$ 是状态空间，$A$ 是动作空间，$P$ 是转移概率，$R$ 是奖励函数，$\gamma$ 是折扣因子
- 态射：环境同态

### 7.4 形式化验证范畴 / Formal Verification Category / Formale Verifikation Kategorie / Catégorie de vérification formelle

**定义 7.4.1 (FV范畴)**
形式化验证范畴 $\mathbf{FV}$ 定义为：

- 对象：$(M, \phi)$，其中 $M$ 是模型，$\phi$ 是性质
- 态射：$(f, \psi): (M_1, \phi_1) \to (M_2, \phi_2)$ 是模型变换 $f: M_1 \to M_2$ 和性质变换 $\psi: \phi_1 \Rightarrow \phi_2$

## 代码实现 / Code Implementation / Code-Implementierung / Implémentation de code

### Rust实现：范畴论基础 / Rust Implementation: Category Theory Foundation

```rust
use std::collections::HashMap;
use std::hash::Hash;

// 态射类型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Morphism<A, B> {
    pub source: A,
    pub target: B,
    pub name: String,
}

impl<A, B> Morphism<A, B> {
    pub fn new(source: A, target: B, name: String) -> Self {
        Morphism { source, target, name }
    }
}

// 范畴定义
pub struct Category<Obj, Mor> 
where
    Obj: Hash + Eq + Clone,
    Mor: Clone,
{
    pub objects: Vec<Obj>,
    pub morphisms: HashMap<(Obj, Obj), Vec<Mor>>,
    pub identity: HashMap<Obj, Mor>,
    pub composition: HashMap<(Mor, Mor), Mor>,
}

impl<Obj, Mor> Category<Obj, Mor>
where
    Obj: Hash + Eq + Clone,
    Mor: Clone,
{
    pub fn new() -> Self {
        Category {
            objects: Vec::new(),
            morphisms: HashMap::new(),
            identity: HashMap::new(),
            composition: HashMap::new(),
        }
    }
    
    pub fn add_object(&mut self, obj: Obj) {
        if !self.objects.contains(&obj) {
            self.objects.push(obj.clone());
        }
    }
    
    pub fn add_morphism(&mut self, source: Obj, target: Obj, morphism: Mor) {
        let key = (source, target);
        self.morphisms.entry(key).or_insert_with(Vec::new).push(morphism);
    }
    
    pub fn set_identity(&mut self, obj: Obj, identity: Mor) {
        self.identity.insert(obj, identity);
    }
    
    pub fn compose(&self, f: &Mor, g: &Mor) -> Option<Mor> {
        self.composition.get(&(f.clone(), g.clone())).cloned()
    }
}

// 函子定义
pub trait Functor<C, D> {
    type ObjectMap;
    type MorphismMap;
    
    fn map_object(&self, obj: C) -> D;
    fn map_morphism(&self, morphism: C) -> D;
}

// 自然变换
pub struct NaturalTransformation<F, G, C, D> 
where
    F: Functor<C, D>,
    G: Functor<C, D>,
{
    pub components: HashMap<C, D>,
}

impl<F, G, C, D> NaturalTransformation<F, G, C, D>
where
    F: Functor<C, D>,
    G: Functor<C, D>,
    C: Hash + Eq + Clone,
    D: Clone,
{
    pub fn new() -> Self {
        NaturalTransformation {
            components: HashMap::new(),
        }
    }
    
    pub fn add_component(&mut self, obj: C, morphism: D) {
        self.components.insert(obj, morphism);
    }
}

// 极限定义
pub struct Limit<F, C> {
    pub limit_object: C,
    pub projections: HashMap<String, C>,
}

impl<F, C> Limit<F, C> {
    pub fn new(limit_object: C) -> Self {
        Limit {
            limit_object,
            projections: HashMap::new(),
        }
    }
    
    pub fn add_projection(&mut self, name: String, morphism: C) {
        self.projections.insert(name, morphism);
    }
}

// 伴随函子
pub struct Adjunction<F, G, C, D> 
where
    F: Functor<C, D>,
    G: Functor<D, C>,
{
    pub unit: NaturalTransformation<F, G, C, C>,
    pub counit: NaturalTransformation<G, F, D, D>,
}

impl<F, G, C, D> Adjunction<F, G, C, D>
where
    F: Functor<C, D>,
    G: Functor<D, C>,
    C: Hash + Eq + Clone,
    D: Hash + Eq + Clone,
{
    pub fn new(
        unit: NaturalTransformation<F, G, C, C>,
        counit: NaturalTransformation<G, F, D, D>,
    ) -> Self {
        Adjunction { unit, counit }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_category_creation() {
        let mut cat = Category::<String, String>::new();
        cat.add_object("A".to_string());
        cat.add_object("B".to_string());
        cat.add_morphism("A".to_string(), "B".to_string(), "f".to_string());
        
        assert_eq!(cat.objects.len(), 2);
        assert!(cat.morphisms.contains_key(&("A".to_string(), "B".to_string())));
    }
    
    #[test]
    fn test_natural_transformation() {
        let mut nt = NaturalTransformation::<String, String, String, String>::new();
        nt.add_component("A".to_string(), "f".to_string());
        
        assert_eq!(nt.components.len(), 1);
    }
}
```

### Haskell实现：类型安全的范畴论 / Haskell Implementation: Type-Safe Category Theory

```haskell
{-# LANGUAGE GADTs, DataKinds, TypeFamilies, RankNTypes #-}

-- 态射类型
data Morphism a b where
  Morphism :: (a -> b) -> Morphism a b

-- 范畴类型类
class Category cat where
  id :: cat a a
  (.) :: cat b c -> cat a b -> cat a c

-- 函数范畴实例
instance Category (->) where
  id = Prelude.id
  (.) = (Prelude..)

-- 函子类型类
class Functor f where
  fmap :: (a -> b) -> f a -> f b

-- 自然变换
type NaturalTransformation f g = forall a. f a -> g a

-- 伴随函子
class (Functor f, Functor g) => Adjunction f g where
  unit :: a -> g (f a)
  counit :: f (g a) -> a

-- 极限
class Functor f => HasLimit f where
  type Limit f :: *
  limit :: Limit f -> f (Limit f)

-- 余极限
class Functor f => HasColimit f where
  type Colimit f :: *
  colimit :: f (Colimit f) -> Colimit f

-- 积
data Product a b = Product a b

instance Functor (Product a) where
  fmap f (Product a b) = Product a (f b)

-- 余积（和）
data Coproduct a b = Left a | Right b

instance Functor (Coproduct a) where
  fmap f (Left a) = Left a
  fmap f (Right b) = Right (f b)

-- 等化子
data Equalizer f g where
  Equalizer :: (a -> b) -> Equalizer f g

-- 余等化子
data Coequalizer f g where
  Coequalizer :: (a -> b) -> Coequalizer f g

-- 拉回
data Pullback f g where
  Pullback :: (a -> b) -> (a -> c) -> Pullback f g

-- 推出
data Pushout f g where
  Pushout :: (b -> a) -> (c -> a) -> Pushout f g

-- 机器学习范畴
data MLObject = MLObject
  { inputSpace :: String
  , outputSpace :: String
  , hypothesisSpace :: String
  , lossFunction :: String
  }

data MLMorphism = MLMorphism
  { dataTransform :: String -> String
  , modelTransform :: String -> String
  }

-- 神经网络范畴
data NNObject = NNObject
  { nodes :: [String]
  , edges :: [(String, String)]
  , activation :: String
  , weights :: [(String, String, Double)]
  }

data NNMorphism = NNMorphism
  { nodeMap :: String -> String
  , edgeMap :: (String, String) -> (String, String)
  , weightMap :: Double -> Double
  }

-- 强化学习范畴
data RLObject = RLObject
  { stateSpace :: [String]
  , actionSpace :: [String]
  , transitionProb :: String -> String -> Double
  , rewardFunction :: String -> String -> Double
  , discountFactor :: Double
  }

data RLMorphism = RLMorphism
  { stateMap :: String -> String
  , actionMap :: String -> String
  , probMap :: Double -> Double
  , rewardMap :: Double -> Double
  }

-- 测试
main :: IO ()
main = do
  let mlObj = MLObject "X" "Y" "H" "L"
  let mlMorph = MLMorphism id id
  
  let nnObj = NNObject ["v1", "v2"] [("v1", "v2")] "sigmoid" [("v1", "v2", 0.5)]
  let nnMorph = NNMorphism id id id
  
  let rlObj = RLObject ["s1", "s2"] ["a1", "a2"] (\_ _ -> 0.5) (\_ _ -> 1.0) 0.9
  let rlMorph = RLMorphism id id id id
  
  print "Category theory implementation completed"
```

## 参考文献 / References / Literatur / Références

1. **Mac Lane, S.** (1998). *Categories for the Working Mathematician*. Springer.
2. **Awodey, S.** (2010). *Category Theory*. Oxford University Press.
3. **Riehl, E.** (2017). *Category Theory in Context*. Dover Publications.
4. **Barr, M. & Wells, C.** (1990). *Category Theory for Computing Science*. Prentice Hall.
5. **Fong, B. & Spivak, D.** (2019). *An Invitation to Applied Category Theory*. Cambridge University Press.

---

*本模块为FormalAI提供了统一的范畴论框架，将AI理论统一在抽象的数学结构下。*

*This module provides FormalAI with a unified categorical framework, unifying AI theories under abstract mathematical structures.*

## 相关章节 / Related Chapters

**前置依赖 / Prerequisites:**

- [0.0 ZFC公理系统](00-set-theory-zfc.md)

**后续依赖 / Follow-ups:**

- [0.2 类型理论](02-type-theory.md)
- [0.3 逻辑演算系统](03-logical-calculus.md)

## 2024/2025 最新进展 / Latest Updates

- 场景范畴与代理交互的函子化建模（占位）。
- 伴随在训练-推理对偶中的应用（占位）。

## Lean 占位模板 / Lean Placeholder

```lean
-- 占位：范畴、函子与自然变换的最小接口
-- TODO: 采用 category_theory 库定义 Category/Functor/NatTrans
```
