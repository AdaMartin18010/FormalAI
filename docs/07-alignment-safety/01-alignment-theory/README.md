# 7.1 对齐理论 / Alignment Theory

## 概述 / Overview

对齐理论研究如何确保AI系统的行为与人类价值观和意图保持一致，为安全AI系统提供理论基础。

Alignment theory studies how to ensure AI system behavior aligns with human values and intentions, providing theoretical foundations for safe AI systems.

## 目录 / Table of Contents

- [7.1 对齐理论 / Alignment Theory](#71-对齐理论--alignment-theory)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [1. 价值学习 / Value Learning](#1-价值学习--value-learning)
    - [1.1 偏好学习 / Preference Learning](#11-偏好学习--preference-learning)
    - [1.2 奖励建模 / Reward Modeling](#12-奖励建模--reward-modeling)
    - [1.3 价值函数逼近 / Value Function Approximation](#13-价值函数逼近--value-function-approximation)
  - [2. 强化学习对齐 / Reinforcement Learning Alignment](#2-强化学习对齐--reinforcement-learning-alignment)
    - [2.1 人类反馈强化学习 / RLHF](#21-人类反馈强化学习--rlhf)
    - [2.2 直接偏好优化 / DPO](#22-直接偏好优化--dpo)
    - [2.3 对比学习 / Contrastive Learning](#23-对比学习--contrastive-learning)
  - [3. 可解释性对齐 / Interpretability Alignment](#3-可解释性对齐--interpretability-alignment)
    - [3.1 概念学习 / Concept Learning](#31-概念学习--concept-learning)
    - [3.2 注意力对齐 / Attention Alignment](#32-注意力对齐--attention-alignment)
    - [3.3 决策树提取 / Decision Tree Extraction](#33-决策树提取--decision-tree-extraction)
  - [4. 鲁棒性对齐 / Robustness Alignment](#4-鲁棒性对齐--robustness-alignment)
    - [4.1 对抗训练 / Adversarial Training](#41-对抗训练--adversarial-training)
    - [4.2 分布偏移 / Distribution Shift](#42-分布偏移--distribution-shift)
    - [4.3 不确定性量化 / Uncertainty Quantification](#43-不确定性量化--uncertainty-quantification)
  - [5. 多智能体对齐 / Multi-Agent Alignment](#5-多智能体对齐--multi-agent-alignment)
    - [5.1 合作博弈 / Cooperative Games](#51-合作博弈--cooperative-games)
    - [5.2 机制设计 / Mechanism Design](#52-机制设计--mechanism-design)
    - [5.3 社会选择理论 / Social Choice Theory](#53-社会选择理论--social-choice-theory)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：偏好学习算法](#rust实现偏好学习算法)
    - [Haskell实现：价值函数学习](#haskell实现价值函数学习)
  - [参考文献 / References](#参考文献--references)

---

## 1. 价值学习 / Value Learning

### 1.1 偏好学习 / Preference Learning

**偏好关系 / Preference Relation:**

$$\succ \subseteq \mathcal{A} \times \mathcal{A}$$

其中 $\mathcal{A}$ 是动作空间。

where $\mathcal{A}$ is the action space.

**偏好学习目标 / Preference Learning Objective:**

$$\mathcal{L}(\theta) = \mathbb{E}_{(a_1, a_2, y) \sim \mathcal{D}} [\ell(f_\theta(a_1, a_2), y)]$$

其中 $y \in \{0,1\}$ 表示偏好。

where $y \in \{0,1\}$ indicates preference.

**Bradley-Terry模型 / Bradley-Terry Model:**

$$P(a_1 \succ a_2) = \frac{\exp(r(a_1))}{\exp(r(a_1)) + \exp(r(a_2))}$$

### 1.2 奖励建模 / Reward Modeling

**奖励函数 / Reward Function:**

$$R: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$$

**奖励建模目标 / Reward Modeling Objective:**

$$\mathcal{L}(\phi) = \mathbb{E}_{(s, a, r^*) \sim \mathcal{D}} [(R_\phi(s, a) - r^*)^2]$$

**奖励不确定性 / Reward Uncertainty:**

$$\sigma_R^2(s, a) = \mathbb{E}_{\phi \sim p(\phi)} [(R_\phi(s, a) - \bar{R}(s, a))^2]$$

### 1.3 价值函数逼近 / Value Function Approximation

**价值函数 / Value Function:**

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, a_t)\right]$$

**价值迭代 / Value Iteration:**

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a) + \gamma V_k(s')]$$

---

## 2. 强化学习对齐 / Reinforcement Learning Alignment

### 2.1 人类反馈强化学习 / RLHF

**RLHF目标 / RLHF Objective:**

$$\mathcal{L}_{\text{RLHF}} = \mathcal{L}_{\text{SFT}} + \beta \mathcal{L}_{\text{RL}}$$

其中：

where:

$$\mathcal{L}_{\text{SFT}} = \mathbb{E}_{(x, y) \sim \mathcal{D}} [-\log \pi_\theta(y|x)]$$

$$\mathcal{L}_{\text{RL}} = \mathbb{E}_{x \sim \mathcal{D}} [\mathbb{E}_{y \sim \pi_\theta} [r_\phi(x, y) - \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})]]$$

**奖励函数学习 / Reward Function Learning:**

$$r_\phi(x, y) = \mathbb{E}_{(y_w, y_l) \sim \mathcal{D}} [\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]$$

### 2.2 直接偏好优化 / DPO

**DPO目标 / DPO Objective:**

$$\mathcal{L}_{\text{DPO}} = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} [-\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)})]$$

**DPO优势 / DPO Advantages:**

- 避免奖励函数学习 / Avoids reward function learning
- 直接优化偏好 / Directly optimizes preferences
- 更稳定的训练 / More stable training

### 2.3 对比学习 / Contrastive Learning

**对比损失 / Contrastive Loss:**

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

其中 $\tau$ 是温度参数。

where $\tau$ is the temperature parameter.

**InfoNCE损失 / InfoNCE Loss:**

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}[\log \frac{\exp(f(x)^T f(x^+)/\tau)}{\sum_{x^-} \exp(f(x)^T f(x^-)/\tau)}]$$

---

## 3. 可解释性对齐 / Interpretability Alignment

### 3.1 概念学习 / Concept Learning

**概念表示 / Concept Representation:**

$$c: \mathcal{X} \rightarrow \{0,1\}$$

**概念瓶颈模型 / Concept Bottleneck Model:**

$$f(x) = g(h(x))$$

其中 $h: \mathcal{X} \rightarrow \mathcal{C}$ 是概念编码器。

where $h: \mathcal{X} \rightarrow \mathcal{C}$ is the concept encoder.

**概念对齐 / Concept Alignment:**

$$\mathcal{L}_{\text{align}} = \mathbb{E}_{(x, c^*) \sim \mathcal{D}} [\text{BCE}(h(x), c^*)]$$

### 3.2 注意力对齐 / Attention Alignment

**注意力权重 / Attention Weights:**

$$\alpha_{ij} = \frac{\exp(\text{score}(q_i, k_j))}{\sum_l \exp(\text{score}(q_i, k_l))}$$

**注意力对齐损失 / Attention Alignment Loss:**

$$\mathcal{L}_{\text{attention}} = \mathbb{E}_{(x, A^*) \sim \mathcal{D}} [\|\alpha(x) - A^*\|_F^2]$$

### 3.3 决策树提取 / Decision Tree Extraction

**决策树 / Decision Tree:**

$$T(x) = \sum_{l \in \text{leaves}} v_l \mathbb{1}_{x \in R_l}$$

其中 $R_l$ 是叶子节点区域。

where $R_l$ is the leaf node region.

**树提取目标 / Tree Extraction Objective:**

$$\min_T \mathbb{E}_{x \sim \mathcal{D}} [(f(x) - T(x))^2] + \lambda \text{complexity}(T)$$

---

## 4. 鲁棒性对齐 / Robustness Alignment

### 4.1 对抗训练 / Adversarial Training

**对抗样本 / Adversarial Examples:**

$$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(f(x), y))$$

**对抗训练目标 / Adversarial Training Objective:**

$$\mathcal{L}_{\text{adv}} = \mathcal{L}_{\text{clean}} + \alpha \mathcal{L}_{\text{adversarial}}$$

**PGD攻击 / PGD Attack:**

$$x_{t+1} = \text{clip}(x_t + \alpha \cdot \text{sign}(\nabla_x \mathcal{L}(f(x_t), y)))$$

### 4.2 分布偏移 / Distribution Shift

**协变量偏移 / Covariate Shift:**

$$P_{\text{train}}(y|x) = P_{\text{test}}(y|x) \text{ but } P_{\text{train}}(x) \neq P_{\text{test}}(x)$$

**重要性加权 / Importance Weighting:**

$$\mathcal{L}_{\text{weighted}} = \mathbb{E}_{x \sim P_{\text{train}}} [\frac{P_{\text{test}}(x)}{P_{\text{train}}(x)} \mathcal{L}(f(x), y)]$$

### 4.3 不确定性量化 / Uncertainty Quantification

**贝叶斯神经网络 / Bayesian Neural Network:**

$$p(\theta|D) \propto p(D|\theta) p(\theta)$$

**预测不确定性 / Predictive Uncertainty:**

$$\text{Var}[f(x)] = \mathbb{E}_{\theta \sim p(\theta|D)} [f_\theta(x)^2] - \mathbb{E}_{\theta \sim p(\theta|D)} [f_\theta(x)]^2$$

---

## 5. 多智能体对齐 / Multi-Agent Alignment

### 5.1 合作博弈 / Cooperative Games

**特征函数 / Characteristic Function:**

$$v: 2^N \rightarrow \mathbb{R}$$

**夏普利值 / Shapley Value:**

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

### 5.2 机制设计 / Mechanism Design

**社会选择函数 / Social Choice Function:**

$$f: \mathcal{P}^n \rightarrow \mathcal{A}$$

其中 $\mathcal{P}$ 是偏好关系集合。

where $\mathcal{P}$ is the set of preference relations.

**激励相容性 / Incentive Compatibility:**

$$u_i(f(\succ_1, \ldots, \succ_n)) \geq u_i(f(\succ_1', \ldots, \succ_n'))$$

### 5.3 社会选择理论 / Social Choice Theory

**阿罗不可能定理 / Arrow's Impossibility Theorem:**

不存在满足所有理想性质的社会选择函数。

No social choice function satisfies all ideal properties.

**投票系统 / Voting Systems:**

- 多数投票 / Majority voting
- 波达计数 / Borda count
- 康多塞方法 / Condorcet method

---

## 代码示例 / Code Examples

### Rust实现：偏好学习算法

```rust
use std::collections::HashMap;
use rand::Rng;

#[derive(Debug, Clone)]
struct PreferenceData {
    query: String,
    preferred: String,
    dispreferred: String,
}

#[derive(Debug)]
struct PreferenceLearner {
    model: HashMap<String, Vec<f64>>,
    learning_rate: f64,
    embedding_dim: usize,
}

impl PreferenceLearner {
    fn new(embedding_dim: usize, learning_rate: f64) -> Self {
        PreferenceLearner {
            model: HashMap::new(),
            learning_rate,
            embedding_dim,
        }
    }
    
    fn get_embedding(&self, text: &str) -> Vec<f64> {
        self.model.get(text).cloned().unwrap_or_else(|| {
            // 默认嵌入
            vec![0.0; self.embedding_dim]
        })
    }
    
    fn update_embedding(&mut self, text: &str, gradient: &[f64]) {
        let current = self.get_embedding(text);
        let updated: Vec<f64> = current.iter()
            .zip(gradient.iter())
            .map(|(c, g)| c + self.learning_rate * g)
            .collect();
        self.model.insert(text.to_string(), updated);
    }
    
    fn compute_preference_score(&self, query: &str, response: &str) -> f64 {
        let query_emb = self.get_embedding(query);
        let response_emb = self.get_embedding(response);
        
        // 计算余弦相似度
        let dot_product: f64 = query_emb.iter()
            .zip(response_emb.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let query_norm: f64 = query_emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        let response_norm: f64 = response_emb.iter().map(|x| x * x).sum::<f64>().sqrt();
        
        if query_norm > 0.0 && response_norm > 0.0 {
            dot_product / (query_norm * response_norm)
        } else {
            0.0
        }
    }
    
    fn train_on_preference(&mut self, data: &PreferenceData) {
        let preferred_score = self.compute_preference_score(&data.query, &data.preferred);
        let dispreferred_score = self.compute_preference_score(&data.query, &data.dispreferred);
        
        // Bradley-Terry模型损失
        let logit = preferred_score - dispreferred_score;
        let probability = 1.0 / (1.0 + (-logit).exp());
        
        // 计算梯度
        let gradient_factor = 1.0 - probability;
        
        // 更新嵌入
        let query_emb = self.get_embedding(&data.query);
        let preferred_emb = self.get_embedding(&data.preferred);
        let dispreferred_emb = self.get_embedding(&data.dispreferred);
        
        // 计算梯度
        let mut query_grad = vec![0.0; self.embedding_dim];
        let mut preferred_grad = vec![0.0; self.embedding_dim];
        let mut dispreferred_grad = vec![0.0; self.embedding_dim];
        
        for i in 0..self.embedding_dim {
            query_grad[i] = gradient_factor * (preferred_emb[i] - dispreferred_emb[i]);
            preferred_grad[i] = gradient_factor * query_emb[i];
            dispreferred_grad[i] = -gradient_factor * query_emb[i];
        }
        
        // 应用梯度更新
        self.update_embedding(&data.query, &query_grad);
        self.update_embedding(&data.preferred, &preferred_grad);
        self.update_embedding(&data.dispreferred, &dispreferred_grad);
    }
    
    fn predict_preference(&self, query: &str, response1: &str, response2: &str) -> f64 {
        let score1 = self.compute_preference_score(query, response1);
        let score2 = self.compute_preference_score(query, response2);
        
        // 返回选择response1的概率
        1.0 / (1.0 + (score2 - score1).exp())
    }
}

#[derive(Debug)]
struct RewardModel {
    preference_learner: PreferenceLearner,
}

impl RewardModel {
    fn new(embedding_dim: usize) -> Self {
        RewardModel {
            preference_learner: PreferenceLearner::new(embedding_dim, 0.01),
        }
    }
    
    fn train(&mut self, training_data: &[PreferenceData]) {
        for data in training_data {
            self.preference_learner.train_on_preference(data);
        }
    }
    
    fn predict_reward(&self, query: &str, response: &str) -> f64 {
        self.preference_learner.compute_preference_score(query, response)
    }
    
    fn predict_preference(&self, query: &str, response1: &str, response2: &str) -> f64 {
        self.preference_learner.predict_preference(query, response1, response2)
    }
}

// DPO训练器
#[derive(Debug)]
struct DPOTrainer {
    model: HashMap<String, Vec<f64>>,
    reference_model: HashMap<String, Vec<f64>>,
    beta: f64,
    learning_rate: f64,
}

impl DPOTrainer {
    fn new(beta: f64, learning_rate: f64) -> Self {
        DPOTrainer {
            model: HashMap::new(),
            reference_model: HashMap::new(),
            beta,
            learning_rate,
        }
    }
    
    fn compute_log_prob(&self, text: &str) -> f64 {
        // 简化的对数概率计算
        let embedding = self.model.get(text).cloned().unwrap_or_else(|| vec![0.0; 10]);
        embedding.iter().sum::<f64>().ln()
    }
    
    fn compute_reference_log_prob(&self, text: &str) -> f64 {
        let embedding = self.reference_model.get(text).cloned().unwrap_or_else(|| vec![0.0; 10]);
        embedding.iter().sum::<f64>().ln()
    }
    
    fn train_step(&mut self, query: &str, preferred: &str, dispreferred: &str) {
        let log_prob_preferred = self.compute_log_prob(preferred);
        let log_prob_dispreferred = self.compute_log_prob(dispreferred);
        let ref_log_prob_preferred = self.compute_reference_log_prob(preferred);
        let ref_log_prob_dispreferred = self.compute_reference_log_prob(dispreferred);
        
        let log_ratio_preferred = log_prob_preferred - ref_log_prob_preferred;
        let log_ratio_dispreferred = log_prob_dispreferred - ref_log_prob_dispreferred;
        
        let logit = self.beta * (log_ratio_preferred - log_ratio_dispreferred);
        let loss = -logit.sigmoid().ln();
        
        // 简化的梯度更新
        let gradient = loss * self.beta;
        
        // 更新模型参数
        self.update_model_parameters(gradient);
    }
    
    fn update_model_parameters(&mut self, gradient: f64) {
        // 简化的参数更新
        for (_, embedding) in self.model.iter_mut() {
            for value in embedding.iter_mut() {
                *value += self.learning_rate * gradient;
            }
        }
    }
}

fn main() {
    // 创建训练数据
    let training_data = vec![
        PreferenceData {
            query: "What is the capital of France?".to_string(),
            preferred: "The capital of France is Paris.".to_string(),
            dispreferred: "I don't know.".to_string(),
        },
        PreferenceData {
            query: "Explain quantum physics.".to_string(),
            preferred: "Quantum physics is a fundamental theory in physics that describes the behavior of matter and energy at the atomic and subatomic level.".to_string(),
            dispreferred: "It's complicated.".to_string(),
        },
    ];
    
    // 训练偏好学习器
    let mut preference_learner = PreferenceLearner::new(10, 0.01);
    for data in &training_data {
        preference_learner.train_on_preference(data);
    }
    
    // 测试偏好预测
    let query = "What is machine learning?";
    let response1 = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.";
    let response2 = "It's a type of AI.";
    
    let preference = preference_learner.predict_preference(query, response1, response2);
    println!("Preference for response1: {:.3}", preference);
    
    // 训练奖励模型
    let mut reward_model = RewardModel::new(10);
    reward_model.train(&training_data);
    
    let reward = reward_model.predict_reward(query, response1);
    println!("Reward for response1: {:.3}", reward);
    
    // DPO训练
    let mut dpo_trainer = DPOTrainer::new(0.1, 0.01);
    for data in &training_data {
        dpo_trainer.train_step(&data.query, &data.preferred, &data.dispreferred);
    }
    
    println!("Training completed!");
}
```

### Haskell实现：价值函数学习

```haskell
import Data.List (foldl')
import Data.Map (Map)
import qualified Data.Map as Map
import System.Random

-- 价值函数类型
data ValueFunction = ValueFunction {
    weights :: [Double],
    bias :: Double
} deriving Show

-- 状态类型
data State = State {
    features :: [Double]
} deriving Show

-- 动作类型
data Action = Action {
    actionId :: String,
    actionFeatures :: [Double]
} deriving Show

-- 偏好数据
data PreferenceData = PreferenceData {
    state :: State,
    preferredAction :: Action,
    dispreferredAction :: Action
} deriving Show

-- 创建价值函数
createValueFunction :: Int -> ValueFunction
createValueFunction dim = ValueFunction {
    weights = replicate dim 0.0,
    bias = 0.0
}

-- 计算价值
computeValue :: ValueFunction -> State -> Double
computeValue vf state =
    let linear = sum (zipWith (*) (weights vf) (features state)) + bias vf
    in 1.0 / (1.0 + exp (-linear))  -- sigmoid激活

-- 计算动作价值
computeActionValue :: ValueFunction -> State -> Action -> Double
computeActionValue vf state action =
    let stateActionFeatures = features state ++ actionFeatures action
        linear = sum (zipWith (*) (weights vf) stateActionFeatures) + bias vf
    in 1.0 / (1.0 + exp (-linear))

-- 偏好学习
preferenceLearning :: ValueFunction -> PreferenceData -> ValueFunction
preferenceLearning vf data =
    let preferredValue = computeActionValue vf (state data) (preferredAction data)
        dispreferredValue = computeActionValue vf (state data) (dispreferredAction data)
        
        -- Bradley-Terry模型
        logit = preferredValue - dispreferredValue
        probability = 1.0 / (1.0 + exp (-logit))
        
        -- 计算梯度
        gradientFactor = 1.0 - probability
        
        -- 更新权重
        newWeights = updateWeights (weights vf) gradientFactor (state data) (preferredAction data) (dispreferredAction data)
        newBias = bias vf + gradientFactor * 0.01  -- 学习率
    in vf { weights = newWeights, bias = newBias }

-- 更新权重
updateWeights :: [Double] -> Double -> State -> Action -> Action -> [Double]
updateWeights weights gradient state preferred dispreferred =
    let stateActionFeatures = features state ++ actionFeatures preferred
        stateActionFeaturesDispreferred = features state ++ actionFeatures dispreferred
        featureDiff = zipWith (-) stateActionFeatures stateActionFeaturesDispreferred
    in zipWith (\w g -> w + 0.01 * gradient * g) weights featureDiff

-- 奖励建模
data RewardModel = RewardModel {
    valueFunction :: ValueFunction,
    uncertainty :: Double
} deriving Show

createRewardModel :: Int -> RewardModel
createRewardModel dim = RewardModel {
    valueFunction = createValueFunction dim,
    uncertainty = 1.0
}

predictReward :: RewardModel -> State -> Action -> Double
predictReward model state action =
    computeActionValue (valueFunction model) state action

predictRewardWithUncertainty :: RewardModel -> State -> Action -> (Double, Double)
predictRewardWithUncertainty model state action =
    let reward = predictReward model state action
        uncertainty = uncertainty model
    in (reward, uncertainty)

-- 训练奖励模型
trainRewardModel :: RewardModel -> [PreferenceData] -> RewardModel
trainRewardModel model trainingData =
    let updatedValueFunction = foldl' preferenceLearning (valueFunction model) trainingData
        -- 计算不确定性
        predictions = map (\data -> 
            (predictReward model (state data) (preferredAction data),
             predictReward model (state data) (dispreferredAction data))) trainingData
        variance = calculateVariance predictions
    in model { 
        valueFunction = updatedValueFunction,
        uncertainty = sqrt variance
    }

calculateVariance :: [(Double, Double)] -> Double
calculateVariance predictions =
    let allRewards = concatMap (\(r1, r2) -> [r1, r2]) predictions
        mean = sum allRewards / fromIntegral (length allRewards)
        squaredDiffs = map (\r -> (r - mean) ^ 2) allRewards
    in sum squaredDiffs / fromIntegral (length squaredDiffs)

-- 对齐评估
data AlignmentMetrics = AlignmentMetrics {
    preferenceAccuracy :: Double,
    rewardCorrelation :: Double,
    safetyScore :: Double
} deriving Show

evaluateAlignment :: ValueFunction -> [PreferenceData] -> AlignmentMetrics
evaluateAlignment vf testData =
    let -- 计算偏好准确率
        preferencePredictions = map (\data -> 
            let preferredValue = computeActionValue vf (state data) (preferredAction data)
                dispreferredValue = computeActionValue vf (state data) (dispreferredAction data)
            in preferredValue > dispreferredValue) testData
        accuracy = fromIntegral (length (filter id preferencePredictions)) / fromIntegral (length testData)
        
        -- 计算奖励相关性
        rewardPairs = map (\data -> 
            (computeActionValue vf (state data) (preferredAction data),
             computeActionValue vf (state data) (dispreferredAction data))) testData
        correlation = calculateCorrelation rewardPairs
        
        -- 计算安全分数
        safetyScore = calculateSafetyScore vf testData
    in AlignmentMetrics {
        preferenceAccuracy = accuracy,
        rewardCorrelation = correlation,
        safetyScore = safetyScore
    }

calculateCorrelation :: [(Double, Double)] -> Double
calculateCorrelation pairs =
    let n = fromIntegral (length pairs)
        sumX = sum (map fst pairs)
        sumY = sum (map snd pairs)
        sumXY = sum (map (\(x, y) -> x * y) pairs)
        sumXX = sum (map (\(x, _) -> x * x) pairs)
        sumYY = sum (map (\(_, y) -> y * y) pairs)
        
        numerator = n * sumXY - sumX * sumY
        denominator = sqrt ((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY))
    in if denominator == 0 then 0 else numerator / denominator

calculateSafetyScore :: ValueFunction -> [PreferenceData] -> Double
calculateSafetyScore vf testData =
    let -- 简化的安全分数计算
        -- 检查是否有极端价值
        allValues = concatMap (\data -> 
            [computeActionValue vf (state data) (preferredAction data),
             computeActionValue vf (state data) (dispreferredAction data)]) testData
        
        maxValue = maximum allValues
        minValue = minimum allValues
        
        -- 安全分数基于价值范围
        safetyScore = 1.0 - (maxValue - minValue) / 2.0
    in max 0.0 (min 1.0 safetyScore)

-- 生成示例数据
generateTrainingData :: Int -> IO [PreferenceData]
generateTrainingData n = do
    gen <- getStdGen
    let (data, _) = foldl' 
            (\(acc, g) i -> 
                let (g1, g2) = split g
                    (stateFeatures, g3) = randomFeatures 3 g1
                    (prefFeatures, g4) = randomFeatures 2 g2
                    (disprefFeatures, g5) = randomFeatures 2 g3
                    
                    state = State { features = stateFeatures }
                    preferred = Action { actionId = "pref_" ++ show i, actionFeatures = prefFeatures }
                    dispreferred = Action { actionId = "dispref_" ++ show i, actionFeatures = disprefFeatures }
                    
                    preferenceData = PreferenceData {
                        state = state,
                        preferredAction = preferred,
                        dispreferredAction = dispreferred
                    }
                in (acc ++ [preferenceData], g4)) 
            ([], gen) [1..n]
    return data
  where
    randomFeatures :: Int -> StdGen -> ([Double], StdGen)
    randomFeatures n g = 
        let (features, g') = foldl' 
                (\(acc, g) _ -> 
                    let (x, g') = randomR (-1.0, 1.0) g
                    in (acc ++ [x], g')) 
                ([], g) [1..n]
        in (features, g')

-- 主函数
main :: IO ()
main = do
    putStrLn "生成训练数据..."
    trainingData <- generateTrainingData 100
    
    putStrLn "训练价值函数..."
    let initialVF = createValueFunction 5
        trainedVF = foldl' preferenceLearning initialVF trainingData
    
    putStrLn "训练奖励模型..."
    let initialModel = createRewardModel 5
        trainedModel = trainRewardModel initialModel trainingData
    
    putStrLn "评估对齐性能..."
    let metrics = evaluateAlignment trainedVF trainingData
    
    putStrLn "对齐评估结果:"
    putStrLn $ "偏好准确率: " ++ show (preferenceAccuracy metrics)
    putStrLn $ "奖励相关性: " ++ show (rewardCorrelation metrics)
    putStrLn $ "安全分数: " ++ show (safetyScore metrics)
    
    putStrLn "\n对齐训练完成！"
```

---

## 参考文献 / References

1. Christiano, P., et al. (2017). Deep reinforcement learning from human preferences. *NIPS*.
2. Rafailov, R., et al. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS*.
3. Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI feedback. *arXiv*.
4. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.
5. Leike, J., et al. (2018). Scalable agent alignment via reward modeling. *arXiv*.
6. Amodei, D., et al. (2016). Concrete problems in AI safety. *arXiv*.

---

*本模块为FormalAI提供了对齐理论的基础，涵盖了从价值学习到多智能体对齐的各个方面，为安全AI系统的设计和评估提供了数学工具。*
