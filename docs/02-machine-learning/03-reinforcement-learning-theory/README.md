# 2.3 强化学习理论 / Reinforcement Learning Theory / Verstärkungslern-Theorie / Théorie de l'apprentissage par renforcement

## 概述 / Overview

强化学习理论研究智能体如何通过与环境交互来学习最优策略，为自主AI系统提供理论基础。

Reinforcement learning theory studies how agents learn optimal policies through interaction with environments, providing theoretical foundations for autonomous AI systems.

## 目录 / Table of Contents

- [2.3 强化学习理论 / Reinforcement Learning Theory / Verstärkungslern-Theorie / Théorie de l'apprentissage par renforcement](#23-强化学习理论--reinforcement-learning-theory--verstärkungslern-theorie--théorie-de-lapprentissage-par-renforcement)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [1. 马尔可夫决策过程 / Markov Decision Processes](#1-马尔可夫决策过程--markov-decision-processes)
    - [1.1 MDP形式化 / MDP Formulation](#11-mdp形式化--mdp-formulation)
    - [1.2 策略与价值函数 / Policy and Value Functions](#12-策略与价值函数--policy-and-value-functions)
    - [1.3 贝尔曼方程 / Bellman Equations](#13-贝尔曼方程--bellman-equations)
  - [2. 动态规划 / Dynamic Programming](#2-动态规划--dynamic-programming)
    - [2.1 策略迭代 / Policy Iteration](#21-策略迭代--policy-iteration)
    - [2.2 价值迭代 / Value Iteration](#22-价值迭代--value-iteration)
    - [2.3 策略评估 / Policy Evaluation](#23-策略评估--policy-evaluation)
  - [3. 蒙特卡洛方法 / Monte Carlo Methods](#3-蒙特卡洛方法--monte-carlo-methods)
    - [3.1 首次访问MC / First-Visit MC](#31-首次访问mc--first-visit-mc)
    - [3.2 每次访问MC / Every-Visit MC](#32-每次访问mc--every-visit-mc)
    - [3.3 探索策略 / Exploration Policies](#33-探索策略--exploration-policies)
  - [4. 时序差分学习 / Temporal Difference Learning](#4-时序差分学习--temporal-difference-learning)
    - [4.1 TD(0)算法 / TD(0) Algorithm](#41-td0算法--td0-algorithm)
    - [4.2 TD(λ)算法 / TD(λ) Algorithm](#42-tdλ算法--tdλ-algorithm)
    - [4.3 SARSA算法 / SARSA Algorithm](#43-sarsa算法--sarsa-algorithm)
  - [5. Q学习 / Q-Learning](#5-q学习--q-learning)
    - [5.1 Q学习算法 / Q-Learning Algorithm](#51-q学习算法--q-learning-algorithm)
    - [5.2 收敛性分析 / Convergence Analysis](#52-收敛性分析--convergence-analysis)
    - [5.3 双Q学习 / Double Q-Learning](#53-双q学习--double-q-learning)
  - [6. 策略梯度方法 / Policy Gradient Methods](#6-策略梯度方法--policy-gradient-methods)
    - [6.1 策略梯度定理 / Policy Gradient Theorem](#61-策略梯度定理--policy-gradient-theorem)
    - [6.2 REINFORCE算法 / REINFORCE Algorithm](#62-reinforce算法--reinforce-algorithm)
    - [6.3 自然策略梯度 / Natural Policy Gradient](#63-自然策略梯度--natural-policy-gradient)
  - [7. Actor-Critic方法 / Actor-Critic Methods](#7-actor-critic方法--actor-critic-methods)
    - [7.1 Actor-Critic框架 / Actor-Critic Framework](#71-actor-critic框架--actor-critic-framework)
    - [7.2 A3C算法 / A3C Algorithm](#72-a3c算法--a3c-algorithm)
    - [7.3 A2C算法 / A2C Algorithm](#73-a2c算法--a2c-algorithm)
  - [8. 深度强化学习 / Deep Reinforcement Learning](#8-深度强化学习--deep-reinforcement-learning)
    - [8.1 DQN算法 / DQN Algorithm](#81-dqn算法--dqn-algorithm)
    - [8.2 DDPG算法 / DDPG Algorithm](#82-ddpg算法--ddpg-algorithm)
    - [8.3 PPO算法 / PPO Algorithm](#83-ppo算法--ppo-algorithm)
  - [9. 多智能体强化学习 / Multi-Agent Reinforcement Learning](#9-多智能体强化学习--multi-agent-reinforcement-learning)
    - [9.1 博弈论基础 / Game Theory Foundations](#91-博弈论基础--game-theory-foundations)
    - [9.2 纳什均衡 / Nash Equilibrium](#92-纳什均衡--nash-equilibrium)
    - [9.3 合作与竞争 / Cooperation and Competition](#93-合作与竞争--cooperation-and-competition)
  - [10. 元强化学习 / Meta Reinforcement Learning](#10-元强化学习--meta-reinforcement-learning)
    - [10.1 快速适应 / Fast Adaptation](#101-快速适应--fast-adaptation)
    - [10.2 元学习算法 / Meta-Learning Algorithms](#102-元学习算法--meta-learning-algorithms)
    - [10.3 迁移学习 / Transfer Learning](#103-迁移学习--transfer-learning)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：Q学习算法](#rust实现q学习算法)
    - [Haskell实现：策略梯度](#haskell实现策略梯度)
  - [参考文献 / References](#参考文献--references)

---

## 相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes

**前置依赖 / Prerequisites / Voraussetzungen / Prérequis:**
- [2.1 统计学习理论](01-statistical-learning-theory/README.md) - 提供学习基础 / Provides learning foundation

**后续应用 / Applications / Anwendungen / Applications:**
- [7.1 对齐理论](../07-alignment-safety/01-alignment-theory/README.md) - 提供学习基础 / Provides learning foundation
- [7.2 价值学习理论](../07-alignment-safety/02-value-learning/README.md) - 提供价值基础 / Provides value foundation

---

## 1. 马尔可夫决策过程 / Markov Decision Processes

### 1.1 MDP形式化 / MDP Formulation

**马尔可夫决策过程 / Markov Decision Process:**

MDP是一个五元组 $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$，其中：

MDP is a 5-tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ where:

- $\mathcal{S}$ 是状态空间 / state space
- $\mathcal{A}$ 是动作空间 / action space
- $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow [0,1]$ 是转移概率函数 / transition probability function
- $\mathcal{R}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$ 是奖励函数 / reward function
- $\gamma \in [0,1]$ 是折扣因子 / discount factor

**转移概率 / Transition Probability:**

$$P(s'|s,a) = \mathbb{P}(S_{t+1} = s' | S_t = s, A_t = a)$$

**奖励函数 / Reward Function:**

$$R(s,a,s') = \mathbb{E}[R_{t+1} | S_t = s, A_t = a, S_{t+1} = s']$$

### 1.2 策略与价值函数 / Policy and Value Functions

**策略 / Policy:**

策略 $\pi$ 是从状态到动作的映射：

Policy $\pi$ is a mapping from states to actions:

$$\pi: \mathcal{S} \rightarrow \mathcal{A}$$

**状态价值函数 / State Value Function:**

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0 = s\right]$$

**动作价值函数 / Action Value Function:**

$$Q^\pi(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_{t+1} | S_0 = s, A_0 = a\right]$$

### 1.3 贝尔曼方程 / Bellman Equations

**状态价值贝尔曼方程 / State Value Bellman Equation:**

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

**动作价值贝尔曼方程 / Action Value Bellman Equation:**

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

**最优贝尔曼方程 / Optimal Bellman Equation:**

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$$

---

## 2. 动态规划 / Dynamic Programming

### 2.1 策略迭代 / Policy Iteration

**策略迭代算法 / Policy Iteration Algorithm:**

1. **策略评估 / Policy Evaluation:**
   $$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$$

2. **策略改进 / Policy Improvement:**
   $$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

**收敛性 / Convergence:**

策略迭代算法在有限步内收敛到最优策略。

Policy iteration converges to optimal policy in finite steps.

### 2.2 价值迭代 / Value Iteration

**价值迭代算法 / Value Iteration Algorithm:**

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$$

**收敛性 / Convergence:**

价值迭代以几何速率收敛：

Value iteration converges at geometric rate:

$$\|V_{k+1} - V^*\|_\infty \leq \gamma \|V_k - V^*\|_\infty$$

### 2.3 策略评估 / Policy Evaluation

**同步策略评估 / Synchronous Policy Evaluation:**

$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$$

**异步策略评估 / Asynchronous Policy Evaluation:**

$$V(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]$$

---

## 3. 蒙特卡洛方法 / Monte Carlo Methods

### 3.1 首次访问MC / First-Visit MC

**首次访问MC / First-Visit MC:**

对于每个状态，只在首次访问时更新价值：

For each state, update value only at first visit:

$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

其中 $G_t$ 是从时间 $t$ 开始的回报。

where $G_t$ is the return starting from time $t$.

**回报计算 / Return Calculation:**

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$$

### 3.2 每次访问MC / Every-Visit MC

**每次访问MC / Every-Visit MC:**

对于每个状态，在每次访问时都更新价值：

For each state, update value at every visit:

$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

### 3.3 探索策略 / Exploration Policies

**ε-贪婪策略 / ε-Greedy Policy:**

$$
\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_a Q(s,a) \\
\frac{\epsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}
$$

**玻尔兹曼策略 / Boltzmann Policy:**

$$\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}$$

---

## 4. 时序差分学习 / Temporal Difference Learning

### 4.1 TD(0)算法 / TD(0) Algorithm

**TD(0)更新 / TD(0) Update:**

$$V(s_t) \leftarrow V(s_t) + \alpha [R_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

**TD误差 / TD Error:**

$$\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

### 4.2 TD(λ)算法 / TD(λ) Algorithm

**TD(λ)更新 / TD(λ) Update:**

$$V(s_t) \leftarrow V(s_t) + \alpha \delta_t e_t(s_t)$$

其中资格迹为：

where eligibility trace is:

$$
e_t(s) = \begin{cases}
\gamma \lambda e_{t-1}(s) + 1 & \text{if } s = s_t \\
\gamma \lambda e_{t-1}(s) & \text{otherwise}
\end{cases}
$$

### 4.3 SARSA算法 / SARSA Algorithm

**SARSA更新 / SARSA Update:**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

**SARSA(λ) / SARSA(λ):**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \delta_t e_t(s_t, a_t)$$

---

## 5. Q学习 / Q-Learning

### 5.1 Q学习算法 / Q-Learning Algorithm

**Q学习更新 / Q-Learning Update:**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

**Q学习特点 / Q-Learning Characteristics:**

- 离策略学习 / Off-policy learning
- 收敛到最优Q函数 / Converges to optimal Q-function
- 不需要策略模型 / No policy model required

### 5.2 收敛性分析 / Convergence Analysis

**收敛条件 / Convergence Conditions:**

1. 所有状态-动作对被访问无限次
2. 学习率满足Robbins-Monro条件

   1. All state-action pairs visited infinitely often
   2. Learning rates satisfy Robbins-Monro conditions

**Robbins-Monro条件 / Robbins-Monro Conditions:**

$$\sum_{t=0}^{\infty} \alpha_t = \infty, \quad \sum_{t=0}^{\infty} \alpha_t^2 < \infty$$

### 5.3 双Q学习 / Double Q-Learning

**双Q学习更新 / Double Q-Learning Update:**

$$Q_A(s_t, a_t) \leftarrow Q_A(s_t, a_t) + \alpha [R_{t+1} + \gamma Q_B(s_{t+1}, \arg\max_{a'} Q_A(s_{t+1}, a')) - Q_A(s_t, a_t)]$$

**优势 / Advantages:**

- 减少Q值过估计 / Reduces Q-value overestimation
- 提高学习稳定性 / Improves learning stability

---

## 6. 策略梯度方法 / Policy Gradient Methods

### 6.1 策略梯度定理 / Policy Gradient Theorem

**策略梯度定理 / Policy Gradient Theorem:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)]$$

其中 $J(\theta) = \mathbb{E}_{\pi_\theta}[V^{\pi_\theta}(s_0)]$。

where $J(\theta) = \mathbb{E}_{\pi_\theta}[V^{\pi_\theta}(s_0)]$.

### 6.2 REINFORCE算法 / REINFORCE Algorithm

**REINFORCE更新 / REINFORCE Update:**

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

其中 $G_t$ 是蒙特卡洛回报。

where $G_t$ is the Monte Carlo return.

**基线方法 / Baseline Method:**

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t))$$

### 6.3 自然策略梯度 / Natural Policy Gradient

**自然策略梯度 / Natural Policy Gradient:**

$$\nabla_\theta^{nat} J(\theta) = F(\theta)^{-1} \nabla_\theta J(\theta)$$

其中 $F(\theta)$ 是Fisher信息矩阵：

where $F(\theta)$ is the Fisher information matrix:

$$F(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T]$$

---

## 7. Actor-Critic方法 / Actor-Critic Methods

### 7.1 Actor-Critic框架 / Actor-Critic Framework

**Actor-Critic更新 / Actor-Critic Update:**

**Actor (策略) / Actor (Policy):**
$$\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \log \pi_\theta(a_t|s_t) \delta_t$$

**Critic (价值函数) / Critic (Value Function):**
$$w_{t+1} = w_t + \alpha_w \delta_t \nabla_w V_w(s_t)$$

其中 $\delta_t = R_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)$。

where $\delta_t = R_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)$.

### 7.2 A3C算法 / A3C Algorithm

**A3C特点 / A3C Characteristics:**

- 异步更新 / Asynchronous updates
- 全局共享参数 / Globally shared parameters
- 多线程并行 / Multi-threaded parallelism

**优势函数 / Advantage Function:**

$$A(s,a) = Q(s,a) - V(s)$$

### 7.3 A2C算法 / A2C Algorithm

**A2C特点 / A2C Characteristics:**

- 同步更新 / Synchronous updates
- 更稳定的训练 / More stable training
- 更好的收敛性 / Better convergence

---

## 8. 深度强化学习 / Deep Reinforcement Learning

### 8.1 DQN算法 / DQN Algorithm

**DQN特点 / DQN Characteristics:**

1. **经验回放 / Experience Replay:**
   $$(s_t, a_t, r_t, s_{t+1}) \rightarrow \text{Replay Buffer}$$

2. **目标网络 / Target Network:**
   $$Q_{target}(s,a) = R + \gamma \max_{a'} Q(s', a'; \theta^-)$$

3. **损失函数 / Loss Function:**
   $$L(\theta) = \mathbb{E}[(R + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s,a; \theta))^2]$$

### 8.2 DDPG算法 / DDPG Algorithm

**DDPG特点 / DDPG Characteristics:**

- 连续动作空间 / Continuous action spaces
- 确定性策略 / Deterministic policy
- Actor-Critic架构 / Actor-Critic architecture

**策略网络 / Policy Network:**
$$\mu_\theta(s) = \arg\max_a Q(s,a)$$

**价值网络 / Value Network:**
$$Q(s,a) = \mathbb{E}[R + \gamma Q(s', \mu_\theta(s'))]$$

### 8.3 PPO算法 / PPO Algorithm

**PPO目标函数 / PPO Objective:**

$$L(\theta) = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$$

其中比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.

where ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.

---

## 9. 多智能体强化学习 / Multi-Agent Reinforcement Learning

### 9.1 博弈论基础 / Game Theory Foundations

**博弈 / Game:**

博弈是一个三元组 $(\mathcal{N}, \mathcal{S}, \mathcal{U})$，其中：

Game is a 3-tuple $(\mathcal{N}, \mathcal{S}, \mathcal{U})$ where:

- $\mathcal{N}$ 是玩家集合 / set of players
- $\mathcal{S} = \prod_{i \in \mathcal{N}} \mathcal{S}_i$ 是策略空间 / strategy space
- $\mathcal{U} = \{u_i\}_{i \in \mathcal{N}}$ 是效用函数集合 / set of utility functions

### 9.2 纳什均衡 / Nash Equilibrium

**纳什均衡 / Nash Equilibrium:**

策略组合 $s^*$ 是纳什均衡，如果：

Strategy profile $s^*$ is Nash equilibrium if:

$$u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*) \quad \forall s_i \in \mathcal{S}_i, \forall i \in \mathcal{N}$$

### 9.3 合作与竞争 / Cooperation and Competition

**囚徒困境 / Prisoner's Dilemma:**

| | 合作 | 背叛 |
|---|---|---|
| 合作 | (3,3) | (0,5) |
| 背叛 | (5,0) | (1,1) |

**重复博弈 / Repeated Games:**

- 触发策略 / Trigger strategies
- 合作演化 / Evolution of cooperation
- 声誉机制 / Reputation mechanisms

---

## 10. 元强化学习 / Meta Reinforcement Learning

### 10.1 快速适应 / Fast Adaptation

**元学习目标 / Meta-Learning Objective:**

$$\min_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} [L_{\mathcal{T}}(U_\theta(\mathcal{D}^{tr}_{\mathcal{T}}))]$$

其中 $U_\theta$ 是适应算法，$\mathcal{D}^{tr}_{\mathcal{T}}$ 是任务 $\mathcal{T}$ 的训练数据。

where $U_\theta$ is the adaptation algorithm and $\mathcal{D}^{tr}_{\mathcal{T}}$ is training data for task $\mathcal{T}$.

### 10.2 元学习算法 / Meta-Learning Algorithms

**MAML算法 / MAML Algorithm:**

$$\theta' = \theta - \alpha \nabla_\theta L_{\mathcal{T}}(\theta)$$

**Reptile算法 / Reptile Algorithm:**

$$\theta \leftarrow \theta + \beta(\theta' - \theta)$$

### 10.3 迁移学习 / Transfer Learning

**知识迁移 / Knowledge Transfer:**

- 预训练策略 / Pre-trained policies
- 技能组合 / Skill composition
- 分层学习 / Hierarchical learning

---

## 代码示例 / Code Examples

### Rust实现：Q学习算法

```rust
use std::collections::HashMap;
use rand::Rng;

#[derive(Debug, Clone)]
struct Environment {
    states: Vec<String>,
    actions: Vec<String>,
    transitions: HashMap<(String, String), Vec<(String, f64)>>,
    rewards: HashMap<(String, String, String), f64>,
}

impl Environment {
    fn new() -> Self {
        let mut env = Environment {
            states: vec!["s0".to_string(), "s1".to_string(), "s2".to_string()],
            actions: vec!["a0".to_string(), "a1".to_string()],
            transitions: HashMap::new(),
            rewards: HashMap::new(),
        };
        
        // 设置转移概率
        env.transitions.insert(("s0".to_string(), "a0".to_string()), 
            vec![("s1".to_string(), 0.8), ("s2".to_string(), 0.2)]);
        env.transitions.insert(("s0".to_string(), "a1".to_string()), 
            vec![("s1".to_string(), 0.2), ("s2".to_string(), 0.8)]);
        env.transitions.insert(("s1".to_string(), "a0".to_string()), 
            vec![("s0".to_string(), 0.9), ("s2".to_string(), 0.1)]);
        env.transitions.insert(("s1".to_string(), "a1".to_string()), 
            vec![("s0".to_string(), 0.1), ("s2".to_string(), 0.9)]);
        env.transitions.insert(("s2".to_string(), "a0".to_string()), 
            vec![("s0".to_string(), 0.7), ("s1".to_string(), 0.3)]);
        env.transitions.insert(("s2".to_string(), "a1".to_string()), 
            vec![("s0".to_string(), 0.3), ("s1".to_string(), 0.7)]);
        
        // 设置奖励
        env.rewards.insert(("s0".to_string(), "a0".to_string(), "s1".to_string()), 1.0);
        env.rewards.insert(("s0".to_string(), "a0".to_string(), "s2".to_string()), -1.0);
        env.rewards.insert(("s0".to_string(), "a1".to_string(), "s1".to_string()), -1.0);
        env.rewards.insert(("s0".to_string(), "a1".to_string(), "s2".to_string()), 1.0);
        env.rewards.insert(("s1".to_string(), "a0".to_string(), "s0".to_string()), 1.0);
        env.rewards.insert(("s1".to_string(), "a0".to_string(), "s2".to_string()), -1.0);
        env.rewards.insert(("s1".to_string(), "a1".to_string(), "s0".to_string()), -1.0);
        env.rewards.insert(("s1".to_string(), "a1".to_string(), "s2".to_string()), 1.0);
        env.rewards.insert(("s2".to_string(), "a0".to_string(), "s0".to_string()), 1.0);
        env.rewards.insert(("s2".to_string(), "a0".to_string(), "s1".to_string()), -1.0);
        env.rewards.insert(("s2".to_string(), "a1".to_string(), "s0".to_string()), -1.0);
        env.rewards.insert(("s2".to_string(), "a1".to_string(), "s1".to_string()), 1.0);
        
        env
    }
    
    fn step(&self, state: &str, action: &str) -> (String, f64) {
        let mut rng = rand::thread_rng();
        let transitions = self.transitions.get(&(state.to_string(), action.to_string())).unwrap();
        
        let random = rng.gen::<f64>();
        let mut cumulative_prob = 0.0;
        
        for (next_state, prob) in transitions {
            cumulative_prob += prob;
            if random <= cumulative_prob {
                let reward = self.rewards.get(&(state.to_string(), action.to_string(), next_state.clone())).unwrap_or(&0.0);
                return (next_state.clone(), *reward);
            }
        }
        
        // 默认返回第一个转移
        let (next_state, _) = &transitions[0];
        let reward = self.rewards.get(&(state.to_string(), action.to_string(), next_state.clone())).unwrap_or(&0.0);
        (next_state.clone(), *reward)
    }
}

#[derive(Debug)]
struct QLearningAgent {
    q_table: HashMap<(String, String), f64>,
    learning_rate: f64,
    discount_factor: f64,
    epsilon: f64,
}

impl QLearningAgent {
    fn new(states: &[String], actions: &[String], learning_rate: f64, discount_factor: f64, epsilon: f64) -> Self {
        let mut q_table = HashMap::new();
        
        for state in states {
            for action in actions {
                q_table.insert((state.clone(), action.clone()), 0.0);
            }
        }
        
        QLearningAgent {
            q_table,
            learning_rate,
            discount_factor,
            epsilon,
        }
    }
    
    fn choose_action(&self, state: &str, actions: &[String]) -> String {
        let mut rng = rand::thread_rng();
        
        if rng.gen::<f64>() < self.epsilon {
            // 探索：随机选择动作
            actions[rng.gen_range(0..actions.len())].clone()
        } else {
            // 利用：选择Q值最大的动作
            let mut best_action = &actions[0];
            let mut best_q_value = self.q_table.get(&(state.to_string(), actions[0].clone())).unwrap_or(&0.0);
            
            for action in actions {
                let q_value = self.q_table.get(&(state.to_string(), action.clone())).unwrap_or(&0.0);
                if q_value > best_q_value {
                    best_q_value = q_value;
                    best_action = action;
                }
            }
            
            best_action.clone()
        }
    }
    
    fn update(&mut self, state: &str, action: &str, reward: f64, next_state: &str, next_actions: &[String]) {
        let current_q = self.q_table.get(&(state.to_string(), action.to_string())).unwrap_or(&0.0);
        
        // 计算下一状态的最大Q值
        let mut max_next_q = 0.0;
        for next_action in next_actions {
            let next_q = self.q_table.get(&(next_state.to_string(), next_action.clone())).unwrap_or(&0.0);
            if next_q > &max_next_q {
                max_next_q = *next_q;
            }
        }
        
        // Q学习更新公式
        let new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q);
        self.q_table.insert((state.to_string(), action.to_string()), new_q);
    }
    
    fn get_policy(&self, states: &[String], actions: &[String]) -> HashMap<String, String> {
        let mut policy = HashMap::new();
        
        for state in states {
            let mut best_action = &actions[0];
            let mut best_q_value = self.q_table.get(&(state.clone(), actions[0].clone())).unwrap_or(&0.0);
            
            for action in actions {
                let q_value = self.q_table.get(&(state.clone(), action.clone())).unwrap_or(&0.0);
                if q_value > best_q_value {
                    best_q_value = q_value;
                    best_action = action;
                }
            }
            
            policy.insert(state.clone(), best_action.clone());
        }
        
        policy
    }
}

fn main() {
    let env = Environment::new();
    let mut agent = QLearningAgent::new(&env.states, &env.actions, 0.1, 0.9, 0.1);
    
    let episodes = 1000;
    let max_steps = 100;
    
    for episode in 0..episodes {
        let mut current_state = "s0".to_string();
        let mut total_reward = 0.0;
        
        for step in 0..max_steps {
            let action = agent.choose_action(&current_state, &env.actions);
            let (next_state, reward) = env.step(&current_state, &action);
            
            agent.update(&current_state, &action, reward, &next_state, &env.actions);
            
            total_reward += reward;
            current_state = next_state;
            
            if step == max_steps - 1 {
                break;
            }
        }
        
        if episode % 100 == 0 {
            println!("Episode {}: Total Reward = {:.2}", episode, total_reward);
        }
    }
    
    // 输出学习到的策略
    let policy = agent.get_policy(&env.states, &env.actions);
    println!("\n学习到的策略:");
    for (state, action) in &policy {
        println!("{} -> {}", state, action);
    }
    
    // 输出Q表
    println!("\nQ表:");
    for ((state, action), q_value) in &agent.q_table {
        println!("Q({}, {}) = {:.3}", state, action, q_value);
    }
}
```

### Haskell实现：策略梯度

```haskell
import System.Random
import Data.List (foldl')
import Data.Map (Map)
import qualified Data.Map as Map

-- 环境定义
data Environment = Environment {
    states :: [String],
    actions :: [String],
    transitions :: Map (String, String) [(String, Double)],
    rewards :: Map (String, String, String) Double
} deriving Show

-- 智能体定义
data Agent = Agent {
    policy :: Map (String, String) Double,  -- 策略参数
    valueFunction :: Map String Double,      -- 价值函数
    learningRate :: Double,
    discountFactor :: Double
} deriving Show

-- 创建环境
createEnvironment :: Environment
createEnvironment = Environment {
    states = ["s0", "s1", "s2"],
    actions = ["a0", "a1"],
    transitions = Map.fromList [
        (("s0", "a0"), [("s1", 0.8), ("s2", 0.2)]),
        (("s0", "a1"), [("s1", 0.2), ("s2", 0.8)]),
        (("s1", "a0"), [("s0", 0.9), ("s2", 0.1)]),
        (("s1", "a1"), [("s0", 0.1), ("s2", 0.9)]),
        (("s2", "a0"), [("s0", 0.7), ("s1", 0.3)]),
        (("s2", "a1"), [("s0", 0.3), ("s1", 0.7)])
    ],
    rewards = Map.fromList [
        (("s0", "a0", "s1"), 1.0),
        (("s0", "a0", "s2"), -1.0),
        (("s0", "a1", "s1"), -1.0),
        (("s0", "a1", "s2"), 1.0),
        (("s1", "a0", "s0"), 1.0),
        (("s1", "a0", "s2"), -1.0),
        (("s1", "a1", "s0"), -1.0),
        (("s1", "a1", "s2"), 1.0),
        (("s2", "a0", "s0"), 1.0),
        (("s2", "a0", "s1"), -1.0),
        (("s2", "a1", "s0"), -1.0),
        (("s2", "a1", "s1"), 1.0)
    ]
}

-- 环境步进
step :: Environment -> String -> String -> IO (String, Double)
step env state action = do
    let transitions = Map.findWithDefault [] (state, action) (transitions env)
    random <- randomRIO (0.0, 1.0)
    
    let (nextState, _) = selectTransition transitions random
    let reward = Map.findWithDefault 0.0 (state, action, nextState) (rewards env)
    
    return (nextState, reward)
  where
    selectTransition :: [(String, Double)] -> Double -> (String, Double)
    selectTransition [] _ = ("s0", 0.0)
    selectTransition ((s, p):rest) r
        | r <= p = (s, p)
        | otherwise = selectTransition rest (r - p)

-- 创建智能体
createAgent :: Double -> Double -> Agent
createAgent lr df = Agent {
    policy = Map.empty,
    valueFunction = Map.empty,
    learningRate = lr,
    discountFactor = df
}

-- 选择动作（ε-贪婪策略）
chooseAction :: Agent -> Environment -> String -> Double -> IO String
chooseAction agent env state epsilon = do
    random <- randomRIO (0.0, 1.0)
    
    if random < epsilon
        then do
            -- 探索：随机选择
            actionIndex <- randomRIO (0, length (actions env) - 1)
            return $ actions env !! actionIndex
        else do
            -- 利用：选择最优动作
            return $ getBestAction agent state (actions env)

-- 获取最优动作
getBestAction :: Agent -> String -> [String] -> String
getBestAction agent state availableActions =
    let actionValues = map (\action -> (action, getPolicyValue agent state action)) availableActions
        bestAction = foldl' (\best (action, value) -> 
            if value > snd best then (action, value) else best) 
            (head actionValues, snd (head actionValues)) actionValues
    in fst bestAction

-- 获取策略值
getPolicyValue :: Agent -> String -> String -> Double
getPolicyValue agent state action = Map.findWithDefault 0.0 (state, action) (policy agent)

-- 策略梯度更新
updatePolicy :: Agent -> String -> String -> Double -> Agent
updatePolicy agent state action advantage =
    let currentValue = getPolicyValue agent state action
        newValue = currentValue + learningRate agent * advantage
        newPolicy = Map.insert (state, action) newValue (policy agent)
    in agent { policy = newPolicy }

-- 计算优势函数
calculateAdvantage :: Agent -> String -> String -> String -> Double -> Double
calculateAdvantage agent state action nextState reward =
    let currentValue = getPolicyValue agent state action
        nextValue = Map.findWithDefault 0.0 nextState (valueFunction agent)
        tdTarget = reward + discountFactor agent * nextValue
    in tdTarget - currentValue

-- 更新价值函数
updateValueFunction :: Agent -> String -> Double -> Agent
updateValueFunction agent state targetValue =
    let currentValue = Map.findWithDefault 0.0 state (valueFunction agent)
        newValue = currentValue + learningRate agent * (targetValue - currentValue)
        newValueFunction = Map.insert state newValue (valueFunction agent)
    in agent { valueFunction = newValueFunction }

-- 训练智能体
trainAgent :: Agent -> Environment -> Int -> Double -> IO Agent
trainAgent agent env episodes epsilon = do
    foldM (\acc episode -> do
        let episodeAgent = trainEpisode acc env epsilon
        if episode `mod` 100 == 0
            then putStrLn $ "Episode " ++ show episode ++ " completed"
            else return ()
        return episodeAgent
    ) agent [1..episodes]

-- 训练单个回合
trainEpisode :: Agent -> Environment -> Double -> IO Agent
trainEpisode agent env epsilon = do
    let initialState = head (states env)
    (finalAgent, _) <- foldM (\(accAgent, currentState) step -> do
        action <- chooseAction accAgent env currentState epsilon
        (nextState, reward) <- step env currentState action
        
        let advantage = calculateAdvantage accAgent currentState action nextState reward
        let updatedAgent = updatePolicy accAgent currentState action advantage
        
        let targetValue = reward + discountFactor accAgent * Map.findWithDefault 0.0 nextState (valueFunction updatedAgent)
        let finalAgent = updateValueFunction updatedAgent currentState targetValue
        
        return (finalAgent, nextState)
    ) (agent, initialState) [1..50]  -- 最多50步
    
    return finalAgent

-- 主函数
main :: IO ()
main = do
    let env = createEnvironment
        agent = createAgent 0.01 0.9
        episodes = 1000
        epsilon = 0.1
    
    putStrLn "开始训练强化学习智能体..."
    trainedAgent <- trainAgent agent env episodes epsilon
    
    putStrLn "\n训练完成！"
    putStrLn "最终策略:"
    mapM_ (\state -> do
        let bestAction = getBestAction trainedAgent state (actions env)
        putStrLn $ state ++ " -> " ++ bestAction
    ) (states env)
    
    putStrLn "\n价值函数:"
    mapM_ (\state -> do
        let value = Map.findWithDefault 0.0 state (valueFunction trainedAgent)
        putStrLn $ state ++ ": " ++ show value
    ) (states env)
```

---

## 参考文献 / References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. Puterman, M. L. (2014). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
3. Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.
4. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3-4), 229-256.
5. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
6. Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
7. Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484-489.

---

*本模块为FormalAI提供了强化学习的理论基础，涵盖了从基础MDP到深度强化学习的各个方面，为自主AI系统的设计和分析提供了数学工具。*
