# 2.3 强化学习理论 / Reinforcement Learning Theory

## 概述 / Overview

强化学习理论研究智能体如何通过与环境的交互来学习最优策略，为FormalAI提供决策和控制的数学基础。

Reinforcement learning theory studies how agents learn optimal policies through interaction with environments, providing mathematical foundations for decision-making and control in FormalAI.

## 目录 / Table of Contents

- [2.3 强化学习理论 / Reinforcement Learning Theory](#23-强化学习理论--reinforcement-learning-theory)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [1. 马尔可夫决策过程 / Markov Decision Processes](#1-马尔可夫决策过程--markov-decision-processes)
    - [1.1 MDP定义 / MDP Definition](#11-mdp定义--mdp-definition)
    - [1.2 策略与价值函数 / Policy and Value Functions](#12-策略与价值函数--policy-and-value-functions)
    - [1.3 最优策略 / Optimal Policy](#13-最优策略--optimal-policy)
  - [2. 动态规划 / Dynamic Programming](#2-动态规划--dynamic-programming)
    - [2.1 贝尔曼方程 / Bellman Equations](#21-贝尔曼方程--bellman-equations)
    - [2.2 值迭代 / Value Iteration](#22-值迭代--value-iteration)
    - [2.3 策略迭代 / Policy Iteration](#23-策略迭代--policy-iteration)
  - [3. 蒙特卡洛方法 / Monte Carlo Methods](#3-蒙特卡洛方法--monte-carlo-methods)
    - [3.1 蒙特卡洛预测 / Monte Carlo Prediction](#31-蒙特卡洛预测--monte-carlo-prediction)
    - [3.2 蒙特卡洛控制 / Monte Carlo Control](#32-蒙特卡洛控制--monte-carlo-control)
  - [4. 时序差分学习 / Temporal Difference Learning](#4-时序差分学习--temporal-difference-learning)
    - [4.1 TD(0)学习 / TD(0) Learning](#41-td0学习--td0-learning)
    - [4.2 Q学习 / Q-Learning](#42-q学习--q-learning)
    - [4.3 SARSA算法 / SARSA Algorithm](#43-sarsa算法--sarsa-algorithm)
  - [5. 策略梯度方法 / Policy Gradient Methods](#5-策略梯度方法--policy-gradient-methods)
    - [5.1 策略梯度定理 / Policy Gradient Theorem](#51-策略梯度定理--policy-gradient-theorem)
    - [5.2 REINFORCE算法 / REINFORCE Algorithm](#52-reinforce算法--reinforce-algorithm)
    - [5.3 自然策略梯度 / Natural Policy Gradient](#53-自然策略梯度--natural-policy-gradient)
  - [6. 演员-评论家方法 / Actor-Critic Methods](#6-演员-评论家方法--actor-critic-methods)
    - [6.1 演员-评论家架构 / Actor-Critic Architecture](#61-演员-评论家架构--actor-critic-architecture)
    - [6.2 A2C算法 / A2C Algorithm](#62-a2c算法--a2c-algorithm)
    - [6.3 A3C算法 / A3C Algorithm](#63-a3c算法--a3c-algorithm)
  - [7. 深度强化学习 / Deep Reinforcement Learning](#7-深度强化学习--deep-reinforcement-learning)
    - [7.1 DQN算法 / DQN Algorithm](#71-dqn算法--dqn-algorithm)
    - [7.2 DDPG算法 / DDPG Algorithm](#72-ddpg算法--ddpg-algorithm)
    - [7.3 PPO算法 / PPO Algorithm](#73-ppo算法--ppo-algorithm)
  - [8. 多智能体强化学习 / Multi-Agent Reinforcement Learning](#8-多智能体强化学习--multi-agent-reinforcement-learning)
    - [8.1 多智能体MDP / Multi-Agent MDP](#81-多智能体mdp--multi-agent-mdp)
    - [8.2 纳什均衡 / Nash Equilibrium](#82-纳什均衡--nash-equilibrium)
    - [8.3 MADDPG算法 / MADDPG Algorithm](#83-maddpg算法--maddpg-algorithm)
  - [9. 逆强化学习 / Inverse Reinforcement Learning](#9-逆强化学习--inverse-reinforcement-learning)
    - [9.1 逆强化学习问题 / Inverse RL Problem](#91-逆强化学习问题--inverse-rl-problem)
    - [9.2 学徒学习 / Apprenticeship Learning](#92-学徒学习--apprenticeship-learning)
    - [9.3 GAIL算法 / GAIL Algorithm](#93-gail算法--gail-algorithm)
  - [10. 元强化学习 / Meta Reinforcement Learning](#10-元强化学习--meta-reinforcement-learning)
    - [10.1 元学习问题 / Meta-Learning Problem](#101-元学习问题--meta-learning-problem)
    - [10.2 MAML-RL / MAML for RL](#102-maml-rl--maml-for-rl)
    - [10.3 RL2算法 / RL2 Algorithm](#103-rl2算法--rl2-algorithm)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：Q学习算法](#rust实现q学习算法)
    - [Haskell实现：策略梯度算法](#haskell实现策略梯度算法)
  - [参考文献 / References](#参考文献--references)

---

## 1. 马尔可夫决策过程 / Markov Decision Processes

### 1.1 MDP定义 / MDP Definition

**马尔可夫决策过程 / Markov Decision Process:**

$M = (S, A, P, R, \gamma)$ 其中：

$M = (S, A, P, R, \gamma)$ where:

- $S$ 是状态空间 / state space
- $A$ 是动作空间 / action space
- $P: S \times A \times S \rightarrow [0,1]$ 是转移概率 / transition probability
- $R: S \times A \times S \rightarrow \mathbb{R}$ 是奖励函数 / reward function
- $\gamma \in [0,1]$ 是折扣因子 / discount factor

**马尔可夫性质 / Markov Property:**

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)$$

### 1.2 策略与价值函数 / Policy and Value Functions

**策略 / Policy:**

$\pi: S \rightarrow \Delta(A)$ 是状态到动作概率分布的映射。

$\pi: S \rightarrow \Delta(A)$ maps states to probability distributions over actions.

**状态价值函数 / State Value Function:**

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t | s_0 = s\right]$$

**动作价值函数 / Action Value Function:**

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t | s_0 = s, a_0 = a\right]$$

### 1.3 最优策略 / Optimal Policy

**最优价值函数 / Optimal Value Function:**

$$V^*(s) = \max_\pi V^\pi(s)$$

**最优动作价值函数 / Optimal Action Value Function:**

$$Q^*(s, a) = \max_\pi Q^\pi(s, a)$$

**最优策略 / Optimal Policy:**

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

---

## 2. 动态规划 / Dynamic Programming

### 2.1 贝尔曼方程 / Bellman Equations

**状态价值贝尔曼方程 / State Value Bellman Equation:**

$$V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

**动作价值贝尔曼方程 / Action Value Bellman Equation:**

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

**最优贝尔曼方程 / Optimal Bellman Equations:**

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$$

$$Q^*(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$$

### 2.2 值迭代 / Value Iteration

**值迭代算法 / Value Iteration Algorithm:**

$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$$

**收敛性 / Convergence:**

值迭代收敛到最优价值函数：

Value iteration converges to optimal value function:

$$\lim_{k \rightarrow \infty} V_k = V^*$$

### 2.3 策略迭代 / Policy Iteration

**策略评估 / Policy Evaluation:**

$$V_{k+1}^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V_k^\pi(s')]$$

**策略改进 / Policy Improvement:**

$$\pi_{k+1}(s) = \arg\max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^{\pi_k}(s')]$$

---

## 3. 蒙特卡洛方法 / Monte Carlo Methods

### 3.1 蒙特卡洛预测 / Monte Carlo Prediction

**首次访问MC / First-Visit MC:**

$$V(s) = \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i(s)$$

其中 $G_i(s)$ 是第 $i$ 次访问状态 $s$ 的回报。

where $G_i(s)$ is the return from the $i$-th visit to state $s$.

**每次访问MC / Every-Visit MC:**

$$V(s) = \frac{1}{N(s)} \sum_{t=1}^{T} \mathbb{I}(s_t = s) G_t$$

### 3.2 蒙特卡洛控制 / Monte Carlo Control

**$\epsilon$-贪心策略 / $\epsilon$-Greedy Policy:**

$$
\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s,a') \\
\frac{\epsilon}{|A|} & \text{otherwise}
\end{cases}
$$

**MC-ES算法 / MC-ES Algorithm:**

1. 生成完整轨迹
2. 计算回报
3. 更新动作价值函数
4. 改进策略

   1. Generate complete trajectory
   2. Calculate returns
   3. Update action value function
   4. Improve policy

---

## 4. 时序差分学习 / Temporal Difference Learning

### 4.1 TD(0)学习 / TD(0) Learning

**TD(0)更新 / TD(0) Update:**

$$V(s_t) \leftarrow V(s_t) + \alpha [r_{t+1} + \gamma V(s_{t+1}) - V(s_t)]$$

**TD误差 / TD Error:**

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

### 4.2 Q学习 / Q-Learning

**Q学习更新 / Q-Learning Update:**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)]$$

**收敛性 / Convergence:**

在适当条件下，Q学习收敛到最优动作价值函数。

Under appropriate conditions, Q-learning converges to optimal action value function.

### 4.3 SARSA算法 / SARSA Algorithm

**SARSA更新 / SARSA Update:**

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha [r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$$

**在线学习 / On-Policy Learning:**

SARSA是在线学习算法，使用当前策略选择动作。

SARSA is an on-policy learning algorithm using current policy to select actions.

---

## 5. 策略梯度方法 / Policy Gradient Methods

### 5.1 策略梯度定理 / Policy Gradient Theorem

**策略梯度定理 / Policy Gradient Theorem:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)]$$

**证明 / Proof:**

使用似然比技巧和重要性采样。

Using likelihood ratio trick and importance sampling.

### 5.2 REINFORCE算法 / REINFORCE Algorithm

**REINFORCE更新 / REINFORCE Update:**

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) G_t$$

其中 $G_t$ 是从时间步 $t$ 开始的回报。

where $G_t$ is the return starting from time step $t$.

**基线 / Baseline:**

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) (G_t - b(s_t))$$

### 5.3 自然策略梯度 / Natural Policy Gradient

**Fisher信息矩阵 / Fisher Information Matrix:**

$$F(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T]$$

**自然梯度 / Natural Gradient:**

$$\tilde{\nabla}_\theta J(\theta) = F(\theta)^{-1} \nabla_\theta J(\theta)$$

---

## 6. 演员-评论家方法 / Actor-Critic Methods

### 6.1 演员-评论家架构 / Actor-Critic Architecture

**演员网络 / Actor Network:**

$\pi_\theta: S \rightarrow \Delta(A)$ 输出动作概率分布。

$\pi_\theta: S \rightarrow \Delta(A)$ outputs action probability distribution.

**评论家网络 / Critic Network:**

$V_\phi: S \rightarrow \mathbb{R}$ 估计状态价值函数。

$V_\phi: S \rightarrow \mathbb{R}$ estimates state value function.

### 6.2 A2C算法 / A2C Algorithm

**优势函数 / Advantage Function:**

$$A(s,a) = Q(s,a) - V(s)$$

**A2C更新 / A2C Update:**

$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t) A(s_t,a_t)$$

$$\phi \leftarrow \phi + \beta \nabla_\phi (V_\phi(s_t) - V_{target})^2$$

### 6.3 A3C算法 / A3C Algorithm

**异步更新 / Asynchronous Update:**

多个智能体并行更新共享参数。

Multiple agents update shared parameters in parallel.

**全局网络 / Global Network:**

$$\theta \leftarrow \theta + \alpha \sum_i \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) A(s_t^i,a_t^i)$$

---

## 7. 深度强化学习 / Deep Reinforcement Learning

### 7.1 DQN算法 / DQN Algorithm

**经验回放 / Experience Replay:**

$$(s_t, a_t, r_{t+1}, s_{t+1}) \sim \mathcal{D}$$

**目标网络 / Target Network:**

$$Q_{target} = r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)$$

**损失函数 / Loss Function:**

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}[(Q(s,a;\theta) - Q_{target})^2]$$

### 7.2 DDPG算法 / DDPG Algorithm

**确定性策略 / Deterministic Policy:**

$$\mu_\theta: S \rightarrow A$$

**演员更新 / Actor Update:**

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \mathcal{D}}[\nabla_a Q(s,a)|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s)]$$

**评论家更新 / Critic Update:**

$$L(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}[(Q(s,a;\phi) - (r + \gamma Q(s',\mu_\theta(s');\phi^-)))^2]$$

### 7.3 PPO算法 / PPO Algorithm

**比率 / Ratio:**

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**裁剪目标 / Clipped Objective:**

$$L(\theta) = \mathbb{E}_t[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$$

---

## 8. 多智能体强化学习 / Multi-Agent Reinforcement Learning

### 8.1 多智能体MDP / Multi-Agent MDP

**多智能体MDP / Multi-Agent MDP:**

$M = (N, S, A_1, \ldots, A_N, P, R_1, \ldots, R_N, \gamma)$ 其中：

$M = (N, S, A_1, \ldots, A_N, P, R_1, \ldots, R_N, \gamma)$ where:

- $N$ 是智能体数量 / number of agents
- $A_i$ 是智能体 $i$ 的动作空间 / action space of agent $i$
- $R_i$ 是智能体 $i$ 的奖励函数 / reward function of agent $i$

### 8.2 纳什均衡 / Nash Equilibrium

**纳什均衡 / Nash Equilibrium:**

策略组合 $(\pi_1^*, \ldots, \pi_N^*)$ 是纳什均衡，如果：

Strategy profile $(\pi_1^*, \ldots, \pi_N^*)$ is a Nash equilibrium if:

$$V_i^{\pi_i^*, \pi_{-i}^*} \geq V_i^{\pi_i, \pi_{-i}^*} \text{ for all } \pi_i$$

### 8.3 MADDPG算法 / MADDPG Algorithm

**集中式训练 / Centralized Training:**

$$Q_i^\mu(s, a_1, \ldots, a_N) = \mathbb{E}[r_i + \gamma Q_i^\mu(s', \mu_1(s_1'), \ldots, \mu_N(s_N'))]$$

**分散式执行 / Decentralized Execution:**

每个智能体只使用自己的策略。

Each agent only uses its own policy.

---

## 9. 逆强化学习 / Inverse Reinforcement Learning

### 9.1 逆强化学习问题 / Inverse RL Problem

**问题定义 / Problem Definition:**

给定专家演示 $\mathcal{D} = \{(s_t, a_t)\}_{t=1}^T$，学习奖励函数 $R$。

Given expert demonstrations $\mathcal{D} = \{(s_t, a_t)\}_{t=1}^T$, learn reward function $R$.

**最大熵IRL / Maximum Entropy IRL:**

$$\max_R \min_\pi -H(\pi) + \mathbb{E}_\pi[R(s,a)] - \mathbb{E}_{\pi_E}[R(s,a)]$$

### 9.2 学徒学习 / Apprenticeship Learning

**特征匹配 / Feature Matching:**

$$\mathbb{E}_{\pi_E}[\phi(s,a)] = \mathbb{E}_\pi[\phi(s,a)]$$

其中 $\phi(s,a)$ 是状态-动作特征。

where $\phi(s,a)$ are state-action features.

**最大边际 / Maximum Margin:**

$$\min_w \frac{1}{2}\|w\|^2 + C \sum_i \xi_i$$

s.t. $w^T \phi(s,a) \geq w^T \phi(s,a') - \xi_i$

### 9.3 GAIL算法 / GAIL Algorithm

**对抗训练 / Adversarial Training:**

$$\min_\pi \max_D \mathbb{E}_\pi[\log D(s,a)] + \mathbb{E}_{\pi_E}[\log(1-D(s,a))]$$

**生成对抗网络 / Generative Adversarial Network:**

使用GAN框架学习策略。

Using GAN framework to learn policy.

---

## 10. 元强化学习 / Meta Reinforcement Learning

### 10.1 元学习问题 / Meta-Learning Problem

**任务分布 / Task Distribution:**

$\mathcal{T} \sim p(\mathcal{T})$ 是任务分布。

$\mathcal{T} \sim p(\mathcal{T})$ is task distribution.

**元学习目标 / Meta-Learning Objective:**

$$\min_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})}[\mathcal{L}_{\mathcal{T}}(\theta)]$$

### 10.2 MAML-RL / MAML for RL

**MAML更新 / MAML Update:**

$$\theta' = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}}(\theta)$$

**元更新 / Meta Update:**

$$\theta \leftarrow \theta - \beta \nabla_\theta \mathbb{E}_{\mathcal{T}}[\mathcal{L}_{\mathcal{T}}(\theta')]$$

### 10.3 RL2算法 / RL2 Algorithm

**循环神经网络 / Recurrent Neural Network:**

$$h_t = f(h_{t-1}, s_t, a_t, r_t)$$

**快速适应 / Fast Adaptation:**

通过RNN隐状态实现快速适应。

Fast adaptation through RNN hidden states.

---

## 代码示例 / Code Examples

### Rust实现：Q学习算法

```rust
use std::collections::HashMap;
use rand::Rng;

# [derive(Debug, Clone)]
struct QLearning {
    q_table: HashMap<(String, String), f64>,
    learning_rate: f64,
    discount_factor: f64,
    epsilon: f64,
}

impl QLearning {
    fn new(learning_rate: f64, discount_factor: f64, epsilon: f64) -> Self {
        QLearning {
            q_table: HashMap::new(),
            learning_rate,
            discount_factor,
            epsilon,
        }
    }

    fn get_q_value(&self, state: &str, action: &str) -> f64 {
        *self.q_table.get(&(state.to_string(), action.to_string())).unwrap_or(&0.0)
    }

    fn update_q_value(&mut self, state: &str, action: &str, reward: f64, next_state: &str) {
        let current_q = self.get_q_value(state, action);

        // 计算最大Q值
        let max_next_q = self.get_max_q_value(next_state);

        // Q学习更新公式
        let new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q);

        self.q_table.insert((state.to_string(), action.to_string()), new_q);
    }

    fn get_max_q_value(&self, state: &str) -> f64 {
        let actions = vec!["up", "down", "left", "right"];
        actions.iter()
            .map(|action| self.get_q_value(state, action))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn choose_action(&self, state: &str) -> String {
        let mut rng = rand::thread_rng();

        // ε-贪心策略
        if rng.gen::<f64>() < self.epsilon {
            // 随机选择
            let actions = vec!["up", "down", "left", "right"];
            actions[rng.gen_range(0..actions.len())].to_string()
        } else {
            // 选择最大Q值的动作
            let actions = vec!["up", "down", "left", "right"];
            let mut best_action = "up";
            let mut best_q = f64::NEG_INFINITY;

            for action in actions {
                let q_value = self.get_q_value(state, action);
                if q_value > best_q {
                    best_q = q_value;
                    best_action = action;
                }
            }

            best_action.to_string()
        }
    }

    fn train(&mut self, episodes: usize) {
        for episode in 0..episodes {
            let mut state = "start";
            let mut total_reward = 0.0;

            for step in 0..100 {
                let action = self.choose_action(state);
                let (next_state, reward) = self.take_action(state, &action);

                self.update_q_value(state, &action, reward, next_state);

                total_reward += reward;
                state = next_state;

                if state == "goal" {
                    break;
                }
            }

            if episode % 100 == 0 {
                println!("Episode {}, Total Reward: {}", episode, total_reward);
            }
        }
    }

    fn take_action(&self, state: &str, action: &str) -> (&str, f64) {
        // 简化的环境模拟
        match (state, action) {
            ("start", "right") => ("goal", 1.0),
            ("start", _) => ("start", -0.1),
            ("goal", _) => ("goal", 0.0),
            _ => (state, -0.1),
        }
    }
}

fn main() {
    let mut q_learning = QLearning::new(0.1, 0.9, 0.1);

    println!("开始Q学习训练...");
    q_learning.train(1000);

    println!("训练完成！");
    println!("Q表大小: {}", q_learning.q_table.len());

    // 测试最优策略
    let test_state = "start";
    let optimal_action = q_learning.choose_action(test_state);
    println!("状态 '{}' 的最优动作: {}", test_state, optimal_action);
}
```

### Haskell实现：策略梯度算法

```haskell
import System.Random
import Data.List (foldl')
import Data.Vector (Vector, fromList, (!), update)

-- 策略网络
data PolicyNetwork = PolicyNetwork {
    weights :: Vector Double,
    inputSize :: Int,
    outputSize :: Int
} deriving Show

-- 创建策略网络
createPolicyNetwork :: Int -> Int -> PolicyNetwork
createPolicyNetwork inputSize outputSize = PolicyNetwork {
    weights = fromList (replicate (inputSize * outputSize) 0.1),
    inputSize = inputSize,
    outputSize = outputSize
}

-- 前向传播
forward :: PolicyNetwork -> Vector Double -> Vector Double
forward network input =
    let weightMatrix = reshape (inputSize network) (outputSize network) (weights network)
        logits = weightMatrix `mult` input
    in softmax logits

-- Softmax函数
softmax :: Vector Double -> Vector Double
softmax x =
    let maxVal = maximum x
        expX = map (\xi -> exp (xi - maxVal)) x
        sumExp = sum expX
    in map (/ sumExp) expX

-- 策略梯度更新
policyGradientUpdate :: PolicyNetwork -> Vector Double -> Vector Double -> Double -> PolicyNetwork
policyGradientUpdate network state action reward =
    let actionProbs = forward network state
        logProb = log (actionProbs ! actionIndex)
        gradient = scale (reward * logProb) state
        newWeights = weights network `add` gradient
    in network { weights = newWeights }
where
    actionIndex = round action

-- REINFORCE算法
reinforce :: PolicyNetwork -> [(Vector Double, Int, Double)] -> PolicyNetwork
reinforce network trajectory =
    let returns = calculateReturns trajectory
        updates = zipWith (\transition ret ->
            policyGradientUpdate network (state transition) (action transition) ret)
            trajectory returns
    in foldl' (\net update -> update) network updates

-- 计算回报
calculateReturns :: [(Vector Double, Int, Double)] -> [Double]
calculateReturns trajectory =
    let rewards = map reward trajectory
        gamma = 0.99
    in scanr (\r acc -> r + gamma * head acc) 0.0 (init rewards)

-- 数据定义
data Transition = Transition {
    state :: Vector Double,
    action :: Int,
    reward :: Double
} deriving Show

-- 示例环境
gridWorld :: Int -> Int -> [(Vector Double, Int, Double)]
gridWorld width height =
    let states = [(x, y) | x <- [0..width-1], y <- [0..height-1]]
        actions = [0, 1, 2, 3] -- up, down, left, right
    in [(stateToVector (x, y), action, rewardForAction (x, y) action) |
        (x, y) <- states, action <- actions]

-- 状态向量化
stateToVector :: (Int, Int) -> Vector Double
stateToVector (x, y) = fromList [fromIntegral x, fromIntegral y]

-- 动作奖励
rewardForAction :: Int -> Int -> Int -> Double
rewardForAction x y action =
    case action of
        0 -> if y > 0 then 1.0 else -1.0  -- up
        1 -> if y < 4 then 1.0 else -1.0  -- down
        2 -> if x > 0 then 1.0 else -1.0  -- left
        3 -> if x < 4 then 1.0 else -1.0  -- right
        _ -> 0.0

-- 训练函数
trainPolicy :: PolicyNetwork -> Int -> PolicyNetwork
trainPolicy network episodes =
    let trajectories = replicate episodes (gridWorld 5 5)
        networks = scanl reinforce network trajectories
    in last networks

-- 主函数
main :: IO ()
main = do
    let initialNetwork = createPolicyNetwork 2 4
    let trainedNetwork = trainPolicy initialNetwork 100

    putStrLn "策略梯度训练完成！"
    putStrLn $ "网络权重数量: " ++ show (length (weights trainedNetwork))

    -- 测试策略
    let testState = fromList [2.0, 2.0]
    let actionProbs = forward trainedNetwork testState
    putStrLn $ "状态 [2,2] 的动作概率: " ++ show actionProbs
```

---

## 参考文献 / References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. Puterman, M. L. (2014). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley.
3. Bertsekas, D. P. (2017). *Dynamic Programming and Optimal Control*. Athena Scientific.
4. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
5. Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*.
6. Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv*.
7. Lowe, R., et al. (2017). Multi-agent actor-critic for mixed cooperative-competitive environments. *NIPS*.
8. Ng, A. Y., & Russell, S. J. (2000). Algorithms for inverse reinforcement learning. *ICML*.
9. Finn, C., et al. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML*.

---

*本模块为FormalAI提供了全面的强化学习理论基础，涵盖了从基础MDP到现代深度强化学习的完整理论体系。*
