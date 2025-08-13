# 8.2 复杂系统 / Complex Systems / Komplexe Systeme / Systèmes complexes

## 概述 / Overview

复杂系统理论研究由大量相互作用的组件组成的系统行为，为FormalAI提供涌现和自组织现象的理论基础。

Complex systems theory studies the behavior of systems composed of large numbers of interacting components, providing theoretical foundations for emergence and self-organization phenomena in FormalAI.

## 目录 / Table of Contents

- [8.2 复杂系统 / Complex Systems / Komplexe Systeme / Systèmes complexes](#82-复杂系统--complex-systems--komplexe-systeme--systèmes-complexes)
  - [概述 / Overview](#概述--overview)
  - [目录 / Table of Contents](#目录--table-of-contents)
  - [相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes](#相关章节--related-chapters--verwandte-kapitel--chapitres-connexes)
  - [1. 系统动力学 / System Dynamics](#1-系统动力学--system-dynamics)
    - [1.1 动力学方程 / Dynamical Equations](#11-动力学方程--dynamical-equations)
    - [1.2 稳定性分析 / Stability Analysis](#12-稳定性分析--stability-analysis)
    - [1.3 混沌理论 / Chaos Theory](#13-混沌理论--chaos-theory)
  - [2. 网络理论 / Network Theory](#2-网络理论--network-theory)
    - [2.1 网络表示 / Network Representation](#21-网络表示--network-representation)
    - [2.2 网络模型 / Network Models](#22-网络模型--network-models)
    - [2.3 网络动力学 / Network Dynamics](#23-网络动力学--network-dynamics)
  - [3. 自组织 / Self-Organization](#3-自组织--self-organization)
    - [3.1 自组织原理 / Self-Organization Principles](#31-自组织原理--self-organization-principles)
    - [3.2 模式形成 / Pattern Formation](#32-模式形成--pattern-formation)
    - [3.3 集体行为 / Collective Behavior](#33-集体行为--collective-behavior)
  - [4. 临界现象 / Critical Phenomena](#4-临界现象--critical-phenomena)
    - [4.1 临界点 / Critical Points](#41-临界点--critical-points)
    - [4.2 标度律 / Scaling Laws](#42-标度律--scaling-laws)
    - [4.3 临界指数 / Critical Exponents](#43-临界指数--critical-exponents)
  - [5. 相变理论 / Phase Transition Theory](#5-相变理论--phase-transition-theory)
    - [5.1 相变类型 / Phase Transition Types](#51-相变类型--phase-transition-types)
    - [5.2 朗道理论 / Landau Theory](#52-朗道理论--landau-theory)
    - [5.3 重正化群 / Renormalization Group](#53-重正化群--renormalization-group)
  - [代码示例 / Code Examples](#代码示例--code-examples)
    - [Rust实现：复杂系统模拟](#rust实现复杂系统模拟)
    - [Haskell实现：复杂系统](#haskell实现复杂系统)
  - [参考文献 / References](#参考文献--references)

---

## 相关章节 / Related Chapters / Verwandte Kapitel / Chapitres connexes

**前置依赖 / Prerequisites / Voraussetzungen / Prérequis:**

- [8.1 涌现理论](01-emergence-theory/README.md) - 提供涌现基础 / Provides emergence foundation

**后续应用 / Applications / Anwendungen / Applications:**

- [8.3 自组织理论](03-self-organization/README.md) - 提供系统基础 / Provides system foundation

---

## 1. 系统动力学 / System Dynamics

### 1.1 动力学方程 / Dynamical Equations

**常微分方程 / Ordinary Differential Equations:**

$$\frac{dx}{dt} = f(x, t)$$

其中 $x$ 是状态向量，$f$ 是动力学函数。

where $x$ is the state vector and $f$ is the dynamical function.

**偏微分方程 / Partial Differential Equations:**

$$\frac{\partial u}{\partial t} = D\nabla^2 u + f(u)$$

其中 $u$ 是场变量，$D$ 是扩散系数。

where $u$ is the field variable and $D$ is the diffusion coefficient.

**耦合系统 / Coupled Systems:**

$$\frac{dx_i}{dt} = f_i(x_1, x_2, ..., x_n) + \sum_{j=1}^n c_{ij}g(x_i, x_j)$$

其中 $c_{ij}$ 是耦合强度。

where $c_{ij}$ is the coupling strength.

### 1.2 稳定性分析 / Stability Analysis

**线性稳定性 / Linear Stability:**

$$\frac{d\delta x}{dt} = J \delta x$$

其中 $J$ 是雅可比矩阵。

where $J$ is the Jacobian matrix.

**李雅普诺夫稳定性 / Lyapunov Stability:**

$$\frac{dV}{dt} \leq 0$$

其中 $V$ 是李雅普诺夫函数。

where $V$ is the Lyapunov function.

**分岔理论 / Bifurcation Theory:**

$$\frac{dx}{dt} = f(x, \mu)$$

其中 $\mu$ 是分岔参数。

where $\mu$ is the bifurcation parameter.

### 1.3 混沌理论 / Chaos Theory

**混沌定义 / Chaos Definition:**

$$\text{Chaos} = \text{Sensitivity} \land \text{Transitivity} \land \text{Density}$$

**李雅普诺夫指数 / Lyapunov Exponent:**

$$\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \left|\frac{\delta x(t)}{\delta x(0)}\right|$$

**分形维数 / Fractal Dimension:**

$$D = \lim_{\epsilon \to 0} \frac{\ln N(\epsilon)}{\ln(1/\epsilon)}$$

## 2. 网络理论 / Network Theory

### 2.1 网络表示 / Network Representation

**邻接矩阵 / Adjacency Matrix:**

$$
A_{ij} = \begin{cases}
1 & \text{if } (i,j) \in E \\
0 & \text{otherwise}
\end{cases}
$$

**度分布 / Degree Distribution:**

$$P(k) = \frac{N_k}{N}$$

其中 $N_k$ 是度为 $k$ 的节点数。

where $N_k$ is the number of nodes with degree $k$.

**聚类系数 / Clustering Coefficient:**

$$C_i = \frac{2E_i}{k_i(k_i-1)}$$

其中 $E_i$ 是节点 $i$ 的邻居间的边数。

where $E_i$ is the number of edges between neighbors of node $i$.

### 2.2 网络模型 / Network Models

**随机网络 / Random Networks:**

$$P(G) = \prod_{i<j} p^{A_{ij}}(1-p)^{1-A_{ij}}$$

**小世界网络 / Small-World Networks:**

$$\text{Small-World} = \text{High Clustering} \land \text{Short Paths}$$

**无标度网络 / Scale-Free Networks:**

$$P(k) \sim k^{-\gamma}$$

其中 $\gamma$ 是幂律指数。

where $\gamma$ is the power-law exponent.

### 2.3 网络动力学 / Network Dynamics

**同步 / Synchronization:**

$$\frac{d\theta_i}{dt} = \omega_i + K \sum_{j=1}^N A_{ij} \sin(\theta_j - \theta_i)$$

**传播动力学 / Spreading Dynamics:**

$$\frac{dI_i}{dt} = \beta S_i \sum_{j=1}^N A_{ij} I_j - \gamma I_i$$

**意见动力学 / Opinion Dynamics:**

$$\frac{dx_i}{dt} = \sum_{j=1}^N A_{ij} (x_j - x_i)$$

## 3. 自组织 / Self-Organization

### 3.1 自组织原理 / Self-Organization Principles

**自组织定义 / Self-Organization Definition:**

$$\text{Self-Organization} = \text{Local Interactions} \land \text{Global Order}$$

**涌现性质 / Emergent Properties:**

$$\text{Emergence} = \text{Collective Behavior} \land \text{Individual Simplicity}$$

**反馈机制 / Feedback Mechanisms:**

$$\frac{dx}{dt} = f(x) + g(x) \cdot x$$

### 3.2 模式形成 / Pattern Formation

**图灵模式 / Turing Patterns:**

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + f(u,v)$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + g(u,v)$$

**螺旋波 / Spiral Waves:**

$$\frac{\partial u}{\partial t} = \nabla^2 u + f(u)$$

**斑图 / Spots:**

$$\text{Pattern} = \text{Localized Structures} \land \text{Regular Spacing}$$

### 3.3 集体行为 / Collective Behavior

**群体同步 / Collective Synchronization:**

$$\text{Sync} = \text{Phase Locking} \land \text{Frequency Locking}$$

**群体智能 / Collective Intelligence:**

$$\text{Intelligence} = \text{Local Rules} \land \text{Global Optimization}$$

**群体决策 / Collective Decision Making:**

$$\text{Decision} = \text{Individual Preferences} \land \text{Group Consensus}$$

## 4. 临界现象 / Critical Phenomena

### 4.1 临界点 / Critical Points

**临界点定义 / Critical Point Definition:**

$$\text{Critical Point} = \text{Scale Invariance} \land \text{Power Law}$$

**序参量 / Order Parameter:**

$$\phi = \langle \text{Local Order} \rangle$$

**相关长度 / Correlation Length:**

$$\xi \sim |T - T_c|^{-\nu}$$

### 4.2 标度律 / Scaling Laws

**幂律分布 / Power Law Distribution:**

$$P(x) \sim x^{-\alpha}$$

**有限尺寸标度 / Finite Size Scaling:**

$$\phi(L) = L^{-\beta/\nu} f(L/\xi)$$

**普适性 / Universality:**

$$\text{Universality} = \text{Same Exponents} \land \text{Different Systems}$$

### 4.3 临界指数 / Critical Exponents

**临界指数定义 / Critical Exponent Definition:**

$$\alpha, \beta, \gamma, \delta, \nu, \eta$$

**标度关系 / Scaling Relations:**

$$2 - \alpha = 2\beta + \gamma = \beta(\delta + 1)$$

**维数关系 / Dimensional Relations:**

$$d\nu = 2 - \alpha$$

## 5. 相变理论 / Phase Transition Theory

### 5.1 相变类型 / Phase Transition Types

**一级相变 / First-Order Phase Transition:**

$$\Delta S \neq 0, \Delta V \neq 0$$

**二级相变 / Second-Order Phase Transition:**

$$\Delta S = 0, \Delta V = 0, \text{but } \frac{\partial S}{\partial T} \to \infty$$

**连续相变 / Continuous Phase Transition:**

$$\text{Continuous} = \text{No Latent Heat} \land \text{Continuous Order Parameter}$$

### 5.2 朗道理论 / Landau Theory

**朗道自由能 / Landau Free Energy:**

$$F = F_0 + a(T-T_c)\phi^2 + b\phi^4$$

**序参量方程 / Order Parameter Equation:**

$$\frac{\partial F}{\partial \phi} = 2a(T-T_c)\phi + 4b\phi^3 = 0$$

**临界指数 / Critical Exponents:**

$$\beta = \frac{1}{2}, \gamma = 1, \delta = 3$$

### 5.3 重正化群 / Renormalization Group

**重正化变换 / Renormalization Transformation:**

$$K' = R(K)$$

其中 $K$ 是耦合常数。

where $K$ is the coupling constant.

**不动点 / Fixed Points:**

$$K^* = R(K^*)$$

**标度不变性 / Scale Invariance:**

$$\text{Scale Invariance} = \text{Fixed Point} \land \text{Power Law}$$

## 代码示例 / Code Examples

### Rust实现：复杂系统模拟

```rust
use std::collections::HashMap;
use std::f64::consts::PI;

// 复杂系统模拟器
struct ComplexSystemSimulator {
    agents: Vec<Agent>,
    network: Network,
    parameters: SystemParameters,
}

impl ComplexSystemSimulator {
    fn new(num_agents: usize, parameters: SystemParameters) -> Self {
        Self {
            agents: (0..num_agents).map(|i| Agent::new(i)).collect(),
            network: Network::new(num_agents),
            parameters,
        }
    }

    // 系统动力学演化
    fn evolve(&mut self, time_steps: usize) -> Vec<SystemState> {
        let mut states = Vec::new();

        for step in 0..time_steps {
            let state = self.get_current_state();
            states.push(state);

            self.update_system();
        }

        states
    }

    // 更新系统
    fn update_system(&mut self) {
        // 更新每个智能体
        for i in 0..self.agents.len() {
            let neighbors = self.network.get_neighbors(i);
            let neighbor_states: Vec<f64> = neighbors.iter()
                .map(|&j| self.agents[j].state)
                .collect();

            self.agents[i].update_state(&neighbor_states, &self.parameters);
        }

        // 更新网络结构（可选）
        if self.parameters.network_evolution {
            self.network.evolve(&self.agents);
        }
    }

    // 获取当前状态
    fn get_current_state(&self) -> SystemState {
        let agent_states: Vec<f64> = self.agents.iter()
            .map(|agent| agent.state)
            .collect();

        let order_parameter = self.calculate_order_parameter(&agent_states);
        let correlation_length = self.calculate_correlation_length(&agent_states);

        SystemState {
            agent_states,
            order_parameter,
            correlation_length,
            network_density: self.network.get_density(),
        }
    }

    // 计算序参量
    fn calculate_order_parameter(&self, states: &[f64]) -> f64 {
        let mean_state = states.iter().sum::<f64>() / states.len() as f64;
        let variance = states.iter()
            .map(|s| (s - mean_state).powi(2))
            .sum::<f64>() / states.len() as f64;

        variance.sqrt()
    }

    // 计算相关长度
    fn calculate_correlation_length(&self, states: &[f64]) -> f64 {
        let mut total_correlation = 0.0;
        let mut count = 0;

        for i in 0..states.len() {
            for j in (i+1)..states.len() {
                let distance = self.network.get_distance(i, j);
                if distance > 0.0 {
                    let correlation = (states[i] - states[j]).abs();
                    total_correlation += correlation / distance;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_correlation / count as f64
        } else {
            0.0
        }
    }
}

// 智能体
struct Agent {
    id: usize,
    state: f64,
    velocity: f64,
}

impl Agent {
    fn new(id: usize) -> Self {
        Self {
            id,
            state: rand::random::<f64>() * 2.0 * PI,
            velocity: 0.0,
        }
    }

    // 更新状态
    fn update_state(&mut self, neighbor_states: &[f64], parameters: &SystemParameters) {
        if neighbor_states.is_empty() {
            return;
        }

        // Kuramoto模型
        let coupling = parameters.coupling_strength;
        let natural_frequency = parameters.natural_frequency;

        let interaction_term = neighbor_states.iter()
            .map(|&neighbor_state| (neighbor_state - self.state).sin())
            .sum::<f64>() / neighbor_states.len() as f64;

        self.velocity = natural_frequency + coupling * interaction_term;
        self.state += self.velocity * parameters.time_step;

        // 保持状态在[0, 2π]范围内
        self.state = self.state.rem_euclid(2.0 * PI);
    }
}

// 网络
struct Network {
    adjacency_matrix: Vec<Vec<bool>>,
    num_nodes: usize,
}

impl Network {
    fn new(num_nodes: usize) -> Self {
        let mut adjacency_matrix = vec![vec![false; num_nodes]; num_nodes];

        // 创建随机网络
        for i in 0..num_nodes {
            for j in (i+1)..num_nodes {
                if rand::random::<f64>() < 0.1 { // 连接概率
                    adjacency_matrix[i][j] = true;
                    adjacency_matrix[j][i] = true;
                }
            }
        }

        Self {
            adjacency_matrix,
            num_nodes,
        }
    }

    // 获取邻居
    fn get_neighbors(&self, node: usize) -> Vec<usize> {
        (0..self.num_nodes)
            .filter(|&j| self.adjacency_matrix[node][j])
            .collect()
    }

    // 获取距离
    fn get_distance(&self, node1: usize, node2: usize) -> f64 {
        if node1 == node2 {
            return 0.0;
        }

        if self.adjacency_matrix[node1][node2] {
            return 1.0;
        }

        // 简化的距离计算
        let mut visited = vec![false; self.num_nodes];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((node1, 0));
        visited[node1] = true;

        while let Some((current, distance)) = queue.pop_front() {
            if current == node2 {
                return distance as f64;
            }

            for neighbor in self.get_neighbors(current) {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back((neighbor, distance + 1));
                }
            }
        }

        f64::INFINITY
    }

    // 获取网络密度
    fn get_density(&self) -> f64 {
        let mut edge_count = 0;
        for i in 0..self.num_nodes {
            for j in (i+1)..self.num_nodes {
                if self.adjacency_matrix[i][j] {
                    edge_count += 1;
                }
            }
        }

        edge_count as f64 / (self.num_nodes * (self.num_nodes - 1) / 2) as f64
    }

    // 网络演化
    fn evolve(&mut self, agents: &[Agent]) {
        // 简化的网络演化：基于智能体状态的相似性
        for i in 0..self.num_nodes {
            for j in (i+1)..self.num_nodes {
                let similarity = (agents[i].state - agents[j].state).abs();
                let connection_probability = (-similarity).exp();

                if rand::random::<f64>() < connection_probability * 0.1 {
                    self.adjacency_matrix[i][j] = true;
                    self.adjacency_matrix[j][i] = true;
                } else if rand::random::<f64>() < 0.01 {
                    self.adjacency_matrix[i][j] = false;
                    self.adjacency_matrix[j][i] = false;
                }
            }
        }
    }
}

// 系统参数
struct SystemParameters {
    coupling_strength: f64,
    natural_frequency: f64,
    time_step: f64,
    network_evolution: bool,
}

// 系统状态
# [derive(Debug)]
struct SystemState {
    agent_states: Vec<f64>,
    order_parameter: f64,
    correlation_length: f64,
    network_density: f64,
}

// 临界现象分析器
struct CriticalPhenomenaAnalyzer {
    data: Vec<SystemState>,
}

impl CriticalPhenomenaAnalyzer {
    fn new(data: Vec<SystemState>) -> Self {
        Self { data }
    }

    // 分析临界现象
    fn analyze_critical_phenomena(&self) -> CriticalAnalysis {
        let order_parameters: Vec<f64> = self.data.iter()
            .map(|state| state.order_parameter)
            .collect();

        let correlation_lengths: Vec<f64> = self.data.iter()
            .map(|state| state.correlation_length)
            .collect();

        CriticalAnalysis {
            power_law_exponent: self.calculate_power_law_exponent(&order_parameters),
            correlation_exponent: self.calculate_correlation_exponent(&correlation_lengths),
            critical_point: self.detect_critical_point(&order_parameters),
        }
    }

    // 计算幂律指数
    fn calculate_power_law_exponent(&self, data: &[f64]) -> f64 {
        // 简化的幂律拟合
        let log_data: Vec<f64> = data.iter()
            .filter(|&&x| x > 0.0)
            .map(|x| x.ln())
            .collect();

        if log_data.len() < 2 {
            return 0.0;
        }

        let mean_log = log_data.iter().sum::<f64>() / log_data.len() as f64;
        let variance = log_data.iter()
            .map(|x| (x - mean_log).powi(2))
            .sum::<f64>() / log_data.len() as f64;

        -mean_log / variance.max(1e-6)
    }

    // 计算相关指数
    fn calculate_correlation_exponent(&self, data: &[f64]) -> f64 {
        // 简化的相关指数计算
        if data.len() < 2 {
            return 0.0;
        }

        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        variance.sqrt()
    }

    // 检测临界点
    fn detect_critical_point(&self, data: &[f64]) -> Option<usize> {
        if data.len() < 3 {
            return None;
        }

        // 寻找最大变化率
        let mut max_change = 0.0;
        let mut critical_point = 0;

        for i in 1..data.len() {
            let change = (data[i] - data[i-1]).abs();
            if change > max_change {
                max_change = change;
                critical_point = i;
            }
        }

        Some(critical_point)
    }
}

// 临界分析结果
# [derive(Debug)]
struct CriticalAnalysis {
    power_law_exponent: f64,
    correlation_exponent: f64,
    critical_point: Option<usize>,
}

// 随机数生成
mod rand {
    pub fn random<T>() -> T where T: std::default::Default {
        T::default()
    }
}

fn main() {
    println!("=== 复杂系统模拟 ===");

    // 创建系统参数
    let parameters = SystemParameters {
        coupling_strength: 0.5,
        natural_frequency: 1.0,
        time_step: 0.01,
        network_evolution: true,
    };

    // 创建复杂系统模拟器
    let mut simulator = ComplexSystemSimulator::new(100, parameters);

    // 运行模拟
    let states = simulator.evolve(1000);
    println!("模拟完成，生成了 {} 个状态", states.len());

    // 分析临界现象
    let analyzer = CriticalPhenomenaAnalyzer::new(states);
    let analysis = analyzer.analyze_critical_phenomena();
    println!("临界分析结果: {:?}", analysis);

    // 显示最终状态
    if let Some(final_state) = states.last() {
        println!("最终序参量: {:.4}", final_state.order_parameter);
        println!("最终相关长度: {:.4}", final_state.correlation_length);
        println!("最终网络密度: {:.4}", final_state.network_density);
    }
}
```

### Haskell实现：复杂系统

```haskell
-- 复杂系统模块
module ComplexSystems where

import Data.List (foldl', sum)
import Data.Vector (Vector)
import qualified Data.Vector as V
import System.Random (Random, random, randomR)

-- 复杂系统模拟器
data ComplexSystemSimulator = ComplexSystemSimulator {
    agents :: [Agent],
    network :: Network,
    parameters :: SystemParameters
} deriving (Show)

-- 智能体
data Agent = Agent {
    agentId :: Int,
    agentState :: Double,
    agentVelocity :: Double
} deriving (Show)

-- 网络
data Network = Network {
    adjacencyMatrix :: [[Bool]],
    numNodes :: Int
} deriving (Show)

-- 系统参数
data SystemParameters = SystemParameters {
    couplingStrength :: Double,
    naturalFrequency :: Double,
    timeStep :: Double,
    networkEvolution :: Bool
} deriving (Show)

-- 系统状态
data SystemState = SystemState {
    agentStates :: [Double],
    orderParameter :: Double,
    correlationLength :: Double,
    networkDensity :: Double
} deriving (Show)

-- 创建新的复杂系统模拟器
newComplexSystemSimulator :: Int -> SystemParameters -> ComplexSystemSimulator
newComplexSystemSimulator numAgents parameters = ComplexSystemSimulator {
    agents = map (\i -> newAgent i) [0..numAgents-1],
    network = newNetwork numAgents,
    parameters = parameters
}

-- 创建新智能体
newAgent :: Int -> Agent
newAgent id = Agent {
    agentId = id,
    agentState = randomValue 0 (2 * pi),
    agentVelocity = 0.0
}
  where
    randomValue min max = min + (max - min) * 0.5 -- 简化的随机数

-- 创建新网络
newNetwork :: Int -> Network
newNetwork numNodes = Network {
    adjacencyMatrix = createAdjacencyMatrix numNodes,
    numNodes = numNodes
}
  where
    createAdjacencyMatrix n =
        [[if i == j then False else randomConnection | j <- [0..n-1]] | i <- [0..n-1]]

    randomConnection = 0.1 > 0.05 -- 简化的随机连接

-- 系统演化
evolve :: ComplexSystemSimulator -> Int -> [SystemState]
evolve simulator timeSteps =
    take timeSteps (iterate updateSystem simulator)
    >>= (\s -> [getCurrentState s])

-- 更新系统
updateSystem :: ComplexSystemSimulator -> ComplexSystemSimulator
updateSystem simulator = simulator {
    agents = map (\agent -> updateAgent agent simulator) (agents simulator)
}

-- 更新智能体
updateAgent :: Agent -> ComplexSystemSimulator -> Agent
updateAgent agent simulator =
    let neighbors = getNeighbors (agentId agent) (network simulator)
        neighborStates = map (\i -> agentState (agents simulator !! i)) neighbors
    in updateAgentState agent neighborStates (parameters simulator)

-- 更新智能体状态
updateAgentState :: Agent -> [Double] -> SystemParameters -> Agent
updateAgentState agent neighborStates params =
    if null neighborStates
        then agent
        else agent {
            agentState = newState,
            agentVelocity = newVelocity
        }
  where
    coupling = couplingStrength params
    naturalFreq = naturalFrequency params
    dt = timeStep params

    interactionTerm = sum (map (\neighborState ->
        sin (neighborState - agentState agent)) neighborStates) / fromIntegral (length neighborStates)

    newVelocity = naturalFreq + coupling * interactionTerm
    newState = (agentState agent + newVelocity * dt) `mod'` (2 * pi)

-- 获取邻居
getNeighbors :: Int -> Network -> [Int]
getNeighbors node network =
    [j | j <- [0..numNodes network-1],
     j /= node && (adjacencyMatrix network !! node !! j)]

-- 获取当前状态
getCurrentState :: ComplexSystemSimulator -> SystemState
getCurrentState simulator = SystemState {
    agentStates = map agentState (agents simulator),
    orderParameter = calculateOrderParameter (map agentState (agents simulator)),
    correlationLength = calculateCorrelationLength simulator,
    networkDensity = calculateNetworkDensity (network simulator)
}

-- 计算序参量
calculateOrderParameter :: [Double] -> Double
calculateOrderParameter states =
    let meanState = sum states / fromIntegral (length states)
        variance = sum (map (\s -> (s - meanState) ^ 2) states) / fromIntegral (length states)
    in sqrt variance

-- 计算相关长度
calculateCorrelationLength :: ComplexSystemSimulator -> Double
calculateCorrelationLength simulator =
    let states = map agentState (agents simulator)
        correlations = [(i, j, correlation states i j) |
            i <- [0..length states-1],
            j <- [i+1..length states-1]]
        validCorrelations = filter (\(_, _, corr) -> corr > 0) correlations
    in if null validCorrelations
        then 0.0
        else sum (map (\(_, _, corr) -> corr) validCorrelations) / fromIntegral (length validCorrelations)
  where
    correlation states i j = abs (states !! i - states !! j)

-- 计算网络密度
calculateNetworkDensity :: Network -> Double
calculateNetworkDensity network =
    let edgeCount = sum (map (\row -> length (filter id row)) (adjacencyMatrix network))
        totalPossibleEdges = numNodes network * (numNodes network - 1) `div` 2
    in fromIntegral edgeCount / fromIntegral totalPossibleEdges

-- 临界现象分析器
data CriticalPhenomenaAnalyzer = CriticalPhenomenaAnalyzer {
    analysisData :: [SystemState]
} deriving (Show)

-- 创建分析器
newCriticalPhenomenaAnalyzer :: [SystemState] -> CriticalPhenomenaAnalyzer
newCriticalPhenomenaAnalyzer data_ = CriticalPhenomenaAnalyzer { analysisData = data_ }

-- 分析临界现象
analyzeCriticalPhenomena :: CriticalPhenomenaAnalyzer -> CriticalAnalysis
analyzeCriticalPhenomena analyzer = CriticalAnalysis {
    powerLawExponent = calculatePowerLawExponent orderParameters,
    correlationExponent = calculateCorrelationExponent correlationLengths,
    criticalPoint = detectCriticalPoint orderParameters
}
  where
    orderParameters = map orderParameter (analysisData analyzer)
    correlationLengths = map correlationLength (analysisData analyzer)

-- 计算幂律指数
calculatePowerLawExponent :: [Double] -> Double
calculatePowerLawExponent data_ =
    let logData = map log (filter (> 0) data_)
    in if length logData < 2
        then 0.0
        else -mean logData / max 1e-6 (variance logData)
  where
    mean xs = sum xs / fromIntegral (length xs)
    variance xs = sum (map (\x -> (x - mean xs) ^ 2) xs) / fromIntegral (length xs)

-- 计算相关指数
calculateCorrelationExponent :: [Double] -> Double
calculateCorrelationExponent data_ =
    if length data_ < 2
        then 0.0
        else sqrt (variance data_)
  where
    variance xs = sum (map (\x -> (x - mean xs) ^ 2) xs) / fromIntegral (length xs)
    mean xs = sum xs / fromIntegral (length xs)

-- 检测临界点
detectCriticalPoint :: [Double] -> Maybe Int
detectCriticalPoint data_ =
    if length data_ < 3
        then Nothing
        else Just criticalPoint
  where
    changes = zipWith (\x y -> abs (x - y)) (tail data_) data_
    (criticalPoint, _) = maximum (zip [0..] changes)

-- 临界分析结果
data CriticalAnalysis = CriticalAnalysis {
    powerLawExponent :: Double,
    correlationExponent :: Double,
    criticalPoint :: Maybe Int
} deriving (Show)

-- 辅助函数
mod' :: Double -> Double -> Double
mod' x y = x - y * fromIntegral (floor (x / y))

sin :: Double -> Double
sin x = x - x^3/6 + x^5/120 -- 简化的sin函数

log :: Double -> Double
log x = if x <= 0 then 0 else x - 1 -- 简化的log函数

-- 示例使用
main :: IO ()
main = do
    putStrLn "=== 复杂系统模拟 ==="

    -- 创建系统参数
    let parameters = SystemParameters {
        couplingStrength = 0.5,
        naturalFrequency = 1.0,
        timeStep = 0.01,
        networkEvolution = True
    }

    -- 创建复杂系统模拟器
    let simulator = newComplexSystemSimulator 100 parameters

    -- 运行模拟
    let states = evolve simulator 1000
    putStrLn $ "模拟完成，生成了 " ++ show (length states) ++ " 个状态"

    -- 分析临界现象
    let analyzer = newCriticalPhenomenaAnalyzer states
    let analysis = analyzeCriticalPhenomena analyzer
    putStrLn $ "临界分析结果: " ++ show analysis

    -- 显示最终状态
    case last states of
        Just finalState -> do
            putStrLn $ "最终序参量: " ++ show (orderParameter finalState)
            putStrLn $ "最终相关长度: " ++ show (correlationLength finalState)
            putStrLn $ "最终网络密度: " ++ show (networkDensity finalState)
        Nothing -> putStrLn "没有状态数据"
```

## 参考文献 / References

1. Strogatz, S. H. (2001). Nonlinear dynamics and chaos: With applications to physics, biology, chemistry, and engineering. Westview Press.
2. Newman, M. E. (2010). Networks: An introduction. Oxford University Press.
3. Barabási, A. L. (2016). Network science. Cambridge University Press.
4. Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. International Symposium on Mathematical Problems in Theoretical Physics.
5. Bak, P. (1996). How nature works: The science of self-organized criticality. Springer.
6. Kauffman, S. A. (1993). The origins of order: Self-organization and selection in evolution. Oxford University Press.
7. Haken, H. (1983). Synergetics: An introduction. Springer.
8. Nicolis, G., & Prigogine, I. (1977). Self-organization in nonequilibrium systems. Wiley.
9. Stanley, H. E. (1971). Introduction to phase transitions and critical phenomena. Oxford University Press.
10. Goldenfeld, N. (1992). Lectures on phase transitions and the renormalization group. CRC Press.

---

*复杂系统理论为FormalAI提供了涌现和自组织现象的理论基础，为理解AI系统的集体行为和智能涌现提供了重要框架。*

*Complex systems theory provides theoretical foundations for emergence and self-organization phenomena in FormalAI, offering important frameworks for understanding collective behavior and intelligent emergence in AI systems.*
