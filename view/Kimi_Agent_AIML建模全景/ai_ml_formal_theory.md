# AI/ML形式化理论与论证

> **理论深度对标**: MIT 6.867, CMU 10-715, Stanford CS229T

---

## 目录

- [AI/ML形式化理论与论证](#aiml形式化理论与论证)
  - [目录](#目录)
  - [1. 核心概念的形式化定义](#1-核心概念的形式化定义)
    - [1.1 学习问题的形式化框架](#11-学习问题的形式化框架)
      - [定义 1.1.1 (监督学习问题)](#定义-111-监督学习问题)
      - [定义 1.1.2 (数据生成过程)](#定义-112-数据生成过程)
      - [定义 1.1.3 (泛化误差 / 期望风险)](#定义-113-泛化误差--期望风险)
      - [定义 1.1.4 (经验风险)](#定义-114-经验风险)
      - [定义 1.1.5 (贝叶斯最优风险)](#定义-115-贝叶斯最优风险)
    - [1.2 损失函数的数学性质](#12-损失函数的数学性质)
      - [定义 1.2.1 (凸函数)](#定义-121-凸函数)
      - [定义 1.2.2 ($L$-Lipschitz连续性)](#定义-122-l-lipschitz连续性)
      - [定义 1.2.3 ($\\beta$-平滑性 / Lipschitz梯度)](#定义-123-beta-平滑性--lipschitz梯度)
      - [定义 1.2.4 ($\\mu$-强凸性)](#定义-124-mu-强凸性)
    - [1.3 常用损失函数的形式化分析](#13-常用损失函数的形式化分析)
      - [定义 1.3.1 (0-1损失)](#定义-131-0-1损失)
      - [定义 1.3.2 (平方损失 / $L\_2$损失)](#定义-132-平方损失--l_2损失)
      - [定义 1.3.3 (绝对值损失 / $L\_1$损失)](#定义-133-绝对值损失--l_1损失)
      - [定义 1.3.4 (Hinge损失 / SVM损失)](#定义-134-hinge损失--svm损失)
      - [定义 1.3.5 (Logistic损失 / 交叉熵损失)](#定义-135-logistic损失--交叉熵损失)
      - [定理 1.3.1 (损失函数的比较)](#定理-131-损失函数的比较)
  - [2. 学习理论基础](#2-学习理论基础)
    - [2.1 PAC学习框架](#21-pac学习框架)
      - [定义 2.1.1 (PAC可学习性)](#定义-211-pac可学习性)
      - [定义 2.1.2 (样本复杂度)](#定义-212-样本复杂度)
      - [定理 2.1.1 (有限假设空间的PAC界)](#定理-211-有限假设空间的pac界)
    - [2.2 VC维理论](#22-vc维理论)
      - [定义 2.2.1 (打散 / Shattering)](#定义-221-打散--shattering)
      - [定义 2.2.2 (VC维)](#定义-222-vc维)
      - [引理 2.2.1 (Sauer-Shelah引理)](#引理-221-sauer-shelah引理)
      - [定理 2.2.1 (VC泛化界)](#定理-221-vc泛化界)
      - [例子 2.2.1 (常见假设空间的VC维)](#例子-221-常见假设空间的vc维)
    - [2.3 Rademacher复杂度](#23-rademacher复杂度)
      - [定义 2.3.1 (经验Rademacher复杂度)](#定义-231-经验rademacher复杂度)
      - [定义 2.3.2 (Rademacher复杂度)](#定义-232-rademacher复杂度)
      - [定理 2.3.1 (基于Rademacher复杂度的泛化界)](#定理-231-基于rademacher复杂度的泛化界)
      - [定理 2.3.2 (Rademacher复杂度的性质)](#定理-232-rademacher复杂度的性质)
      - [定理 2.3.3 (线性函数的Rademacher复杂度)](#定理-233-线性函数的rademacher复杂度)
    - [2.4 覆盖数与度量熵](#24-覆盖数与度量熵)
      - [定义 2.4.1 ($\\epsilon$-覆盖)](#定义-241-epsilon-覆盖)
      - [定义 2.4.2 (覆盖数)](#定义-242-覆盖数)
      - [定义 2.4.3 (度量熵)](#定义-243-度量熵)
      - [定义 2.4.4 ($L\_2(P\_n)$覆盖数)](#定义-244-l_2p_n覆盖数)
      - [定理 2.4.1 (Dudley熵积分)](#定理-241-dudley熵积分)
      - [定理 2.4.2 (基于覆盖数的泛化界)](#定理-242-基于覆盖数的泛化界)
  - [3. 优化理论](#3-优化理论)
    - [3.1 凸优化基础](#31-凸优化基础)
      - [定义 3.1.1 (凸优化问题)](#定义-311-凸优化问题)
      - [定义 3.1.2 (凸集)](#定义-312-凸集)
      - [定理 3.1.1 (凸优化的一阶最优性条件)](#定理-311-凸优化的一阶最优性条件)
      - [定理 3.1.2 (强凸函数的唯一最优性)](#定理-312-强凸函数的唯一最优性)
    - [3.2 梯度下降收敛性](#32-梯度下降收敛性)
      - [算法 3.2.1 (梯度下降 / GD)](#算法-321-梯度下降--gd)
      - [定理 3.2.1 (凸函数的GD收敛性)](#定理-321-凸函数的gd收敛性)
      - [定理 3.2.2 (强凸函数的GD收敛性)](#定理-322-强凸函数的gd收敛性)
      - [定理 3.2.3 (非凸函数的GD收敛性)](#定理-323-非凸函数的gd收敛性)
    - [3.3 随机梯度下降 (SGD)](#33-随机梯度下降-sgd)
      - [定义 3.3.1 (随机梯度)](#定义-331-随机梯度)
      - [算法 3.3.1 (SGD)](#算法-331-sgd)
      - [假设 3.3.1 (SGD的标准假设)](#假设-331-sgd的标准假设)
      - [定理 3.3.1 (凸函数的SGD收敛性)](#定理-331-凸函数的sgd收敛性)
      - [定理 3.3.2 (强凸函数的SGD收敛性)](#定理-332-强凸函数的sgd收敛性)
    - [3.4 自适应优化器](#34-自适应优化器)
      - [算法 3.4.1 (AdaGrad)](#算法-341-adagrad)
      - [定理 3.4.1 (AdaGrad的收敛性)](#定理-341-adagrad的收敛性)
      - [算法 3.4.2 (RMSprop)](#算法-342-rmsprop)
      - [算法 3.4.3 (Adam)](#算法-343-adam)
      - [定理 3.4.2 (Adam的收敛性分析)](#定理-342-adam的收敛性分析)
      - [定理 3.4.3 (Adam的泛化问题)](#定理-343-adam的泛化问题)
    - [3.5 二阶优化方法](#35-二阶优化方法)
      - [算法 3.5.1 (牛顿法)](#算法-351-牛顿法)
      - [定理 3.5.1 (牛顿法的局部二次收敛)](#定理-351-牛顿法的局部二次收敛)
      - [算法 3.5.2 (拟牛顿法 - L-BFGS)](#算法-352-拟牛顿法---l-bfgs)
      - [算法 3.5.3 (自然梯度)](#算法-353-自然梯度)
      - [定理 3.5.2 (自然梯度的不变性)](#定理-352-自然梯度的不变性)
  - [4. 深度学习理论](#4-深度学习理论)
    - [4.1 神经网络的表达能力](#41-神经网络的表达能力)
      - [定义 4.1.1 (前馈神经网络)](#定义-411-前馈神经网络)
      - [定义 4.1.2 (激活函数)](#定义-412-激活函数)
      - [定理 4.1.1 (通用近似定理 - Cybenko 1989)](#定理-411-通用近似定理---cybenko-1989)
      - [定理 4.1.2 (Barron的近似界)](#定理-412-barron的近似界)
      - [定理 4.1.3 (深度分离 - Telgarsky 2015, 2016)](#定理-413-深度分离---telgarsky-2015-2016)
    - [4.2 深度vs宽度的权衡](#42-深度vs宽度的权衡)
      - [定义 4.2.1 (网络复杂度度量)](#定义-421-网络复杂度度量)
      - [定理 4.2.1 (深度效率 - Eldan \& Shamir 2016)](#定理-421-深度效率---eldan--shamir-2016)
      - [定理 4.2.2 (宽度的表达能力)](#定理-422-宽度的表达能力)
      - [定理 4.2.3 (最优深度-宽度权衡)](#定理-423-最优深度-宽度权衡)
    - [4.3 过参数化与隐式正则化](#43-过参数化与隐式正则化)
      - [定义 4.3.1 (过参数化)](#定义-431-过参数化)
      - [定理 4.3.1 (过参数化神经网络的插值)](#定理-431-过参数化神经网络的插值)
      - [定义 4.3.2 (隐式正则化)](#定义-432-隐式正则化)
      - [定理 4.3.2 (线性模型的隐式正则化)](#定理-432-线性模型的隐式正则化)
      - [定理 4.3.3 (矩阵分解的隐式正则化)](#定理-433-矩阵分解的隐式正则化)
      - [猜想 4.3.1 (神经网络的隐式正则化)](#猜想-431-神经网络的隐式正则化)
    - [4.4 神经正切核 (NTK) 理论](#44-神经正切核-ntk-理论)
      - [定义 4.4.1 (神经正切核)](#定义-441-神经正切核)
      - [定义 4.4.2 (无限宽度极限)](#定义-442-无限宽度极限)
      - [定理 4.4.1 (NTK在无限宽度下的恒定性)](#定理-441-ntk在无限宽度下的恒定性)
      - [定理 4.4.2 (NTK regime下的训练动态)](#定理-442-ntk-regime下的训练动态)
      - [定理 4.4.3 (NTK的泛化界)](#定理-443-ntk的泛化界)
      - [定理 4.4.4 (NTK的显式形式)](#定理-444-ntk的显式形式)
    - [4.5 双下降现象 (Double Descent)](#45-双下降现象-double-descent)
      - [定义 4.5.1 (经典U型风险曲线)](#定义-451-经典u型风险曲线)
      - [定义 4.5.2 (双下降曲线)](#定义-452-双下降曲线)
      - [定理 4.5.1 (线性模型的双下降)](#定理-451-线性模型的双下降)
      - [定理 4.5.2 (随机特征模型的双下降)](#定理-452-随机特征模型的双下降)
      - [定理 4.5.3 (神经网络的双下降)](#定理-453-神经网络的双下降)
      - [定理 4.5.4 (样本-wise双下降)](#定理-454-样本-wise双下降)
  - [5. 泛化理论](#5-泛化理论)
    - [5.1 泛化误差上界推导](#51-泛化误差上界推导)
      - [定义 5.1.1 (泛化间隙)](#定义-511-泛化间隙)
      - [定理 5.1.1 (一致稳定性泛化界)](#定理-511-一致稳定性泛化界)
      - [定理 5.1.2 (SGD的稳定性)](#定理-512-sgd的稳定性)
      - [定理 5.1.3 (神经网络的泛化界 - NTK)](#定理-513-神经网络的泛化界---ntk)
      - [定理 5.1.4 (PAC-Bayes泛化界)](#定理-514-pac-bayes泛化界)
    - [5.2 正则化技术的理论分析](#52-正则化技术的理论分析)
      - [定义 5.2.1 ($L\_2$正则化 / 权重衰减)](#定义-521-l_2正则化--权重衰减)
      - [定理 5.2.1 ($L\_2$正则化的泛化效应)](#定理-521-l_2正则化的泛化效应)
      - [定义 5.2.2 ($L\_1$正则化 / Lasso)](#定义-522-l_1正则化--lasso)
      - [定理 5.2.2 (Lasso的稀疏性)](#定理-522-lasso的稀疏性)
      - [定理 5.2.3 (早停的正则化效应)](#定理-523-早停的正则化效应)
    - [5.3 Dropout的贝叶斯解释](#53-dropout的贝叶斯解释)
      - [定义 5.3.1 (Dropout)](#定义-531-dropout)
      - [定理 5.3.1 (Dropout作为自适应正则化)](#定理-531-dropout作为自适应正则化)
      - [定理 5.3.2 (Dropout的贝叶斯解释)](#定理-532-dropout的贝叶斯解释)
    - [5.4 数据增强的理论效应](#54-数据增强的理论效应)
      - [定义 5.4.1 (数据增强)](#定义-541-数据增强)
      - [定理 5.4.1 (数据增强的隐式正则化)](#定理-541-数据增强的隐式正则化)
      - [定理 5.4.2 (不变性学习)](#定理-542-不变性学习)
      - [定理 5.4.3 (Mixup的理论分析)](#定理-543-mixup的理论分析)
      - [定理 5.4.4 (数据增强的泛化界)](#定理-544-数据增强的泛化界)
  - [附录：核心定理汇总](#附录核心定理汇总)
    - [A.1 集中不等式](#a1-集中不等式)
    - [A.2 优化收敛率总结](#a2-优化收敛率总结)
    - [A.3 泛化界总结](#a3-泛化界总结)
  - [参考文献](#参考文献)

---

## 1. 核心概念的形式化定义

### 1.1 学习问题的形式化框架

#### 定义 1.1.1 (监督学习问题)

一个**监督学习问题**由以下五元组定义：

$$\mathcal{L} = (\mathcal{X}, \mathcal{Y}, \mathcal{D}, \mathcal{H}, \ell)$$

其中：

- **输入空间** $\mathcal{X} \subseteq \mathbb{R}^d$：特征向量的集合
- **输出空间** $\mathcal{Y}$：标签空间（分类问题中 $\mathcal{Y} = \{0, 1, \ldots, K-1\}$，回归问题中 $\mathcal{Y} \subseteq \mathbb{R}$）
- **数据分布** $\mathcal{D}$：定义在 $\mathcal{X} \times \mathcal{Y}$ 上的联合概率分布
- **假设空间** $\mathcal{H} = \{h: \mathcal{X} \to \mathcal{Y}\}$：从输入到输出的映射函数集合
- **损失函数** $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$：衡量预测误差的函数

#### 定义 1.1.2 (数据生成过程)

训练集 $S = \{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}$ 中的样本独立同分布(i.i.d.)地从 $\mathcal{D}$ 中采样：

$$(x_i, y_i) \stackrel{\text{i.i.d.}}{\sim} \mathcal{D}, \quad i = 1, 2, \ldots, n$$

#### 定义 1.1.3 (泛化误差 / 期望风险)

假设 $h \in \mathcal{H}$ 的**泛化误差**（或期望风险）定义为：

$$R(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(h(x), y)] = \int_{\mathcal{X} \times \mathcal{Y}} \ell(h(x), y) \, d\mathcal{D}(x, y)$$

#### 定义 1.1.4 (经验风险)

基于训练集 $S$ 的**经验风险**定义为：

$$\hat{R}_S(h) = \frac{1}{n} \sum_{i=1}^{n} \ell(h(x_i), y_i)$$

#### 定义 1.1.5 (贝叶斯最优风险)

**贝叶斯最优风险**是所有可测函数中的最小风险：

$$R^* = \inf_{h \text{ 可测}} R(h)$$

达到此风险的函数 $h^*$ 称为**贝叶斯最优分类器**。

---

### 1.2 损失函数的数学性质

#### 定义 1.2.1 (凸函数)

函数 $f: \mathbb{R}^d \to \mathbb{R}$ 是**凸函数**，如果对于所有 $x, y \in \mathbb{R}^d$ 和 $\lambda \in [0, 1]$：

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

等价地，若 $f$ 二阶可微，则 $f$ 凸当且仅当Hessian矩阵半正定：

$$\nabla^2 f(x) \succeq 0, \quad \forall x \in \mathbb{R}^d$$

#### 定义 1.2.2 ($L$-Lipschitz连续性)

函数 $f: \mathbb{R}^d \to \mathbb{R}$ 是**$L$-Lipschitz连续**的，如果：

$$|f(x) - f(y)| \leq L \|x - y\|, \quad \forall x, y \in \mathbb{R}^d$$

若 $f$ 可微，则等价于：

$$\|\nabla f(x)\| \leq L, \quad \forall x \in \mathbb{R}^d$$

#### 定义 1.2.3 ($\beta$-平滑性 / Lipschitz梯度)

函数 $f: \mathbb{R}^d \to \mathbb{R}$ 是**$\beta$-平滑**的（具有$\beta$-Lipschitz梯度），如果：

$$\|\nabla f(x) - \nabla f(y)\| \leq \beta \|x - y\|, \quad \forall x, y \in \mathbb{R}^d$$

等价地，对于凸函数 $f$：

$$f(y) \leq f(x) + \nabla f(x)^\top(y-x) + \frac{\beta}{2}\|y-x\|^2$$

#### 定义 1.2.4 ($\mu$-强凸性)

函数 $f: \mathbb{R}^d \to \mathbb{R}$ 是**$\mu$-强凸**的，如果：

$$f(y) \geq f(x) + \nabla f(x)^\top(y-x) + \frac{\mu}{2}\|y-x\|^2, \quad \forall x, y \in \mathbb{R}^d$$

等价地，若 $f$ 二阶可微：

$$\nabla^2 f(x) \succeq \mu I, \quad \forall x \in \mathbb{R}^d$$

**条件数**：$\kappa = \frac{\beta}{\mu}$ 衡量函数的"良态"程度。

---

### 1.3 常用损失函数的形式化分析

#### 定义 1.3.1 (0-1损失)

对于二分类问题：

$$\ell_{0-1}(y, \hat{y}) = \mathbb{1}[y \neq \hat{y}] = \begin{cases} 1 & \text{if } y \neq \hat{y} \\ 0 & \text{if } y = \hat{y} \end{cases}$$

**性质**：

- 非凸、不连续
- 直接优化是NP-hard问题

#### 定义 1.3.2 (平方损失 / $L_2$损失)

$$\ell_{sq}(y, \hat{y}) = (y - \hat{y})^2$$

**性质**：

- 凸、$\beta$-平滑（$\beta = 2$）
- 对异常值敏感

#### 定义 1.3.3 (绝对值损失 / $L_1$损失)

$$\ell_{abs}(y, \hat{y}) = |y - \hat{y}|$$

**性质**：

- 凸、1-Lipschitz
- 非平滑（在0点不可微）
- 对异常值更鲁棒

#### 定义 1.3.4 (Hinge损失 / SVM损失)

$$\ell_{hinge}(y, f(x)) = \max(0, 1 - y \cdot f(x))$$

其中 $y \in \{-1, +1\}$。

**性质**：

- 凸、1-Lipschitz
- 非平滑（在 $y \cdot f(x) = 1$ 处）

#### 定义 1.3.5 (Logistic损失 / 交叉熵损失)

$$\ell_{log}(y, f(x)) = \log(1 + e^{-y \cdot f(x)})$$

**性质**：

- 凸、平滑
- 是0-1损失的凸上界

#### 定理 1.3.1 (损失函数的比较)

对于 $y \cdot f(x) \geq 0$：

$$\ell_{0-1}(y, f(x)) \leq \ell_{hinge}(y, f(x)) \leq \ell_{log}(y, f(x))$$

**证明**：

当 $y \cdot f(x) \geq 1$ 时，所有损失均为0或接近0。

当 $0 \leq y \cdot f(x) < 1$ 时：

- $\ell_{0-1} = 0$（若 $y \cdot f(x) > 0$）或 $1$（若 $y \cdot f(x) = 0$）
- $\ell_{hinge} = 1 - y \cdot f(x) \in (0, 1]$
- $\ell_{log} = \log(1 + e^{-y \cdot f(x)}) \geq 1 - y \cdot f(x)$（由 $\log(1+e^{-z}) \geq 1-z$ 对 $z \in [0,1]$）

因此不等式成立。$\square$

---

## 2. 学习理论基础

### 2.1 PAC学习框架

#### 定义 2.1.1 (PAC可学习性)

假设空间 $\mathcal{H}$ 是**PAC可学习**的，如果存在算法 $\mathcal{A}$ 和多项式函数 $poly(\cdot, \cdot, \cdot)$，使得对于任意：

- 精度参数 $\epsilon > 0$
- 置信参数 $\delta > 0$
- 分布 $\mathcal{D}$ 在 $\mathcal{X} \times \mathcal{Y}$ 上

当样本数 $n \geq poly(1/\epsilon, 1/\delta, d)$ 时，算法 $\mathcal{A}$ 输出假设 $h_S$ 满足：

$$\mathbb{P}_{S \sim \mathcal{D}^n}\left[R(h_S) \leq R^* + \epsilon\right] \geq 1 - \delta$$

若 $R^* = 0$（可实现情形），称为**强PAC可学习**。

#### 定义 2.1.2 (样本复杂度)

**样本复杂度** $n_{\mathcal{H}}(\epsilon, \delta)$ 是满足PAC条件的最小样本数。

#### 定理 2.1.1 (有限假设空间的PAC界)

若 $|\mathcal{H}| < \infty$ 且损失函数有界 $\ell \in [0, 1]$，则对于任意 $\epsilon, \delta > 0$，当：

$$n \geq \frac{\log|\mathcal{H}| + \log(1/\delta)}{2\epsilon^2}$$

时，以至少 $1-\delta$ 的概率：

$$R(\hat{h}_S) \leq \min_{h \in \mathcal{H}} R(h) + \epsilon$$

其中 $\hat{h}_S = \arg\min_{h \in \mathcal{H}} \hat{R}_S(h)$ 是经验风险最小化(ERM)解。

**证明**：

**步骤1**：一致收敛

我们需要证明经验风险一致收敛于期望风险：

$$\mathbb{P}\left[\sup_{h \in \mathcal{H}} |\hat{R}_S(h) - R(h)| > \epsilon\right] \leq \delta$$

**步骤2**：联合界

$$\mathbb{P}\left[\sup_{h \in \mathcal{H}} |\hat{R}_S(h) - R(h)| > \epsilon\right] \leq \sum_{h \in \mathcal{H}} \mathbb{P}\left[|\hat{R}_S(h) - R(h)| > \epsilon\right]$$

**步骤3**：Hoeffding不等式

对于固定 $h$，$\hat{R}_S(h) = \frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i)$ 是 $n$ 个i.i.d.有界随机变量的平均。

由Hoeffding不等式：

$$\mathbb{P}\left[|\hat{R}_S(h) - R(h)| > \epsilon\right] \leq 2e^{-2n\epsilon^2}$$

**步骤4**：综合

$$\mathbb{P}\left[\sup_{h \in \mathcal{H}} |\hat{R}_S(h) - R(h)| > \epsilon\right] \leq 2|\mathcal{H}|e^{-2n\epsilon^2}$$

令此 $\leq \delta$，解得：

$$n \geq \frac{\log|\mathcal{H}| + \log(2/\delta)}{2\epsilon^2}$$

**步骤5**：ERM的性能

设 $h^* = \arg\min_{h \in \mathcal{H}} R(h)$，则：

$$
\begin{aligned}
R(\hat{h}_S) - R(h^*) &= [R(\hat{h}_S) - \hat{R}_S(\hat{h}_S)] + [\hat{R}_S(\hat{h}_S) - \hat{R}_S(h^*)] + [\hat{R}_S(h^*) - R(h^*)] \\
&\leq [R(\hat{h}_S) - \hat{R}_S(\hat{h}_S)] + [\hat{R}_S(h^*) - R(h^*)] \quad \text{(由ERM定义)} \\
&\leq 2\sup_{h \in \mathcal{H}} |\hat{R}_S(h) - R(h)|
\end{aligned}
$$

因此，当一致收敛误差 $\leq \epsilon/2$ 时，$R(\hat{h}_S) - R(h^*) \leq \epsilon$。$\square$

---

### 2.2 VC维理论

#### 定义 2.2.1 (打散 / Shattering)

假设空间 $\mathcal{H}$ **打散**点集 $C = \{x_1, \ldots, x_m\} \subseteq \mathcal{X}$，如果对于所有 $2^m$ 种标签赋值 $(y_1, \ldots, y_m) \in \{0, 1\}^m$，存在 $h \in \mathcal{H}$ 使得：

$$h(x_i) = y_i, \quad \forall i = 1, \ldots, m$$

#### 定义 2.2.2 (VC维)

**VC维** $d_{VC}(\mathcal{H})$ 是 $\mathcal{H}$ 能打散的最大点集的大小：

$$d_{VC}(\mathcal{H}) = \max\{m : \exists C \subseteq \mathcal{X}, |C| = m, \mathcal{H} \text{ 打散 } C\}$$

若 $\mathcal{H}$ 能打散任意大的点集，则 $d_{VC}(\mathcal{H}) = \infty$。

#### 引理 2.2.1 (Sauer-Shelah引理)

若 $d_{VC}(\mathcal{H}) = d$，则对于任意 $m \geq d$：

$$\Pi_{\mathcal{H}}(m) \leq \sum_{i=0}^{d} \binom{m}{i} \leq \left(\frac{em}{d}\right)^d$$

其中 $\Pi_{\mathcal{H}}(m) = \max_{|C|=m} |\{(h(x_1), \ldots, h(x_m)) : h \in \mathcal{H}\}|$ 是增长函数。

#### 定理 2.2.1 (VC泛化界)

对于假设空间 $\mathcal{H}$ 满足 $d_{VC}(\mathcal{H}) = d$，以至少 $1-\delta$ 的概率：

$$R(h) \leq \hat{R}_S(h) + O\left(\sqrt{\frac{d \log(n/d) + \log(1/\delta)}{n}}\right)$$

对所有 $h \in \mathcal{H}$ 一致成立。

**证明概要**：

**步骤1**：对称化

$$
\mathbb{P}\left[\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_S(h)| > \epsilon\right] \leq 2\mathbb{P}\left[\sup_{h \in \mathcal{H}} |\hat{R}_S(h) - \hat{R}_{S'}(h)| > \epsilon/2\right]
$$

其中 $S'$ 是"幽灵样本"。

**步骤2**：条件于样本的复杂度

对于固定 $S \cup S'$（$2n$ 个点），$\mathcal{H}$ 最多产生 $\Pi_{\mathcal{H}}(2n)$ 种不同的分类。

**步骤3**：联合界与Hoeffding

$$
\mathbb{P}\left[\sup_{h \in \mathcal{H}} |\hat{R}_S(h) - \hat{R}_{S'}(h)| > \epsilon/2 \bigg| S \cup S'\right] \leq \Pi_{\mathcal{H}}(2n) \cdot 2e^{-n\epsilon^2/8}
$$

**步骤4**：应用Sauer-Shelah引理

$$\Pi_{\mathcal{H}}(2n) \leq \left(\frac{2en}{d}\right)^d$$

**步骤5**：综合并解 $\epsilon$

$$2\left(\frac{2en}{d}\right)^d e^{-n\epsilon^2/8} \leq \delta$$

取对数并解得：

$$\epsilon = O\left(\sqrt{\frac{d\log(n/d) + \log(1/\delta)}{n}}\right)$$

$\square$

#### 例子 2.2.1 (常见假设空间的VC维)

| 假设空间 | VC维 |
|---------|------|
| $\mathbb{R}^d$ 中的线性分类器 | $d+1$ |
| $\mathbb{R}^d$ 中的齐次线性分类器 | $d$ |
| 轴对齐矩形 | $2d$ |
| $k$ 项单调合取 | $k$ |
| 深度为 $k$ 的决策树 | $O(2^k)$ |
| 神经网络（待讨论） | 复杂依赖结构 |

---

### 2.3 Rademacher复杂度

#### 定义 2.3.1 (经验Rademacher复杂度)

给定样本 $S = (x_1, \ldots, x_n)$，函数类 $\mathcal{F}$ 的**经验Rademacher复杂度**定义为：

$$\hat{\mathfrak{R}}_S(\mathcal{F}) = \mathbb{E}_{\sigma}\left[\sup_{f \in \mathcal{F}} \frac{1}{n}\sum_{i=1}^{n} \sigma_i f(x_i)\right]$$

其中 $\sigma = (\sigma_1, \ldots, \sigma_n)$ 是i.i.d. Rademacher随机变量（$\mathbb{P}[\sigma_i = +1] = \mathbb{P}[\sigma_i = -1] = 1/2$）。

#### 定义 2.3.2 (Rademacher复杂度)

**Rademacher复杂度**是经验复杂度的期望：

$$\mathfrak{R}_n(\mathcal{F}) = \mathbb{E}_S[\hat{\mathfrak{R}}_S(\mathcal{F})]$$

#### 定理 2.3.1 (基于Rademacher复杂度的泛化界)

以至少 $1-\delta$ 的概率，对所有 $f \in \mathcal{F}$：

$$\mathbb{E}[f(x)] \leq \frac{1}{n}\sum_{i=1}^{n} f(x_i) + 2\mathfrak{R}_n(\mathcal{F}) + O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right)$$

**证明概要**：

**步骤1**：McDiarmid不等式

函数 $\Phi(S) = \sup_{f \in \mathcal{F}} \left|\mathbb{E}[f] - \frac{1}{n}\sum_{i=1}^n f(x_i)\right|$ 满足有界差分条件（假设 $f \in [0, 1]$）：

$$|\Phi(S) - \Phi(S')| \leq \frac{1}{n}$$

由McDiarmid不等式：

$$\mathbb{P}[\Phi(S) - \mathbb{E}[\Phi(S)] > t] \leq e^{-2nt^2}$$

**步骤2**：期望的上界

$$\mathbb{E}[\Phi(S)] \leq 2\mathfrak{R}_n(\mathcal{F})$$

这通过对称化论证得到。

**步骤3**：综合

令 $t = \sqrt{\frac{\log(1/\delta)}{2n}}$，则：

$$\Phi(S) \leq 2\mathfrak{R}_n(\mathcal{F}) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

以至少 $1-\delta$ 的概率成立。$\square$

#### 定理 2.3.2 (Rademacher复杂度的性质)

设 $\mathcal{F}, \mathcal{G}$ 是函数类，$c \in \mathbb{R}$，$\phi$ 是 $L$-Lipschitz函数：

1. **缩放**：$\mathfrak{R}_n(c\mathcal{F}) = |c| \cdot \mathfrak{R}_n(\mathcal{F})$

2. **求和**：$\mathfrak{R}_n(\mathcal{F} + \mathcal{G}) \leq \mathfrak{R}_n(\mathcal{F}) + \mathfrak{R}_n(\mathcal{G})$

3. **Lipschitz复合**（Ledoux-Talagrand收缩）：
   $$\mathfrak{R}_n(\phi \circ \mathcal{F}) \leq L \cdot \mathfrak{R}_n(\mathcal{F})$$

4. **与VC维的关系**：对于二分类，$\mathfrak{R}_n(\mathcal{H}) \leq O\left(\sqrt{\frac{d_{VC}(\mathcal{H})}{n}}\right)$

#### 定理 2.3.3 (线性函数的Rademacher复杂度)

设 $\mathcal{F} = \{x \mapsto w^\top x : \|w\|_2 \leq W, \|x\|_2 \leq X\}$，则：

$$\mathfrak{R}_n(\mathcal{F}) \leq \frac{WX}{\sqrt{n}}$$

**证明**：

$$
\begin{aligned}
\hat{\mathfrak{R}}_S(\mathcal{F}) &= \mathbb{E}_{\sigma}\left[\sup_{\|w\|_2 \leq W} \frac{1}{n}\sum_{i=1}^{n} \sigma_i w^\top x_i\right] \\
&= \mathbb{E}_{\sigma}\left[\sup_{\|w\|_2 \leq W} w^\top \left(\frac{1}{n}\sum_{i=1}^{n} \sigma_i x_i\right)\right] \\
&= \frac{W}{n} \mathbb{E}_{\sigma}\left[\left\|\sum_{i=1}^{n} \sigma_i x_i\right\|_2\right] \\
&\leq \frac{W}{n} \sqrt{\mathbb{E}_{\sigma}\left[\left\|\sum_{i=1}^{n} \sigma_i x_i\right\|_2^2\right]} \quad \text{(Jensen不等式)} \\
&= \frac{W}{n} \sqrt{\sum_{i=1}^{n} \|x_i\|_2^2} \quad \text{(由 } \mathbb{E}[\sigma_i \sigma_j] = \delta_{ij}\text{)} \\
&\leq \frac{WX}{\sqrt{n}}
\end{aligned}
$$

$\square$

---

### 2.4 覆盖数与度量熵

#### 定义 2.4.1 ($\epsilon$-覆盖)

设 $(\mathcal{F}, d)$ 是度量空间。子集 $C \subseteq \mathcal{F}$ 是**$\epsilon$-覆盖**，如果对于所有 $f \in \mathcal{F}$，存在 $c \in C$ 使得 $d(f, c) \leq \epsilon$。

#### 定义 2.4.2 (覆盖数)

**$\epsilon$-覆盖数** $N(\mathcal{F}, d, \epsilon)$ 是最小$\epsilon$-覆盖的大小。

#### 定义 2.4.3 (度量熵)

**度量熵**定义为：

$$\mathcal{H}(\mathcal{F}, d, \epsilon) = \log N(\mathcal{F}, d, $\epsilon$)$$

#### 定义 2.4.4 ($L_2(P_n)$覆盖数)

对于样本 $S = (x_1, \ldots, x_n)$，定义经验 $L_2$ 度量：

$$d_{P_n}(f, g) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (f(x_i) - g(x_i))^2}$$

#### 定理 2.4.1 (Dudley熵积分)

$$
\mathfrak{R}_n(\mathcal{F}) \leq \inf_{\alpha > 0} \left\{4\alpha + 12\int_{\alpha}^{\sup_{f \in \mathcal{F}} \|f\|_{\infty}} \sqrt{\frac{\log N(\mathcal{F}, d_{P_n}, \epsilon)}{n}} d\epsilon\right\}
$$

#### 定理 2.4.2 (基于覆盖数的泛化界)

设 $\mathcal{F}$ 是函数类，$f \in [0, 1]$ 对所有 $f \in \mathcal{F}$。以至少 $1-\delta$ 的概率：

$$
\mathbb{E}[f(x)] \leq \frac{1}{n}\sum_{i=1}^{n} f(x_i) + O\left(\sqrt{\frac{\log N(\mathcal{F}, d_{P_n}, \epsilon) + \log(1/\delta)}{n}} + \epsilon\right)
$$

---



## 3. 优化理论

### 3.1 凸优化基础

#### 定义 3.1.1 (凸优化问题)

**凸优化问题**具有形式：

$$\min_{x \in \mathcal{C}} f(x)$$

其中：

- $f: \mathbb{R}^d \to \mathbb{R}$ 是凸函数
- $\mathcal{C} \subseteq \mathbb{R}^d$ 是凸集

#### 定义 3.1.2 (凸集)

集合 $\mathcal{C}$ 是**凸集**，如果对于所有 $x, y \in \mathcal{C}$ 和 $\lambda \in [0, 1]$：

$$\lambda x + (1-\lambda)y \in \mathcal{C}$$

#### 定理 3.1.1 (凸优化的一阶最优性条件)

设 $f$ 是凸且可微，$\mathcal{C}$ 是凸集。$x^* \in \mathcal{C}$ 是最优解当且仅当：

$$\nabla f(x^*)^\top (x - x^*) \geq 0, \quad \forall x \in \mathcal{C}$$

若 $\mathcal{C} = \mathbb{R}^d$（无约束），则最优性条件简化为：

$$\nabla f(x^*) = 0$$

#### 定理 3.1.2 (强凸函数的唯一最优性)

若 $f$ 是 $\mu$-强凸，则存在唯一全局最优解 $x^*$，且：

$$f(x) - f(x^*) \geq \frac{\mu}{2}\|x - x^*\|^2, \quad \forall x$$

---

### 3.2 梯度下降收敛性

#### 算法 3.2.1 (梯度下降 / GD)

输入：初始点 $x_0$，步长 $\eta$，迭代次数 $T$

对于 $t = 0, 1, \ldots, T-1$：
$$x_{t+1} = x_t - \eta \nabla f(x_t)$$

输出：$x_T$ 或 $\bar{x} = \frac{1}{T}\sum_{t=0}^{T-1} x_t$

#### 定理 3.2.1 (凸函数的GD收敛性)

设 $f$ 是凸、$\beta$-平滑，且最优解 $x^*$ 满足 $\|x^*\| \leq D$。选择 $\eta = \frac{1}{\beta}$，则：

$$f\left(\frac{1}{T}\sum_{t=0}^{T-1} x_t\right) - f(x^*) \leq \frac{\beta D^2}{2T}$$

即达到 $\epsilon$-最优需要 $T = O\left(\frac{\beta D^2}{\epsilon}\right)$ 次迭代。

**证明**：

**步骤1**：平滑性的二次上界

由 $\beta$-平滑性：

$$f(x_{t+1}) \leq f(x_t) + \nabla f(x_t)^\top(x_{t+1} - x_t) + \frac{\beta}{2}\|x_{t+1} - x_t\|^2$$

代入 $x_{t+1} - x_t = -\frac{1}{\beta}\nabla f(x_t)$：

$$f(x_{t+1}) \leq f(x_t) - \frac{1}{2\beta}\|\nabla f(x_t)\|^2$$

**步骤2**：凸性条件

由凸性：

$$f(x_t) \leq f(x^*) + \nabla f(x_t)^\top(x_t - x^*)$$

**步骤3**：距离递推

$$
\begin{aligned}
\|x_{t+1} - x^*\|^2 &= \|x_t - \eta\nabla f(x_t) - x^*\|^2 \\
&= \|x_t - x^*\|^2 - 2\eta\nabla f(x_t)^\top(x_t - x^*) + \eta^2\|\nabla f(x_t)\|^2 \\
&\leq \|x_t - x^*\|^2 - 2\eta(f(x_t) - f(x^*)) + \eta^2\|\nabla f(x_t)\|^2
\end{aligned}
$$

**步骤4**：综合并累加

取 $\eta = 1/\beta$，利用步骤1的结果：

$$f(x_t) - f(x_{t+1}) \geq \frac{1}{2\beta}\|\nabla f(x_t)\|^2$$

累加 $t = 0$ 到 $T-1$：

$$\sum_{t=0}^{T-1} (f(x_t) - f(x^*)) \leq \frac{\beta}{2}\|x_0 - x^*\|^2 \leq \frac{\beta D^2}{2}$$

由Jensen不等式：

$$f\left(\frac{1}{T}\sum_{t=0}^{T-1} x_t\right) - f(x^*) \leq \frac{1}{T}\sum_{t=0}^{T-1} (f(x_t) - f(x^*)) \leq \frac{\beta D^2}{2T}$$

$\square$

#### 定理 3.2.2 (强凸函数的GD收敛性)

设 $f$ 是 $\mu$-强凸、$\beta$-平滑（条件数 $\kappa = \beta/\mu$）。选择 $\eta = \frac{1}{\beta}$，则：

$$\|x_T - x^*\|^2 \leq \left(1 - \frac{1}{\kappa}\right)^T \|x_0 - x^*\|^2$$

或等价地：

$$f(x_T) - f(x^*) \leq \frac{\beta}{2}\left(1 - \frac{1}{\kappa}\right)^T \|x_0 - x^*\|^2$$

达到 $\epsilon$-最优需要 $T = O\left(\kappa \log\frac{1}{\epsilon}\right)$ 次迭代（线性收敛）。

**证明**：

**步骤1**：强凸性-平滑性联合不等式

对于 $\mu$-强凸、$\beta$-平滑函数：

$$(\nabla f(x) - \nabla f(y))^\top(x - y) \geq \frac{\mu\beta}{\mu + \beta}\|x - y\|^2 + \frac{1}{\mu + \beta}\|\nabla f(x) - \nabla f(y)\|^2$$

**步骤2**：距离收缩

$$
\begin{aligned}
\|x_{t+1} - x^*\|^2 &= \|x_t - \eta\nabla f(x_t) - x^*\|^2 \\
&= \|x_t - x^*\|^2 - 2\eta\nabla f(x_t)^\top(x_t - x^*) + \eta^2\|\nabla f(x_t)\|^2 \\
&\leq \|x_t - x^*\|^2 - 2\eta\left(\mu\|x_t - x^*\|^2 + \frac{1}{\beta}\|\nabla f(x_t)\|^2\right) + \eta^2\|\nabla f(x_t)\|^2
\end{aligned}
$$

取 $\eta = 1/\beta$：

$$\|x_{t+1} - x^*\|^2 \leq \left(1 - \frac{\mu}{\beta}\right)\|x_t - x^*\|^2 = \left(1 - \frac{1}{\kappa}\right)\|x_t - x^*\|^2$$

**步骤3**：递推

$$\|x_T - x^*\|^2 \leq \left(1 - \frac{1}{\kappa}\right)^T \|x_0 - x^*\|^2$$

**步骤4**：函数值收敛

由平滑性：

$$f(x_T) - f(x^*) \leq \frac{\beta}{2}\|x_T - x^*\|^2 \leq \frac{\beta}{2}\left(1 - \frac{1}{\kappa}\right)^T \|x_0 - x^*\|^2$$

$\square$

#### 定理 3.2.3 (非凸函数的GD收敛性)

设 $f$ 是 $\beta$-平滑（不必凸），$\eta = 1/\beta$。则：

$$\min_{0 \leq t < T} \|\nabla f(x_t)\|^2 \leq \frac{2\beta(f(x_0) - f(x^*))}{T}$$

达到 $\|\nabla f(x)\| \leq \epsilon$ 需要 $T = O(1/\epsilon^2)$ 次迭代。

---

### 3.3 随机梯度下降 (SGD)

#### 定义 3.3.1 (随机梯度)

对于经验风险 $\hat{R}(w) = \frac{1}{n}\sum_{i=1}^n \ell(w; z_i)$，**随机梯度**为：

$$g_t = \nabla \ell(w_t; z_{i_t})$$

其中 $i_t \sim \text{Uniform}(\{1, \ldots, n\})$。

#### 算法 3.3.1 (SGD)

输入：初始点 $w_0$，步长序列 $\{\eta_t\}$

对于 $t = 0, 1, \ldots, T-1$：

1. 随机采样 $i_t \sim \text{Uniform}(\{1, \ldots, n\})$
2. $w_{t+1} = w_t - \eta_t \nabla \ell(w_t; z_{i_t})$

输出：$\bar{w}_T = \frac{1}{T}\sum_{t=0}^{T-1} w_t$ 或最后迭代 $w_T$

#### 假设 3.3.1 (SGD的标准假设)

1. **无偏性**：$\mathbb{E}[g_t | w_t] = \nabla f(w_t)$
2. **有界方差**：$\mathbb{E}[\|g_t - \nabla f(w_t)\|^2 | w_t] \leq \sigma^2$
3. **有界梯度**（可选）：$\|\nabla \ell(w; z)\| \leq G$

#### 定理 3.3.1 (凸函数的SGD收敛性)

设 $f$ 是凸，$\mathbb{E}[\|g_t\|^2] \leq G^2$，$\|w^*\| \leq D$。选择 $\eta_t = \frac{D}{G\sqrt{T}}$，则：

$$\mathbb{E}\left[f\left(\frac{1}{T}\sum_{t=0}^{T-1} w_t\right)\right] - f(w^*) \leq \frac{DG}{\sqrt{T}}$$

达到 $\epsilon$-最优需要 $T = O\left(\frac{D^2G^2}{\epsilon^2}\right)$ 次迭代。

**证明**：

**步骤1**：距离递推

$$
\begin{aligned}
\|w_{t+1} - w^*\|^2 &= \|w_t - \eta_t g_t - w^*\|^2 \\
&= \|w_t - w^*\|^2 - 2\eta_t g_t^\top(w_t - w^*) + \eta_t^2\|g_t\|^2
\end{aligned}
$$

**步骤2**：取条件期望

$$\mathbb{E}[\|w_{t+1} - w^*\|^2 | w_t] = \|w_t - w^*\|^2 - 2\eta_t \nabla f(w_t)^\top(w_t - w^*) + \eta_t^2 \mathbb{E}[\|g_t\|^2 | w_t]$$

由凸性 $\nabla f(w_t)^\top(w_t - w^*) \geq f(w_t) - f(w^*)$：

$$\mathbb{E}[\|w_{t+1} - w^*\|^2 | w_t] \leq \|w_t - w^*\|^2 - 2\eta_t(f(w_t) - f(w^*)) + \eta_t^2 G^2$$

**步骤3**：重排并累加

$$2\eta_t(f(w_t) - f(w^*)) \leq \|w_t - w^*\|^2 - \mathbb{E}[\|w_{t+1} - w^*\|^2 | w_t] + \eta_t^2 G^2$$

取全期望并累加 $t = 0$ 到 $T-1$：

$$2\sum_{t=0}^{T-1} \eta_t \mathbb{E}[f(w_t) - f(w^*)] \leq \|w_0 - w^*\|^2 + G^2\sum_{t=0}^{T-1} \eta_t^2$$

**步骤4**：常数步长

取 $\eta_t = \eta$：

$$\frac{1}{T}\sum_{t=0}^{T-1} \mathbb{E}[f(w_t)] - f(w^*) \leq \frac{D^2}{2\eta T} + \frac{\eta G^2}{2}$$

最优选择 $\eta = \frac{D}{G\sqrt{T}}$：

$$\mathbb{E}\left[f\left(\frac{1}{T}\sum_{t=0}^{T-1} w_t\right)\right] - f(w^*) \leq \frac{DG}{\sqrt{T}}$$

$\square$

#### 定理 3.3.2 (强凸函数的SGD收敛性)

设 $f$ 是 $\mu$-强凸，$\mathbb{E}[\|g_t\|^2] \leq G^2$。选择 $\eta_t = \frac{2}{\mu(t+1)}$，则：

$$\mathbb{E}[f(\bar{w}_T)] - f(w^*) \leq \frac{2G^2}{\mu T}$$

其中 $\bar{w}_T = \frac{2}{T(T+1)}\sum_{t=0}^{T-1} (t+1)w_t$ 是加权平均。

---

### 3.4 自适应优化器

#### 算法 3.4.1 (AdaGrad)

输入：初始点 $w_0$，学习率 $\eta$

初始化：$G_0 = 0$

对于 $t = 0, 1, \ldots, T-1$：

1. 计算随机梯度 $g_t$
2. 累积梯度：$G_{t+1} = G_t + g_t \odot g_t$
3. 更新：$w_{t+1} = w_t - \frac{\eta}{\sqrt{G_{t+1} + \epsilon}} \odot g_t$

#### 定理 3.4.1 (AdaGrad的收敛性)

设 $f$ 是凸，$\|g_t\|_\infty \leq G_\infty$。AdaGrad满足：

$$\sum_{t=0}^{T-1} (f(w_t) - f(w^*)) \leq \frac{1}{2\eta}\|w_0 - w^*\|^2 + \eta \sum_{i=1}^{d} \sqrt{\sum_{t=0}^{T-1} g_{t,i}^2}$$

对于稀疏梯度，AdaGrad可实现 $O(1/\sqrt{T})$ 收敛率，且自适应于几何结构。

#### 算法 3.4.2 (RMSprop)

输入：初始点 $w_0$，学习率 $\eta$，衰减率 $\beta \in (0, 1)$

初始化：$v_0 = 0$

对于 $t = 0, 1, \ldots, T-1$：

1. 计算随机梯度 $g_t$
2. 指数移动平均：$v_{t+1} = \beta v_t + (1-\beta) g_t \odot g_t$
3. 更新：$w_{t+1} = w_t - \frac{\eta}{\sqrt{v_{t+1} + \epsilon}} \odot g_t$

#### 算法 3.4.3 (Adam)

输入：初始点 $w_0$，学习率 $\eta$，衰减率 $\beta_1, \beta_2 \in (0, 1)$

初始化：$m_0 = 0, v_0 = 0$

对于 $t = 0, 1, \ldots, T-1$：

1. 计算随机梯度 $g_t$
2. 一阶矩估计：$m_{t+1} = \beta_1 m_t + (1-\beta_1) g_t$
3. 二阶矩估计：$v_{t+1} = \beta_2 v_t + (1-\beta_2) g_t \odot g_t$
4. 偏差修正：$\hat{m}_{t+1} = \frac{m_{t+1}}{1-\beta_1^{t+1}}$, $\hat{v}_{t+1} = \frac{v_{t+1}}{1-\beta_2^{t+1}}$
5. 更新：$w_{t+1} = w_t - \frac{\eta}{\sqrt{\hat{v}_{t+1}} + \epsilon} \odot \hat{m}_{t+1}$

#### 定理 3.4.2 (Adam的收敛性分析)

Adam的原始论文声称收敛，但后续研究发现：

**Reddi et al. (2018)** 证明Adam在某些凸问题上可能不收敛。

**修正版本 (AMSGrad)**：
$$\hat{v}_{t+1} = \max(\hat{v}_t, v_{t+1})$$

AMSGrad保证收敛：

$$\sum_{t=1}^{T} (f(w_t) - f(w^*)) \leq O(\sqrt{T})$$

#### 定理 3.4.3 (Adam的泛化问题)

**Wilson et al. (2017)** 发现：

- Adam在训练集上收敛更快
- 但泛化性能往往不如带动量的SGD

**可能原因**：

1. 自适应方法引入隐式偏差
2. 尖锐极小值 vs 平坦极小值
3. 解的"锐度"影响泛化

---

### 3.5 二阶优化方法

#### 算法 3.5.1 (牛顿法)

$$w_{t+1} = w_t - [\nabla^2 f(w_t)]^{-1} \nabla f(w_t)$$

#### 定理 3.5.1 (牛顿法的局部二次收敛)

设 $f$ 是 $\mu$-强凸、$\beta$-平滑，且Hessian是 $L$-Lipschitz：

$$\|\nabla^2 f(x) - \nabla^2 f(y)\| \leq L\|x - y\|$$

若 $\|w_0 - w^*\| \leq \frac{\mu}{L}$，则：

$$\|w_{t+1} - w^*\| \leq \frac{L}{2\mu} \|w_t - w^*\|^2$$

即局部二次收敛。

#### 算法 3.5.2 (拟牛顿法 - L-BFGS)

避免显式计算Hessian，通过梯度差分近似：

$$s_t = w_{t+1} - w_t, \quad y_t = \nabla f(w_{t+1}) - \nabla f(w_t)$$

近似Hessian逆满足割线条件：

$$H_{t+1} y_t = s_t$$

#### 算法 3.5.3 (自然梯度)

$$w_{t+1} = w_t - \eta F(w_t)^{-1} \nabla f(w_t)$$

其中 $F(w)$ 是Fisher信息矩阵：

$$F(w) = \mathbb{E}_{x \sim p(x|w)}[\nabla \log p(x|w) \nabla \log p(x|w)^\top]$$

#### 定理 3.5.2 (自然梯度的不变性)

自然梯度在参数重参数化下保持不变，对应于在分布空间中的最速下降。

---



## 4. 深度学习理论

### 4.1 神经网络的表达能力

#### 定义 4.1.1 (前馈神经网络)

一个**$L$层前馈神经网络** $f: \mathbb{R}^d \to \mathbb{R}^k$ 定义为：

$$f(x) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x + b_1) \cdots) + b_{L-1}) + b_L$$

其中：

- $W_l \in \mathbb{R}^{d_l \times d_{l-1}}$ 是权重矩阵
- $b_l \in \mathbb{R}^{d_l}$ 是偏置向量
- $\sigma: \mathbb{R} \to \mathbb{R}$ 是逐元素激活函数
- $d_0 = d$（输入维度），$d_L = k$（输出维度）

#### 定义 4.1.2 (激活函数)

常见激活函数：

**Sigmoid**：$\sigma(z) = \frac{1}{1 + e^{-z}}$

**ReLU**：$\sigma(z) = \max(0, z)$

**Tanh**：$\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

#### 定理 4.1.1 (通用近似定理 - Cybenko 1989)

设 $\sigma$ 是非常数的连续sigmoidal函数（即 $\lim_{z \to -\infty} \sigma(z) = 0$，$\lim_{z \to +\infty} \sigma(z) = 1$）。

对于任意紧集 $K \subset \mathbb{R}^d$，任意连续函数 $f: K \to \mathbb{R}$，和任意 $\epsilon > 0$，存在单隐藏层神经网络 $f_{\theta}$ 使得：

$$\sup_{x \in K} |f(x) - f_{\theta}(x)| < \epsilon$$

**证明概要**：

**步骤1**：函数空间的稠密性

证明形如 $G(x) = \sum_{j=1}^{N} \alpha_j \sigma(w_j^\top x + b_j)$ 的函数在 $C(K)$ 中稠密。

**步骤2**：Hahn-Banach定理

若 $G$ 不稠密，则存在非零有界线性泛函 $L$ 使得 $L(G) = 0$ 对所有 $G$ 成立。

**步骤3**：Riesz表示

$L(f) = \int_K f(x) d\mu(x)$ 对某个有限符号测度 $\mu$。

**步骤4**：矛盾

$\int_K \sigma(w^\top x + b) d\mu(x) = 0$ 对所有 $w, b$ 成立。

通过变量替换和Fourier分析，可推出 $\mu = 0$，矛盾。$\square$

#### 定理 4.1.2 (Barron的近似界)

设 $f: \mathbb{R}^d \to \mathbb{R}$ 满足Fourier条件：

$$C_f = \int_{\mathbb{R}^d} \|\omega\|_2 |\hat{f}(\omega)| d\omega < \infty$$

其中 $\hat{f}$ 是 $f$ 的Fourier变换。则存在单隐藏层神经网络 $f_N$ 有 $N$ 个神经元使得：

$$\int_{\|x\| \leq r} (f(x) - f_N(x))^2 dx \leq O\left(\frac{r^d C_f^2}{N}\right)$$

**关键洞察**：近似误差以 $O(1/N)$ 衰减，与维度 $d$ 无关（通过 $C_f$ 隐式依赖）。

#### 定理 4.1.3 (深度分离 - Telgarsky 2015, 2016)

存在函数 $f: [0, 1]^d \to \mathbb{R}$ 可以被深度为 $O(k)$、宽度为 $O(1)$ 的网络以误差 $\epsilon$ 近似，但浅层网络需要宽度 $\Omega(2^k)$ 才能达到相同精度。

**直观**：某些函数具有层次结构，深度网络可以高效地利用这种结构。

---

### 4.2 深度vs宽度的权衡

#### 定义 4.2.1 (网络复杂度度量)

- **深度**：层数 $L$
- **宽度**：最大层宽度 $W = \max_l d_l$
- **参数量**：$P = \sum_{l=1}^{L} (d_l \cdot d_{l-1} + d_l)$

#### 定理 4.2.1 (深度效率 - Eldan & Shamir 2016)

存在函数 $f: \mathbb{R}^d \to \mathbb{R}$ 可以被三层网络以 $\epsilon$ 误差近似，但任何两层网络需要 $\Omega(e^{\Omega(d)})$ 个神经元才能达到相同精度。

#### 定理 4.2.2 (宽度的表达能力)

对于ReLU网络，宽度 $W$ 和深度 $L$ 的表达能力满足：

- 固定宽度 $W$，增加深度 $L$ 可以逼近任意连续函数（当 $L \to \infty$）
- 固定深度 $L$，增加宽度 $W$ 也可以逼近任意连续函数（当 $W \to \infty$）

#### 定理 4.2.3 (最优深度-宽度权衡)

对于近似 $d$ 维Lipschitz函数到精度 $\epsilon$：

- 浅层网络：需要 $O(\epsilon^{-d})$ 参数（维度灾难）
- 深度网络：在某些结构化函数上，需要 $O(poly(d, \log(1/\epsilon)))$ 参数

---

### 4.3 过参数化与隐式正则化

#### 定义 4.3.1 (过参数化)

神经网络是**过参数化**的，如果参数量 $P$ 远大于样本数 $n$：

$$P \gg n$$

#### 定理 4.3.1 (过参数化神经网络的插值)

对于足够宽的ReLU网络，以高概率存在参数使得网络完美插值训练数据：

$$f_{\theta}(x_i) = y_i, \quad \forall i = 1, \ldots, n$$

#### 定义 4.3.2 (隐式正则化)

**隐式正则化**指优化算法（如GD、SGD）倾向于收敛到具有特定性质的解，即使目标函数中没有显式正则化项。

#### 定理 4.3.2 (线性模型的隐式正则化)

对于线性回归 $\min_w \|Xw - y\|^2$，使用梯度下降从零初始化收敛到最小范数解：

$$w_{GD} = \arg\min_w \|w\|_2 \quad \text{s.t.} \quad Xw = y$$

**证明**：

GD更新：$w_{t+1} = w_t - \eta X^\top(Xw_t - y)$

注意到 $w_t$ 始终在 $X^\top$ 的列空间中：$w_t = X^\top \alpha_t$

在收敛时 $Xw_\infty = y$，且 $w_\infty = X^\top \alpha_\infty$。

最小范数解为 $w^* = X^\top(XX^\top)^{-1}y$（当 $XX^\top$ 可逆）。

可以验证 $w_\infty = w^*$。$\square$

#### 定理 4.3.3 (矩阵分解的隐式正则化)

对于矩阵感知问题，使用梯度下降学习低秩矩阵 $M = UV^\top$ 隐式偏好低秩解。

**Gunasekar et al. (2017)**：对于足够小的初始化，GD收敛到核范数最小化解：

$$\min_M \|M\|_* \quad \text{s.t.} \quad \mathcal{A}(M) = b$$

#### 猜想 4.3.1 (神经网络的隐式正则化)

对于过参数化神经网络，SGD隐式偏好"简单"解，这可能解释良好的泛化性能。

可能的隐式正则化：

- 参数范数小
- 解的"平坦性"
- 低复杂度表示

---

### 4.4 神经正切核 (NTK) 理论

#### 定义 4.4.1 (神经正切核)

考虑参数化为 $f(x; \theta)$ 的神经网络，其中 $\theta \in \mathbb{R}^P$。定义**神经正切核**：

$$\Theta(x, x'; \theta) = \nabla_\theta f(x; \theta)^\top \nabla_\theta f(x'; \theta)$$

#### 定义 4.4.2 (无限宽度极限)

对于宽度为 $m$ 的网络，当 $m \to \infty$，在适当初始化下：

$$\Theta(x, x'; \theta_0) \xrightarrow{m \to \infty} \Theta_{\infty}(x, x')$$

其中 $\Theta_{\infty}$ 是确定性核函数。

#### 定理 4.4.1 (NTK在无限宽度下的恒定性)

对于适当初始化的无限宽度网络，在训练过程中NTK保持不变：

$$\Theta(x, x'; \theta_t) \approx \Theta_{\infty}(x, x'), \quad \forall t$$

#### 定理 4.4.2 (NTK regime下的训练动态)

在NTK regime（无限宽度、小学习率），网络训练等价于核回归：

$$f_t(x) = \Theta_{\infty}(x, X) \Theta_{\infty}(X, X)^{-1}(I - e^{-\eta \Theta_{\infty}(X, X)t})Y$$

其中 $\Theta_{\infty}(X, X)_{ij} = \Theta_{\infty}(x_i, x_j)$。

**证明概要**：

**步骤1**：线性化网络

对于小参数变化，线性近似：

$$f(x; \theta) \approx f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^\top (\theta - \theta_0)$$

**步骤2**：梯度流方程

$$\frac{d\theta_t}{dt} = -\nabla_\theta \hat{R}(\theta_t) = -\frac{1}{n}\sum_{i=1}^{n} (f(x_i; \theta_t) - y_i) \nabla_\theta f(x_i; \theta_t)$$

**步骤3**：NTK恒定性

在无限宽度下，$\nabla_\theta f(x; \theta_t) \approx \nabla_\theta f(x; \theta_0)$，因此：

$$\frac{df_t(x)}{dt} \approx -\frac{1}{n}\Theta_{\infty}(x, X)(f_t(X) - Y)$$

**步骤4**：求解ODE

这是一个线性ODE，解为：

$$f_t(X) = (I - e^{-\eta \Theta_{\infty}(X, X)t/n})Y$$

泛化到任意点 $x$：

$$f_t(x) = \Theta_{\infty}(x, X)\Theta_{\infty}(X, X)^{-1}(I - e^{-\eta \Theta_{\infty}(X, X)t/n})Y$$

$\square$

#### 定理 4.4.3 (NTK的泛化界)

在NTK regime下，训练后的网络满足：

$$R(f_T) \leq \hat{R}(f_T) + O\left(\sqrt{\frac{\text{tr}(\Theta_{\infty}(X, X)^{-1})Y^\top Y + \log(1/\delta)}{n}}\right)$$

#### 定理 4.4.4 (NTK的显式形式)

对于单隐藏层ReLU网络，无限宽度NTK为：

$$\Theta_{\infty}(x, x') = x^\top x' \cdot \kappa_0\left(\frac{x^\top x'}{\|x\|\|x'\|}\right) + \kappa_1\left(\frac{x^\top x'}{\|x\|\|x'\|}\right)$$

其中：

$$\kappa_0(u) = \frac{1}{\pi}(\pi - \arccos(u))$$

$$\kappa_1(u) = \frac{1}{\pi}(u(\pi - \arccos(u)) + \sqrt{1-u^2})$$

---

### 4.5 双下降现象 (Double Descent)

#### 定义 4.5.1 (经典U型风险曲线)

传统统计学习理论预测：

- 欠参数化：模型复杂度 $\uparrow$ $\Rightarrow$ 测试误差 $\downarrow$
- 过拟合：模型复杂度 $\uparrow$ $\Rightarrow$ 测试误差 $\uparrow$

形成U型曲线，最优在"插值阈值"附近。

#### 定义 4.5.2 (双下降曲线)

**Belkin et al. (2019)** 发现：

1. **第一下降**（经典）：模型复杂度增加 $\to$ 测试误差下降
2. **插值阈值**：模型刚好插值训练数据
3. **第二下降**（现代）：超过插值阈值后，测试误差再次下降

形成"双下降"曲线。

#### 定理 4.5.1 (线性模型的双下降)

考虑线性回归 $y = Xw^* + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma^2 I)$。

设 $\hat{w}$ 是最小范数插值解：

$$\hat{w} = \arg\min_w \|w\|_2 \quad \text{s.t.} \quad Xw = y$$

当 $d > n$（过参数化），风险为：

$$\mathbb{E}[\|\hat{w} - w^*\|^2] = \frac{\sigma^2 n}{d - n - 1} + \|w^*\|^2 \frac{d - n}{d}$$

**分析**：

- 当 $d \to n^+$（刚超过插值阈值）：风险发散
- 当 $d \to \infty$：风险 $\to \sigma^2$（噪声水平）

#### 定理 4.5.2 (随机特征模型的双下降)

考虑随机特征模型：

$$f(x) = \frac{1}{\sqrt{N}}\sum_{j=1}^{N} a_j \sigma(w_j^\top x)$$

其中 $w_j \sim \mathcal{N}(0, I)$ 固定，只优化 $a_j$。

**Mei & Montanari (2019)** 证明在高维极限下存在双下降，并给出了渐近风险的闭式表达。

#### 定理 4.5.3 (神经网络的双下降)

对于足够宽的两层神经网络，实验观察到双下降现象。

**理论解释**：

1. **NTK regime**：无限宽度下等价于核方法，风险由核性质决定
2. **特征学习regime**：有限宽度下，网络学习有用特征

#### 定理 4.5.4 (样本-wise双下降)

增加训练样本数也可能导致双下降：

- 小样本：高方差
- 中等样本：可能增加测试误差（对抗性样本）
- 大样本：收敛到贝叶斯最优

---

## 5. 泛化理论

### 5.1 泛化误差上界推导

#### 定义 5.1.1 (泛化间隙)

**泛化间隙**定义为期望风险与经验风险之差：

$$\text{Gen}(h) = R(h) - \hat{R}_S(h)$$

#### 定理 5.1.1 (一致稳定性泛化界)

设算法 $\mathcal{A}$ 是 $\beta$-一致稳定的，即对于相邻训练集 $S$ 和 $S'$（相差一个样本）：

$$\sup_z |\ell(\mathcal{A}(S); z) - \ell(\mathcal{A}(S'); z)| \leq \beta$$

则以至少 $1-\delta$ 的概率：

$$R(\mathcal{A}(S)) \leq \hat{R}_S(\mathcal{A}(S)) + \beta + (2n\beta + M)\sqrt{\frac{\log(1/\delta)}{2n}}$$

其中 $M$ 是损失的上界。

#### 定理 5.1.2 (SGD的稳定性)

对于凸、$L$-Lipschitz、$\beta$-平滑的损失函数，使用学习率 $\eta_t \leq 2/\beta$ 的SGD是均匀稳定的，稳定参数为：

$$\beta_{SGD} \leq \frac{2L^2}{n}\sum_{t=1}^{T} \eta_t$$

对于 $\eta_t = 1/\sqrt{t}$：

$$\beta_{SGD} = O\left(\frac{L^2 \sqrt{T}}{n}\right)$$

#### 定理 5.1.3 (神经网络的泛化界 - NTK)

对于宽度为 $m$ 的两层网络，在NTK regime下，以高概率：

$$R(f) \leq \hat{R}(f) + O\left(\sqrt{\frac{y^\top (\Theta_{\infty}(X,X))^{-1} y}{n}}\right)$$

#### 定理 5.1.4 (PAC-Bayes泛化界)

设 $P$ 是先验分布，$Q$ 是后验分布（可能依赖于数据）。以至少 $1-\delta$ 的概率：

$$\mathbb{E}_{h \sim Q}[R(h)] \leq \mathbb{E}_{h \sim Q}[\hat{R}_S(h)] + \sqrt{\frac{KL(Q\|P) + \log(2\sqrt{n}/\delta)}{2n}}$$

**关键洞察**：泛化界依赖于后验与先验的KL散度，而非参数数量。

---

### 5.2 正则化技术的理论分析

#### 定义 5.2.1 ($L_2$正则化 / 权重衰减)

目标函数：

$$\hat{R}_{reg}(w) = \hat{R}(w) + \frac{\lambda}{2}\|w\|_2^2$$

#### 定理 5.2.1 ($L_2$正则化的泛化效应)

$L_2$正则化限制参数范数，从而限制假设空间的Rademacher复杂度：

$$\mathfrak{R}_n(\{x \mapsto w^\top x : \|w\|_2 \leq W\}) \leq \frac{WX}{\sqrt{n}}$$

#### 定义 5.2.2 ($L_1$正则化 / Lasso)

$$\hat{R}_{reg}(w) = \hat{R}(w) + \lambda\|w\|_1$$

#### 定理 5.2.2 (Lasso的稀疏性)

在高维线性模型 $y = Xw^* + \epsilon$ 中，若 $w^*$ 是 $k$-稀疏（$k$ 个非零元素），Lasso以高概率恢复真实支撑集：

$$\|\hat{w}_{Lasso} - w^*\|_2^2 = O\left(\frac{k \log d}{n}\right)$$

#### 定理 5.2.3 (早停的正则化效应)

对于梯度下降，早停等价于 $L_2$正则化。

**证明概要**：

线性回归中，GD迭代 $t$ 步的解为：

$$w_t = \sum_{j=1}^{d} \frac{1 - (1 - \eta \lambda_j)^t}{\lambda_j} v_j v_j^\top X^\top y$$

其中 $\lambda_j, v_j$ 是 $X^\top X$ 的特征值/向量。

这类似于岭回归：

$$w_{ridge} = \sum_{j=1}^{d} \frac{1}{\lambda_j + \lambda} v_j v_j^\top X^\top y$$

两者都对小特征值方向进行收缩。$\square$

---

### 5.3 Dropout的贝叶斯解释

#### 定义 5.3.1 (Dropout)

训练时，以概率 $p$ 随机将神经元输出置零：

$$\tilde{h} = m \odot h, \quad m_i \sim \text{Bernoulli}(1-p)$$

测试时，使用期望输出：$h_{test} = (1-p)h$。

#### 定理 5.3.1 (Dropout作为自适应正则化)

对于单层线性网络，Dropout等价于：

$$\hat{R}_{dropout}(W) = \mathbb{E}_{m}\left[\frac{1}{n}\sum_{i=1}^{n} \ell(W(m \odot x_i), y_i)\right]$$

对于平方损失，近似于：

$$\hat{R}_{dropout}(W) \approx \hat{R}(W) + \frac{p}{1-p}\sum_{j=1}^{d} \|w_j\|^2 \cdot \text{Var}(x_j)$$

#### 定理 5.3.2 (Dropout的贝叶斯解释)

**Gal & Ghahramani (2016)**：Dropout可以解释为对权重的不确定性进行建模的变分推断。

设 $q(w)$ 是变分分布，最小化：

$$KL(q(w) \| p(w|D)) \approx -\mathbb{E}_{q(w)}[\log p(D|w)] + KL(q(w) \| p(w))$$

Dropout对应特定的 $q(w)$ 选择。

---

### 5.4 数据增强的理论效应

#### 定义 5.4.1 (数据增强)

**数据增强**通过对训练样本应用随机变换 $T$ 来扩充数据集：

$$S_{aug} = \{(T(x_i), y_i) : i = 1, \ldots, n, T \sim \mathcal{T}\}$$

常见变换：

- 图像：旋转、平移、缩放、翻转、颜色抖动
- 文本：同义词替换、回译

#### 定理 5.4.1 (数据增强的隐式正则化)

数据增强等价于在原始数据上添加正则化项：

$$\hat{R}_{aug}(h) = \mathbb{E}_{T \sim \mathcal{T}}\left[\frac{1}{n}\sum_{i=1}^{n} \ell(h(T(x_i)), y_i)\right]$$

对于小扰动，Taylor展开：

$$\hat{R}_{aug}(h) \approx \hat{R}(h) + \frac{1}{2}\mathbb{E}_{T}\left[\text{tr}(\nabla^2 \ell \cdot \text{Cov}(T(x) - x))\right]$$

#### 定理 5.4.2 (不变性学习)

如果真实标签在变换群 $G$ 下不变：

$$y = h^*(x) = h^*(g(x)), \quad \forall g \in G$$

则数据增强鼓励学习器学习不变表示：

$$h(x) \approx h(g(x)), \quad \forall g \in G$$

#### 定理 5.4.3 (Mixup的理论分析)

**Mixup**通过凸组合生成虚拟样本：

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

其中 $\lambda \sim \text{Beta}(\alpha, \alpha)$。

**Zhang et al. (2018)** 证明Mixup：

1. 扩大训练分布支持
2. 鼓励模型在样本间线性行为
3. 提供对对抗样本的鲁棒性

#### 定理 5.4.4 (数据增强的泛化界)

设变换集合 $\mathcal{T}$ 的大小为 $M$，则增强后的泛化界：

$$R(h) \leq \hat{R}_{aug}(h) + O\left(\sqrt{\frac{d_{VC} \log(Mn) + \log(1/\delta)}{Mn}}\right)$$

其中 $Mn$ 是有效样本量。

---

## 附录：核心定理汇总

### A.1 集中不等式

**Hoeffding不等式**：设 $X_1, \ldots, X_n$ 是i.i.d.，$X_i \in [a, b]$，则：

$$\mathbb{P}\left[\left|\frac{1}{n}\sum_{i=1}^{n} X_i - \mathbb{E}[X]\right| > t\right] \leq 2e^{-2nt^2/(b-a)^2}$$

**Bernstein不等式**：设 $X_1, \ldots, X_n$ 是i.i.d.，$|X_i| \leq M$，$\text{Var}(X_i) = \sigma^2$，则：

$$\mathbb{P}\left[\left|\frac{1}{n}\sum_{i=1}^{n} X_i - \mathbb{E}[X]\right| > t\right] \leq 2\exp\left(-\frac{nt^2}{2\sigma^2 + 2Mt/3}\right)$$

**McDiarmid不等式**：设 $f$ 满足有界差分条件：

$$|f(x_1, \ldots, x_i, \ldots, x_n) - f(x_1, \ldots, x_i', \ldots, x_n)| \leq c_i$$

则：

$$\mathbb{P}[|f(X_1, \ldots, X_n) - \mathbb{E}[f]| > t] \leq 2\exp\left(-\frac{2t^2}{\sum_{i=1}^{n} c_i^2}\right)$$

### A.2 优化收敛率总结

| 方法 | 凸 | 强凸 | 非凸 |
|-----|-----|-----|-----|
| GD | $O(1/T)$ | $O(\kappa \log(1/\epsilon))$ | $O(1/\sqrt{T})$ (梯度范数) |
| SGD | $O(1/\sqrt{T})$ | $O(1/T)$ | $O(1/\sqrt{T})$ |
| 牛顿法 | - | 局部二次 | 局部二次 |

### A.3 泛化界总结

| 方法 | 界 |
|-----|-----|
| 有限假设空间 | $O\left(\sqrt{\frac{\log|\mathcal{H}| + \log(1/\delta)}{n}}\right)$ |
| VC维 | $O\left(\sqrt{\frac{d_{VC} \log(n/d_{VC}) + \log(1/\delta)}{n}}\right)$ |
| Rademacher | $O\left(\mathfrak{R}_n(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{n}}\right)$ |
| PAC-Bayes | $O\left(\sqrt{\frac{KL(Q\|P) + \log(n/\delta)}{n}}\right)$ |
| 稳定性 | $O\left(\beta + \sqrt{\frac{\log(1/\delta)}{n}}\right)$ |

---

## 参考文献

1. Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press.

2. Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2018). *Foundations of Machine Learning* (2nd ed.). MIT Press.

3. Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.

4. Bubeck, S. (2015). Convex Optimization: Algorithms and Complexity. *Foundations and Trends in Machine Learning*, 8(3-4), 231-357.

5. Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias-variance trade-off. *PNAS*, 116(32), 15849-15854.

6. Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks. *NeurIPS*.

7. Bartlett, P. L., & Mendelson, S. (2002). Rademacher and Gaussian Complexities: Risk Bounds and Structural Results. *JMLR*, 3, 463-482.

---

*文档生成时间：2024年*
*理论深度对标：MIT 6.867, CMU 10-715, Stanford CS229T*
