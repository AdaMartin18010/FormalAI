# AI/ML 知识体系可视化表征

> 本文档系统性地呈现人工智能与机器学习的知识体系，包含思维导图、对比矩阵、决策树和概念关系图等多种可视化形式。

---

## 目录

- [AI/ML 知识体系可视化表征](#aiml-知识体系可视化表征)
  - [目录](#目录)
  - [1. AI/ML知识全景思维导图](#1-aiml知识全景思维导图)
    - [1.1 完整知识体系概览](#11-完整知识体系概览)
    - [1.2 深度学习专项知识图谱](#12-深度学习专项知识图谱)
  - [2. 多维概念对比矩阵](#2-多维概念对比矩阵)
    - [2.1 主流深度学习框架对比](#21-主流深度学习框架对比)
    - [2.2 监督学习算法对比](#22-监督学习算法对比)
    - [2.3 生成模型对比](#23-生成模型对比)
    - [2.4 优化器对比](#24-优化器对比)
    - [2.5 大语言模型对比](#25-大语言模型对比)
  - [3. 决策树图](#3-决策树图)
    - [3.1 模型选择决策树](#31-模型选择决策树)
    - [3.2 技术栈选型决策树](#32-技术栈选型决策树)
    - [3.3 部署架构决策树](#33-部署架构决策树)
  - [4. 推理归纳证明决策树](#4-推理归纳证明决策树)
    - [4.1 机器学习问题求解思维流程](#41-机器学习问题求解思维流程)
    - [4.2 模型诊断决策树](#42-模型诊断决策树)
    - [4.3 超参数调优决策流程](#43-超参数调优决策流程)
  - [5. 概念关系图](#5-概念关系图)
    - [5.1 核心概念依赖关系图](#51-核心概念依赖关系图)
    - [5.2 先修知识图谱](#52-先修知识图谱)
    - [5.3 算法演进时间线](#53-算法演进时间线)
  - [附录：快速参考卡片](#附录快速参考卡片)
    - [A. 损失函数速查表](#a-损失函数速查表)
    - [B. 激活函数选择指南](#b-激活函数选择指南)
    - [C. 评估指标选择](#c-评估指标选择)

---

## 1. AI/ML知识全景思维导图

### 1.1 完整知识体系概览

```mermaid
mindmap
  root((AI/ML<br/>知识体系))
    基础理论
      数学基础
        线性代数
          向量与矩阵
          特征值分解
          奇异值分解
        概率统计
          概率分布
          贝叶斯定理
          假设检验
        微积分
          梯度与偏导
          链式法则
          优化理论
        信息论
          熵与互信息
          KL散度
          交叉熵
      机器学习理论
        学习理论
          PAC学习框架
          VC维理论
          偏差-方差权衡
        优化理论
          凸优化
          非凸优化
          约束优化
        统计学习
          经验风险最小化
          结构风险最小化
          正则化理论

    核心算法
      监督学习
        分类算法
          逻辑回归
          支持向量机
          决策树
          随机森林
          朴素贝叶斯
          K近邻
        回归算法
          线性回归
          多项式回归
          岭回归/Lasso
          弹性网络
        集成方法
          Bagging
          Boosting
            AdaBoost
            Gradient Boosting
            XGBoost
            LightGBM
            CatBoost
          Stacking
      无监督学习
        聚类算法
          K-Means
          层次聚类
          DBSCAN
          高斯混合模型
        降维技术
          PCA
          t-SNE
          UMAP
          自编码器
        关联规则
          Apriori
          FP-Growth
      深度学习
        神经网络基础
          前馈神经网络
          卷积神经网络
          循环神经网络
          注意力机制
        高级架构
          Transformer
          ResNet
          GAN
          VAE
          Diffusion
        训练技术
          反向传播
          批量归一化
           dropout
          学习率调度
      强化学习
        基础概念
          MDP
          价值函数
          策略梯度
        算法分类
          值迭代
          策略迭代
          Actor-Critic
          DQN/PPO/A3C

    工具框架
      深度学习框架
        PyTorch
          动态图
          自动求导
          分布式训练
        TensorFlow
          静态图
          Keras API
          TFX部署
        JAX
          函数变换
          XLA编译
          自动向量化
      ML工具库
        Scikit-learn
          传统ML
          预处理
          模型评估
        XGBoost/LightGBM
          梯度提升
          特征重要度
        Hugging Face
          Transformers
          Datasets
          Tokenizers
      数据处理
        NumPy/Pandas
        Dask/Ray
        Spark MLlib
      部署工具
        ONNX
        TensorRT
        TorchServe
        MLflow

    应用场景
      计算机视觉
        图像分类
        目标检测
        图像分割
        人脸识别
        OCR
      自然语言处理
        文本分类
        机器翻译
        问答系统
        文本生成
        信息抽取
      语音处理
        语音识别
        语音合成
        声纹识别
      推荐系统
        协同过滤
        内容推荐
        混合推荐
      其他应用
        自动驾驶
        医疗诊断
        金融风控
        游戏AI
```

### 1.2 深度学习专项知识图谱

```mermaid
mindmap
  root((深度学习<br/>核心架构))
    基础组件
      激活函数
        ReLU/LeakyReLU
        Sigmoid/Tanh
        GELU/Swish
        Softmax
      损失函数
        交叉熵损失
        均方误差
        对比损失
        感知损失
      优化器
        SGD + Momentum
        Adam/AdamW
        RMSprop
        LAMB/LARS
      正则化
        L1/L2正则
        Dropout
        数据增强
        早停策略

    网络架构
      CNN家族
        LeNet → AlexNet
        VGGNet
        ResNet/DenseNet
        EfficientNet
        Vision Transformer
      RNN家族
        Vanilla RNN
        LSTM
        GRU
        Bi-directional
        Seq2Seq
      Transformer家族
        Encoder-only
          BERT
          RoBERTa
          ALBERT
        Decoder-only
          GPT系列
          LLaMA
          Claude
        Encoder-Decoder
          T5
          BART
          UL2
      生成模型
        VAE
          编码器-解码器
          重参数化技巧
          隐空间插值
        GAN
          生成器
          判别器
          训练技巧
        Diffusion
          前向扩散
          反向去噪
          条件生成

    训练技术
      优化策略
        学习率调度
          Warmup
          Cosine Annealing
          Step Decay
        梯度处理
          梯度裁剪
          梯度累积
          混合精度
      分布式训练
        数据并行
        模型并行
        流水线并行
        ZeRO优化
      迁移学习
        特征提取
        微调
        适配器
        LoRA/QLoRA
```

---

## 2. 多维概念对比矩阵

### 2.1 主流深度学习框架对比

| 维度 | PyTorch | TensorFlow | JAX |
|------|---------|------------|-----|
| **开发公司** | Meta (Facebook) | Google | Google |
| **发布年份** | 2016 | 2015 | 2018 |
| **计算图类型** | 动态图 (Eager) | 静态图 (Graph) | 函数变换 (Functional) |
| **调试体验** | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐⭐ 良好 | ⭐⭐⭐⭐ 良好 |
| **生产部署** | ⭐⭐⭐⭐ 良好 | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐⭐ 一般 |
| **研究友好度** | ⭐⭐⭐⭐⭐ 极高 | ⭐⭐⭐⭐ 高 | ⭐⭐⭐⭐⭐ 极高 |
| **生态系统** | 丰富 (torchvision等) | 最丰富 (Keras/TFX) | 快速增长 |
| **GPU支持** | CUDA原生 | CUDA/XLA | XLA优化 |
| **TPU支持** | 有限 | 原生支持 | 原生支持 |
| **分布式训练** | DDP/FSDP | Strategy API | pmap/pjit |
| **代码风格** | Pythonic直观 | 配置化 | 函数式编程 |
| **典型应用** | 研究/原型 | 生产/大规模 | 科研/高性能计算 |
| **学习曲线** | 平缓 | 较陡 | 中等 |
| **社区活跃度** | 极高 | 高 | 快速增长 |

### 2.2 监督学习算法对比

| 算法 | 适用场景 | 优点 | 缺点 | 时间复杂度 | 空间复杂度 | 可解释性 |
|------|----------|------|------|------------|------------|----------|
| **逻辑回归** | 二分类、概率预测 | 简单快速、可解释性强 | 只能处理线性问题 | O(nd) | O(d) | ⭐⭐⭐⭐⭐ |
| **SVM** | 高维数据、小样本 | 泛化能力强、核技巧灵活 | 大数据集慢、调参复杂 | O(n²d)~O(n³d) | O(n²)~O(n³) | ⭐⭐⭐ |
| **决策树** | 混合类型数据 | 直观可解释、无需归一化 | 易过拟合、不稳定 | O(nd log n) | O(n) | ⭐⭐⭐⭐⭐ |
| **随机森林** | 通用分类回归 | 准确率高、抗过拟合 | 训练慢、黑盒 | O(k·n·d·log n) | O(k·n) | ⭐⭐⭐ |
| **XGBoost** | 竞赛/表格数据 | 速度快、准确率高 | 调参复杂、易过拟合 | O(k·n·d) | O(n) | ⭐⭐⭐ |
| **KNN** | 小数据集、推荐 | 简单无训练、适应性强 | 预测慢、维度灾难 | O(1)训练/O(nd)预测 | O(nd) | ⭐⭐⭐⭐ |
| **朴素贝叶斯** | 文本分类、垃圾邮件 | 极快、小数据友好 | 特征独立性假设 | O(nd) | O(d) | ⭐⭐⭐⭐ |

> n=样本数, d=特征数, k=树数量/迭代次数

### 2.3 生成模型对比

| 维度 | VAE (变分自编码器) | GAN (生成对抗网络) | Diffusion (扩散模型) |
|------|-------------------|-------------------|---------------------|
| **核心思想** | 概率编码解码 | 对抗博弈 | 逐步去噪 |
| **训练稳定性** | ⭐⭐⭐⭐⭐ 稳定 | ⭐⭐ 不稳定 | ⭐⭐⭐⭐ 较稳定 |
| **生成质量** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐⭐ 高 | ⭐⭐⭐⭐⭐ 最高 |
| **训练速度** | ⭐⭐⭐⭐⭐ 快 | ⭐⭐⭐⭐ 较快 | ⭐⭐ 慢 |
| **推理速度** | ⭐⭐⭐⭐⭐ 快 | ⭐⭐⭐⭐⭐ 快 | ⭐⭐ 慢 |
| **模式覆盖** | ⭐⭐⭐⭐ 较好 | ⭐⭐ 易模式坍塌 | ⭐⭐⭐⭐⭐ 全面 |
| **隐空间** | 连续可解释 | 无显式隐空间 | 无显式隐空间 |
| **条件生成** | ⭐⭐⭐⭐ 容易 | ⭐⭐⭐⭐ 容易 | ⭐⭐⭐⭐⭐ 容易 |
| **数学基础** | 变分推断 | 博弈论 | 随机过程 |
| **代表模型** | β-VAE, VQ-VAE | StyleGAN, BigGAN | Stable Diffusion, DALL-E |
| **主要应用** | 降维、异常检测 | 图像生成、风格迁移 | 高质量图像/视频生成 |
| **训练技巧** | KL权重调度 | WGAN-GP, Spectral Norm | DDIM加速、Classifier Guidance |

### 2.4 优化器对比

| 优化器 | 更新公式特点 | 内存需求 | 收敛速度 | 泛化能力 | 超参数敏感度 | 最佳适用场景 |
|--------|-------------|----------|----------|----------|-------------|-------------|
| **SGD** | θ = θ - η·∇L | 低 (1x) | 慢 | ⭐⭐⭐⭐⭐ 最优 | 中等 | 大规模训练、追求泛化 |
| **Momentum** | 累加速度向量 | 低 (2x) | 中等 | ⭐⭐⭐⭐⭐ 最优 | 中等 | 非凸优化、逃离局部最优 |
| **AdaGrad** | 自适应学习率 | 高 (d+1)x | 快(初期) | ⭐⭐⭐ 一般 | 低 | 稀疏梯度、NLP |
| **RMSprop** | 指数移动平均梯度 | 中 (2x) | 快 | ⭐⭐⭐⭐ 良好 | 中等 | RNN、非平稳目标 |
| **Adam** | Momentum + RMSprop | 中 (3x) | ⭐⭐⭐⭐⭐ 最快 | ⭐⭐⭐ 一般 | 低 | 默认选择、快速收敛 |
| **AdamW** | Adam + 解耦权重衰减 | 中 (3x) | 快 | ⭐⭐⭐⭐ 良好 | 低 | Transformer、大模型训练 |
| **LAMB** | 分层自适应+动量 | 高 (3x) | 快 | ⭐⭐⭐⭐ 良好 | 低 | 超大batch训练 |
| **LARS** | 层级自适应学习率 | 低 (2x) | 中等 | ⭐⭐⭐⭐ 良好 | 中等 | 对比学习、自监督 |

**优化器选择建议：**

- 小模型/快速实验 → Adam/AdamW
- 大模型/生产部署 → AdamW + 学习率调度
- 追求最优泛化 → SGD + Momentum + 充分训练
- 超大batch → LAMB/LARS
- 计算机视觉 → SGD + Momentum (经典)
- NLP/Transformer → AdamW

### 2.5 大语言模型对比

| 特性 | GPT-4/GPT-4o | LLaMA 3 | Claude 3 | Gemini |
|------|-------------|---------|----------|--------|
| **开发公司** | OpenAI | Meta | Anthropic | Google |
| **模型规模** | 未公开 (~1.8T MoE) | 8B/70B/405B | 未公开 | 1.5B-1.5T |
| **架构类型** | Decoder-only | Decoder-only | Decoder-only | MoE |
| **上下文长度** | 128K (4o: 1M) | 128K | 200K | 1M+ |
| **多模态** | ✅ 图像/语音/视频 | ❌ 纯文本 | ✅ 图像 | ✅ 图像/视频/音频 |
| **开源** | ❌ API only | ✅ 开源权重 | ❌ API only | 部分开源(Gemma) |
| **训练数据** | 未公开 | 15T tokens | 未公开 | 多模态数据 |
| **安全性** | ⭐⭐⭐⭐ 良好 | ⭐⭐⭐ 依赖微调 | ⭐⭐⭐⭐⭐ 最高 | ⭐⭐⭐⭐ 良好 |
| **推理成本** | 高 | 可控(自托管) | 高 | 中等 |
| **推理速度** | 中等 | 快(小模型) | 中等 | 快 |
| **代码能力** | ⭐⭐⭐⭐⭐ 最强 | ⭐⭐⭐⭐ 强 | ⭐⭐⭐⭐⭐ 强 | ⭐⭐⭐⭐ 强 |
| **长文本处理** | ⭐⭐⭐⭐ 良好 | ⭐⭐⭐⭐ 良好 | ⭐⭐⭐⭐⭐ 最强 | ⭐⭐⭐⭐⭐ 最强 |
| **数学推理** | ⭐⭐⭐⭐⭐ 最强 | ⭐⭐⭐⭐ 强 | ⭐⭐⭐⭐⭐ 强 | ⭐⭐⭐⭐⭐ 强 |
| **API可用性** | ⭐⭐⭐⭐⭐ 全球 | 自托管 | ⭐⭐⭐ 受限地区 | ⭐⭐⭐⭐⭐ 全球 |

---

## 3. 决策树图

### 3.1 模型选择决策树

```mermaid
flowchart TD
    A[开始: 机器学习问题] --> B{数据类型?}

    B -->|结构化/表格数据| C{样本量?}
    B -->|图像数据| D[深度学习: CNN/ViT]
    B -->|文本数据| E[深度学习: Transformer/RNN]
    B -->|时间序列| F{序列长度?}
    B -->|音频/语音| G[深度学习: 专用架构]

    C -->|小样本 < 10K| H{特征维度?}
    C -->|中等样本 10K-100K| I{问题类型?}
    C -->|大样本 > 100K| J{计算资源?}

    H -->|低维 < 100| K[传统ML: SVM/朴素贝叶斯]
    H -->|高维 > 100| L[降维后: PCA + 分类器]

    I -->|分类| M[集成方法: XGBoost/LightGBM]
    I -->|回归| N[集成方法: XGBoost/随机森林]
    I -->|聚类| O[无监督: K-Means/DBSCAN]

    J -->|有限| P[分布式: Spark MLlib]
    J -->|充足| Q[深度学习: TabNet/FT-Transformer]

    D --> R{任务类型?}
    R -->|分类| S[ResNet/EfficientNet]
    R -->|检测| T[YOLO/RCNN系列]
    R -->|分割| U[UNet/Mask R-CNN]

    E --> V{任务类型?}
    V -->|分类| W[BERT/RoBERTa]
    V -->|生成| X[GPT/LLaMA/Claude]
    V -->|翻译| Y[T5/mBART]

    F -->|短序列| Z[传统: ARIMA/Prophet]
    F -->|长序列| AA[深度学习: LSTM/Transformer]

    G --> AB{任务?}
    AB -->|识别| AC[Whisper/Wav2Vec]
    AB -->|合成| AD[Tacotron/WaveNet]
```

### 3.2 技术栈选型决策树

```mermaid
flowchart TD
    A[项目需求分析] --> B{项目阶段?}

    B -->|研究/原型| C{偏好?}
    B -->|生产部署| D{部署环境?}
    B -->|边缘设备| E{硬件平台?}

    C -->|快速实验| F[PyTorch + Jupyter]
    C -->|可复现研究| G[JAX + Haiku/Flax]
    C -->|Keras API偏好| H[TensorFlow + Keras]

    D -->|云平台| I{云服务提供商?}
    D -->|本地服务器| J{容器化?}
    D -->|混合部署| K[MLflow + ONNX]

    I -->|AWS| L[SageMaker + PyTorch/TF]
    I -->|GCP| M[Vertex AI + TensorFlow]
    I -->|Azure| N[Azure ML + PyTorch]

    J -->|Docker| O[TensorFlow Serving/TorchServe]
    J -->|K8s| P[Kubeflow + KServe]

    E -->|NVIDIA GPU| Q[TensorRT + ONNX]
    E -->|ARM/移动| R[TensorFlow Lite/CoreML]
    E -->|浏览器| S[TensorFlow.js/ONNX.js]
    E -->|FPGA/ASIC| T[专用SDK]

    F --> U[配套工具: WandB + Hydra]
    G --> V[配套工具: Optax + Orbax]
    H --> W[配套工具: TensorBoard + TFX]
```

### 3.3 部署架构决策树

```mermaid
flowchart TD
    A[部署需求分析] --> B{延迟要求?}

    B -->|实时 < 100ms| C{并发量?}
    B -->|近实时 100ms-1s| D{模型大小?}
    B -->|批处理 > 1s| E{数据量?}

    C -->|低 < 100 QPS| F[单机GPU + REST API]
    C -->|中 100-1000 QPS| G[负载均衡 + 多实例]
    C -->|高 > 1000 QPS| H{模型可压缩?}

    H -->|是| I[模型量化 + TensorRT]
    H -->|否| J[模型并行 + 分布式推理]

    D -->|小 < 100MB| K[边缘部署 + 本地推理]
    D -->|中 100MB-1GB| L[模型分片 + 流式加载]
    D -->|大 > 1GB| M[模型并行 + 专家系统]

    E -->|TB级| N[Spark + 批处理推理]
    E -->|PB级| O[Dataflow + 分布式计算]

    F --> P[技术: FastAPI + TorchServe]
    G --> Q[技术: K8s + KServe + HPA]
    I --> R[技术: ONNX Runtime + TensorRT]
    J --> S[技术: Megatron/DeepSpeed]
    K --> T[技术: TensorFlow Lite/CoreML]
    L --> U[技术: 模型分片 + 增量加载]
    M --> V[技术: MoE + 路由系统]
    N --> W[技术: Spark ML + Delta Lake]
    O --> X[技术: Beam + BigQuery]
```

---

## 4. 推理归纳证明决策树

### 4.1 机器学习问题求解思维流程

```mermaid
flowchart TD
    subgraph 问题定义阶段
        A1[业务问题识别] --> A2[转化为ML问题]
        A2 --> A3{可解性评估}
        A3 -->|不可行| A4[重新设计或放弃]
        A3 -->|可行| A5[定义成功指标]
    end

    subgraph 数据探索阶段
        B1[数据收集] --> B2[数据质量检查]
        B2 --> B3{数据充足?}
        B3 -->|否| B4[数据增强/采集]
        B3 -->|是| B5[EDA分析]
        B5 --> B6[特征工程]
    end

    subgraph 模型开发阶段
        C1[基线模型] --> C2[快速迭代]
        C2 --> C3{性能达标?}
        C3 -->|否| C4[诊断分析]
        C4 --> C5{问题类型?}
        C5 -->|欠拟合| C6[增加模型容量]
        C5 -->|过拟合| C7[正则化/数据增强]
        C5 -->|数据问题| C8[数据清洗]
        C3 -->|是| C9[模型优化]
    end

    subgraph 验证部署阶段
        D1[交叉验证] --> D2[测试集评估]
        D2 --> D3{泛化良好?}
        D3 -->|否| D4[返回调优]
        D3 -->|是| D5[模型解释]
        D5 --> D6[部署上线]
        D6 --> D7[监控反馈]
    end

    A5 --> B1
    B6 --> C1
    C9 --> D1
    D7 --> D8{性能下降?}
    D8 -->|是| D9[模型重训练]
    D9 --> C1
    D8 -->|否| D10[持续监控]
```

### 4.2 模型诊断决策树

```mermaid
flowchart TD
    A[模型表现不佳] --> B{训练误差?}

    B -->|高| C[欠拟合问题]
    B -->|低| D{验证误差?}

    C --> E{模型复杂度?}
    E -->|太低| F[增加模型容量]
    E -->|足够| G{训练充分?}

    G -->|不充分| H[延长训练/调整学习率]
    G -->|充分| I{特征质量?}

    I -->|差| J[特征工程/特征选择]
    I -->|好| K[检查数据标签]

    D -->|高| L[过拟合问题]
    D -->|低| M[模型表现良好]

    L --> N{训练数据量?}
    N -->|少| O[数据增强/收集]
    N -->|足够| P{正则化?}

    P -->|无| Q[添加Dropout/L2]
    P -->|有| R[增加正则强度]

    F --> S[重新训练]
    H --> S
    J --> S
    O --> S
    Q --> S
    R --> S
    K --> T[修正标签后重训]

    S --> U{问题解决?}
    U -->|否| A
    U -->|是| V[模型就绪]
```

### 4.3 超参数调优决策流程

```mermaid
flowchart TD
    A[超参数调优] --> B{调参预算?}

    B -->|有限 < 50次| C[网格搜索/随机搜索]
    B -->|中等 50-200次| D[贝叶斯优化]
    B -->|充足 > 200次| E[Population Based Training]

    C --> F{参数类型?}
    F -->|离散| G[网格搜索]
    F -->|连续| H[随机搜索]

    D --> I[Optuna/Ray Tune]
    E --> J[ASHA/Hyperband]

    K[关键超参数] --> L[学习率]
    K --> M[Batch Size]
    K --> N[正则化系数]
    K --> O[网络深度/宽度]

    L --> P{优化器?}
    P -->|Adam| Q[1e-4 ~ 1e-2]
    P -->|SGD| R[1e-3 ~ 1e-1]

    M --> S{硬件限制?}
    S -->|显存小| T[8-32]
    S -->|显存大| U[64-512]

    N --> V{过拟合程度?}
    V -->|严重| W[1e-4 ~ 1e-2]
    V -->|轻微| X[1e-6 ~ 1e-4]

    G --> Y[应用调优结果]
    H --> Y
    I --> Y
    J --> Y
```

---

## 5. 概念关系图

### 5.1 核心概念依赖关系图

```mermaid
graph TD
    subgraph 数学基础
        A1[线性代数] --> A2[矩阵运算]
        A1 --> A3[特征分解]
        A4[微积分] --> A5[梯度计算]
        A4 --> A6[优化理论]
        A7[概率论] --> A8[贝叶斯推断]
        A7 --> A9[分布理论]
    end

    subgraph 机器学习基础
        B1[监督学习] --> B2[分类]
        B1 --> B3[回归]
        B4[无监督学习] --> B5[聚类]
        B4 --> B6[降维]
        B7[强化学习] --> B8[MDP]
        B7 --> B9[策略优化]
    end

    subgraph 深度学习
        C1[神经网络] --> C2[前馈网络]
        C1 --> C3[卷积网络]
        C1 --> C4[循环网络]
        C5[训练技术] --> C6[反向传播]
        C5 --> C7[优化器]
        C5 --> C8[正则化]
        C9[高级架构] --> C10[Transformer]
        C9 --> C11[GAN]
        C9 --> C12[Diffusion]
    end

    subgraph 应用
        D1[计算机视觉] --> D2[图像分类]
        D1 --> D3[目标检测]
        D4[NLP] --> D5[文本分类]
        D4 --> D6[机器翻译]
        D4 --> D7[文本生成]
    end

    A2 --> C6
    A3 --> B6
    A5 --> C6
    A6 --> C7
    A8 --> B1
    A9 --> B4

    B2 --> C2
    B3 --> C2
    C2 --> C3
    C2 --> C4
    C6 --> C5
    C7 --> C5
    C8 --> C5
    C3 --> D1
    C4 --> D4
    C10 --> D4
    C10 --> D7
```

### 5.2 先修知识图谱

```mermaid
flowchart TD
    subgraph Level_0_基础数学
        L0_1[高中数学]
        L0_2[Python编程基础]
    end

    subgraph Level_1_数学进阶
        L1_1[线性代数]
        L1_2[概率统计]
        L1_3[微积分]
        L1_4[Python科学计算]
    end

    subgraph Level_2_机器学习入门
        L2_1[监督学习基础]
        L2_2[无监督学习基础]
        L2_3[Scikit-learn]
        L2_4[数据预处理]
    end

    subgraph Level_3_深度学习
        L3_1[神经网络基础]
        L3_2[PyTorch/TensorFlow]
        L3_3[CNN架构]
        L3_4[RNN/LSTM]
        L3_5[优化算法]
    end

    subgraph Level_4_高级主题
        L4_1[Transformer]
        L4_2[GAN/VAE]
        L4_3[强化学习]
        L4_4[大语言模型]
    end

    subgraph Level_5_专业应用
        L5_1[计算机视觉]
        L5_2[NLP]
        L5_3[推荐系统]
        L5_4[多模态AI]
    end

    L0_1 --> L1_1
    L0_1 --> L1_2
    L0_1 --> L1_3
    L0_2 --> L1_4

    L1_1 --> L2_1
    L1_2 --> L2_1
    L1_2 --> L2_2
    L1_4 --> L2_3
    L2_1 --> L2_4

    L2_1 --> L3_1
    L2_3 --> L3_2
    L3_1 --> L3_3
    L3_1 --> L3_4
    L1_3 --> L3_5

    L3_3 --> L4_1
    L3_4 --> L4_1
    L3_2 --> L4_2
    L3_5 --> L4_3
    L4_1 --> L4_4

    L4_1 --> L5_1
    L4_1 --> L5_2
    L4_4 --> L5_2
    L2_2 --> L5_3
    L4_1 --> L5_4
    L4_2 --> L5_4
```

### 5.3 算法演进时间线

```mermaid
timeline
    title AI/ML 算法演进历程

    section 1950s-1980s
        1958 : 感知机 Perceptron
        1965 : 第一个神经网络
        1972 : K-Means聚类
        1986 : 反向传播算法
        1989 : CNN诞生 LeNet

    section 1990s-2000s
        1995 : SVM
        1997 : LSTM
        1998 : LeNet-5实用化
        2001 : 随机森林
        2006 : 深度学习复兴
        2009 : ImageNet数据集

    section 2010s
        2012 : AlexNet突破
        2013 : Word2Vec
        2014 : GAN / VAE
        2014 : Adam优化器
        2015 : ResNet / BatchNorm
        2016 : AlphaGo
        2017 : Transformer
        2018 : BERT / GPT-1

    section 2020s
        2020 : GPT-3
        2020 : Vision Transformer
        2021 : CLIP / DALL-E
        2022 : ChatGPT / Stable Diffusion
        2023 : GPT-4 / LLaMA
        2024 : GPT-4o / Claude 3 / Gemini
```

---

## 附录：快速参考卡片

### A. 损失函数速查表

| 问题类型 | 推荐损失函数 | 公式 |
|---------|-------------|------|
| 二分类 | Binary Cross Entropy | -[y·log(p) + (1-y)·log(1-p)] |
| 多分类 | Categorical Cross Entropy | -Σ y_i·log(p_i) |
| 回归 | MSE | (y - ŷ)² |
| 回归(异常值鲁棒) | MAE | \|y - ŷ\| |
| 排序 | Hinge Loss | max(0, 1 - y·ŷ) |

### B. 激活函数选择指南

| 场景 | 推荐激活函数 | 原因 |
|-----|-------------|------|
| 隐藏层默认 | ReLU | 计算快、缓解梯度消失 |
| 深层网络 | GELU/Swish | 平滑、性能更好 |
| 输出层(分类) | Softmax | 概率归一化 |
| 输出层(回归) | Linear | 无约束输出 |
| RNN/LSTM | Tanh | 输出范围控制 |

### C. 评估指标选择

| 问题类型 | 平衡数据 | 不平衡数据 |
|---------|---------|-----------|
| 二分类 | Accuracy / F1 | Precision-Recall AUC |
| 多分类 | Macro F1 | Weighted F1 |
| 回归 | RMSE / MAE | MAPE / R² |
| 排序 | NDCG / MAP | MRR |

---

*文档版本: 1.0 | 最后更新: 2024年*
*本知识图谱持续更新，建议结合实践项目深化理解*
