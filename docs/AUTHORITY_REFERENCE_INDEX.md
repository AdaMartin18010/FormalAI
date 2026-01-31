# 权威引用索引 / Authority Reference Index

**创建日期**：2025-02-01  
**目的**：为 FormalAI 各主题模块提供权威引用锚点，支撑概念定义溯源与对标验证  
**维护**：每季度更新一次，与 QUARTERLY_UPDATE_CHECKLIST 联动

---

## 一、使用说明

- 每个主题模块在定义概念、论证主张时，应优先引用本索引中的**一手来源**；
- 引用格式：`[索引编号]` 或 直接标注完整引用；
- 新增主题时，需在本索引中补充对应锚点。

---

## 二、按主题分类的权威源

### 2.1 知识图谱与 Ontology

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| KG-01 | 课程 | Stanford CS520: Knowledge Graphs (web.stanford.edu/class/cs520) | **最近可及版本**：Spring 2022；**大纲**：有向标注图、RDF/OWL 数据模型、知识获取、推理算法、演化维护、用户交互；视频公开(YouTube) |
| KG-02 | 博客 | Stanford AI Lab: Introduction to Knowledge Graphs | 定义、应用场景 |
| KG-03 | 标准 | W3C OWL 2 / RDF 1.1 | 本体与知识表示规范 |
| KG-04 | 论文 | 各 CS520 客座讲座论文 | 见课程网站 |

### 2.2 形式化验证

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| FV-01 | 教材 | Clarke et al., Model Checking, MIT Press 2018 | 模型检测经典 |
| FV-02 | 教材 | Baier & Katoen, Principles of Model Checking, MIT Press 2008 | 时序逻辑、自动机 |
| FV-03 | 论文 | Cousot & Cousot (1977), Abstract Interpretation, POPL | 抽象解释奠基 |
| FV-04 | 论文 | Hoare (1969), Axiomatic basis for computer programming, CACM | 霍尔逻辑 |
| FV-05 | 工具/研究 | MIT Grove, Perennial (Coq/Iris) | 分布式系统形式化验证 |
| FV-06 | 课程 | Stanford CS259 Software Verification (web.stanford.edu/class/cs259) | **最近可及**；软件验证、程序分析 |

### 2.3 深度学习与机器学习

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| DL-01 | 教材 | Goodfellow, Bengio, Courville, Deep Learning, MIT Press 2016, 版次1 | Part I 基础(Ch2-3)、Part II 深度网络(Ch6-9)、Part III 优化(Ch8)、Part IV 应用(Ch12-15)；深度学习的数学基础 |
| DL-02 | 课程 | Stanford CS230, CS221 | 深度学习、AI 原理；CS230 讲义可参考 course.stanford.edu |
| DL-03 | 课程 | MIT 6.S965 TinyML and Efficient Deep Learning | 高效深度学习 |
| DL-04 | 课程 | Oxford ML (cs.ox.ac.uk/teaching/courses/ml) | **2025-2026 活跃**；Seth Flaxman 主讲；监督/无监督学习、神经网络 |
| DL-05 | 课程/暑期 | Oxford ML Summer School (oxfordml.school) | DeepMind 支持；MLx Fundamentals/GenAI/Health 等专题 |
| DL-06 | 论文 | Kaplan et al., Scaling Laws for Neural Language Models, 2020 | 原始 Scaling Law |
| DL-07 | 论文 | Hoffmann et al. (2022), Training Compute-Optimal Large Language Models | Chinchilla |

### 2.4 Scaling Law

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| SL-01 | 论文 | Hoffmann et al. (2022), Chinchilla | 计算最优、20 tokens/param |
| SL-02 | 论文 | Sardana et al. (2024), Beyond Chinchilla-Optimal, arXiv 2401.00448 | 推理成本扩展 |
| SL-03 | 论文 | Porian et al. (2024), NeurIPS 2024 | Kaplan vs Hoffmann 差异解释 |

### 2.5 意识理论

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| CO-01 | 论文 | Chalmers (1995), Facing Up to the Problem of Consciousness | 难问题 |
| CO-02 | 论文 | Tononi et al., IIT 3.0, PLoS Comp Bio 2016 | 整合信息理论 |
| CO-03 | 论文 | Dehaene, Consciousness and the Brain, 2014 | GWT/全局工作空间 |
| CO-04 | 论文 | Nature (2025), Adversarial testing of GNWT and IIT | PubMed 40307561 |
| CO-05 | 论文 | Phua et al. (2025), Testing Consciousness Theories on AI, arXiv 2512.19155 | GWT+HOT 互补 |
| CO-06 | 论文 | RCUET (2025), arXiv 2505.01464 | AI 意识形式化 |

### 2.6 大语言模型与 AI 代理

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| LLM-01 | 课程 | Berkeley CS294 Advanced LLM Agents (Spring 2025, rdi.berkeley.edu) | **大纲**：LLM 推理、搜索与规划、代码生成与验证、定理证明与自动形式化；Dawn Song 主讲 |
| LLM-02 | 课程 | MIT 6.4110 Representation, Inference and Reasoning in AI (airr.mit.edu) | 表示与推理 |

### 2.7 强化学习

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| RL-01 | 课程 | CMU 10-703 Deep Reinforcement Learning (cmudeeprl.github.io/703website_f25/) | **Fall 2025 活跃**；深度学习+RL、模仿学习、内在好奇心、机器人学习；Katerina Fragkiadaki, Aviral Kumar 主讲 |

### 2.8 中文权威教材（补充）

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| CN-01 | 教材 | 李航, 统计学习方法, 清华大学出版社 2012 | |
| CN-02 | 教材 | 周志华, 机器学习, 清华大学出版社 2016 | |
| CN-03 | 教材 | 邱锡鹏, 神经网络与深度学习, 机械工业出版社 2020 | |

---

## 三、会议/期刊锚点

- **NeurIPS, ICML, ICLR**：机器学习、深度学习
- **CAV, POPL, PLDI, LICS**：形式化方法、程序验证
- **Nature, Nature Neuroscience, Neuron**：意识、神经科学
- **ACL, EMNLP, NAACL**：NLP、语言模型

---

## 四、更新记录

| 日期 | 更新内容 |
|------|----------|
| 2025-02-01 | 初版创建 |
| 2025-02-01 | 补充 Goodfellow 章节、Stanford CS520/Berkeley CS294 课程大纲要点（BENCHMARKING_REPORT Q1 待办闭环） |
| 2025-02-01 | Stanford CS520 标注"最近可及 Spring 2022"；新增 CMU 10-703 Deep RL (RL-01) |
| 2025-02-01 | 新增 Stanford CS259 软件验证 (FV-06) |
| 2025-02-01 | 新增 Oxford ML (DL-04)、Oxford ML Summer School (DL-05)；Kaplan/Hoffmann 重编号为 DL-06/07 |

---

**维护者**：FormalAI 项目组
