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
| FV-12 | 课程 | Stanford CS256 Formal Methods for Reactive Systems (cs256.stanford.edu) | 反应式系统形式化方法；LTL/CTL、自动机、演绎验证、模型检测 |
| FV-13 | 课程 | CMU 15-311 Logic and Mechanized Reasoning (csd.cmu.edu/course/15311) | 命题/一阶/高阶逻辑、Lean 交互式定理证明、SAT/SMT/一阶定理证明器 |
| FV-14 | 课程 | CMU 15-414 Bug Catching: Automated Program Verification (cs.cmu.edu/~15414) | 程序验证原理与算法、Why3 演绎验证平台、决策过程与模型检测 |
| FV-07 | 工具 | ProofNet++ (2025), arXiv 2505.24230 | 神经符号证明验证系统，结合LLM与形式验证 |
| FV-08 | 工具 | GenBaB (2025), α,β-CROWN框架 | 扩展神经网络验证到非线性激活(Sigmoid, Tanh, Sine, GeLU) |
| FV-09 | 工具 | Marabou 2.0 (2025) | 综合神经网络形式化分析器，VNN-COMP 2023/2024获奖者 |
| FV-10 | 框架 | 运行时监控框架 (2025), arXiv 2507.11987 | 轻量级神经证书运行时验证，用于CPS控制 |
| FV-11 | 论文 | Neural Model Checking (2025), NeurIPS 2025 | 联合验证安全性和活性，使用约束求解器训练 |

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
| LLM-02 | 课程 | MIT 6.4110 Representation, Inference and Reasoning in AI (airr.mit.edu, Spring 2025) | **2025活跃**；表示与推理、AI推理机制 |
| LLM-03 | 课程 | Stanford CS229: Machine Learning (Fall 2025) | **2025活跃**；Moses Charikar, Carlos Guestrin, Andrew Ng主讲；监督/无监督/强化学习 |
| LLM-04 | 课程 | Stanford CS238V/AA228V: Validation of Safety Critical Systems | **2025活跃**；形式化方法、时序逻辑、模型检测、自主系统验证 |
| LLM-05 | 课程/暑期 | ESSLLI 2024 Large Language Models, Knowledge, and Reasoning (damir.cavar.me/ESSLLI24_LLM_KG) | LLM 与知识表示、本体/知识图谱、描述逻辑、神经符号建模 |
| LLM-06 | 课程 | MIT 24.S90 Demystifying Large Language Models (linguistics.mit.edu/s90_f24) | LLM 能力与局限、复杂推理与语言、训练数据关系、与人类语言习得比较 |
| LLM-07 | 课程 | Cambridge L98 Introduction to Computational Semantics (cl.cam.ac.uk/teaching/2425/L98) | **2024-25**；形式语义方法、计算语义 |
| LLM-08 | 课程 | Purdue 592 Can Machines Think? Reasoning with LLMs (Fall 2024) | 符号与神经语言模型、形式演绎推理、零样本/少样本/CoT、神经符号推理 |

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

### 2.9 W3C标准与规范

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| W3C-01 | 标准 | W3C OWL 2 Web Ontology Language Primer (Second Edition) | OWL 2类、属性、个体、数据值定义；语义Web应用标准 |
| W3C-02 | 标准 | W3C RDF 1.1 Concepts and Abstract Syntax | RDF语义模型、三元组结构、图数据模型 |
| W3C-03 | 标准 | W3C Knowledge Graph标准 | 知识图谱表示与查询标准 |

### 2.10 认知科学与学习理论

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| COG-01 | 理论 | Schema Theory (Bartlett 1932, Anderson 1970s) | 知识以层次化schema组织，包含槽位和默认值 |
| COG-02 | 理论 | Cognitive Load Theory (Sweller 1988) | 内在负荷、外在负荷、相关负荷；工作记忆限制 |
| COG-03 | 理论 | Spaced Repetition Theory | 分布式学习优于集中学习；工作记忆资源恢复 |
| COG-04 | 论文 | Nature (2022), The science of effective learning with spacing and retrieval practice | 间隔重复与检索练习的科学基础 |
| COG-05 | 论文 | Educ Psychol Rev: Spacing vs interleaving (Rohrer et al.), discriminative-contrast hypothesis | 间隔练习与交错练习区分；间隔≈认知负荷/工作记忆恢复，交错≈区分对比 |

### 2.11 类型论与范畴论

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| TT-01 | 课程 | Cambridge CAT Category Theory (cl.cam.ac.uk/teaching/2425/CAT) | **2024-25**；范畴论 Part II 课程 |
| TT-02 | 课程 | CMU B619 Modern Dependent Types (carloangiuli.com/courses/b619-sp24) | 依赖类型理论、Agda/Coq/Lean、外延/内涵/同伦/立方类型论 |
| TT-03 | 课程 | CMU 80-518/818 Topics in Logic: Type Theory (Awodey, awodey.github.io/typetheory) | **Spring 2025**；类型论形式系统、范畴语义、Martin-Löf、同伦类型论；先修范畴论 |

### 2.12 AI 对齐与安全

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| AL-01 | 课程 | Stanford MS&E338 Aligning Superintelligence (web.stanford.edu/class/msande338) | 超智能对齐、奖励黑客/不确定性/学习、CIRL；先修研究生 ML 与智能体 |
| AL-02 | 课程 | Stanford CS120 Introduction to AI Safety (web.stanford.edu/class/cs120) | **Fall 2025**；可靠、伦理、对齐的 AI；可解释性、鲁棒性、评估；OpenAI/Anthropic/UCB 客座 |
| AL-03 | 课程 | Stanford CS362 Research in AI Alignment (scottviteri.com/teaching/courses/cs362) | **Fall 2024**；对齐研究研讨、哲学与技术结合；需熟悉 LLM 与强化学习 |
| AL-04 | 课程 | Oxford AI Safety and Alignment (robots.ox.ac.uk/~fazl/aisaa/, Oct 2025) | **5 天集中**；对齐问题、前沿对齐方法、可解释性、监测、社会技术层面；Bengio、DeepMind 客座 |
| AL-05 | 课程/机构 | BlueDot Impact / EA Cambridge Technical Alignment Curriculum (eacambridge.org/technical-alignment-curriculum) | 12 单元；RLHF、可扩展监督、机制可解释性、治理；6000+ 校友，OpenAI/Anthropic/DeepMind 等 |

### 2.13 可解释 AI (XAI)

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| XAI-01 | 课程 | Harvard Explainable AI: From Simple Predictors to Complex Generative Models | 研究生；可解释模型、事后解释、机制可解释性、LLM 与扩散模型应用 |
| XAI-02 | 课程 | UW CSE574 Explainable Artificial Intelligence (courses.cs.washington.edu/courses/cse574) | **Spring 2025**；可解释模型、特征归因、反事实/概念解释、人机协作 |
| XAI-03 | 课程 | Cambridge L193 Explainable Artificial Intelligence (cl.cam.ac.uk/teaching/2425/L193) | **2024-25**；硕士级可解释 AI |
| XAI-04 | 课程/专项 | Duke Explainable AI (XAI) Specialization, Coursera | Interpretable ML、Developing XAI；回归/树/规则、机制可解释性、鲁棒性/隐私 |

### 2.14 神经符号 AI

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| NS-01 | 暑期学校 | Neuro-Symbolic AI Summer School NSSS 2024 (neurosymbolic.github.io/nsss2024) | 逻辑推理、神经网络、神经符号方法教程与软件演示；2023 注册 3500+ |
| NS-02 | 课程 | CMU 10-747 Neuro-Symbolic AI (cs.cmu.edu/~pradeepr/747, Fall 2025) | 神经符号 AI；先修概率、统计、机器学习基础 |
| NS-03 | 课程/资源 | ASU Neuro Symbolic AI (neurosymbolic.asu.edu) | 研究生课、YouTube 教程、Advances in Neuro Symbolic Reasoning and Learning 教材 |

### 2.15 统计学习理论（补充）

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| SLT-01 | 课程 | MIT 9.520/6.7910 Statistical Learning Theory and Applications (poggio-lab.mit.edu/9-520) | **Fall 2025**；线性预测、核方法、SVM、神经网络；逼近、优化与深度网络学习理论 |
| SLT-02 | 课程 | MIT 6.790 Machine Learning (gradml.mit.edu) | PAC、VC 维、可学习性、基本定理 |
| SLT-03 | 课程 | NYU CSCI-GA.2566 Foundations of Machine Learning (cs.nyu.edu/~mohri/ml25) | 概率工具、集中不等式、PAC、Rademacher 复杂度、VC 维、SVM、核方法、Boosting、强化学习；教材 Mohri et al. 2018 |
| SLT-04 | 教材 | Mohri, Rostamizadeh, Talwalkar, Foundations of Machine Learning, 2nd ed., MIT Press 2018 | PAC 模型、有限/无穷假设集学习保证、Rademacher 复杂度、增长函数、VC 维；与 [SLT-03] 课程配套 |
| SLT-05 | 课程 | MIT 18.465 Topics in Statistics: Statistical Learning Theory | PAC、VC、Rademacher 等统计学习理论 |
| SLT-06 | 课程 | USC CSCI 699 Advanced Topics in Machine Learning | 研究生级统计学习与可学习性；PAC、VC 维等 |

### 2.16 计算理论（01.3 用）

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| COMP-01 | 教材 | Sipser, Introduction to the Theory of Computation, 3rd ed., Cengage 2013 | 自动机、可计算性、复杂度（P/NP）；标准本科/研究生教材 |

### 2.17 因果推理（02.4 用）

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| CAUS-01 | 教材 | Pearl, Causality, Cambridge University Press 2000 (2nd ed. 2009) | 因果图、do-演算、识别、混淆控制 |
| CAUS-02 | 教材 | Pearl et al., Causal Inference in Statistics: A Primer, Wiley 2016 | 因果推断入门；与 UCLA 教学材料配套 |

### 2.18 涌现与复杂性（08 用）

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| EMER-01 | 课程/平台 | Santa Fe Institute Complexity Explorer (complexityexplorer.org) | Introduction to Complexity (Melanie Mitchell)、动力系统与混沌、涌现；免费在线 |

### 2.19 具身/边缘/量子/元学习/社会/哲学（10~12,15~17,20 用）

| 编号 | 类型 | 来源 | 说明 |
|------|------|------|------|
| EDGE-01 | 课程 | [DL-03] MIT 6.S965 TinyML | 边缘/嵌入式 ML，与绿色 AI 重叠 |
| META-01 | 论文 | Finn et al., Model-Agnostic Meta-Learning (MAML), ICML 2017 | 元学习经典；少样本学习 |
| SOC-01 | 综述/课程 | 多智能体系统、社会 AI（见 ACL/AAAI 相关 tutorial） | 社会 AI 与多智能体；可补充具体课程链接 |

---

## 三、会议/期刊锚点

### 3.1 顶级会议

- **NeurIPS, ICML, ICLR**：机器学习、深度学习
- **CAV, POPL, PLDI, LICS**：形式化方法、程序验证
- **ACL, EMNLP, NAACL**：NLP、语言模型
- **CVPR, ICCV, ECCV**：计算机视觉、多模态AI
- **AAAI, IJCAI**：人工智能综合会议

### 3.2 顶级期刊

- **Nature, Science**：跨学科顶级期刊，AI相关突破性研究
- **Nature Neuroscience, Neuron**：意识、神经科学
- **Journal of Machine Learning Research (JMLR)**：机器学习理论
- **Artificial Intelligence (AIJ)**：人工智能综合期刊
- **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**：模式识别与机器学习

---

## 四、更新记录

| 日期 | 更新内容 |
|------|----------|
| 2025-02-01 | 初版创建 |
| 2025-02-01 | 补充 Goodfellow 章节、Stanford CS520/Berkeley CS294 课程大纲要点（BENCHMARKING_REPORT Q1 待办闭环） |
| 2025-02-01 | Stanford CS520 标注"最近可及 Spring 2022"；新增 CMU 10-703 Deep RL (RL-01) |
| 2025-02-01 | 新增 Stanford CS259 软件验证 (FV-06) |
| 2025-02-01 | 新增 Oxford ML (DL-04)、Oxford ML Summer School (DL-05)；Kaplan/Hoffmann 重编号为 DL-06/07 |
| 2025-02-02 | 补充形式化验证2025最新工具(FV-07~FV-11)：ProofNet++、GenBaB、Marabou 2.0、运行时监控框架 |
| 2025-02-02 | 补充MIT 6.4110、Stanford CS229/CS238V课程(LLM-02~LLM-04) |
| 2025-02-02 | 新增W3C标准索引(W3C-01~W3C-03)：OWL 2、RDF 1.1、知识图谱标准 |
| 2025-02-02 | 新增认知科学理论索引(COG-01~COG-04)：Schema理论、认知负荷理论、间隔重复理论 |
| 2025-02-02 | 扩展形式化验证(FV-12~FV-14)：Stanford CS256、CMU 15-311/15-414 |
| 2025-02-02 | 扩展大语言模型(LLM-05~LLM-08)：ESSLLI 2024、MIT 24.S90、Cambridge L98、Purdue 592 |
| 2025-02-02 | 新增类型论与范畴论(TT-01~TT-03)：Cambridge CAT、CMU B619、CMU 80-518/818 |
| 2025-02-02 | 新增AI对齐与安全(AL-01~AL-03)：Stanford MS&E338、CS120、CS362 |
| 2025-02-02 | 新增可解释AI(XAI-01~XAI-04)：Harvard、UW CSE574、Cambridge L193、Duke Coursera |
| 2025-02-02 | 新增神经符号AI(NS-01~NS-03)：NSSS 2024、CMU 10-747、ASU |
| 2025-02-02 | 新增统计学习理论(SLT-01~SLT-03)：MIT 9.520/6.790、NYU CSCI-GA.2566 |
| 2025-02-02 | 扩展统计学习：SLT-04 Mohri et al. 教材；对齐：AL-04 Oxford、AL-05 BlueDot/EA Cambridge；认知：COG-05 间隔vs交错 |
| 2025-02-02 | 新增 SLT-05 MIT 18.465、SLT-06 USC CSCI 699 统计学习课程（权威对标与认知优化计划） |

---

**维护者**：FormalAI 项目组
**关联文档**：

- [概念定义溯源表](../concepts/DEFINITION_SOURCE_TABLE.md)
- [季度更新检查清单](QUARTERLY_UPDATE_CHECKLIST.md)
