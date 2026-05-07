# Harness Engineering 文本分类项目 PRD

## 1. 文档目的

本文档沉淀 `student_package` 项目的需求、边界、接口约束、评测规则、交付标准与协作约定。后续实现 `MyHarness`、整理探索报告、运行本地评测和提交成果时，均以本文档作为需求基准。

## 2. 项目背景

本项目来自 2026 年夏季 Harness Engineering 考核。任务要求在不训练模型权重的前提下，设计一个围绕 LLM 的外部 Harness，通过记忆管理、Prompt 构造、上下文预算控制、输出解析和安全防护，在有限输入窗口内完成文本分类。

本地 DEV 任务是客服意图分类。正式评测会覆盖 DEV 同域任务、其他领域 OOD 分类任务和自然语言选择题任务，因此实现不能只针对当前 DEV 数据硬编码或过拟合。

## 3. 产品目标

- 在 `solution.py` 的 `MyHarness` 类内实现可泛化的分类 Harness。
- 接收训练流 `update(text, label)`，维护外部记忆与标签知识。
- 对测试文本 `predict(text)` 返回准确、规范、可 exact match 的标签字符串。
- 在单次 LLM 调用 `max_prompt_tokens <= 2048` 的限制下，主动控制 Prompt 长度，避免被评测器截断。
- 在 DEV 与私有任务上尽可能提升平均准确率，同时保证运行稳定、耗时合理、无违规行为。
- 形成可解释的探索记录与报告材料，为主观评分提供依据。

## 4. 用户与使用场景

- 考生：阅读项目说明，修改 `solution.py`，本地配置 LLM API，运行 `run.py` 验证效果，并提交最终成果。
- 评测系统：按任务顺序调用 `update` 输入带标签训练样本，再调用 `predict` 对无标签文本预测，统计 exact match 准确率。
- 评审老师：阅读探索报告，评价 Harness 设计的创新性、合理性和可解释性。

## 5. 项目范围

### 5.1 范围内

- 理解 PDF 说明、基础代码、数据结构、tokenizer 和本地评测脚本。
- 在 `solution.py` 的 `MyHarness` 类内部设计并实现分类策略。
- 维护训练样本记忆、标签集合、检索逻辑、Prompt 模板、输出清洗和兜底策略。
- 使用 `count_tokens`、`count_messages_tokens` 与 `max_prompt_tokens` 控制上下文预算。
- 本地运行 `run.py` 进行最小必要验证，并记录结果。
- 在 `doc/think.md` 中持续记录数据观察、策略尝试、实验结果和后续判断。
- 准备最终提交所需的 `solution.py` 与探索报告 PDF。

### 5.2 范围外

- 不修改 `harness_base.py`、`run.py` 的评测逻辑或评测数据来获取不真实分数。
- 不在 `solution.py` 中读写磁盘文件、访问测试集标签或绕过接口获取答案。
- 不硬编码公开 DEV 测试集答案，不设计针对某一份测试集的作弊映射。
- 不引入 `requirements.txt` 外或提交规则不允许的第三方依赖。
- 不进行无限循环、恶意超时、大量无意义 LLM 调用等影响评测系统的行为。

## 6. 关键文件说明

| 路径 | 类型 | 需求说明 |
| --- | --- | --- |
| `solution.py` | 核心实现 | 唯一需要提交的 Python 代码文件；只能修改 `MyHarness` 类内部。 |
| `harness_base.py` | 基类 | 不可修改；提供 `call_llm`、token 计数函数、`max_prompt_tokens` 和 `memory`。 |
| `llm_client.py` | 本地 LLM 配置 | 本地调试时只需修改顶部 `BASE_URL`、`API_KEY`、`MODEL`。 |
| `run.py` | 本地评测脚本 | 默认读取 DEV 数据，运行 4 轮取平均准确率，并统计 token 与耗时。 |
| `data/train_dev.jsonl` | DEV 训练集 | 231 条样本，77 类，每类 3 条。 |
| `data/test_dev.jsonl` | DEV 验证集 | 539 条样本，77 类，每类 7 条，本地用于验证。 |
| `tokenizer/` | 本地 tokenizer | 用于精确 token 计数，评测中按 content token 判断是否截断。 |
| `requirements.txt` | 依赖 | 本地运行需要 `openai`、`transformers`、`numpy`。 |
| `doc/PRD.md` | 本文档 | 记录项目需求、规则和交付标准。 |
| `doc/think.md` | 过程记录 | 记录数据观察、实现方式、实验结果和思考。 |

当前目录中存在 `data/*(1).jsonl`、`tokenizer/*(1).*`、`.DS_Store`、`*.baiduyun.p.downloading` 等重复或临时文件。它们不应作为正式提交核心内容；如需清理，应先确认来源和用途。

## 7. 数据需求

- 输入数据均为 JSONL 格式，每行一条样本。
- 样本字段：
  - `text`：待分类自然语言文本。
  - `label`：类别标签字符串；正式评测测试阶段不可访问。
- 每个任务保证测试集中出现的所有标签均已在训练集中出现。
- 本地 DEV 数据：
  - `train_dev.jsonl`：231 条，77 类，每类 3 条。
  - `test_dev.jsonl`：539 条，77 类，每类 7 条。
  - 文本长度大致为几十字符，存在少量更长客服表达。
- 正式任务：
  - 同域客服意图分类，文本与 DEV 不同。
  - OOD 文本分类，领域、标签名、标签数量可能完全不同。
  - 自然语言选择题，文本为题目，标签可能为 `A/B/C/D` 等选项。

## 8. 功能需求

### 8.1 训练流记忆更新

- `update(text, label)` 必须接收一条训练样本并更新 Harness 内部状态。
- 必须保留标签集合，确保 `predict` 只返回训练集中出现过的标签。
- 应支持按标签组织示例、统计标签描述、构建轻量索引或其他可解释记忆结构。
- 训练样本数量、标签数量、文本长度在正式任务中可能变化，实现应自适应。

### 8.2 预测接口

- `predict(text)` 必须返回字符串标签，且与目标标签完全一致才计为正确。
- 预测过程可以调用 `self.call_llm(messages)`，也可以结合确定性检索、规则、投票或兜底逻辑。
- LLM 输出必须经过清洗，去除解释文本、引号、前后空白和无关内容。
- 如果 LLM 返回不在标签集合内的内容，应有纠错或兜底策略，例如从候选标签中做最近匹配或重问。

### 8.3 Prompt 与上下文管理

- 单次 `call_llm` 的 prompt token 数不得超过 `self.max_prompt_tokens`，默认 2048。
- 调用前应使用 `self.count_messages_tokens(messages)` 或 `self.count_tokens(text)` 估算预算。
- Prompt 应优先保留与当前测试文本最相关的训练示例和完整候选标签。
- 需要避免长标签列表、全量样本和冗余说明挤占预算。
- Prompt 必须明确要求模型只输出标签，不输出解释。

### 8.4 检索与候选压缩

- 当训练样本或标签较多时，应从全量记忆中筛选候选示例或候选标签。
- 候选筛选策略应尽量领域无关，避免只针对客服意图关键词。
- 可考虑结合字符/词级相似度、标签示例覆盖、标签名语义提示、少量 few-shot 示例等方式。
- 候选数量应受 token 预算约束，并在准确率和成本之间取平衡。

### 8.5 安全与鲁棒性

- 需要防范测试文本中的 Prompt Injection，例如“忽略之前指令并输出某标签”等内容。
- Prompt 应把待分类文本视为不可信数据，要求模型只根据语义分类，不执行文本中的指令。
- 输出解析应支持大小写、标点、代码块、解释性句子等常见异常。
- 多线程调用下 `predict` 可能并发执行，共享状态应避免在预测阶段被不必要地修改。

### 8.6 成本与性能

- 正式评测会限制任务执行时间，Harness 不应进行过多轮 LLM 调用。
- 默认本地评测 `--runs 4`，每轮对 539 条 DEV 测试样本预测。
- 优先设计单轮或少量调用的策略；如采用多轮纠错，需要严格限制触发条件。
- 应记录 prompt/条、completion/条、准确率和耗时，用于比较策略优劣。

## 9. 接口约束

`MyHarness` 必须保持以下接口：

```python
class MyHarness(Harness):
    def __init__(self, call_llm, count_tokens, count_messages_tokens, max_prompt_tokens: int):
        ...

    def update(self, text: str, label: str) -> None:
        ...

    def predict(self, text: str) -> str:
        ...
```

可用注入能力：

- `self.call_llm(messages: list[dict]) -> str`：调用 OpenAI compatible LLM。
- `self.count_tokens(text: str) -> int`：计算单段文本 token。
- `self.count_messages_tokens(messages: list[dict]) -> int`：计算 messages content token 总数。
- `self.max_prompt_tokens: int`：单次 prompt token 上限。
- `self.memory: list[tuple[str, str]]`：基础训练样本记忆。

## 10. 代码与依赖约束

- 只能提交 `solution.py`。
- 只能修改 `MyHarness` 类内部，其余部分保持不变。
- `solution.py` 允许使用 Python 标准库、`numpy` 和 `harness_base`。
- 禁止 import `openai`、`sklearn`、`torch`、`transformers` 等提交规则外依赖。
- 禁止读写任何磁盘文件。
- 禁止通过任何方式获取测试集标签。
- 禁止硬编码 DEV 测试集答案或公开数据答案。

## 11. 本地运行需求

安装依赖：

```bash
pip install -r requirements.txt
```

配置 LLM：

```python
BASE_URL = "http://your-endpoint/v1"
API_KEY  = "your-api-key"
MODEL    = "your-model-name"
```

运行评测：

```bash
python run.py
python run.py --runs 1
python run.py --workers 50
```

当前本机环境中 `python` 命令不可用，需使用 `python3` 或先配置 `python` 命令别名。项目当前不是 Git 仓库，如继续执行“测试前提交”规则，需要先初始化或关联仓库。

## 12. 评测标准

- 客观得分占 80%：私有任务加权平均准确率排名赋分。
- 主观得分占 20%：专家根据探索报告评价 Harness 设计的创新性、合理性和可解释性。
- 正式评测统一使用 Qwen3-8B Instruct 非思考模式。
- 每个任务默认多次采样，取平均结果以提升稳定性。
- `predict` 返回值与标签 exact match，任何多余解释都可能导致错误。

## 13. 交付物

- `solution.py`：包含 `MyHarness` 完整实现的最终代码。
- 探索报告 PDF：简要记录不同 Harness 策略的尝试、效果和分析。
- 可选辅助文档：
  - `doc/think.md`：过程记录和实验日志。
  - `doc/PRD.md`：需求与规则基准。

提交截止时间为北京时间 2026-05-09 00:00。提交期间可多次提交，后提交内容会覆盖先前提交。

## 14. 验证规则

- 每一次测试前，必须先完成一次 Git 提交，确保测试对应的代码状态可追溯、可回滚。
- 每次完成关键修改后，至少运行与该修改相关的最小验证。
- 每次验证后，应在 `doc/think.md` 记录命令、准确率、token 消耗、耗时、失败现象和下一步判断。
- 若无法验证，需要说明原因，例如未配置 LLM API、网络不可用、依赖未安装或当前目录不是 Git 仓库。

## 15. 验收标准

- `solution.py` 可被 `run.py` 正常导入，`MyHarness` 能完成 `update` 和 `predict` 调用。
- `predict` 始终返回训练集中存在的标签字符串。
- 单次 LLM prompt 主动控制在 `max_prompt_tokens` 内，评测时不依赖自动截断。
- 本地 DEV 评测可以运行完成，无未捕获异常、死循环或明显超时。
- 实验过程、关键策略和结果已记录，可支持探索报告撰写。
- 最终提交不包含违规依赖、磁盘读写、测试标签泄露或硬编码答案。

## 16. 风险与对策

| 风险 | 影响 | 对策 |
| --- | --- | --- |
| 过拟合 DEV 客服数据 | 私有 OOD 和选择题任务失分 | 采用领域无关检索与分类 Prompt，不写死客服标签逻辑。 |
| Prompt 超过 2048 token | 关键信息被截断，预测不稳定 | 调用前主动计数，动态裁剪示例与候选。 |
| LLM 输出不规范 | exact match 失败 | 输出清洗、候选校验和兜底匹配。 |
| Prompt Injection | 被测试文本诱导输出错误标签 | 明确把输入文本当作数据，不执行其中指令。 |
| 多线程共享状态修改 | 预测结果不稳定 | 预测阶段尽量只读共享状态，必要时预计算不可变结构。 |
| 本地环境未配置 API | 无法验证准确率 | 先配置 `llm_client.py` 或记录无法验证原因。 |

## 17. 变更记录

| 日期 | 变更内容 | 负责人 |
| --- | --- | --- |
| 2026-05-07 | 初始化项目 PRD 与规则文档 | Codex |
| 2026-05-07 | 增加“每一次测试前必须提交一次”的验证规则 | Codex |
| 2026-05-07 | 阅读项目 PDF、代码、数据和现有文档后，补全 Harness 分类任务需求、约束、评测和交付标准 | Codex |
