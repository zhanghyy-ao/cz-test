# 模块 03：Token 预算控制

_负责把每次 LLM 调用的输入限制在 `self.max_prompt_tokens` 内，默认目标为 `2048` token。_

---

## 功能边界

预算控制模块不决定“哪个标签正确”，而是决定 Prompt 中每个区块能占多少 token，以及超预算时按什么顺序裁剪。它是本项目稳定性的硬约束模块。

| 输入 | 输出 |
| --- | --- |
| 当前文本、候选标签、示例、规则、Prompt 模板 | 不超预算的 `messages` 或可组装材料 |

## 核心功能

| 功能 | 说明 |
| --- | --- |
| 预算分配 | 给系统指令、候选标签、示例、输入文本分配 token |
| 调用前校验 | 使用 `self.count_messages_tokens(messages)` 确认不超限 |
| 逐级裁剪 | 超预算时删除低优先级内容 |
| 安全余量 | 预留 100-200 token，避免模板和 tokenizer 差异 |
| 日志指标 | 本地实验记录平均 prompt token |

## 推荐预算表

| Prompt 区块 | 默认 token | 说明 |
| --- | ---: | --- |
| 系统指令 | 120-180 | 包括任务、禁止执行输入指令、只输出标签 |
| 输出约束 | 40-80 | exact match、候选内选择 |
| 候选标签 | 150-400 | 标签多时只放 top-k |
| 相似正例 | 400-700 | few-shot 主要来源 |
| Hard negatives | 150-350 | 混淆类辅助 |
| 当前输入 | 300-700 | 长文本需截断或抽取证据 |
| 余量 | 100-200 | 防止边界溢出 |

## 裁剪优先级

1. 删除最低分额外示例
2. 删除较长 hard negative
3. 删除低优先级 cheatsheet 规则
4. 减少每个标签的示例数量
5. 缩短当前输入文本
6. 缩小候选标签数量，但不得低于安全下限

## 技术方案

### 硬预算循环

```python
def fit_messages(messages, state):
    while self.count_messages_tokens(messages) > self.max_prompt_tokens:
        if state["extra_examples"]:
            state["extra_examples"].pop()
        elif state["hard_negatives"]:
            state["hard_negatives"].pop()
        elif state["rules"]:
            state["rules"].pop()
        else:
            state["input_text"] = shorten_text(state["input_text"])
        messages = compose_messages(state)
    return messages
```

### 文本截断策略

| 输入类型 | 截断策略 |
| --- | --- |
| 短文本 | 不截断 |
| 中长文本 | 保留开头和结尾 |
| 带选项题 | 必须保留选项区域 |
| 长文档分类 | 抽取包含标签关键词、转折词、结论词的句子 |

### 预算状态

```python
budget_state = {
    "candidate_labels": candidates,
    "positive_examples": positives,
    "hard_negatives": negatives,
    "rules": rules,
    "input_text": text,
    "reserved": 150
}
```

## 与其他模块的接口

预算控制器最好接收结构化材料，而不是一段已经拼好的长字符串。这样超预算时能按模块裁剪。

```python
messages = self._compose_messages(prompt_state)
messages = self._fit_budget(messages, prompt_state)
```

## 风险与对策

| 风险 | 表现 | 对策 |
| --- | --- | --- |
| 只按字符数估算 | tokenizer 下仍超限 | 必须用 `count_messages_tokens` |
| 候选裁剪过度 | 真标签被删 | 设置候选数量下限，并保留兜底 |
| 当前输入被截坏 | 关键信息丢失 | 优先删示例，再截输入 |
| 预算逻辑复杂 | 代码难调试 | 统一用 `prompt_state` 管理各区块 |

## 参考内容

| 资料 | 价值 | 链接 |
| --- | --- | --- |
| LLMLingua | Prompt 压缩和预算控制思想 | https://github.com/microsoft/LLMLingua |
| LongLLMLingua | 长上下文信息压缩和重排 | https://arxiv.org/abs/2310.06839 |
| TALE | Token-budget-aware reasoning | https://arxiv.org/abs/2412.18547 |
| Budget Guidance | 通过预算控制推理长度 | https://arxiv.org/abs/2506.13752 |
| OpenAI tiktoken | token 计数工具参考 | https://github.com/openai/tiktoken |

