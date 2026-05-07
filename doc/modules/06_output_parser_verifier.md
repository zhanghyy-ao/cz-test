# 模块 06：输出解析与验证

_负责把 LLM 原始输出变成训练集中出现过的合法标签，并在异常时触发兜底或二次验证。_

---

## 功能边界

输出解析模块是 exact match 分类任务的最后防线。它不负责决定 Prompt 内容，但必须保证 `predict` 返回值是合法标签字符串。

| 输入 | 输出 |
| --- | --- |
| LLM 原始输出、候选标签、全量合法标签、检索分数 | 最终标签 |

## 核心功能

| 功能 | 说明 |
| --- | --- |
| 清洗输出 | 去除空白、引号、代码块、解释前缀 |
| 标签匹配 | 精确匹配候选标签或全量标签 |
| JSON 解析 | 若模型输出 JSON，抽取 `label` |
| 近似映射 | 输出稍有变体时映射回合法标签 |
| 异常兜底 | 输出不可用时返回检索最高分标签 |
| 二次验证 | 必要时让 LLM 在更小候选内重选 |

## 解析顺序

推荐按从严格到宽松的顺序解析：

1. 原始输出去空白后是否等于某个候选标签
2. 去除引号、句号、代码块后是否等于某个候选标签
3. JSON 中的 `label` 是否为候选标签
4. 输出文本是否包含唯一候选标签
5. 输出是否匹配全量合法标签
6. 使用 `difflib.get_close_matches` 做近似匹配
7. 返回检索分数最高的候选标签

## 技术方案

```python
def parse_label(raw, candidates, all_labels, fallback):
    text = normalize(raw)
    if text in candidates:
        return text

    label = try_parse_json_label(text)
    if label in candidates:
        return label

    contained = [x for x in candidates if x in text]
    if len(contained) == 1:
        return contained[0]

    close = difflib.get_close_matches(text, candidates, n=1, cutoff=0.75)
    if close:
        return close[0]

    return fallback
```

## Verifier 触发条件

| 条件 | 是否建议触发 |
| --- | --- |
| 输出非法且解析失败 | 是 |
| top1/top2 检索分数极接近 | 可选 |
| 模型输出多个候选标签 | 是 |
| 候选标签数很少且输入复杂 | 可选 |
| 默认每条样本都 verifier | 否 |

## Verifier Prompt

```text
下面是同一文本的候选标签，请只选择一个最合适的标签。
只能输出标签，不要解释。

候选标签：
1. ...
2. ...

文本：
<<<
...
>>>
```

Verifier 应只接收 top 3-5 候选和少量关键示例，避免二次调用也超预算。

## 兜底策略

| 兜底 | 适用 | 说明 |
| --- | --- | --- |
| 检索 top1 | 默认 | 不依赖模型输出，稳定 |
| 全量最近训练样例标签 | 文本高度相似 | 对 DEV 近似表达有效 |
| 标签名近似 | 选择题或标签语义明确 | 对 `A/B/C/D`、短标签有效 |
| 多数类 | 最后兜底 | 仅在无候选时使用 |

## 风险与对策

| 风险 | 表现 | 对策 |
| --- | --- | --- |
| 包含匹配误判 | 输出解释中提到多个标签 | 只有唯一候选出现时才用包含匹配 |
| 近似匹配过度 | 错误映射到相似标签 | 设置较高 cutoff，优先候选内匹配 |
| verifier 成本过高 | 评测慢 | 只在异常和低置信度触发 |

## 参考内容

| 资料 | 价值 | 链接 |
| --- | --- | --- |
| OpenAI Evals | LLM 系统输出校验和评测思路 | https://github.com/openai/evals |
| lm-evaluation-harness | 分类评测需要稳定任务格式和输出处理 | https://github.com/EleutherAI/lm-evaluation-harness |
| Promptfoo | 可对非法输出率做回归测试 | https://www.promptfoo.dev/docs/guides/ |
| Self-Consistency | 可作为少量困难样本的多输出验证参考 | https://arxiv.org/abs/2203.11171 |

