# 模块 02：候选检索与压缩

_负责从全量标签和训练样例中筛出最可能有用的一小部分上下文。_

---

## 功能边界

候选检索模块在 `predict(text)` 中运行。它读取外部记忆，输出候选标签、相似正例、hard negatives 和可选混淆规则。它不直接调用 LLM，也不负责 token 预算的最终裁剪。

| 输入 | 输出 |
| --- | --- |
| 当前文本、标签集合、样例记忆、标签原型 | `candidate_labels`、`selected_examples`、`hard_negatives`、`rules` |

## 核心功能

| 功能 | 说明 |
| --- | --- |
| 候选标签召回 | 从所有标签中选择 top-k |
| 相似样例选择 | 从候选标签下选择 few-shot 样例 |
| Hard negative 选择 | 选择相似但标签不同的样例，帮助模型区分边界 |
| 选择题识别 | 若标签像 `A/B/C/D`，优先保留所有选项标签 |
| 去重与多样性 | 避免 Prompt 被同类重复样例占满 |

## 技术方案

### 组合打分

推荐使用多个轻量信号组合，不依赖 embedding：

```text
label_score =
  0.55 * max_example_similarity
+ 0.20 * avg_top_example_similarity
+ 0.15 * label_name_similarity
+ 0.10 * token_overlap
```

其中 `example_similarity` 可由字符 n-gram Jaccard、`difflib.SequenceMatcher` 和词重叠加权得到。

### 相似度函数

| 方法 | 适用 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 字符 n-gram Jaccard | 中文、短文本、客服表达 | 无依赖，跨语言稳 | 对长文本局部匹配不够敏感 |
| 词重叠 | 英文、选择题、标签名 | 简单可解释 | 中文无空格时较弱 |
| `difflib.SequenceMatcher` | 短文本近似匹配 | 标准库可用 | 大规模样例时较慢 |
| 简化 TF-IDF | OOD 文本、英文长文 | 比纯重叠更稳 | 代码复杂度更高 |

### 候选数量

| 场景 | 推荐候选标签数 |
| --- | ---: |
| 标签数小于等于 8 | 全量保留 |
| 选择题 `A/B/C/D` | 全量保留 |
| 标签数 9-30 | top 8-12 |
| 标签数 31-100 | top 12-20 |
| 标签数超过 100 | top 16-24，并减少每类示例数 |

### 示例选择

优先选每个高分候选标签下最相似的 1 个样例，再用剩余预算补充 top 全局样例。这样可以避免一个标签占满 few-shot 区域。

```python
def select_examples(text, candidates):
    selected = []
    for label in candidates:
        best = best_example_for_label(text, label)
        if best:
            selected.append(best)
    selected.extend(global_top_examples(text, candidates))
    return dedupe(selected)
```

## Hard negatives

Hard negative 是“看起来像当前文本，但标签不同”的样例。它们对多意图客服分类、近义标签和选择题尤其有用。

| 选择条件 | 说明 |
| --- | --- |
| 与当前文本相似度高 | 模型容易混淆 |
| 标签不同 | 明确展示边界 |
| 候选标签排名靠前 | 避免引入无关标签 |
| 文本短而典型 | 减少 token 成本 |

## 与其他模块的接口

```python
retrieval_result = {
    "candidate_labels": list[str],
    "positive_examples": list[tuple[str, str, float]],
    "hard_negatives": list[tuple[str, str, float]],
    "rules": list[str]
}
```

## 预算影响

检索模块应返回“按优先级排序”的候选内容，让预算控制器可以从低优先级开始删。

| 内容 | 优先级 | 说明 |
| --- | ---: | --- |
| 候选标签 | 最高 | 没有标签空间就无法保证 exact match |
| 每个 top 标签的最佳样例 | 高 | 提供标签语义 |
| hard negatives | 中 | 提升混淆类边界 |
| 额外相似样例 | 中低 | 预算足够时补充 |
| 长规则和长样例 | 低 | 优先裁剪 |

## 风险与对策

| 风险 | 表现 | 对策 |
| --- | --- | --- |
| 召回漏掉真标签 | LLM 无法选择正确答案 | 候选数不要过小，保留确定性兜底 |
| 相似度偏词面 | 同义表达召回失败 | 结合标签名、样例最高分和平均分 |
| Prompt 示例单一 | 模型偏向某一标签 | 每个候选先选一个代表样例 |

## 参考内容

| 资料 | 价值 | 链接 |
| --- | --- | --- |
| In-Context Learning for Text Classification with Many Labels | 大标签集下先检索候选标签，缓解上下文限制 | https://arxiv.org/abs/2309.10954 |
| Active Example Selection for In-Context Learning | 示例选择会显著影响 ICL 效果 | https://github.com/ChicagoHAI/active-example-selection |
| Retrieval-augmented Multi-label Text Classification | 检索增强可改善低资源和长尾分类 | https://arxiv.org/abs/2305.13058 |
| KATE | 用近邻样例做 in-context example selection | https://arxiv.org/abs/2209.11684 |

