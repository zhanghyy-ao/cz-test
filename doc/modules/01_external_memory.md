# 模块 01：外部记忆管理

_负责在 `update(text, label)` 阶段构建可检索、可压缩、可解释的任务记忆。_

---

## 功能边界

外部记忆管理模块接收训练流中的有标签样本，并将它们组织成后续 `predict` 可直接使用的结构。它不负责调用 LLM，也不负责最终分类；它的目标是让检索模块能快速获得标签、样例、标签原型和混淆信息。

| 输入 | 输出 | 不负责 |
| --- | --- | --- |
| `text`, `label` | 标签集合、按标签样例、轻量索引、统计信息 | LLM 调用、Prompt 拼接、测试标签读取 |

## 核心功能

| 功能 | 说明 | 实现建议 |
| --- | --- | --- |
| 标签注册 | 维护所有合法标签 | `self.labels` 保留插入顺序，`self.label_set` 用于 O(1) 判断 |
| 样例保存 | 保存训练样本原文和标签 | `self.example_memory: list[tuple[str, str]]` |
| 按标签分组 | 支持按标签检索示例 | `self.by_label[label].append(text)` |
| 标签原型 | 保存标签的代表性词面特征 | 字符 n-gram、词集合、长度统计 |
| 混淆缓存 | 保存容易混淆的标签对 | 本地验证或 verifier 反馈后更新 |

## 推荐数据结构

```python
self.labels = []
self.label_set = set()
self.by_label = {}
self.example_memory = []
self.label_profiles = {}
self.confusions = {}
```

`label_profiles` 可以采用不依赖第三方库的轻量结构：

```python
self.label_profiles[label] = {
    "char_ngrams": Counter(),
    "tokens": Counter(),
    "lengths": [],
    "examples": []
}
```

## 技术方案

### 字符 n-gram 记忆

中文客服意图、英文分类、自然语言选择题都可能出现，因此字符 n-gram 比单纯空格分词更稳。建议使用 `2-4` gram，并过滤空白。

```python
def char_ngrams(text, ns=(2, 3, 4)):
    s = "".join(text.split())
    grams = []
    for n in ns:
        grams.extend(s[i:i+n] for i in range(max(0, len(s) - n + 1)))
    return grams
```

### 标签原型

每个标签只有少量样例时，原型不应写成长摘要，而应保存可计算的特征。候选检索时可将当前文本与每个标签下的所有样例比对，取最高分、平均分和覆盖分。

| 原型字段 | 用途 |
| --- | --- |
| `char_ngrams` | 跨语言相似度 |
| `tokens` | 英文、选项题、标签名匹配 |
| `examples` | few-shot Prompt 来源 |
| `lengths` | 判断输入是否异常短/长 |

### 记忆写入门控

如果未来在本地验证中收集错误样本，不要把所有预测过程都写入记忆。只写入以下内容：

- 预测错误且两个标签高相似
- LLM 输出非法标签
- 低置信度并触发 verifier
- 某个标签长期被误判为另一个标签

## 与 `solution.py` 的接口

```python
def update(self, text: str, label: str) -> None:
    if label not in self.label_set:
        self.label_set.add(label)
        self.labels.append(label)
        self.by_label[label] = []
        self.label_profiles[label] = new_profile()

    self.example_memory.append((text, label))
    self.by_label[label].append(text)
    update_profile(self.label_profiles[label], text)
```

## 预算影响

外部记忆可以很大，但进入 Prompt 的只能是精选视图。该模块应保存全量训练样本，但不应默认把全量样本交给 Prompt 组装器。

| 记忆内容 | 是否进入 Prompt | 原因 |
| --- | --- | --- |
| 全量样例 | 否 | 会超预算并稀释注意力 |
| 候选标签名 | 是 | 保证输出空间受控 |
| 相似 top-k 样例 | 是 | 提供任务内判别依据 |
| 标签统计原型 | 通常否 | 用于检索打分即可 |
| 混淆规则 | 视预算进入 | 可提升边界判断 |

## 风险与对策

| 风险 | 表现 | 对策 |
| --- | --- | --- |
| 记忆污染 | 错误经验反复进入 Prompt | 只在有可靠反馈时写入混淆规则 |
| 过拟合 DEV | 针对客服标签写死规则 | 使用领域无关特征和通用检索 |
| 并发修改 | 多线程预测时修改共享状态 | `predict` 默认只读记忆，反馈写入谨慎使用 |

## 参考内容

| 资料 | 价值 | 链接 |
| --- | --- | --- |
| MemGPT | 将上下文窗口视作主存，外部存储视作可分页长期记忆 | https://arxiv.org/abs/2310.08560 |
| ACE | 将外部经验整理成可演化 playbook | https://arxiv.org/abs/2510.04618 |
| Dynamic Cheatsheet | 测试时维护短 cheatsheet 改善黑盒 LLM 表现 | https://github.com/suzgunmirac/dynamic-cheatsheet |
| Meta-Harness | Harness 决定存储、检索和呈现哪些信息 | https://arxiv.org/abs/2603.28052 |

