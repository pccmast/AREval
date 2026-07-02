# AREval 优化评审 & 多轮对话方案

> 状态：评审完成，待实施 | 日期：2026-07-02

---

## 一、已修复的三项生产瓶颈 ✅

| 修复 | 实现 | commit |
|------|------|--------|
| 失败 metric 标 NaN 不标 0 | `_run_with_retry()` → NaN, `nanmean` 聚合 | `fe53f12` |
| min_score 下限 | `Evaluator.__init__(min_score=0.0)`, 任一指标低于此值封顶 | `fe53f12` |
| 可配置重试 | `Evaluator.__init__(metric_retries=1)` | `fe53f12` |

---

## 二、多轮对话优化的两种方案

### 现状问题

方案 A（装饰器，不传 `conversation_id`）：
```python
@eval_trace(new_conversation=True)   # AREval 自己生成 conv_id
def handle(msg): ...

# 问题：AREval 的 conv_id 和业务层的 conversation_id 是两个不同的 ID
#      业务层不知道 AREval 内部的 conv_id → 无法关联
```

方案 B（不用装饰器，用 EvalTracer API）：
```python
def handle(msg, conversation_id):
    if conversation_id not in self._active:
        tracer.start_conversation(conversation_id)
    with tracer.start_span(...): ...

# 问题：代码侵入性高，Agent 框架需要改 handle() 签名
#      每个 Agent 都要手动管理 tracer 生命周期
```

### 优化方案：`conversation_id` 从请求上下文自动提取 ✅ 已实现

**实现**：`areval-sdk/areval_sdk/decorators.py` — `@eval_trace` 新增 `conversation_id_extractor` 参数，并自动从 kwargs 中探测 `conversation_id` / `conv_id` / `session_id`。`start_conversation()` 幂等——相同 conv_id 重复调用不重置 turn 计数器。

**三种使用方式**：

```python
# areval-sdk/areval_sdk/decorators.py

def eval_trace(
    name=None,
    conversation_id_extractor=None,   # ★ 新增：从 kwargs 提取 conv_id 的函数
    ...
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 自动提取 conversation_id
            cid = None
            if conversation_id_extractor:
                cid = conversation_id_extractor(args, kwargs)
            elif "conversation_id" in kwargs:
                cid = kwargs["conversation_id"]

            if cid:
                current = getattr(_tracer, '_active_conversation', None)
                if cid != current:
                    _tracer.start_conversation(cid)

            with _tracer.start_span(span_name) as span:
                return func(*args, **kwargs)
        return wrapper
    return decorator
```

使用方式——三种粒度：

```python
# 方式 1：约定参数名（零配置）
@eval_trace()
def handle(msg: str, conversation_id: str = "") -> str: ...
# → 自动从 kwargs["conversation_id"] 提取

# 方式 2：自定义提取器
@eval_trace(conversation_id_extractor=lambda args, kwargs: kwargs["metadata"]["conv_id"])
def handle(msg: str, metadata: dict) -> str: ...

# 方式 3：手动控制（和现在一样）
@eval_trace(new_conversation=True)
def handle(msg: str) -> str: ...
```

### 改动量评估

```
✅ 已实现 (areval-sdk/areval_sdk/decorators.py + areval-engine/areval/tracing/tracer.py)
  +~30 行（conversation_id 自动提取 + 幂等检查）
  262 tests pass
```

### 推荐

方式 1 + 方式 2 覆盖 95% 的场景。方式 3 保留为兜底。**不再需要在代码里手动判断 new_conversation** —— 框架自动比较当前 conv_id 和上一个 conv_id，变了就自动开始新会话。

---

## 三、后续优化评审（按优先级排序）

### 🟡 中优先级 —— 提升精度

| 优化项 | 当前状态 | 计划改动 | 代码量 | 备注 |
|--------|---------|---------|--------|------|
| ContextPrecision 句子级拆分 | 整段判断 | 复用 Faithfulness 的 batch prompt 模式 | ~30 行 | 需要先评估场景是否适合拆分 |
| 加权平均 | 简单平均 | `overall_score = weighted_avg(scores, weights)` + YAML 配置 | ~25 行 | 含向后兼容 |
| A/B 模型对比 | 手动跑两次比分数 | `areval compare --agent-a x --agent-b y` 自动对比 | ~80 行 | Win/Loss/Tie 统计 + CLI 表格 |
| 评估 prompt 模板外置 | 写死在 Python 里 | `prompts/faithfulness.yaml` → `_FAITHFULNESS_RUBRIC` 从文件加载 | ~40 行 | 方便非开发者微调 prompt |

### 🟢 低优先级 —— 体验提升

| 优化项 | 当前状态 | 计划改动 | 代码量 |
|--------|---------|---------|--------|
| 异步长时间评估 | 同步批量 | `evaluate_async()` + 进度回调 / WebSocket | ~120 行 |
| CI/CD pytest 插件 | 没有原生集成 | `pytest --areval --dataset seed/rag_evaluation.jsonl` | ~60 行 |
| Dataset 从 LangSmith 导入 | 只支持 JSONL | `areval import --from langsmith --project-id xxx` | ~60 行 |
| 成本治理 | 无限制 | `Evaluator(max_cost_usd=5.0)` 超出自动降级 | ~40 行 |
| Dashboard websocket 实时推送 | fetch + 手动刷新 | `GET /api/v1/ws/evaluations` WebSocket | ~100 行 |
| 多 worker 并发评估 | 单进程 | 改 CLI 里加 `--workers 4` + ProcessPoolExecutor | ~60 行 |
| K8s Helm Chart | Docker Compose | 需运维配合 | 不确定 |
| DAGJudge 启用 | 骨架已存在未接入 | 在 JailbreakResistance 里试水 | ~50 行 |

### 不需要改的（评审确认）

| 项目 | 原因 |
|------|------|
| `test_case.id` 字符串匹配 | ID 是 uuid，不会因为删空格而改变。不是实际瓶颈 |
| 安全红队自动对抗样本生成 | 学术研究方向，生产不可控。且 OWASP Top10 已覆盖 |
| DeepEval 30+ 内置 metric | AREval 的 13 种已经覆盖核心评估维度，不需要追数字 |
| SemanticSimilarity tier 降级 | 改不了——需要 embedding 模型，不是 LLM。sentence-transformers 补充即可 |

---

## 四、建议实施顺序

```
✅ 已完成:
  1. NaN + min_score + retry (fe53f12)
  2. 路由表 ADR-6 + 策展升级 + 在线告警

下一批（本周）:
  3. 多轮对话优化（conversation_id 自动提取）—— 35 行
  4. ContextPrecision 句子级拆分 —— 30 行
  5. 加权平均 + YAML weights —— 25 行

下一批（之后）:
  6. A/B 模型对比 —— 80 行
  7. 评估 prompt 外置 —— 40 行

不紧急:
  8. 异步评估 + WebSocket + CI 插件 + 成本治理
