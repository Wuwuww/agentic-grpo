# agentic-grpo

值班排障代理上的 GRPO。采样可以打到 OpenAI 兼容的大模型 API；被更新的不是模型权重，而是写进系统提示里的策略参数。算法对齐 [verl](https://github.com/volcengine/verl) 的 GRPO，不复制它的 FSDP / vLLM 训练。

## 场景

告警进来之后，代理只能通过 `query_logs`、`query_metrics`、`query_changes` 看证据，再输出一个根因 id。奖励是规则算的：根因对不对、有没有查到关键证据、有没有乱查。同一条告警采样一组轨迹，优势只在组内比较，没有 critic。

这不是数学题 GRPO 的再打包。轨迹是多步工具调用，奖励看的是取证是否够用。

## 和 verl 的对应关系

| verl | 这里 |
| --- | --- |
| `algorithm.adv_estimator=grpo` | 组内减均值，默认再除无偏标准差；组大小为 1 时均值 0、标准差 1 |
| `actor_rollout_ref.rollout.n` | 每条告警的采样数，必须 ≥ 2 |
| clipped policy loss | 序列级 PPO clip，`clip_ratio_c` 处理负优势 |
| `kl_loss_type=low_var_kl` | Schulman k3，相对初始策略参数 |
| `actor.optim` | 只作用在 `investigate` / `commit` |
| Megatron / FSDP 权重更新 | 明确跳过，日志里是 `llm_weights=skipped` |

`log π` 不是大模型的 token logprob，而是策略参数对轨迹特征的线性分数。GRPO 的梯度对这两个参数是真的；对 API 后面的模型权重是假的。下一步采样会把新参数写进系统提示，所以行为会变。

## 运行

没有 API key 时用脚本后端，它按当前 `investigate` 决定查证还是直接猜：

```bash
cd /home/ubuntu/cursor_workspace/agentic-grpo
PYTHONPATH=src python -m agentic_grpo --backend scripted
```

有密钥时，采样改走真实接口，参数更新方式不变：

```bash
export LLM_API_KEY=sk-...
export LLM_API_BASE=https://api.openai.com/v1   # 也可换成 DeepSeek / 通义兼容地址
PYTHONPATH=src python -m agentic_grpo --backend api
```

配置在 `configs/oncall_grpo.json`。指标写到 `runs/oncall/metrics.jsonl`，参数写到 `runs/oncall/policy_params.json`。

## Search-R1

同一套 GRPO 上加了 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 的三件事，场景换成值班手册问答：

- 模型自己决定何时输出 `<search>`。检索引擎把手册片段包进 `<information>` 再交回模型，直到 `<answer>`。
- 奖励只有答案的精确匹配。格式对不对、检索了几次，都不加分。
- `<information>` 里的 token 不进入策略损失和 KL。不掩码时，梯度会跟着段落长度走。

被更新的仍是提示里的 `search_rate`，不是模型权重。题面里已经写明答案的那一题不需要检索；另外四题的答案只在手册里。脚本后端会在每组里留一个样本去检索，这样组内优势一开始不为零。

```bash
PYTHONPATH=src python -m agentic_grpo.search --backend scripted
```

配置在 `configs/search_r1_grpo.json`，对应 Search-R1 的 `n_agent=5`、`state_masking=true`、`kl_loss_coef=0.001`。有密钥时把 `--backend` 换成 `api`，采样走真实接口，权重更新仍然跳过。

脚本后端 8 个 epoch 的结果：精确匹配从 0.36 到 1.00，`search_rate` 从 0.05 到约 0.98。题面里已经写明答案的那题始终不检索，所以平均检索次数停在 0.80。匹配率到 1.00 之后，相对初始策略的 KL 会把 `search_rate` 轻轻拉回。

```bash
PYTHONPATH=src python -m pytest -q
```

## 不要这样讲

不要说这是用 verl 或 Search-R1 训了一个 Qwen。权重没有回传。脚本后端是为了在没有密钥时把优势、clip 和参数步进跑通；它知道标准答案，只用来检查学习信号，不是排障能力本身。
