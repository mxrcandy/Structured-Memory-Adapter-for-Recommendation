# LETTER Amazon Sports 正式训练数据

## 实施方案

复用 `prepare_rec_amazon_data.py` 的 Beauty 流程：Stage 1 全量商品文本，Stage 2 每用户 S[:-1]，SFT 用 S[:-2] 预测 S[-2]，测试用 S[:-1] 预测 S[-1]。无额外验证集、滑窗、混合通用数据或 domain token。输入取已完成的 LETTER JSON；不改原始文件。

Toys 沿用 Yelp 的 max_history_items=300（仅 1 位用户超限）；Sports 不截断历史（最长 295）。Stage 1 超长文本按现有规则裁剪尾部，所有训练模板和 benchmark prompt 加生成四层 SID 必须实测不超过 2048 tokens。seed=42，分片每片至多 1000 行。

已完成的接入步骤：

1. 给现有 builder 增加 Toys / Sports CLI 选项，添加历史截断与任务注册测试。
2. 生成 `output_amazon_letter_toys` 和 `output_amazon_letter_sports`。
3. 注册 `DATA_PROFILE=amazon_letter_toys/amazon_letter_sports` 的三个训练阶段，以及 `amazon_toys/amazon_sports` 测试任务、E2E 评测路由和评测目录。
4. 全量校验输入哈希、划分/目标/无答案泄露、SID/PID 映射、parquet schema、分片覆盖和 2048 长度预算；检查真实 loader/evaluator 可加载，运行相关回归测试。仅构建和注册，不启动模型训练或 GPU 评测，不修改依赖。

## 构建命令

从项目根目录，在现有 openonerec 环境执行：

```bash
python process_data/prepare_rec_amazon_data.py --dataset Sports
```

生成器拒绝覆盖非空目录。构建统计见本目录 `summary.json`；全量审核结果见 `validation_report.json` 和 `benchmark_validation.json`。

## 完成结果（2026-10-10）

处理逻辑和输出 schema 复用 Beauty，SID 为四层各 256，加 sid_begin/sid_end。PID=原 item ID+1，SID 映射保留碰撞，候选按保留的训练历史中出现次数降序、PID 升序排序，评测使用 most_popular_originally。不使用测试末次事件计算频次。Toys 截断作用于 Stage2/benchmark 的 S[:-1] 和 SFT 的 S[:-2] 各自的最近 300 项，目标事件保持不变。

| 数据集 | Stage1 | Stage2 | SFT | 测试 | 分片数 Stage1/Stage2/SFT |
| --- | ---: | ---: | ---: | ---: | --- |
| Sports | 18,357 | 35,598 | 35,598 | 35,598 | 19 / 36 / 36 |

| 数据集 | Stage1 最大 tokens | Stage2 | SFT | SFT 自定义模板 | 测试含生成预算 | 历史截断用户 | 商品文本截断 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sports | 2048 | 1771 | 1826 | 1829 | 1831 | 0 | 14 |

summary 的 training_interactions / history_items 沿用 Beauty 实现，统计的是原始 S[:-1]（截断前），不是实际保留事件数。Toys 原 148,185 个训练历史事件，截断 249 个，保留 147,936；Sports 保留 260,739 个。intermediate/items.parquet 保存完整商品文本，即使 Stage1 文本被裁剪也可追溯。

## 已注册入口

| 数据集 | 训练 DATA_PROFILE | 测试任务 TASK_TYPES |
| --- | --- | --- |
| Sports | amazon_letter_sports | amazon_sports |

训练 profile 在 `pretrain/scripts/data_profiles.sh` 注册，分别指向三套独立 file_list；默认 MAX_LENGTH=2048、Qwen3 SID token 范围 151669–152694。`e2e_rec_eval.sh` 已接入两套评测路由，`benchmarks/eval_amazon.sh` 根据 profile/task 自动选择对应 benchmark_data。测试任务在 benchmark registry/config 注册，默认全量用户、生成四个 SID code tokens。

本次只处理数据并注册，没有启动训练、提交调度任务或运行 GPU 推理。训练使用已扩展四层×256词表的模型；测试环境仍是现有 openonerec_benchmark。

```bash
# 按现有训练入口选择 profile（实际 MODEL_DIR 等参数按实验设置）
DATA_PROFILE=amazon_letter_sports MODEL_DIR=/path/to/expanded/model bash pretrain/scripts/train_stage1_itemic.sh
# 评测已训练导出的模型
DATA_PROFILE=amazon_letter_sports MODEL_PATH=/path/to/converted bash benchmarks/eval_amazon.sh
```

## 审核复跑

```bash
conda activate openonerec
PYTHONDONTWRITEBYTECODE=1 python process_data/verify_amazon_letter.py --dataset Sports
PYTHONDONTWRITEBYTECODE=1 python -m unittest process_data.tests.test_amazon_data pretrain.tests.test_amazon_eval_submission -v
conda activate openonerec_benchmark
PYTHONDONTWRITEBYTECODE=1 python process_data/tests/verify_amazon_output.py --dataset Sports
```

完整产物审核检查源哈希、Beauty schema、逐用户目标/历史、测试 messages 无 assistant 答案、训练频次映射、分片无重复遗漏。已有独立全量审核脚本已扩展 Toys/Sports 和显式历史截断，使用真实 tokenizer 重测全部长度，使用真实 loader/evaluator 做 CPU 集成检查。

用真实答案 SID 模拟预测时两套 SID pass@1 均为 1；PID pass@1 Toys=0.995106、Sports=0.995955。差异来自 SID 碰撞及确定性的热门 PID 映射，不能当作模型推荐成绩。详细记录：各输出目录 summary.json、validation_report.json、benchmark_validation.json。
