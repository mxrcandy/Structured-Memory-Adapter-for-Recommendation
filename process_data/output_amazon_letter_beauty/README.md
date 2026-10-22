# LETTER Amazon Beauty

构建入口：`process_data/prepare_rec_amazon_data.py`。运行环境为已有的 `openonerec`，没有安装依赖。

```bash
conda activate openonerec
python process_data/prepare_rec_amazon_data.py
```

默认输出 `process_data/output_amazon_letter_beauty/`。脚本拒绝覆盖非空目录；重新构建时使用新的 `--output_dir`。输入文件 SHA256、seed、长度分布和截断数量写入 `summary.json`。任务与 UUID 确定性生成，训练分片以 seed=42 打乱；benchmark 按数字 uid 排序。

## 划分与格式

每个用户原始序列为 `s`：

| 数据 | 内容 | 样本数 |
| --- | --- | ---: |
| Stage 1 | 全部商品 SID + title + description，segments 格式 | 12,101 |
| Stage 2 | `s[:-1]` 的纯 SID 序列，segments 格式 | 22,363 |
| SFT | 指令 + `s[:-2]`，assistant 输出 `s[-2]` | 22,363 |
| Benchmark | 指令 + `s[:-1]`，metadata 答案为 `s[-1]`，messages 无答案 | 22,363 |

没有滑动窗口、domain token、用户画像或通用数据混合。测试只保留每个用户最后一次交互；商品目录是全量的，测试商品的文本可用于 Stage 1。重复交互按时间位置划分，不额外删除训练历史中出现过的商品。

SID 使用 LETTER 的全部四层，映射为 `<|sid_begin|><s_a_i><s_b_j><s_c_k><s_d_l><|sid_end|>`，每层 256。保留原始碰撞：12,101 商品对应 12,088 个唯一 SID。无 domain token。

PID/IID = 原始 item_id + 1，因为现有 evaluator 将 0 当无效值。`intermediate/items.parquet` 同时保存 item_id、pid、sid、sid_codes 和完整原文，映射可逆。`intermediate/train_sequences.parquet` 保存训练序列及单独测试 target，不能直接当作训练输入。

`benchmark_data/sid2pid.json` 为实际 amazon_beauty evaluator 使用的映射；`sid2iid.json` 为相同映射的 iid 字段兼容版本。key 是四层 code 的 base-8192 整数编码，映射值为候选列表；count 和 count_after_downsample 都只统计 `s[:-1]`，未做 downsample。候选按训练次数降序、PID 升序排列；Beauty 使用 most_popular_originally，固定选择第一个，避免并列时随机变化。旧 RecIF 三层编码保持原样。

## 目录

```text
output_amazon_letter_beauty/
  tasks/pretrain_item_understand.parquet
  tasks/pretrain_amazon_seq_rec.parquet
  tasks/sft_amazon_seq_rec.parquet
  split_data_pretrain_item_understand/  # 13 个分片 + file_list.json
  split_data_pretrain/                  # 23 个分片 + file_list.json
  split_data_sft/                       # 23 个分片 + file_list.json
  intermediate/items.parquet
  intermediate/train_sequences.parquet
  benchmark_data/amazon_beauty/amazon_beauty_test.parquet
  benchmark_data/sid2pid.json
  benchmark_data/sid2iid.json
  special_tokens.json
  summary.json
```

## 长度与训练接入

注册 4×256 SID tokens 和两端标记后，用 Qwen3-0.6B tokenizer 测量，训练长度包含末尾 pad。Stage 1 最长 2048（6 条长文本尾部截断，完整原文保留）；Stage 2 最长 1219；SFT 最长 1274（benchmark 自定义模板下 1277）；benchmark prompt 加生成四层 SID 最长 1279。所有用户历史完整保留。

`DATA_PROFILE=amazon_letter_beauty` 已指向三个独立训练分片目录，并默认 `MAX_LENGTH=2048`，显式环境变量可覆盖。训练脚本原有 RecIF profile 默认不变。

**尚未启动训练，也未扩展模型权重。** 训练前用现有 `pretrain/tools/model_converter/expand_qwen3_vocab.py` 为 Qwen3-0.6B 扩词表，参数必须为 `--itemic_layer_n 4 --vocab_size_per_layer 256`。训练显式指定 `MODEL_DIR`，后续阶段指定对应 checkpoint；`ITEMIC_START_INDEX` / `ITEMIC_END_INDEX` 需按扩展后的 tokenizer 设置（含首尾标记的连续范围），不能沿用 RecIF 默认的 8192×3 范围。若使用 memory adapter，其相关 SID 范围和结构参数也应单独核对。本次只完成数据和评测任务接入。

通用混合留到下一版：当前 `prepare_mixed_data.py` 会 glob `pretrain_*`，直接混合会把 Stage 1 一并引入 Stage 2，使用前应增加明确任务筛选。

## Benchmark

任务名称为 `amazon_beauty`，已注册到原有 recommendation evaluator，生成长度为 4。数据路径为输出根目录的 `benchmark_data`，不用改名成 video。

在 `openonerec_benchmark` 评测环境中，使用训练并导出的四层 SID 模型：

```bash
conda activate openonerec_benchmark
cd benchmarks
PYTHONPATH=. python scripts/ray-vllm/evaluate.py \
  --task_types amazon_beauty \
  --model_path /path/to/trained/converted \
  --data_dir ../process_data/output_amazon_letter_beauty/benchmark_data \
  --output_dir /path/to/amazon_beauty_results \
  --sample_size 3000 \
  --num_beams 32 --num_return_sequences 32 \
  --max_model_len 2048
```

`sample_size=3000` 沿用现有 head(3000) 规则，首次保存 `amazon_beauty/amazon_beauty_test_sample_3000.parquet`，后续复用。当前只生成完整测试文件，没有预先生成 3000 样本缓存。

用真实答案模拟预测时 SID pass@1=1，PID pass@1≈0.999016：保留 SID 碰撞并固定选择热门 PID 导致 22 个测试 target 无法命中。这是映射策略的结果，不是模型跑分。
