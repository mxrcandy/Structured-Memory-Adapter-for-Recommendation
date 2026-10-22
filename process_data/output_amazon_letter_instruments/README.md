# LETTER Instruments

Independent Instruments corpus, using the same protocol as Beauty. No sliding
windows, domain token, or cross-catalog SID disambiguation.

For each original user sequence `s`:

| Stage | Contents | Rows |
| --- | --- | ---: |
| Stage 1 | All item SID + title + description | 9,922 |
| Stage 2 | Pure SID sequence `s[:-1]` | 24,772 |
| SFT | Instruction + `s[:-2]`, answer `s[-2]` | 24,772 |
| Benchmark | Instruction + `s[:-1]`, target `s[-1]` | 24,772 |

Source values are `LETTER_Instruments_ItemUnderstand_Pretrain`,
`LETTER_Instruments_Sequence_Pretrain`, `LETTER_Instruments_sft`, and
`LETTER_Instruments_benchmark`. UUIDs use the Instruments namespace.
Sequence and SFT sources enable ITM during structured training; item understanding
and general samples do not.

SID tokens use a/b/c/d, 256 values each, with sid_begin/sid_end wrappers. The
existing `hf_models/Qwen3-0.6B_itemic` tokenizer/model vocabulary can be reused.
PID/IID = raw item ID + 1; zero is reserved. There are 9,897 unique SIDs for
9,922 items. Collisions are retained; mapping counts use training histories only,
ordered by popularity then PID, matching Beauty's `most_popular_originally` policy.

All item text and user histories fit MAX_LENGTH=2048 without truncation.
Stage-1 maximum is 1,480 tokens, Stage-2 maximum 1,303, and benchmark prompt plus
four generated codes at most 1,363. Full counts and hashes are in `summary.json`.

## Build

From the repository root in `openonerec`:

```bash
python process_data/prepare_rec_amazon_data.py --dataset Instruments
```

The builder refuses nonempty outputs. Stage shards and repository-relative
`file_list.json` manifests are under `split_data_pretrain_item_understand`,
`split_data_pretrain`, and `split_data_sft`. Full task rows are under `tasks`;
catalog/sequences are under `intermediate`.

## Training and Evaluation

Use `DATA_PROFILE=amazon_letter_instruments` with the training shell entry points from an allocated compute job. Scheduler submission files are site-specific and are not included.
The profile selects Instruments shards, MAX_LENGTH=2048 and SID IDs 151669-152694.
Train a separate Instruments Stage 1 and pass its converted checkpoint to e2e.

The registered task is `amazon_instruments`. Its files are:

```text
benchmark_data/amazon_instruments/amazon_instruments_test.parquet
benchmark_data/sid2pid.json
benchmark_data/sid2iid.json
```

`eval_amazon.sh` selects this corpus's benchmark directory from the task, unless
BENCHMARK_DATA_DIR is explicitly overridden. Standalone evaluation also accepts
DATA_PROFILE=amazon_letter_instruments or TASK_TYPES=amazon_instruments.
Beauty and Instruments catalogs/mappings must be evaluated independently.
