# Whetstone

Whetstone is a from-scratch post-training laboratory for **verifiable reasoning**. Due to limitations of computational resources, only relatively small models and datasets are supported and tested.

## Benchmark

`Qwen/Qwen3-0.6B-Base` on OpenAI GSM8K dataset:

| Template  |    N | Accuracy | Parse | No-ans. | Wrong | Tok/resp |
| --------- | ---: | -------: | ----: | ------: | ----: | -------: |
| zero-shot |  200 |    0.355 | 0.540 |   0.440 | 0.185 |    256.8 |
| zero-shot | 1319 |    0.342 | 0.552 |   0.430 | 0.210 |    251.9 |
| few-shot  |  200 |    0.530 | 0.985 |   0.010 | 0.455 |    101.3 |
| few-shot  | 1319 |    0.536 | 0.991 |   0.005 | 0.455 |     97.5 |

```bash
uv run scripts/run_eval.py --config configs/eval/gsm8k_bench_fewshot.yaml \
  --set dataset.limit=null --run-name gsm8k_bench_fewshot_full
uv run scripts/run_eval.py --config configs/eval/gsm8k_bench_fewshot.yaml 

uv run scripts/run_eval.py --config configs/eval/gsm8k_bench.yaml \
  --set dataset.limit=null --run-name gsm8k_bench_zeroshot_full
uv run scripts/run_eval.py --config configs/eval/gsm8k_bench.yaml
```

