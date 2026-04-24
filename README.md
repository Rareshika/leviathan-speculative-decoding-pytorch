# Speculative Decoding in PyTorch

A PyTorch implementation of speculative decoding from Leviathan et al., with
a KV-cached baseline and a benchmark harness for studying practical speedups
across draft-target model pairings. Across the pairings benchmarked here on
an NVIDIA L4 GPU, the best setup reached `1.88x` speedup over the baseline.

## Overview

Speculative decoding is an inference-time acceleration technique in which a
smaller draft model proposes tokens that a larger target model then verifies,
while preserving the target model's distribution. In this project, I
implemented speculative decoding for decoder-only language models together
with a KV-cached autoregressive baseline and a benchmarking pipeline that
exports run summaries to CSV. I used that setup to evaluate several
draft-target pairings from the same model family and to study how model
alignment affects practical speedup under memory-bandwidth constraints.

## What This Project Implements

- Implemented a KV-cached autoregressive baseline decoder so the comparison
  against speculative decoding stays controlled and fair, rather than relying
  on `.generate()`.
- Built a speculative decoder with the full draft, verify, and commit loop,
  while preserving the target model's distribution.
- Added a benchmark harness that measures latency, throughput, and realized
  speedup across model pairings.
- Tracked empirical acceptance rate (`alpha`) and the draft-to-target cost
  coefficient (`c`) to connect the theory to observed performance.
- Exported benchmark summaries to `CSV` for easier comparison across runs and
  configurations.

## Why This Project Matters

- Reproduces the core mechanics of a recent inference paper in plain PyTorch,
  rather than treating speculative decoding as a black-box feature.
- Builds a compact but reproducible benchmark workflow for measuring
  throughput, latency, acceptance rate, and realized speedup.
- Shows how speculative decoding behaves in practice across real model
  families, including both strong positive results and clear failure cases.

## Repository Structure

```text
.
├── main.py
├── src/
│   ├── decoders.py
│   ├── benchmark.py
│   └── utils.py
└── requirements.txt
```

- `main.py`: benchmark entry point
- `src/decoders.py`: baseline + speculative decoding logic
- `src/benchmark.py`: timing loop and summary reporting
- `src/utils.py`: tokenization, cache helpers, timing helpers

## Experimental Setup

The reported benchmarks were run in Google Colab Pro on a single NVIDIA L4
GPU. I used 20 prompts from the Hugging Face `wikitext-2-raw-v1` test split,
shuffled with a fixed seed, filtered out short entries and section headers,
and truncated each prompt to 50 tokens. Each decoder was then asked to
generate up to 128 new tokens per prompt over two full passes through the
prompt set, following a short warmup run.

### Hardware

Most experiments were conducted on an L4 GPU, which is a useful setting for
speculative decoding because the method is especially relevant when inference
is more memory-bandwidth-bound than compute-bound. Preliminary runs on an
A100 showed smaller but still consistent speedups, which matches the
expectation that higher memory bandwidth reduces the relative bottleneck.

- Google Colab Pro
- NVIDIA L4 HIGH-RAM

```text
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA L4                      Off | 00000000:00:04.0 Off |                    0 |
| N/A   45C    P0              27W /  72W |      0MiB / 23034MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

### Models

| Target Model                | Draft Model                 | Target Params | Draft Params | Notes                                                      |
| --------------------------- | --------------------------- | ------------- | ------------ | ---------------------------------------------------------- |
| `openai-community/gpt2-xl`  | `openai-community/gpt2`     | 1.5B          | 120M         | Early same-family GPT-2 pairing                            |
| `EleutherAI/pythia-6.9b-v0` | `EleutherAI/pythia-160m-v0` | 6.9B          | 160M         | Same-family pair trained on the same corpus                |
| `EleutherAI/pythia-6.9b-v0` | `EleutherAI/pythia-70m-v0`  | 6.9B          | 70M          | More aggressive draft compression within the Pythia family |
| `facebook/opt-6.7b`         | `facebook/opt-125m`         | 6.7B          | 125M         | Same-family OPT pairing                                    |
| `HuggingFaceTB/SmolLM2-1.7B`| `HuggingFaceTB/SmolLM2-135M`| 1.7B          | 135M         | Same-family small-model failure case                       |
| `Qwen/Qwen2.5-7B`           | `Qwen/Qwen2.5-0.5B`         | 7B            | 0.5B         | Same-family modern decoder-only pairing                    |
| `bigscience/bloom-7b1`      | `bigscience/bloom-560m`     | 7B            | 560M         | Same-family BLOOM pairing                                  |

Using same-family models matters because the draft model needs to approximate
the target model's token distribution closely enough for its proposals to be
accepted frequently. When the two models are trained on similar data with
similar objectives, acceptance rates are usually higher, and the observed
speedup is more likely to reflect the algorithm itself rather than plain
distribution mismatch.

### Dataset and Prompts

- Prompt source: the Hugging Face `wikitext-2-raw-v1` test split.
- Prompt selection: 20 prompts sampled after shuffling with seed `42`,
  filtering out short entries, and skipping Wikipedia-style section headers.
- Prompt length: truncated to 50 tokens.
- Generation length: capped at 128 new tokens per prompt.

### Decode Settings

- `gamma ∈ {1, 3, 5, 7, 9}`: draft-length sweep used in the reported results.
- `temperature = 0.0`: generation is run in greedy mode.
- `max_tokens = 128`: maximum number of new tokens generated per prompt.
- `iterations = 2`: each decoder is evaluated over two passes through the
  prompt set.
- `warmup = 10`: a short untimed generation on the first prompt is used to
  avoid charging one-time startup overhead to the benchmark.

## Results

### At A Glance

This smaller table is meant to make the main outcomes easier to scan before
getting into the full gamma sweep below. Each row shows the best-performing
setting for one draft-target pairing.

| Target Model  | Draft Model    | Best Gamma | Best Speedup | Alpha At Best Gamma | `c`   | Quick Read          |
| ------------- | -------------- | ---------- | ------------ | ------------------- | ----- | ------------------- |
| `gpt2-xl`     | `gpt2`         | 5          | 1.48x        | 0.583               | 0.255 | Strong gain         |
| `pythia-6.9b` | `pythia-160m`  | 5          | 1.58x        | 0.533               | 0.050 | Strong gain         |
| `pythia-6.9b` | `pythia-70m`   | 5          | 1.88x        | 0.461               | 0.023 | Best overall result |
| `opt-6.7b`    | `opt-125m`     | 5          | 1.74x        | 0.535               | 0.047 | Very strong gain    |
| `bloom-7b1`   | `bloom-560m`   | 3          | 1.17x        | 0.678               | 0.154 | Modest gain         |
| `SmolLM2-1.7B`| `SmolLM2-135M` | 1          | 0.53x        | 0.833               | 0.306 | Clear slowdown      |
| `Qwen2.5-7B`  | `Qwen2.5-0.5B` | 3          | 0.97x        | 0.563               | 0.137 | Near break-even     |

### Key Takeaways

- The strongest result came from `pythia-6.9b` + `pythia-70m`, which reached
  `1.88x` speedup at `gamma = 5`. It is a good example of how powerful a
  very small draft model can be when it stays well aligned with the target.
- The most useful pattern across pairings was the tradeoff between `alpha`,
  `c`, and `gamma`: low draft cost creates room for more aggressive drafting,
  but the gains only hold if the acceptance rate stays high enough as
  `gamma` increases.
- The clearest failure case was `SmolLM2-1.7B` + `SmolLM2-135M`, where
  speculative decoding was slower than the baseline across the entire gamma
  sweep. That run made it clear that high acceptance alone is not enough: if
  the draft cost is too high and the baseline is already fast, speculation
  can still be a net loss.

### Full Results

| Target Model  | Draft Model    | Gamma | Avg TPS | Avg Latency (s) | Alpha | `c`   | Speedup vs Baseline | Notes                 |
| ------------- | -------------- | ----- | ------- | --------------- | ----- | ----- | ------------------- | --------------------- |
| `gpt2-xl`     | `gpt2`         | `-`   | 25.40   | 5.04            | N/A   | `-`   | N/A                 | Baseline              |
| `gpt2-xl`     | `gpt2`         | 1     | 29.34   | 4.36            | 0.814 | 0.255 | 1.16x               | -                     |
| `gpt2-xl`     | `gpt2`         | 3     | 37.42   | 3.42            | 0.695 | 0.255 | 1.47x               | -                     |
| `gpt2-xl`     | `gpt2`         | 5     | 37.58   | 3.41            | 0.583 | 0.255 | 1.48x               | Best for this pair    |
| `gpt2-xl`     | `gpt2`         | 7     | 35.14   | 3.64            | 0.506 | 0.255 | 1.38x               | -                     |
| `gpt2-xl`     | `gpt2`         | 9     | 33.23   | 3.85            | 0.454 | 0.255 | 1.31x               | -                     |
| `pythia-6.9b` | `pythia-160m`  | `-`   | 18.14   | 7.06            | N/A   | `-`   | N/A                 | Baseline              |
| `pythia-6.9b` | `pythia-160m`  | 1     | 22.33   | 5.73            | 0.791 | 0.050 | 1.23x               | -                     |
| `pythia-6.9b` | `pythia-160m`  | 3     | 28.44   | 4.50            | 0.640 | 0.050 | 1.57x               | -                     |
| `pythia-6.9b` | `pythia-160m`  | 5     | 28.58   | 4.48            | 0.533 | 0.050 | 1.58x               | Best for this pair    |
| `pythia-6.9b` | `pythia-160m`  | 7     | 26.72   | 4.79            | 0.453 | 0.050 | 1.47x               | -                     |
| `pythia-6.9b` | `pythia-160m`  | 9     | 24.34   | 5.26            | 0.394 | 0.050 | 1.34x               | -                     |
| `pythia-6.9b` | `pythia-70m`   | `-`   | 18.08   | 7.08            | N/A   | `-`   | N/A                 | Baseline              |
| `pythia-6.9b` | `pythia-70m`   | 1     | 24.91   | 5.14            | 0.729 | 0.023 | 1.38x               | -                     |
| `pythia-6.9b` | `pythia-70m`   | 3     | 33.01   | 3.88            | 0.574 | 0.023 | 1.83x               | -                     |
| `pythia-6.9b` | `pythia-70m`   | 5     | 33.99   | 3.77            | 0.461 | 0.023 | 1.88x               | Best overall          |
| `pythia-6.9b` | `pythia-70m`   | 7     | 32.37   | 3.95            | 0.386 | 0.023 | 1.79x               | -                     |
| `pythia-6.9b` | `pythia-70m`   | 9     | 30.35   | 4.22            | 0.332 | 0.023 | 1.68x               | -                     |
| `opt-6.7b`    | `opt-125m`     | `-`   | 18.18   | 7.04            | N/A   | `-`   | N/A                 | Baseline              |
| `opt-6.7b`    | `opt-125m`     | 1     | 23.89   | 5.36            | 0.792 | 0.047 | 1.31x               | -                     |
| `opt-6.7b`    | `opt-125m`     | 3     | 31.24   | 4.10            | 0.645 | 0.047 | 1.72x               | -                     |
| `opt-6.7b`    | `opt-125m`     | 5     | 31.72   | 4.03            | 0.535 | 0.047 | 1.74x               | Best for this pair    |
| `opt-6.7b`    | `opt-125m`     | 7     | 30.04   | 4.26            | 0.461 | 0.047 | 1.65x               | -                     |
| `opt-6.7b`    | `opt-125m`     | 9     | 27.91   | 4.59            | 0.406 | 0.047 | 1.53x               | -                     |
| `bloom-7b1`   | `bloom-560m`   | `-`   | 17.05   | 7.51            | N/A   | `-`   | N/A                 | Baseline              |
| `bloom-7b1`   | `bloom-560m`   | 1     | 17.07   | 7.50            | 0.821 | 0.154 | 1.00x               | Barely above baseline |
| `bloom-7b1`   | `bloom-560m`   | 3     | 19.92   | 6.43            | 0.678 | 0.154 | 1.17x               | Best for this pair    |
| `bloom-7b1`   | `bloom-560m`   | 5     | 19.00   | 6.74            | 0.568 | 0.154 | 1.11x               | -                     |
| `bloom-7b1`   | `bloom-560m`   | 7     | 17.44   | 7.34            | 0.499 | 0.154 | 1.02x               | Near break-even       |
| `bloom-7b1`   | `bloom-560m`   | 9     | 15.65   | 8.18            | 0.433 | 0.154 | 0.92x               | Slower than baseline  |
| `SmolLM2-1.7B`| `SmolLM2-135M` | `-`   | 38.32   | 3.34            | N/A   | `-`   | N/A                 | Baseline              |
| `SmolLM2-1.7B`| `SmolLM2-135M` | 1     | 20.25   | 6.32            | 0.833 | 0.306 | 0.53x               | Clear slowdown        |
| `SmolLM2-1.7B`| `SmolLM2-135M` | 3     | 20.86   | 6.14            | 0.714 | 0.306 | 0.54x               | Best for this pair    |
| `SmolLM2-1.7B`| `SmolLM2-135M` | 5     | 18.75   | 6.83            | 0.607 | 0.306 | 0.49x               | -                     |
| `SmolLM2-1.7B`| `SmolLM2-135M` | 7     | 16.13   | 7.94            | 0.514 | 0.306 | 0.42x               | -                     |
| `SmolLM2-1.7B`| `SmolLM2-135M` | 9     | 13.98   | 9.16            | 0.443 | 0.306 | 0.36x               | Worst slowdown        |
| `Qwen2.5-7B`  | `Qwen2.5-0.5B` | `-`   | 17.11   | 7.48            | N/A   | `-`   | N/A                 | Baseline              |
| `Qwen2.5-7B`  | `Qwen2.5-0.5B` | 1     | 15.54   | 8.24            | 0.731 | 0.137 | 0.91x               | Slower than baseline  |
| `Qwen2.5-7B`  | `Qwen2.5-0.5B` | 3     | 16.62   | 7.70            | 0.563 | 0.137 | 0.97x               | Closest to break-even |
| `Qwen2.5-7B`  | `Qwen2.5-0.5B` | 5     | 15.29   | 8.37            | 0.474 | 0.137 | 0.89x               | Slower than baseline  |
| `Qwen2.5-7B`  | `Qwen2.5-0.5B` | 7     | 12.90   | 9.92            | 0.385 | 0.137 | 0.75x               | Slower than baseline  |
| `Qwen2.5-7B`  | `Qwen2.5-0.5B` | 9     | 11.16   | 11.47           | 0.328 | 0.137 | 0.65x               | Slowest setting       |

## Interpreting The Results

Compared with the headline results in the original paper, the speedups here
are smaller, and that is expected for this setup. I am benchmarking
general-purpose decoder-only models used out of the box, rather than
draft-target pairs tuned for a single task or explicitly optimized for
alignment. In that sense, this project is less of a best-case reproduction
and more of a practical study of how speculative decoding behaves across real
model families.

### Why My Numbers Are Lower Than The Paper

My experiments use general decoder-only models that were not fine-tuned to
closely match each other's output distributions. Because speculative decoding
depends heavily on draft-target alignment, this lowers the acceptance rate,
especially as `gamma` increases. That, in turn, lowers the realized speedup
relative to the paper's more favorable setup.

### Hardware Context

Absolute throughput depends heavily on the hardware and software stack. Since
these runs were performed on Google Colab L4 High-RAM, the most meaningful
comparisons are the within-repo ones: baseline versus speculative for the
same pairing, and then trends in `alpha`, `c`, and optimal `gamma` across
pairings.

### What To Compare Fairly

- The fairest comparison in this repo is between each speculative setup and
  its own KV-cached autoregressive baseline, using the same prompts, token
  budget, decode settings, and hardware.
- Across pairings, it makes more sense to compare trends in acceptance rate
  (`alpha`), cost coefficient (`c`), and realized speedup than to compare raw
  throughput numbers in isolation.
- These results should be read as a practical benchmark of speculative
  decoding under general-purpose model pairings, not as a direct replication
  of the paper's best-case headline numbers.

## Failed Experiments and Lessons Learned

- My first KV-caching implementation broke exact token-for-token parity with
  the baseline. The logic itself was sound, but the way I truncated the cache
  introduced tiny floating-point differences that were sometimes enough to
  flip the greedy argmax. The main lesson here was that mathematically
  equivalent decoding paths do not always remain bit-for-bit identical in
  practice.

- I initially ran experiments on a Kaggle T4, where runtimes were long and
  speculative decoding often looked like a slowdown. At first I was not sure
  whether the problem was in the implementation or in the environment, but
  moving to more suitable hardware made the first consistent speedups visible.
  That was a useful reminder that speculative decoding is highly dependent on
  hardware, especially through the interaction between `alpha`, `c`, and the
  choice of `gamma`.

### Pairings That Failed

The `HuggingFaceTB/SmolLM2-1.7B` + `HuggingFaceTB/SmolLM2-135M` pairing was
the clearest example of speculative decoding failing outright in this
benchmark. Even with a fairly high acceptance rate at low `gamma`, the draft
model was still too expensive relative to an already fast baseline. The
result was a slowdown across the entire sweep, from `0.53x` at `gamma = 1`
to `0.36x` at `gamma = 9`.

This case was especially useful because it showed that `alpha` is not enough
on its own. The pairing started with `alpha ≈ 0.83`, which looks promising in
isolation, but `c = 0.31` was simply too large. Once the target model is
already fast, speculative decoding has much less room to recover the overhead
of the draft and verification steps.

The `Qwen/Qwen2.5-7B` + `Qwen/Qwen2.5-0.5B` pairing was one of the more
surprising negative results. Even though both models come from the same
family, speculative decoding was slower than the baseline. I expected this
setup to behave more like the `pythia-6.9b` + `pythia-160m` pairing, but the
empirical acceptance rate (`alpha`) ended up being much lower than expected.

In practice, the 0.5B draft model was not close enough to the 7B target to
make its proposals both cheap and reliable. This shows up clearly in the
measurements: `alpha` starts around `0.73` at `gamma = 1` and drops to about
`0.56` at `gamma = 3`, while `c = 0.13` means the draft calls are still
expensive enough that the lower acceptance rate wipes out the benefit.

### Observations

- Same-family models are the most natural starting point for speculative
  decoding, but family membership alone is not enough; the draft still has to
  stay close to the target model in practice.
- Alignment matters because acceptance rate is one of the strongest drivers of
  realized speedup, but it is not enough on its own if the draft model is too
  expensive relative to the baseline.
- Adding KV caching can break exact token-for-token agreement even in
  mathematically sound implementations, simply because tiny floating-point
  differences can flip greedy decoding decisions.

## How To Run

This project is meant to be run on a GPU-enabled environment such as Google
Colab or a local CUDA setup. Before starting a run, I usually edit the main
benchmark constants in `main.py`, especially `TARGET_MODEL`, `DRAFT_MODEL`,
`GAMMA`, `TOKEN_LIMIT`, and the output path for the benchmark summary.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then launch the benchmark:

```bash
python main.py
```

By default, the script will:

- load the target and draft models defined in `main.py`
- sample and tokenize prompts from `wikitext-2-raw-v1`
- estimate `c` automatically if it is left as `None`
- benchmark both the KV-cached baseline and the speculative decoder
- append the run summary to `benchmark_summary.csv`

If you want to test a different pairing, the only things you need to change
are usually `TARGET_MODEL`, `DRAFT_MODEL`, and `GAMMA` in `main.py`. If you
want to reproduce the exact results reported in this README, you should also
set `iterations=2` in the `benchmarker.run(...)` call and rerun the script
for each pairing and `gamma` value.

## Future Improvements

- Add an n-gram-based draft model as a lightweight non-neural baseline.
- Explore cross-tokenizer support so draft and target models from different
  families can be evaluated together.
- Try a hierarchical speculative decoding setup in which a cheaper heuristic
  or intermediate model runs before the main draft model.

## References

- Leviathan et al., ["Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192)
- [wikitext](https://huggingface.co/datasets/Salesforce/wikitext)
- [gpt2](https://huggingface.co/openai-community/gpt2)
- [gpt2-xl](https://huggingface.co/openai-community/gpt2-xl)
- [pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b-v0)
- [pythia-160m](https://huggingface.co/EleutherAI/pythia-160m-v0)
- [pythia-70m](https://huggingface.co/EleutherAI/pythia-70m-v0)
- [opt-6.7b](https://huggingface.co/facebook/opt-6.7b)
- [opt-125m](https://huggingface.co/facebook/opt-125m)
- [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B)
- [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
- [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [bigscience/bloom-7b1](https://huggingface.co/bigscience/bloom-7b1)
- [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m)
