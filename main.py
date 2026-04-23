import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src import SpeculativeDecoder, AutoregressiveDecoder, Benchmarker
from src.utils import tokenize_and_trim_prompt, estimate_cost_coefficient

PROMPT_TOKEN_LENGTH = 50
TOKEN_LIMIT = 128  # Number of tokens to generate per prompt.
GAMMA = 7  # Draft length for speculative decoding.
TARGET_MODEL = "openai-community/gpt2-xl"
DRAFT_MODEL = "openai-community/gpt2"
C_VALUE = None  # Leave as None to auto-estimate from one cached decode step.
DEVICE = "cuda"
BENCHMARK_SUMMARY_PATH = "benchmark_summary.csv"
BASELINE_DECODER_NAME = "Baseline"


def main():
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        dtype=torch.float16,
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL,
        dtype=torch.float16,
    )

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test").shuffle(seed=42)

    # Filter out empty lines and Wikipedia section headers.
    raw_prompts = [x["text"][:] for x in ds if len(x["text"]) > 200 and not x["text"].startswith(" =")][:20]

    tokenized_prompts = [
        tokenize_and_trim_prompt(prompt, tokenizer, max_length=PROMPT_TOKEN_LENGTH)
        for prompt in raw_prompts
    ]

    estimated_c_value = C_VALUE
    c_estimate = None

    baseline = AutoregressiveDecoder(
        target_model,
        tokenizer,
        device=DEVICE,
        model_name=TARGET_MODEL,
        c_value=estimated_c_value,
    )
    speculative = SpeculativeDecoder(
        target_model,
        draft_model,
        tokenizer,
        gamma=GAMMA,
        device=DEVICE,
        target_model_name=TARGET_MODEL,
        draft_model_name=DRAFT_MODEL,
        c_value=estimated_c_value,
    )

    if estimated_c_value is None and tokenized_prompts:
        c_estimate = estimate_cost_coefficient(
            draft_model=draft_model,
            target_model=target_model,
            model_inputs=tokenized_prompts[0],
            num_trials=10,
            warmup_runs=3,
            mode="decode_step",
        )
        estimated_c_value = c_estimate["c_value"]
        baseline.c_value = estimated_c_value
        speculative.c_value = estimated_c_value

        print(
            "Estimated c-value: "
            f"{estimated_c_value:.4f} "
            f"(draft {c_estimate['draft_time_s'] * 1000:.2f} ms, "
            f"target {c_estimate['target_time_s'] * 1000:.2f} ms, "
            f"mode={c_estimate['mode']})"
        )

    benchmarker = Benchmarker(
        decoders={BASELINE_DECODER_NAME: baseline, f"Speculative (Gamma={GAMMA})": speculative},
        prompts=tokenized_prompts,
        summary_filename=BENCHMARK_SUMMARY_PATH,
        baseline_decoder_name=BASELINE_DECODER_NAME,
    )

    print("Starting benchmarks...")
    benchmarker.run(max_tokens=TOKEN_LIMIT, iterations=3)


if __name__ == "__main__":
    main()
