from __future__ import annotations

import time

import torch


def tokenize_and_trim_prompt(prompt: str, tokenizer, max_length: int):
    encoded_prompt = tokenizer(prompt)

    input_ids = encoded_prompt.input_ids[:max_length]
    attention_mask = encoded_prompt.attention_mask[:max_length]

    return {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.tensor([attention_mask]),
    }


def logits_to_probs(logits, temperature: float = 1.0):
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")

    if temperature == 0.0:
        greedy_ids = logits.argmax(dim=-1, keepdim=True)
        probas = torch.zeros_like(logits)
        probas.scatter_(dim=-1, index=greedy_ids, value=1.0)
        return probas

    return torch.softmax(logits / temperature, dim=-1)


def sample_token(logits, temperature: float = 1.0, probs=None):
    if probs is None:
        probs = logits_to_probs(logits, temperature)

    if temperature == 0.0:
        return probs.argmax(dim=-1)

    sample_id = torch.multinomial(probs, num_samples=1)
    return sample_id.squeeze(-1)


def sample_and_get_probas(logits, temperature=1.0):
    """Backward-compatible wrapper around the split sampling helpers."""
    probas = logits_to_probs(logits, temperature)
    sample_id = sample_token(logits, temperature, probs=probas)
    return sample_id, probas


def truncate_kv_cache(past_key_values, n_keep):
    past_key_values.crop(n_keep)
    return past_key_values


def estimate_cost_coefficient(
    draft_model,
    target_model,
    model_inputs,
    num_trials: int = 10,
    warmup_runs: int = 3,
    mode: str = "decode_step",
    use_cache: bool = True,
) -> dict[str, float | int | str]:
    """
    Estimate c = t_draft / t_target by timing repeated forward passes on the same inputs.

    mode="decode_step" measures a single cached next-token step by:
    1. prefilling on all but the last token
    2. timing one forward pass on the last token

    mode="prefill" measures a single forward pass over the full prompt.
    """
    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative.")
    if mode not in {"decode_step", "prefill"}:
        raise ValueError("mode must be either 'decode_step' or 'prefill'.")

    draft_time_s = _average_forward_time(
        draft_model,
        model_inputs,
        num_trials=num_trials,
        warmup_runs=warmup_runs,
        mode=mode,
        use_cache=use_cache,
    )
    target_time_s = _average_forward_time(
        target_model,
        model_inputs,
        num_trials=num_trials,
        warmup_runs=warmup_runs,
        mode=mode,
        use_cache=use_cache,
    )

    if target_time_s <= 0:
        raise ValueError("target_model timing came out as zero, so c could not be estimated.")

    return {
        "c_value": draft_time_s / target_time_s,
        "draft_time_s": draft_time_s,
        "target_time_s": target_time_s,
        "num_trials": num_trials,
        "warmup_runs": warmup_runs,
        "mode": mode,
    }


def _average_forward_time(
    model,
    model_inputs,
    num_trials: int,
    warmup_runs: int,
    mode: str,
    use_cache: bool,
) -> float:
    device = _get_model_device(model)
    prepared_inputs = _prepare_model_inputs(model_inputs, device)
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for _ in range(warmup_runs):
                _run_timed_forward(model, prepared_inputs, device=device, mode=mode, use_cache=use_cache)

            total_time_s = 0.0
            for _ in range(num_trials):
                total_time_s += _run_timed_forward(
                    model,
                    prepared_inputs,
                    device=device,
                    mode=mode,
                    use_cache=use_cache,
                )
    finally:
        model.train(was_training)

    return total_time_s / num_trials


def _run_timed_forward(model, model_inputs, device, mode: str, use_cache: bool) -> float:
    if mode == "prefill":
        _synchronize_device(device)
        start_time = time.perf_counter()
        model(**model_inputs, use_cache=use_cache)
        _synchronize_device(device)
        return time.perf_counter() - start_time

    prefix_inputs, decode_inputs = _split_inputs_for_decode_step(model_inputs)
    prefix_output = model(**prefix_inputs, use_cache=True)
    _synchronize_device(device)

    _synchronize_device(device)
    start_time = time.perf_counter()
    model(
        **decode_inputs,
        past_key_values=prefix_output.past_key_values,
        use_cache=use_cache,
    )
    _synchronize_device(device)
    return time.perf_counter() - start_time


def _split_inputs_for_decode_step(model_inputs):
    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("decode_step mode requires model_inputs['input_ids'].")
    if input_ids.shape[-1] < 2:
        raise ValueError("decode_step mode requires at least 2 tokens in the prompt.")

    prefix_inputs = {}
    decode_inputs = {"input_ids": input_ids[..., -1:]}

    for key, value in model_inputs.items():
        if not torch.is_tensor(value):
            prefix_inputs[key] = value
            continue

        if value.shape[-1] == input_ids.shape[-1]:
            prefix_inputs[key] = value[..., :-1]
            decode_inputs[key] = value[..., -1:]
        else:
            prefix_inputs[key] = value

    decode_inputs.pop("attention_mask", None)
    return prefix_inputs, decode_inputs


def _prepare_model_inputs(model_inputs, device):
    if torch.is_tensor(model_inputs):
        return {"input_ids": model_inputs.to(device)}

    if not isinstance(model_inputs, dict):
        raise TypeError("model_inputs must be a tensor or a mapping of model keyword arguments.")

    prepared_inputs = {}
    for key, value in model_inputs.items():
        if torch.is_tensor(value):
            prepared_inputs[key] = value.to(device)
        else:
            prepared_inputs[key] = value

    return prepared_inputs


def _get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _synchronize_device(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
