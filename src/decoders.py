from __future__ import annotations

import torch

from .utils import sample_and_get_probas, truncate_kv_cache


def _infer_model_name(model, fallback: str | None = None):
    config = getattr(model, "config", None)
    model_name = getattr(config, "_name_or_path", None)
    return model_name or fallback or model.__class__.__name__


def _count_parameters(model):
    num_parameters = getattr(model, "num_parameters", None)
    if callable(num_parameters):
        return int(num_parameters())

    return int(sum(parameter.numel() for parameter in model.parameters()))


class SpeculativeDecoder:
    def __init__(
        self,
        target_model,
        draft_model,
        tokenizer,
        gamma: int = 4,
        device: str = "cuda",
        target_model_name: str | None = None,
        draft_model_name: str | None = None,
        target_num_parameters: int | None = None,
        draft_num_parameters: int | None = None,
        c_value: float | None = None,
    ):
        self.target_model = target_model.to(device)
        self.draft_model = draft_model.to(device)
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.device = device
        self.vocab_size = len(self.tokenizer)

        self.target_model_name = target_model_name or _infer_model_name(self.target_model)
        self.draft_model_name = draft_model_name or _infer_model_name(self.draft_model)
        self.target_num_parameters = target_num_parameters or _count_parameters(self.target_model)
        self.draft_num_parameters = draft_num_parameters or _count_parameters(self.draft_model)
        self.c_value = c_value

    def generate(self, encoded_prompt, max_tokens: int = 128, temperature: float = 0.0):
        """Generate text with speculative decoding and return the run's empirical alpha."""
        input_ids = encoded_prompt["input_ids"].squeeze(0).to(self.device)
        prompt_length = input_ids.shape[0]
        total_length = prompt_length + max_tokens

        sequence = torch.zeros(total_length + self.gamma, dtype=torch.long, device=self.device)
        sequence[:prompt_length] = input_ids

        current_length = prompt_length
        drafted_tokens = 0
        accepted_tokens = 0

        with torch.no_grad():
            draft_past, target_past = self._prefill_caches(input_ids)

            while current_length < total_length:
                remaining_tokens = total_length - current_length

                # Speculative decoding always commits at least one token per round.
                if remaining_tokens == 1:
                    current_input = sequence[current_length - 1].unsqueeze(0).unsqueeze(0)
                    output = self.target_model(current_input, past_key_values=target_past, use_cache=True)
                    target_past = output.past_key_values

                    last_token_logits = output.logits[0, -1, :self.vocab_size]
                    final_id, _ = sample_and_get_probas(last_token_logits, temperature)
                    sequence[current_length] = final_id
                    current_length += 1
                    continue

                draft_steps = min(self.gamma, remaining_tokens - 1)
                draft_ids, draft_probas, draft_past = self._draft_tokens(
                    draft_steps,
                    draft_past,
                    sequence,
                    current_length,
                    temperature,
                )

                target_probas, target_past = self._verify_drafted_tokens(
                    draft_steps,
                    target_past,
                    sequence,
                    current_length,
                    temperature,
                )

                n_accepted = self._speculative_sampling_accepts(
                    draft_steps,
                    target_probas,
                    draft_probas,
                    draft_ids,
                )

                drafted_tokens += draft_steps
                accepted_tokens += n_accepted

                valid_length, draft_past, target_past = self._commit_round(
                    n_accepted,
                    draft_steps,
                    draft_past,
                    target_past,
                    target_probas,
                    draft_probas,
                    sequence,
                    current_length,
                    total_length,
                )

                current_length = valid_length

        alpha = accepted_tokens / drafted_tokens if drafted_tokens > 0 else None
        generated_ids = sequence[prompt_length:total_length]
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text, alpha

    def get_summary_fields(self):
        return {
            "model_name": f"{self.target_model_name} + {self.draft_model_name}",
            "number_of_parameters": f"{self.target_num_parameters} + {self.draft_num_parameters}",
            "c": f"{self.c_value:.4f}" if self.c_value is not None else "N/A",
            "gamma": str(self.gamma),
        }

    def _prefill_caches(self, prompt_ids):
        prompt_length = prompt_ids.shape[0]

        draft_out = self.draft_model(prompt_ids.unsqueeze(0), use_cache=True)
        draft_past = truncate_kv_cache(draft_out.past_key_values, prompt_length - 1)

        target_out = self.target_model(prompt_ids.unsqueeze(0), use_cache=True)
        target_past = truncate_kv_cache(target_out.past_key_values, prompt_length - 1)

        return draft_past, target_past

    def _draft_tokens(self, draft_steps, draft_past, sequence, current_length, temperature):
        draft_ids = torch.empty(draft_steps, dtype=torch.long, device=self.device)
        draft_probas = torch.empty((draft_steps, self.vocab_size), device=self.device)

        for i in range(draft_steps):
            current_input = sequence[current_length + i - 1].unsqueeze(0).unsqueeze(0)
            output = self.draft_model(current_input, past_key_values=draft_past, use_cache=True)
            draft_past = output.past_key_values

            last_token_logits = output.logits[0, -1, :self.vocab_size]
            next_token_id, probas = sample_and_get_probas(last_token_logits, temperature)

            sequence[current_length + i] = next_token_id
            draft_ids[i] = next_token_id
            draft_probas[i] = probas

        return draft_ids, draft_probas, draft_past

    def _verify_drafted_tokens(self, draft_steps, target_past, sequence, current_length, temperature):
        target_input = sequence[current_length - 1 : current_length + draft_steps].unsqueeze(0)
        output = self.target_model(target_input, past_key_values=target_past, use_cache=True)
        target_past = output.past_key_values

        logits = output.logits[0, -(draft_steps + 1) :, :self.vocab_size]
        _, target_probas = sample_and_get_probas(logits, temperature)

        return target_probas, target_past

    def _speculative_sampling_accepts(self, draft_steps, target_probas, draft_probas, draft_ids):
        indices = torch.arange(draft_steps, device=self.device)
        p = target_probas[indices, draft_ids]
        q = draft_probas[indices, draft_ids]
        acceptance_ratio = torch.clamp(p / q, max=1.0)
        accepted_mask = torch.rand(draft_steps, device=self.device) <= acceptance_ratio
        return int(accepted_mask.cumprod(dim=0).sum().item())

    def _get_comit_distribtuion(self, n_accepted, draft_steps, target_probas, draft_probas):
        adjusted_distribution = target_probas[n_accepted, :]
        if n_accepted < draft_steps:
            adjusted_distribution = self._build_rejection_distribution(
                target_probas[n_accepted, :],
                draft_probas[n_accepted, :],
            )
        return adjusted_distribution

    def _write_commited_token(
        self,
        commited_token,
        n_accepted,
        draft_steps,
        draft_past,
        sequence,
        current_length,
        total_length,
    ):
        sequence[current_length + n_accepted] = commited_token

        valid_length = current_length + n_accepted + 1
        if n_accepted == draft_steps and valid_length < total_length:
            last_draft_input = sequence[current_length + draft_steps - 1].unsqueeze(0).unsqueeze(0)
            draft_out = self.draft_model(last_draft_input, past_key_values=draft_past, use_cache=True)
            draft_past = draft_out.past_key_values
        return valid_length, draft_past

    def _realign_caches(self, draft_past, target_past, valid_length):
        return truncate_kv_cache(draft_past, valid_length - 1), truncate_kv_cache(target_past, valid_length - 1)

    def _commit_round(
        self,
        n_accepted,
        draft_steps,
        draft_past,
        target_past,
        target_probas,
        draft_probas,
        sequence,
        current_length,
        total_length,
    ):
        commit_distribution = self._get_comit_distribtuion(
            n_accepted,
            draft_steps,
            target_probas,
            draft_probas,
        )

        commited_token = torch.multinomial(commit_distribution, num_samples=1).squeeze()

        valid_length, draft_past = self._write_commited_token(
            commited_token,
            n_accepted,
            draft_steps,
            draft_past,
            sequence,
            current_length,
            total_length,
        )

        draft_past, target_past = self._realign_caches(draft_past, target_past, valid_length)

        return valid_length, draft_past, target_past

    def _build_rejection_distribution(self, target_probas, draft_probas):
        adjusted_distribution = torch.clamp(target_probas - draft_probas, min=0.0)
        total_mass = adjusted_distribution.sum()

        if total_mass <= 0:
            return target_probas

        return adjusted_distribution / total_mass


class AutoregressiveDecoder:
    def __init__(
        self,
        target_model,
        tokenizer,
        device: str = "cuda",
        model_name: str | None = None,
        num_parameters: int | None = None,
        c_value: float | None = None,
    ):
        self.target_model = target_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = len(self.tokenizer)

        self.model_name = model_name or _infer_model_name(self.target_model)
        self.num_parameters = num_parameters or _count_parameters(self.target_model)
        self.c_value = c_value

    def generate(self, encoded_prompt, max_tokens: int = 128, temperature: float = 0.0):
        """Generate text with standard autoregressive decoding."""
        input_ids = encoded_prompt["input_ids"].squeeze(0).to(self.device)
        prompt_length = input_ids.shape[0]
        total_length = prompt_length + max_tokens

        if max_tokens <= 0:
            return "", None

        sequence = torch.zeros(total_length, dtype=torch.long, device=self.device)
        sequence[:prompt_length] = input_ids
        current_length = prompt_length

        with torch.no_grad():
            output = self.target_model(sequence[:prompt_length].unsqueeze(0), use_cache=True)
            past_key_values = output.past_key_values

            last_token_logits = output.logits[0, -1, :self.vocab_size]
            next_token_id, _ = sample_and_get_probas(last_token_logits, temperature)
            sequence[current_length] = next_token_id
            current_length += 1

            while current_length < total_length:
                current_input = sequence[current_length - 1].unsqueeze(0).unsqueeze(0)
                output = self.target_model(current_input, past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values

                last_token_logits = output.logits[0, -1, :self.vocab_size]
                next_token_id, _ = sample_and_get_probas(last_token_logits, temperature)
                sequence[current_length] = next_token_id
                current_length += 1

        generated_ids = sequence[prompt_length:total_length]
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text, None

    def get_summary_fields(self):
        return {
            "model_name": self.model_name,
            "number_of_parameters": str(self.num_parameters),
            "c": f"{self.c_value:.4f}" if self.c_value is not None else "N/A",
            "gamma": "N/A",
        }
