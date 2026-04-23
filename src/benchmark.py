import csv
import os
import time
from datetime import datetime

import torch


class Benchmarker:
    def __init__(
        self,
        decoders,
        prompts,
        summary_filename="benchmark_summary.csv",
        baseline_decoder_name=None,
    ):
        self.decoders = decoders
        self.prompts = prompts
        self.summary_filename = summary_filename
        self.baseline_decoder_name = baseline_decoder_name

    def run(self, max_tokens=128, iterations=3, warmup_tokens=10):
        run_at = datetime.now().isoformat(timespec="seconds")
        summary_rows = []
        baseline_name = self._resolve_baseline_name()

        for name, decoder in self.decoders.items():
            print(f"--- Benchmarking {name} ---")
            self._run_warmup(decoder, warmup_tokens)

            total_latency_s = 0.0
            total_tokens = 0
            alpha_sum = 0.0
            alpha_count = 0
            num_samples = 0

            for _ in range(iterations):
                for prompt in self.prompts:
                    latency_s, alpha = self._benchmark_prompt(decoder, prompt, max_tokens=max_tokens)
                    total_latency_s += latency_s
                    total_tokens += max_tokens
                    num_samples += 1

                    if alpha is not None:
                        alpha_sum += alpha
                        alpha_count += 1

            avg_tps = total_tokens / total_latency_s if total_latency_s > 0 else 0.0
            avg_latency_s = total_latency_s / num_samples if num_samples > 0 else 0.0
            avg_alpha = alpha_sum / alpha_count if alpha_count > 0 else None

            row = {
                "run_at": run_at,
                "decoder_name": name,
                "num_samples": num_samples,
                "prompt_count": len(self.prompts),
                "iterations": iterations,
                "max_tokens": max_tokens,
                "avg_tps": avg_tps,
                "avg_latency_s": avg_latency_s,
                "alpha": avg_alpha,
                "speedup": None,
            }
            row.update(decoder.get_summary_fields())
            summary_rows.append(row)

        self._annotate_speedups(summary_rows, baseline_name)
        self._print_summary(summary_rows)
        self._save_summary(summary_rows)
        return summary_rows

    def _resolve_baseline_name(self):
        if not self.decoders:
            raise ValueError("decoders cannot be empty.")

        if self.baseline_decoder_name is None:
            if len(self.decoders) == 1:
                return next(iter(self.decoders))
            raise ValueError(
                "baseline_decoder_name must be provided when benchmarking multiple decoders."
            )

        if self.baseline_decoder_name not in self.decoders:
            available = ", ".join(self.decoders)
            raise ValueError(
                f"Unknown baseline decoder '{self.baseline_decoder_name}'. Available decoders: {available}."
            )

        return self.baseline_decoder_name

    def _annotate_speedups(self, summary_rows, baseline_name):
        baseline_tps = None
        for row in summary_rows:
            if row["decoder_name"] == baseline_name:
                baseline_tps = row["avg_tps"]
                break

        if baseline_tps is None:
            raise ValueError(f"Baseline decoder '{baseline_name}' was not benchmarked.")

        for row in summary_rows:
            if row["decoder_name"] == baseline_name:
                row["speedup"] = None
            else:
                row["speedup"] = row["avg_tps"] / baseline_tps if baseline_tps > 0 else None

    def _run_warmup(self, decoder, warmup_tokens):
        if not self.prompts or warmup_tokens <= 0:
            return

        decoder.generate(self.prompts[0], max_tokens=warmup_tokens)
        self._synchronize_decoder(decoder)

    def _benchmark_prompt(self, decoder, prompt, max_tokens):
        self._synchronize_decoder(decoder)
        start_time = time.perf_counter()
        _, alpha = decoder.generate(prompt, max_tokens=max_tokens)
        self._synchronize_decoder(decoder)
        latency_s = time.perf_counter() - start_time
        return latency_s, alpha

    def _synchronize_decoder(self, decoder):
        if not torch.cuda.is_available():
            return

        device = getattr(decoder, "device", None)
        if device is None:
            return

        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _print_summary(self, summary_rows):
        print("\n" + "=" * 90)
        print(f"{'BENCHMARK SUMMARY':^90}")
        print("=" * 90)

        header = (
            f"{'Model Name':<25} | {'Samples':<7} | {'Avg TPS':<10} | "
            f"{'Avg Latency (s)':<15} | {'Avg Alpha':<10} | {'Speedup':<8}"
        )
        print(header)
        print("-" * len(header))

        for row in summary_rows:
            alpha_str = f"{row['alpha']:.2f}" if row["alpha"] is not None else "N/A"
            speedup_str = f"{row['speedup']:.2f}x" if row["speedup"] is not None else "N/A"
            print(
                f"{row['decoder_name']:<25} | "
                f"{row['num_samples']:<7} | "
                f"{row['avg_tps']:<10.2f} | "
                f"{row['avg_latency_s']:<15.2f} | "
                f"{alpha_str:<10} | "
                f"{speedup_str:<8}"
            )

        print("=" * 90)

    def _save_summary(self, summary_rows):
        if not self.summary_filename or not summary_rows:
            return

        file_exists = os.path.exists(self.summary_filename)
        with open(self.summary_filename, "a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "run_at",
                    "decoder_name",
                    "model_name",
                    "number_of_parameters",
                    "c",
                    "gamma",
                    "num_samples",
                    "prompt_count",
                    "iterations",
                    "max_tokens",
                    "avg_tps",
                    "avg_latency_s",
                    "alpha",
                    "speedup",
                ],
            )
            if not file_exists:
                writer.writeheader()

            for row in summary_rows:
                writer.writerow(
                    {
                        "run_at": row["run_at"],
                        "decoder_name": row["decoder_name"],
                        "model_name": row["model_name"],
                        "number_of_parameters": row["number_of_parameters"],
                        "c": row["c"],
                        "gamma": row["gamma"],
                        "num_samples": row["num_samples"],
                        "prompt_count": row["prompt_count"],
                        "iterations": row["iterations"],
                        "max_tokens": row["max_tokens"],
                        "avg_tps": f"{row['avg_tps']:.4f}",
                        "avg_latency_s": f"{row['avg_latency_s']:.4f}",
                        "alpha": f"{row['alpha']:.4f}" if row["alpha"] is not None else "N/A",
                        "speedup": f"{row['speedup']:.4f}" if row["speedup"] is not None else "N/A",
                    }
                )

        print(f"Summary saved to: {os.path.abspath(self.summary_filename)}")
