"""Measure a single full prefill at each prompt length."""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from benchmarks.workloads import RequestSpec, generate_different_len_requests
from nanovllm import LLM


def run_prefill(llm: LLM, request: RequestSpec) -> float:
    seq_id = llm.add_request(request.prompt_token_ids, request.sampling_params)
    start_time = perf_counter()
    outputs, processed_tokens = llm.step()
    elapsed = perf_counter() - start_time

    input_tokens = len(request.prompt_token_ids)
    if processed_tokens != input_tokens:
        raise RuntimeError(f"Expected {input_tokens} prefill tokens in one step, got {processed_tokens}")
    if len(outputs) != 1 or outputs[0][0] != seq_id or not llm.is_finished():
        raise RuntimeError("The request must finish in one prefill step")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("~/eecs442_scratch/models/Qwen3-0.6B"))
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--warmup-runs", type=int, default=3, help="Untimed requests per input length.")
    parser.add_argument("--repeats", type=int, default=10, help="Measured requests per input length.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--summary-output", type=Path, default=Path("results/prefill.summary.json"))
    args = parser.parse_args()

    max_model_len = 4096
    if any(not 1 <= length <= max_model_len for length in args.input_lengths):
        parser.error(f"--input-lengths values must be between 1 and {max_model_len}")
    if args.warmup_runs < 0 or args.repeats <= 0:
        parser.error("--warmup-runs must be >= 0 and --repeats must be > 0")

    model_path = args.model.expanduser()
    llm = LLM(str(model_path), enforce_eager=True, max_model_len=max_model_len)
    measurements = []
    print(f"{'Input tokens':>12} {'Median step (ms)':>17} {'Input tok/s':>14}", flush=True)
    for index, input_length in enumerate(args.input_lengths):
        requests = generate_different_len_requests(
            [input_length] * (args.warmup_runs + args.repeats),
            seed=args.seed + index,
        )
        for request in requests[:args.warmup_runs]:
            run_prefill(llm, request)
        samples = [run_prefill(llm, request) for request in requests[args.warmup_runs:]]
        median_elapsed = median(samples)
        median_ms = median_elapsed * 1000
        throughput = input_length / median_elapsed
        measurements.append({
            "input_tokens": input_length,
            "median_prefill_step_ms": median_ms,
            "input_tokens_per_second": throughput,
            "step_times_ms": [elapsed * 1000 for elapsed in samples],
        })
        print(f"{input_length:12d} {median_ms:17.3f} {throughput:14.2f}", flush=True)

    summary_output = args.summary_output.expanduser()
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8") as output_file:
        json.dump({
            "configuration": {
                "model": str(model_path),
                "input_lengths": args.input_lengths,
                "warmup_runs": args.warmup_runs,
                "repeats": args.repeats,
                "seed": args.seed,
            },
            "measurements": measurements,
        }, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    print(f"Summary: {summary_output}")


if __name__ == "__main__":
    main()
