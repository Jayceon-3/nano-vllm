import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Callable

import numpy as np

from nanovllm import LLM, SamplingParams
from nanovllm.engine.llm_engine import StepTrace
from benchmarks.trace import ITLAnalysis, StepRecord, analyze_longest_itls
from benchmarks.workloads import RequestSpec, generate_random_requests


@dataclass
class RequestResult:
    request_id: int
    seq_id: int
    arrival_time: float
    submitted_time: float
    completed_time: float
    output_tokens: int
    token_times: list[float] = field(default_factory=list)

    @property
    def submission_delay(self) -> float:
        return self.submitted_time - self.arrival_time

    @property
    def ttft(self) -> float | None:
        if not self.token_times:
            return None
        return self.token_times[0] - self.arrival_time

    @property
    def itls(self) -> list[float]:
        return [current - previous for previous, current in zip(self.token_times, self.token_times[1:])]

    @property
    def end_to_end_latency(self) -> float:
        return self.completed_time - self.arrival_time


def run_benchmark(
    llm: LLM,
    request_list: list[RequestSpec],
    on_step: Callable[[StepRecord], None] | None = None,
) -> tuple[list[RequestResult], float]:
    """Run a workload sorted by arrival time, returning results in completion order.

    All timestamps are seconds relative to the start of this call. Elapsed time
    includes idle waits and ends when the last request completes.
    Token timestamps are observed on the CPU after each batch is postprocessed.
    TTFT and end-to-end latency include the delay from planned arrival to submission.
    """
    if not request_list:
        return [], 0.0

    next_request_idx = 0
    pending_requests: dict[int, tuple[RequestSpec, float]] = {}
    token_times: dict[int, list[float]] = {}
    results = []
    start_time = time.perf_counter()
    step_id = 0

    def record_token(seq_id: int, timestamp: float):
        token_times[seq_id].append(timestamp - start_time)

    def record_step(trace: StepTrace):
        if on_step is not None:
            on_step(StepRecord(
                step_id=step_id,
                start_time=trace.start_time - start_time,
                end_time=trace.end_time - start_time,
                phase="prefill" if trace.is_prefill else "decode",
                selected_request_ids=tuple(
                    pending_requests[seq_id][0].request_id for seq_id in trace.selected_seq_ids
                ),
                decode_waiting_request_ids=tuple(
                    pending_requests[seq_id][0].request_id for seq_id in trace.decode_waiting_seq_ids
                ),
                processed_tokens=trace.processed_tokens,
                preempted_request_ids=tuple(
                    pending_requests[seq_id][0].request_id for seq_id in trace.preempted_seq_ids
                ),
            ))

    while next_request_idx < len(request_list) or not llm.is_finished():
        while (
            next_request_idx < len(request_list)
            and request_list[next_request_idx].arrival_time <= time.perf_counter() - start_time
        ):
            request = request_list[next_request_idx]
            submitted_time = time.perf_counter() - start_time
            seq_id = llm.add_request(request.prompt_token_ids, request.sampling_params)
            pending_requests[seq_id] = (request, submitted_time)
            token_times[seq_id] = []
            next_request_idx += 1

        if llm.is_finished():
            if next_request_idx < len(request_list):
                next_arrival = request_list[next_request_idx].arrival_time
                wait_time = start_time + next_arrival - time.perf_counter()
                if wait_time > 0:
                    time.sleep(wait_time)
            continue

        outputs, _ = llm.step(
            on_token=record_token,
            on_step=record_step if on_step is not None else None,
        )
        step_id += 1
        for seq_id, token_ids in outputs:
            request, submitted_time = pending_requests.pop(seq_id)
            request_token_times = token_times.pop(seq_id)
            if len(request_token_times) != len(token_ids):
                raise RuntimeError(f"Token timestamp count does not match output count for request {request.request_id}")
            results.append(RequestResult(
                request_id=request.request_id,
                seq_id=seq_id,
                arrival_time=request.arrival_time,
                submitted_time=submitted_time,
                completed_time=request_token_times[-1],
                output_tokens=len(token_ids),
                token_times=request_token_times,
            ))

    return results, results[-1].completed_time


def print_latency_summary(name: str, values: list[float]):
    """Print latency samples in milliseconds using linearly interpolated percentiles."""
    if not values:
        print(f"{name}: N/A (no samples)")
        return

    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    print(
        f"{name} (ms): mean={fmean(values) * 1000:.3f}, "
        f"P50={p50 * 1000:.3f}, P95={p95 * 1000:.3f}, P99={p99 * 1000:.3f}, "
        f"max={max(values) * 1000:.3f}, samples={len(values)}"
    )


def print_itl_analysis(analyses: list[ITLAnalysis], steps: list[StepRecord]):
    if not analyses:
        print("Longest ITLs: no intervals selected")
        return

    steps_by_id = {step.step_id: step for step in steps}
    for rank, interval in enumerate(analyses, start=1):
        print(
            f"Longest ITL #{rank}: request={interval.request_id}, "
            f"tokens={interval.from_token}->{interval.to_token}, "
            f"window=[{interval.start_time:.6f}, {interval.end_time:.6f}]s, "
            f"duration={interval.duration * 1000:.3f}ms"
        )
        print(
            f"  Overlap: prefill={interval.prefill_time * 1000:.3f}ms, "
            f"decode={interval.decode_time * 1000:.3f}ms, "
            f"outside_steps={interval.outside_step_time * 1000:.3f}ms, "
            f"prefill_while_waiting={interval.prefill_wait_time * 1000:.3f}ms, "
            f"preemptions={interval.preemption_count}, "
            f"request_preemptions={interval.request_preemption_count}"
        )
        for step_id in interval.step_ids:
            step = steps_by_id[step_id]
            overlap = min(interval.end_time, step.end_time) - max(interval.start_time, step.start_time)
            print(
                f"  step={step.step_id}, phase={step.phase}, "
                f"window=[{step.start_time:.6f}, {step.end_time:.6f}]s, "
                f"duration={step.duration * 1000:.3f}ms, overlap={overlap * 1000:.3f}ms, "
                f"batch_size={step.batch_size}, processed_tokens={step.processed_tokens}, "
                f"request_waiting={interval.request_id in step.decode_waiting_request_ids}, "
                f"request_selected={interval.request_id in step.selected_request_ids}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark random arrivals or all requests arriving at time zero.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arrival-pattern",
        choices=("random", "all-at-once"),
        default="random",
        help="Request arrival pattern.",
    )
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--time-interval", type=int, default=30, help="Random-arrival window in seconds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="~/eecs442_scratch/models/Qwen3-0.6B")
    parser.add_argument("--trace-output", type=Path, help="Enable step tracing and save the observations as JSON.")
    parser.add_argument("--top-itls", type=int, default=5, help="Number of longest token intervals to analyze when tracing.")
    args = parser.parse_args()
    if args.top_itls < 0:
        parser.error("--top-itls must be >= 0")

    request_list = generate_random_requests(
        num_seqs=args.num_seqs,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        time_interval=args.time_interval,
        seed=args.seed,
        arrival_pattern=args.arrival_pattern,
    )

    print(f"\nArrival pattern {args.arrival_pattern}, requests={len(request_list)}, seed={args.seed}", flush=True)
    path = os.path.expanduser(args.model)
    llm = LLM(path, enforce_eager=False, max_model_len=4096)
    llm.generate(["Benchmark: "], SamplingParams())
    step_records = []
    results, elapsed_time = run_benchmark(
        llm,
        request_list,
        on_step=step_records.append if args.trace_output is not None else None,
    )
    total_tokens = sum(result.output_tokens for result in results)
    throughput = total_tokens / elapsed_time if elapsed_time > 0 else 0.0
    print(
        f"Requests: {len(results)}/{len(request_list)}, Total: {total_tokens}tok, "
        f"Time: {elapsed_time:.2f}s, Throughput: {throughput:.2f}tok/s"
    )
    print_latency_summary("Submission delay", [result.submission_delay for result in results])
    print_latency_summary("TTFT", [result.ttft for result in results if result.ttft is not None])
    # Each adjacent-token interval contributes one sample across all requests.
    print_latency_summary("ITL (all token intervals)", [interval for result in results for interval in result.itls])
    print_latency_summary("End-to-end latency", [result.end_to_end_latency for result in results])
    if args.trace_output is not None:
        analyses = analyze_longest_itls(
            {result.request_id: result.token_times for result in results},
            step_records,
            top_k=args.top_itls,
        )
        print_itl_analysis(analyses, step_records)
        trace_output = args.trace_output.expanduser()
        trace_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "configuration": {**vars(args), "model": path, "trace_output": str(trace_output)},
            "elapsed_time": elapsed_time,
            "total_output_tokens": total_tokens,
            "preemption_count": sum(len(step.preempted_request_ids) for step in step_records),
            "requests": [asdict(result) for result in results],
            "steps": [asdict(step) for step in step_records],
            "longest_itls": [asdict(interval) for interval in analyses],
        }
        with trace_output.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
        print(f"Trace saved to {trace_output}")


if __name__ == "__main__":
    main()
