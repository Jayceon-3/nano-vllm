import os
import time
from dataclasses import dataclass
from statistics import fmean

from nanovllm import LLM, SamplingParams
from benchmarks.workloads import RequestSpec, generate_random_requests


@dataclass
class RequestResult:
    request_id: int
    seq_id: int
    arrival_time: float
    submitted_time: float
    completed_time: float
    output_tokens: int


def run_benchmark(llm: LLM, request_list: list[RequestSpec]) -> tuple[list[RequestResult], float]:
    """Run a workload sorted by arrival time, returning results in completion order.

    All timestamps are seconds relative to the start of this call. Elapsed time
    includes idle waits and ends when the last request completes.
    """
    if not request_list:
        return [], 0.0

    next_request_idx = 0
    pending_requests: dict[int, tuple[RequestSpec, float]] = {}
    results = []
    start_time = time.perf_counter()

    while next_request_idx < len(request_list) or not llm.is_finished():
        while (
            next_request_idx < len(request_list)
            and request_list[next_request_idx].arrival_time <= time.perf_counter() - start_time
        ):
            request = request_list[next_request_idx]
            submitted_time = time.perf_counter() - start_time
            seq_id = llm.add_request(request.prompt_token_ids, request.sampling_params)
            pending_requests[seq_id] = (request, submitted_time)
            next_request_idx += 1

        if llm.is_finished():
            if next_request_idx < len(request_list):
                next_arrival = request_list[next_request_idx].arrival_time
                wait_time = start_time + next_arrival - time.perf_counter()
                if wait_time > 0:
                    time.sleep(wait_time)
            continue

        outputs, _ = llm.step()
        completed_time = time.perf_counter() - start_time
        for seq_id, token_ids in outputs:
            request, submitted_time = pending_requests.pop(seq_id)
            results.append(RequestResult(
                request_id=request.request_id,
                seq_id=seq_id,
                arrival_time=request.arrival_time,
                submitted_time=submitted_time,
                completed_time=completed_time,
                output_tokens=len(token_ids),
            ))

    return results, results[-1].completed_time


def main():
    num_seqs = 256
    max_input_len = 1024
    max_output_len = 1024
    time_interval = 30
    seed = 0

    path = os.path.expanduser("~/eecs442_scratch/models/Qwen3-0.6B")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    request_list = generate_random_requests(
        num_seqs=num_seqs,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        time_interval=time_interval,
        seed=seed,
    )

    llm.generate(["Benchmark: "], SamplingParams())
    results, elapsed_time = run_benchmark(llm, request_list)
    total_tokens = sum(result.output_tokens for result in results)
    throughput = total_tokens / elapsed_time if elapsed_time > 0 else 0.0
    print(
        f"Requests: {len(results)}/{len(request_list)}, Total: {total_tokens}tok, "
        f"Time: {elapsed_time:.2f}s, Throughput: {throughput:.2f}tok/s"
    )
    if results:
        latencies = [result.completed_time - result.arrival_time for result in results]
        submission_delays = [result.submitted_time - result.arrival_time for result in results]
        print(f"End-to-end latency: mean={fmean(latencies):.3f}s, max={max(latencies):.3f}s")
        print(f"Submission delay: mean={fmean(submission_delays):.3f}s, max={max(submission_delays):.3f}s")


if __name__ == "__main__":
    main()
