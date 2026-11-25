# Online Inference Benchmark

## Experiment Goals

Compare the baseline scheduler under random arrivals and all requests arriving at
time zero, using the same request IDs, prompt tokens, and sampling parameters.

## File Responsibilities

- `bench_online.py`: Simulate requests arriving over time, run the benchmark, and summarize results.
- `workloads.py`: Generate request input lengths, output lengths, and arrival times.
- `bench.py` in the repository root: The existing offline benchmark.

## Environment and Baseline Version

Record the GPU, model, dependency versions, and baseline commit.

## Workload Configuration

Defaults are 256 requests, seed 0, input and output lengths independently sampled
from 100 to 1024 tokens, and integer random arrival times in `[0, 30)` seconds.
The `all-at-once` pattern assigns every arrival time to zero and submits requests
in request-ID order. Both patterns consume the same random draws, keeping each
request's contents identical across runs.

The all-at-once pattern follows the arrival behavior of the root `bench.py`.
Request contents come from this workload generator, whose random-draw order
differs from the root benchmark. Both comparison runs use the token-observation
and timing definitions below.

## How to Run

Run from the repository root after activating the environment.

```bash
source init_env.sh
python -m benchmarks.bench_online --arrival-pattern random
python -m benchmarks.bench_online --arrival-pattern all-at-once
```

Each invocation runs one experiment and initializes and warms up its own model.
Use identical workload arguments for both runs to compare the same requests.
Each run prints its arrival pattern and full metric summary. Initialization and
warmup are outside the measured interval.

Adjust the workload configuration through command-line arguments.

```bash
python -m benchmarks.bench_online --arrival-pattern random --num-seqs 256 --time-interval 30 --seed 0
```

`--max-input-len`, `--max-output-len`, and `--model` configure the length limits
and model path. Running without arguments retains the random-arrival experiment.

## Metric Definitions

All request timestamps are seconds relative to the benchmark start, after model
initialization and warmup. Token timestamps are CPU observations taken after a
batch's postprocessing. Completion time is the last output token's timestamp.

| Metric | Definition |
| --- | --- |
| Submission delay | Actual submission time minus planned arrival time |
| TTFT | First output token time minus planned arrival time |
| ITL | Time between adjacent output tokens of the same request |
| End-to-end latency | Completion time minus planned arrival time |
| Output throughput | Actual total output tokens divided by time from benchmark start to last completion, including idle waits |

Latency summaries report the mean, P50, P95, P99, maximum, and sample count in
milliseconds. Percentiles use NumPy's linear interpolation. Submission delay,
TTFT, and end-to-end latency contribute one sample per request. ITL pools every
adjacent-token interval across requests, so longer outputs contribute more
samples. Single-token outputs contribute zero ITL samples. Empty summaries show
`N/A`, and an empty workload has zero output throughput.

## Scheduler Observations

Enable step tracing by choosing an output path. Run each arrival pattern
separately with identical workload arguments.

```bash
python -m benchmarks.bench_online --arrival-pattern random --trace-output results/random.json
python -m benchmarks.bench_online --arrival-pattern all-at-once --trace-output results/all_at_once.json
```

The JSON file contains the configuration, per-request token timestamps, all step
records, total preemption count, and analysis of the longest ITLs. Collection
uses in-memory records during execution. Analysis, console reporting, and JSON
writing happen after the measured run. Each invocation writes the chosen file.

| Step field | Definition |
| --- | --- |
| `step_id` | Zero-based step index within the measured run |
| `start_time`, `end_time` | Benchmark-relative CPU times from before scheduling through completion-result collection |
| `phase` | `prefill` or `decode` |
| `selected_request_ids` | Workload request IDs selected in this step |
| `decode_waiting_request_ids` | Snapshot of workload request IDs in `scheduler.running` before scheduling |
| `processed_tokens` | Actual scheduled input tokens, excluding cached prefixes during prefill, or one input token per selected decode request |
| `preempted_request_ids` | Request IDs preempted during this step, retaining repeated events |

Batch size is the length of `selected_request_ids`. Duration is `end_time` minus
`start_time`. The decode-waiting snapshot represents requests that have completed
prefill and are awaiting another decode iteration. KV allocation is checked
during scheduling. Token observations share the step end timestamp. Step timing
includes scheduling, data preparation, model execution, sampling, postprocessing,
and completion-result collection. Observer callbacks execute after the end
timestamp. Tracing introduces CPU overhead, which can affect measured latency.

`--top-itls` selects how many longest intervals to analyze, defaulting to 5. Token
positions in this analysis are one-based. Each interval lists its overlapping
steps and clips each overlap to the interval boundaries. The report includes
prefill time, decode time, time outside recorded steps, and `prefill_wait_time`.
The last quantity counts prefill overlap while the affected request was in the
decode-waiting snapshot and absent from that step's selected batch. Preemption
counts refer to events in overlapping steps, with a separate count for the
affected request. These observations locate scheduling delays and cache-related
interruptions for further investigation.

Use `--top-itls 0` to save the full trace without selecting intervals. Omitting
`--trace-output` runs the core metrics benchmark with step tracing disabled.

## Experiment Results

To be documented.
