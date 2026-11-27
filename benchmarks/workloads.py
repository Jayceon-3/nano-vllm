from dataclasses import dataclass
from math import isfinite
from random import Random
from nanovllm import SamplingParams


@dataclass
class RequestSpec:
  request_id: int
  arrival_time: float
  prompt_token_ids: list[int]
  sampling_params: SamplingParams


def generate_random_requests(
    num_seqs: int = 256,
    max_input_len: int = 1024,
    max_output_len: int = 1024,
    time_interval: int = 30,
    seed: int = 0,
    arrival_pattern: str = "random",
):
    """Generate identical request contents across random and all-at-once arrivals."""
    if num_seqs < 0:
        raise ValueError("num_seqs must be >= 0")
    if time_interval <= 0:
        raise ValueError("time_interval must be > 0")
    if max_input_len < 100:
        raise ValueError("max_input_len must be >= 100")
    if max_output_len < 100:
        raise ValueError("max_output_len must be >= 100")
    if arrival_pattern not in ("random", "all-at-once"):
        raise ValueError("arrival_pattern must be 'random' or 'all-at-once'")

    rng = Random(seed)
    request_list =[]

    for i in range(num_seqs):
      prompt_token_ids = [rng.randint(0, 10000) for _ in range(rng.randint(100, max_input_len))]
      sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=rng.randint(100, max_output_len))
      arrival_time = rng.randrange(0, time_interval)
      request = RequestSpec(
        request_id = i,
        arrival_time = arrival_time,
        prompt_token_ids = prompt_token_ids,
        sampling_params = sampling_params,
      )
      request_list.append(request)

    # Draw the same random values in both modes before changing arrival times.
    if arrival_pattern == "all-at-once":
        for request in request_list:
            request.arrival_time = 0

    request_list.sort(key=lambda request: (request.arrival_time, request.request_id))
    return request_list


def generate_long_prefill_requests(delay_ms: float, seed: int = 0) -> list[RequestSpec]:
    """Insert B's long prefill while A is generating a long output."""
    if not isfinite(delay_ms) or delay_ms <= 0:
        raise ValueError("--long-prefill-delay-ms must be finite and > 0")

    rng = Random(seed)
    requests = [
        (0, 0.0, 32, 1024),  # A
        (1, delay_ms / 1000, 3072, 16),  # B
    ]
    return [
        RequestSpec(
            request_id=request_id,
            arrival_time=arrival_time,
            prompt_token_ids=[rng.randint(0, 10000) for _ in range(prompt_length)],
            sampling_params=SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_length),
        )
        for request_id, arrival_time, prompt_length, output_length in requests
    ]


def generate_different_len_requests(
    input_lengths: list[int],
    seed: int = 0,
) -> list[RequestSpec]:
    """Generate fresh single-output requests for the prefill length sweep."""
    rng = Random(seed)
    return [
        RequestSpec(
            request_id=request_id,
            arrival_time=0.0,
            prompt_token_ids=[rng.randint(0, 10000) for _ in range(input_length)],
            sampling_params=SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=1),
        )
        for request_id, input_length in enumerate(input_lengths)
    ]
