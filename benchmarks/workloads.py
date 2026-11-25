from dataclasses import dataclass
from random import Random
from nanovllm import SamplingParams


@dataclass
class RequestSpec:
  request_id: int
  arrival_time: int
  prompt_token_ids: list[int]
  sampling_params: SamplingParams


def generate_random_requests(
    num_seqs: int = 256,
    max_input_len: int = 1024,
    max_output_len: int = 1024,
    time_interval: int = 30,
    seed: int = 0,
):
    if num_seqs < 0:
        raise ValueError("num_seqs must be >= 0")
    if time_interval <= 0:
        raise ValueError("time_interval must be > 0")
    if max_input_len < 100:
        raise ValueError("max_input_len must be >= 100")
    if max_output_len < 100:
        raise ValueError("max_output_len must be >= 100")

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

    request_list.sort(key=lambda request: (request.arrival_time, request.request_id))
    return request_list
