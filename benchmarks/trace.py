"""Step observations and attribution of long inter-token intervals."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from heapq import nlargest
from typing import Literal


@dataclass(frozen=True)
class StepRecord:
    step_id: int
    start_time: float
    end_time: float
    phase: Literal["prefill", "decode"]
    selected_request_ids: tuple[int, ...]
    decode_waiting_request_ids: tuple[int, ...]
    processed_tokens: int
    preempted_request_ids: tuple[int, ...]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def batch_size(self) -> int:
        return len(self.selected_request_ids)


@dataclass(frozen=True)
class ITLAnalysis:
    request_id: int
    from_token: int
    to_token: int
    start_time: float
    end_time: float
    prefill_time: float
    decode_time: float
    outside_step_time: float
    prefill_wait_time: float
    preemption_count: int
    request_preemption_count: int
    step_ids: tuple[int, ...]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def analyze_longest_itls(
    token_times: Mapping[int, Sequence[float]],
    steps: Sequence[StepRecord],
    top_k: int = 5,
) -> list[ITLAnalysis]:
    """Attribute the longest intervals using clipped step-time overlaps.

    Token positions are one-based. Prefill wait time counts overlap with prefill
    steps where this request was in the running queue before scheduling and was
    absent from the selected batch. Preemption counts cover overlapping steps.
    All timestamps must use the same benchmark-relative clock.
    """
    if top_k < 0:
        raise ValueError("top_k must be >= 0")

    intervals = (
        (request_id, token_index, previous, current)
        for request_id, times in token_times.items()
        for token_index, (previous, current) in enumerate(zip(times, times[1:]), start=1)
    )
    longest_intervals = nlargest(top_k, intervals, key=lambda interval: interval[3] - interval[2])
    analyses = []

    for request_id, from_token, start_time, end_time in longest_intervals:
        prefill_time = decode_time = prefill_wait_time = 0.0
        preemption_count = request_preemption_count = 0
        step_ids = []

        for step in steps:
            overlap = min(end_time, step.end_time) - max(start_time, step.start_time)
            if overlap <= 0:
                continue

            step_ids.append(step.step_id)
            preemption_count += len(step.preempted_request_ids)
            request_preemption_count += step.preempted_request_ids.count(request_id)
            if step.phase == "prefill":
                prefill_time += overlap
                if request_id in step.decode_waiting_request_ids and request_id not in step.selected_request_ids:
                    prefill_wait_time += overlap
            else:
                decode_time += overlap

        analyses.append(ITLAnalysis(
            request_id=request_id,
            from_token=from_token,
            to_token=from_token + 1,
            start_time=start_time,
            end_time=end_time,
            prefill_time=prefill_time,
            decode_time=decode_time,
            outside_step_time=max(0.0, end_time - start_time - prefill_time - decode_time),
            prefill_wait_time=prefill_wait_time,
            preemption_count=preemption_count,
            request_preemption_count=request_preemption_count,
            step_ids=tuple(step_ids),
        ))

    return analyses
