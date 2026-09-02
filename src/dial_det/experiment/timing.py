"""Clean timing protocol with no synthetic background GPU activity."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import statistics
import time
from typing import Callable


@dataclass(frozen=True)
class TimingSummary:
    warmup_iterations: int
    repetitions: int
    observations: int
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    all_ms: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _synchronize_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def measure_callable(
    function: Callable[[], object],
    *,
    warmup_iterations: int = 100,
    repetitions: int = 5,
    observations_per_repetition: int = 1,
    synchronize_cuda: bool = True,
) -> TimingSummary:
    if warmup_iterations < 0 or repetitions <= 0 or observations_per_repetition <= 0:
        raise ValueError("invalid timing counts")
    for _ in range(warmup_iterations):
        function()
    measurements: list[float] = []
    for _ in range(repetitions):
        for _ in range(observations_per_repetition):
            if synchronize_cuda:
                _synchronize_cuda()
            start = time.perf_counter_ns()
            function()
            if synchronize_cuda:
                _synchronize_cuda()
            measurements.append((time.perf_counter_ns() - start) / 1e6)
    return TimingSummary(
        warmup_iterations=warmup_iterations,
        repetitions=repetitions,
        observations=len(measurements),
        median_ms=float(statistics.median(measurements)),
        mean_ms=float(statistics.mean(measurements)),
        min_ms=float(min(measurements)),
        max_ms=float(max(measurements)),
        all_ms=tuple(float(value) for value in measurements),
    )
