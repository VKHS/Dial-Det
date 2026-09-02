from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ExperimentProtocol:
    dataset: str
    detector_stream: str
    seeds: tuple[int, ...]
    precision: str
    batch_size: int
    warmup_iterations: int
    timing_repetitions: int
    calibration_sample_count: int | None
    utility_sample_count: int | None
    audit_role: str

    def validate(self) -> None:
        if not self.dataset or not self.detector_stream or not self.seeds:
            raise ValueError("dataset, detector stream, and seeds must be specified")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.batch_size <= 0 or self.warmup_iterations < 0 or self.timing_repetitions <= 0:
            raise ValueError("invalid timing protocol")
        for count in (self.calibration_sample_count, self.utility_sample_count):
            if count is not None and count <= 0:
                raise ValueError("sample counts must be positive when supplied")
