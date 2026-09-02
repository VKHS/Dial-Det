# Timing protocol

The article protocol uses batch size 1, FP16, 100 warm-up iterations, all
evaluation images, five repetitions, and stage-wise time/memory traces.

`dial_det.experiment.timing` performs explicit optional CUDA synchronization
before and after each observation. It never starts a background matrix
multiplication, utilization pulse, idle-prevention process, or unrelated CUDA
workload.

Record at least:

- GPU/CPU model;
- driver, CUDA, framework, and package versions;
- precision and batch size;
- input resolution and candidate/proposal caps;
- warm-up and repetition counts;
- synchronization policy;
- selector, graph/evidence, solver, ROI/head, and total time;
- peak allocated and reserved memory;
- summary statistic and confidence interval procedure.
