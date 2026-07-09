# Telemetry logs

This folder holds the raw telemetry logs underlying Table I of the manuscript:
three continuous three-hour benchmarks (staggered / parallel / sequential
scheduling) on a Raspberry Pi 5, sampled every 0.5 s, comprising 63,952
samples in total.

Columns: `Timestamp, FPS, Leaf_Lat_ms, Pest_Lat_ms, CPU_%, RAM_MB, Temp_C,
Freq_MHz, Leaf_Detections, Pest_Detections`.

The logs are released exactly as recorded: the frame-rate filter of Section
III-C has not been applied. Every entry of Table I, the per-inference
intervals of Section IV-C, the latency-distribution statistics of Section V,
and the per-cycle energy estimates of Section IV-D are recoverable from these
files directly.
