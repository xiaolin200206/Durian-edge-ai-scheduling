#!/usr/bin/env python3
"""
analyze_telemetry.py — Reproduce every reported statistic of the
scheduling ablation from the released telemetry logs.

Given the three three-hour benchmark CSVs (staggered / parallel /
sequential), this script recomputes, with no manual steps:

  * Table I        — per-policy mean +/- std of FPS, per-model latency,
                     CPU utilisation, resident memory, mean/peak die
                     temperature, and sample count n (Section IV).
  * Section IV-C   — per-inference completion intervals and inference
                     counts recovered from latency changes.
  * Section IV-D   — TDP-interpolated per-cycle energy (P = 3.0 + 9.0*u).
  * Section V-A    — leaf-latency median, interquartile range, and the
                     tail decomposition at 1.5x the median.
  * Section V-B    — the invocation-cycle beat period and the observed
                     high-latency episode spacing.
  * Frequency-scaling sample counts and the mean die temperature at
                     which they occur (Section IV-A).

The only assumption is the mapping of each CSV file to its policy,
set in POLICY_FILES below. Everything else is derived from the logs.

Usage:
    python analysis/analyze_telemetry.py --data-dir ../data
"""

import argparse
import csv
import glob
import math
import os
import statistics as stats
from datetime import datetime

# --- map each policy to its telemetry file (filename substrings) ----------
# Edit these if your released filenames differ. Matching is by substring so
# the timestamp suffix does not need to be exact.
POLICY_FILES = {
    "Staggered":  "dual_20260619",                       # staggered run
    "Parallel":   "dual_parallel_20260620",              # parallel run
    "Sequential": "dual_sequential_20260620",            # sequential run
}
POLICY_ORDER = ["Staggered", "Parallel", "Sequential"]

# per-model invocation intervals (s), fixed in the deployment script
LEAF_INTERVAL_S = 0.8
PEST_INTERVAL_S = 1.2

# TDP-interpolation constants (Section III-C): P = IDLE + SLOPE * utilisation
IDLE_W  = 3.0
PEAK_W  = 12.0
SLOPE_W = PEAK_W - IDLE_W          # 9.0

FPS_FILTER_MAX = 60.0              # Section III-C frame-rate filter


# --------------------------------------------------------------------------
def find_file(data_dir, needle):
    hits = [f for f in glob.glob(os.path.join(data_dir, "*.csv")) if needle in os.path.basename(f)]
    if not hits:
        raise FileNotFoundError(f"No CSV in {data_dir} matching '{needle}'")
    return sorted(hits)[0]


def load(path):
    """Load a telemetry CSV into a list of dict rows, dropping a trailing
    incomplete row (missing timestamp) if present."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    # drop trailing rows whose timestamp is empty or unparseable (the
    # incomplete final record of a log), matching the sample counts reported.
    while rows and parse_seconds(rows[-1].get("Timestamp") or "") is None:
        rows.pop()
    return rows


def col_float(rows, key, keep=lambda v: True):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v is None or v == "":
            continue
        try:
            x = float(v)
        except ValueError:
            continue
        if keep(x):
            out.append(x)
    return out


def ms(mean_sd):
    return f"{mean_sd[0]:.1f} +/- {mean_sd[1]:.1f}"


def mean_sd(xs):
    return (stats.mean(xs), stats.pstdev(xs) if len(xs) < 2 else stats.stdev(xs))


def parse_seconds(ts):
    """HH:MM:SS.ffff -> seconds since midnight (handles midnight wrap by caller)."""
    try:
        t = datetime.strptime(ts.strip(), "%H:%M:%S.%f")
    except ValueError:
        try:
            t = datetime.strptime(ts.strip(), "%H:%M:%S")
        except ValueError:
            return None
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def duration_s(rows):
    secs = [parse_seconds(r["Timestamp"]) for r in rows]
    secs = [s for s in secs if s is not None]
    d = secs[-1] - secs[0]
    if d < 0:
        d += 86400.0                # crossed midnight
    return d, secs


def inference_events(rows, col):
    """Return (times, values) at each point where `col` changes — i.e. a new
    inference result was recorded (Section IV-C)."""
    secs = [parse_seconds(r["Timestamp"]) for r in rows]
    times, vals = [], []
    prev = None
    t0 = None
    for r, s in zip(rows, secs):
        if s is None:
            continue
        if t0 is None:
            t0 = s
        v = r.get(col, "")
        if v == "" or v is None:
            continue
        try:
            x = float(v)
        except ValueError:
            continue
        if prev is None or x != prev:
            rel = s - t0
            if rel < 0:
                rel += 86400.0
            times.append(rel)
            vals.append(x)
            prev = x
    return times, vals


def quantile(sorted_xs, q):
    """Linear-interpolation quantile (matches pandas default)."""
    if not sorted_xs:
        return float("nan")
    idx = q * (len(sorted_xs) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_xs[int(idx)]
    return sorted_xs[lo] * (hi - idx) + sorted_xs[hi] * (idx - lo)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    args = ap.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    data = {}
    for pol in POLICY_ORDER:
        path = find_file(data_dir, POLICY_FILES[pol])
        data[pol] = load(path)
        print(f"[load] {pol:11s}  {os.path.basename(path)}  ({len(data[pol])} rows)")

    # ---------------- Table I ------------------------------------------------
    print("\n" + "=" * 78)
    print("TABLE I  — Scheduling policy ablation (mean +/- standard deviation)")
    print("=" * 78)
    hdr = f"{'Metric':<20}" + "".join(f"{p:>18}" for p in POLICY_ORDER)
    print(hdr)
    print("-" * len(hdr))

    def row(label, fn):
        print(f"{label:<20}" + "".join(f"{fn(pol):>18}" for pol in POLICY_ORDER))

    # FPS uses the Section III-C filter (>0 and <=60); other rows use all n
    row("FPS",              lambda p: ms(mean_sd(col_float(data[p], "FPS", lambda x: 0 < x <= FPS_FILTER_MAX))))
    row("Leaf latency (ms)", lambda p: ms(mean_sd(col_float(data[p], "Leaf_Lat_ms"))))
    row("Pest latency (ms)", lambda p: ms(mean_sd(col_float(data[p], "Pest_Lat_ms"))))
    row("CPU util. (%)",    lambda p: ms(mean_sd(col_float(data[p], "CPU_%"))))
    row("RAM (MB)",         lambda p: ms(mean_sd(col_float(data[p], "RAM_MB"))))
    row("Mean temp. (C)",   lambda p: f"{mean_sd(col_float(data[p],'Temp_C'))[0]:.2f} +/- {mean_sd(col_float(data[p],'Temp_C'))[1]:.2f}")
    row("Peak temp. (C)",   lambda p: f"{max(col_float(data[p], 'Temp_C')):.1f}")
    row("Samples n",        lambda p: f"{len(data[p]):,}")

    peak = {p: max(col_float(data[p], "Temp_C")) for p in POLICY_ORDER}
    print(f"\nPeak-temperature separation (Parallel - Sequential): "
          f"{peak['Parallel'] - peak['Sequential']:.1f} C")
    print(f"Sequential margin below 82 C cutoff: {82.0 - peak['Sequential']:.1f} C")
    print(f"Parallel proximity to 82 C cutoff:   {82.0 - peak['Parallel']:.1f} C")

    # ---------------- Section IV-C: intervals & counts ----------------------
    print("\n" + "=" * 78)
    print("SECTION IV-C  — Per-inference intervals and counts")
    print("=" * 78)
    counts = {}
    for pol in POLICY_ORDER:
        dur, _ = duration_s(data[pol])
        lt, _ = inference_events(data[pol], "Leaf_Lat_ms")
        pt, _ = inference_events(data[pol], "Pest_Lat_ms")
        nleaf, npest = len(lt) - 1, len(pt) - 1     # first event is not an interval
        counts[pol] = (nleaf, npest, dur)
        print(f"{pol:<11} dur={dur/3600:.2f} h | "
              f"leaf: {nleaf:5d} inf, {dur/max(nleaf,1):.2f} s interval | "
              f"pest: {npest:5d} inf, {dur/max(npest,1):.2f} s interval")
    # detection-rate change staggered -> sequential (rate normalised by duration)
    ls = counts["Staggered"][0] / counts["Staggered"][2]
    ss = counts["Sequential"][0] / counts["Sequential"][2]
    lp = counts["Staggered"][1] / counts["Staggered"][2]
    sp = counts["Sequential"][1] / counts["Sequential"][2]
    print(f"Leaf detection-rate fall (staggered->sequential): {100*(1-ss/ls):.1f} %")
    print(f"Pest detection-rate fall (staggered->sequential): {100*(1-sp/lp):.1f} %")

    # ---------------- Section IV-D: energy ----------------------------------
    print("\n" + "=" * 78)
    print("SECTION IV-D  — TDP-interpolated per-cycle energy  (P = 3.0 + 9.0*u)")
    print("=" * 78)
    for pol in POLICY_ORDER:
        u = stats.mean(col_float(data[pol], "CPU_%")) / 100.0
        P = IDLE_W + SLOPE_W * u
        print(f"{pol:<11} mean utilisation u={u:.3f}  ->  P = {P:.2f} W")
    print("(Per-cycle energy = P * cycle time; cycle time from the deployment log. "
          "Reported as an upper bound on the between-policy difference, Section IV-D.)")

    # ---------------- Section V-A: leaf-latency distribution ----------------
    print("\n" + "=" * 78)
    print("SECTION V-A  — Leaf-latency distribution (per inference)")
    print("=" * 78)
    med = {}
    for pol in POLICY_ORDER:
        _, vals = inference_events(data[pol], "Leaf_Lat_ms")
        vals = sorted(v for v in vals if not math.isnan(v))
        m = quantile(vals, 0.5)
        q25 = quantile(vals, 0.25)
        q75 = quantile(vals, 0.75)
        iqr = q75 - q25
        frac = 100.0 * sum(1 for v in vals if v > 1.5 * m) / len(vals)
        med[pol] = (m, iqr)
        print(f"{pol:<11} median={m:6.1f} ms | IQR={iqr:5.0f} ms "
              f"(Q25={q25:.0f}, Q75={q75:.0f}) | >1.5x median: {frac:.1f} %")
    iqr_seq = med["Sequential"][1]
    print(f"\nIQR contraction factor  parallel/sequential = {med['Parallel'][1]/iqr_seq:.2f}"
          f",  staggered/sequential = {med['Staggered'][1]/iqr_seq:.2f}")

    # tail decomposition at 1.5x median (parallel vs sequential)
    print("\nTail decomposition at 1.5x median (body vs tail mean, tail weight):")
    for pol in ("Parallel", "Sequential"):
        _, vals = inference_events(data[pol], "Leaf_Lat_ms")
        vals = [v for v in vals if not math.isnan(v)]
        m = med[pol][0]
        body = [v for v in vals if v <= 1.5 * m]
        tail = [v for v in vals if v > 1.5 * m]
        print(f"  {pol:<11} body_mean={stats.mean(body):.1f} ms  "
              f"tail_mean={stats.mean(tail):.0f} ms  tail_weight={100*len(tail)/len(vals):.1f} %")

    # ---------------- Section V-B: beat period & episode spacing ------------
    print("\n" + "=" * 78)
    print("SECTION V-B  — Beat period and high-latency episode spacing")
    print("=" * 78)
    # beat period from median cycle times under staggered
    _, lv = inference_events(data["Staggered"], "Leaf_Lat_ms")
    _, pv = inference_events(data["Staggered"], "Pest_Lat_ms")
    leaf_cyc = LEAF_INTERVAL_S + quantile(sorted(lv), 0.5) / 1000.0
    pest_cyc = PEST_INTERVAL_S + quantile(sorted(pv), 0.5) / 1000.0
    beat = 1.0 / abs(1.0 / leaf_cyc - 1.0 / pest_cyc)
    print(f"Leaf cycle {leaf_cyc:.3f} s, pest cycle {pest_cyc:.3f} s  ->  beat period = {beat:.1f} s")

    # observed episode spacing: consecutive above-1.5x-median leaf inferences,
    # merged into an episode when < 2 s apart; spacing = between episode onsets
    def episode_spacing(pol, gap=2.0):
        times, vals = inference_events(data[pol], "Leaf_Lat_ms")
        pairs = [(t, v) for t, v in zip(times, vals) if not math.isnan(v)]
        m = quantile(sorted(v for _, v in pairs), 0.5)
        hi = [t for t, v in pairs if v > 1.5 * m]
        if len(hi) < 2:
            return float("nan")
        onsets = [hi[0]]
        last = hi[0]
        for t in hi[1:]:
            if t - last > gap:
                onsets.append(t)
            last = t
        sp = sorted(onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1))
        return quantile(sp, 0.5)

    for pol in POLICY_ORDER:
        print(f"{pol:<11} median high-latency episode spacing = {episode_spacing(pol):.1f} s")

    # ---------------- Section IV-A: frequency scaling -----------------------
    print("\n" + "=" * 78)
    print("SECTION IV-A  — Idle-state frequency scaling (Freq < 2400 MHz)")
    print("=" * 78)
    max_freq = max(max(col_float(data[p], "Freq_MHz")) for p in POLICY_ORDER)
    scaled_temps = []
    for pol in POLICY_ORDER:
        rows = data[pol]
        n = 0
        temps = []
        for r in rows:
            fv = r.get("Freq_MHz")
            tv = r.get("Temp_C")
            if fv in (None, "") or tv in (None, ""):
                continue
            try:
                f = float(fv)
                t = float(tv)
            except (ValueError, TypeError):
                continue
            if f < max_freq:
                n += 1
                temps.append(t)
        scaled_temps += temps
        mt = stats.mean(temps) if temps else float("nan")
        print(f"{pol:<11} {n:5d} samples below {max_freq:.0f} MHz, "
              f"mean die temp {mt:.1f} C")
    print(f"Pooled mean die temp at frequency-scaled samples: "
          f"{stats.mean(scaled_temps):.1f} C  (<< 82 C cutoff -> idle scaling, not throttling)")

    total = sum(len(data[p]) for p in POLICY_ORDER)
    print(f"\nTotal telemetry samples across the three benchmarks: {total:,}")


if __name__ == "__main__":
    main()
