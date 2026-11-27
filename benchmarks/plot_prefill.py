"""Plot throughput and median step latency from a single-request prefill sweep."""

import argparse
import json
import shutil
import subprocess
from math import isfinite
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Summary JSON from benchmarks.bench_prefill.")
    parser.add_argument("--output", type=Path, required=True, help="PNG path for the figure.")
    args = parser.parse_args()

    summary_path = args.summary.expanduser()
    output_path = args.output.expanduser()
    if output_path.suffix.lower() != ".png" or output_path.resolve() == summary_path.resolve():
        parser.error("--output must be a separate PNG path")

    try:
        with summary_path.open(encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        measurements = sorted(
            (
                int(row["input_tokens"]),
                float(row["input_tokens_per_second"]) / 1000,
                float(row["median_prefill_step_ms"]),
            )
            for row in summary["measurements"]
        )
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        parser.error(f"could not read prefill measurements from {summary_path}: {error}")
    if not measurements or any(
        length <= 0 or not (isfinite(throughput) and throughput > 0 and isfinite(latency) and latency > 0)
        for length, throughput, latency in measurements
    ):
        parser.error("expected at least one prefill measurement with positive, finite values")

    gnuplot = shutil.which("gnuplot")
    if gnuplot is None:
        parser.error("gnuplot is required to render the figure")

    peak_length, peak_throughput, peak_latency = max(measurements, key=lambda row: row[1])
    data = "\n".join(f"{length} {throughput} {latency}" for length, throughput, latency in measurements)
    script = f"""set terminal pngcairo size 1080,760 font 'Sans,12'
$Measurements << EOD
{data}
EOD
set multiplot layout 2,1 title 'Single-request prefill sweep'
set lmargin 12
set rmargin 5
set xrange [0:*]
set yrange [0:*]
set grid ytics lc rgb '#DCE2E8'
set key off
set arrow from {peak_length}, graph 0 to {peak_length}, graph 1 nohead dt 2 lc rgb '#C05B23' lw 2 back
set label 'Measured peak at {peak_length} tokens ({peak_throughput:.2f}k tok/s)' at graph 0.03, graph 0.9
set ylabel 'Input throughput (k tok/s)'
plot $Measurements using 1:2 with linespoints lw 2 pt 7 ps 0.65 lc rgb '#2166AC'
unset label
set xlabel 'Prompt length (tokens)'
set ylabel 'Median prefill step (ms)'
plot $Measurements using 1:3 with linespoints lw 2 pt 7 ps 0.65 lc rgb '#2166AC'
unset multiplot
"""
    png = subprocess.run([gnuplot], input=script.encode(), stdout=subprocess.PIPE, check=True).stdout
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)
    print(f"Measured peak at {peak_length} tokens ({peak_throughput * 1000:,.2f} tok/s, {peak_latency:.2f} ms).")
    print(f"Figure: {output_path}")


if __name__ == "__main__":
    main()
