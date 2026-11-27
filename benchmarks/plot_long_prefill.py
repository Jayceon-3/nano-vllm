"""Plot request A's inter-token latency around request B's long prefill."""

import argparse
import json
import shutil
import subprocess
from io import StringIO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Trace JSON from the long-prefill workload.")
    parser.add_argument("--output", type=Path, required=True, help="PNG path for the figure.")
    args = parser.parse_args()

    trace_path = args.trace.expanduser()
    output_path = args.output.expanduser()
    if trace_path.resolve() == output_path.resolve():
        parser.error("the figure and trace must use different paths")
    if output_path.suffix.lower() != ".png":
        parser.error("--output must be a PNG path")
    with trace_path.open(encoding="utf-8") as trace_file:
        trace = json.load(trace_file)
    if trace["configuration"]["arrival_pattern"] != "long-prefill" or not trace.get("steps"):
        parser.error("expected a long-prefill run recorded with --trace-output")

    requests = {request["request_id"]: request for request in trace["requests"]}
    if 0 not in requests or 1 not in requests:
        parser.error("the trace must contain completed requests A (0) and B (1)")
    request_a, request_b = requests[0], requests[1]
    a_token_times = [float(timestamp) for timestamp in request_a["token_times"]]
    arrival = float(request_b["arrival_time"])
    if len(a_token_times) < 2:
        parser.error("request A must produce at least two tokens")
    if not a_token_times[1] <= arrival < a_token_times[-1]:
        parser.error(
            f"B arrived at {arrival * 1000:.1f} ms, outside A's observed decode window "
            f"[{a_token_times[1] * 1000:.1f}, {a_token_times[-1] * 1000:.1f}) ms. "
            "Adjust --long-prefill-delay-ms and rerun the benchmark."
        )

    # A step that runs B up through B's first output token is prefill work for B.
    prefill_steps = [
        step for step in trace["steps"]
        if request_b["request_id"] in step["selected_request_ids"]
        and step["start_time"] < request_b["token_times"][0]
    ]
    if not prefill_steps or request_a["request_id"] not in prefill_steps[0]["decode_waiting_request_ids"]:
        parser.error("the trace does not show A waiting for decode when B's prefill starts")

    gnuplot = shutil.which("gnuplot")
    if gnuplot is None:
        parser.error("gnuplot is required to render the figure")
    script = StringIO()
    script.write("""set terminal pngcairo size 1200,520 font 'Sans,12'
set title "Request A's ITL during long prompt insertion"
set xlabel 'Time (s)'
set ylabel 'ITL (ms)'
set yrange [0:*]
set grid ytics
set key outside right top
""")
    script.write(f"set arrow from {arrival}, graph 0 to {arrival}, graph 1 nohead dashtype 2 lc rgb '#9A4B00' front\n")
    for step in prefill_steps:
        start, end = float(step["start_time"]), float(step["end_time"])
        script.write(
            f"set object rectangle from {start}, graph 0 to {end}, graph 1 "
            "fc rgb '#FFCF8A' fs solid 0.45 noborder behind\n"
        )
    script.write("""plot '-' using 1:2 with lines lw 2 lc rgb '#2166AC' title "A's ITL", \\
     NaN with lines dashtype 2 lc rgb '#9A4B00' title "B's planned arrival", \\
     NaN with boxes lc rgb '#FFCF8A' fs solid 0.45 noborder title "B's prefill"
""")
    for start, end in zip(a_token_times, a_token_times[1:]):
        script.write(f"{end} {(end - start) * 1000}\n")
    script.write("e\n")

    png = subprocess.run([gnuplot], input=script.getvalue().encode(), stdout=subprocess.PIPE, check=True).stdout
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)
    print(f"A was decoding at B's arrival ({arrival * 1000:.1f} ms). Figure: {output_path}")


if __name__ == "__main__":
    main()
