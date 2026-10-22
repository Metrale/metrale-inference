#!/usr/bin/env python3
"""Print a benchmark CSV (from `examples/bench --out`) as the Markdown tables the README embeds.

    python3 scripts/bench_table.py assets/bench/private.csv

One table per class; rows are tools, kzx first; columns: packed bytes, ratio, compress and
decompress MB/s. Numbers are copied verbatim from the CSV, so the README never carries a value
the benchmark did not print.
"""
import csv
import sys

ORDER = ["kzx", "deflate-9", "zstd-19-long", "zstd-22-ultra-long27", "xz-9e", "bzip2-9", "brotli-11"]
LABEL = {"ptx": "PTX", "cubin": "cubin (SASS ELF)", "body": "postcard body", "kbp": "full `.kbp`"}


def main(path):
    rows = list(csv.DictReader(open(path, newline="")))
    classes = []
    for r in rows:
        key = (r["class"], r["mode"])
        if key not in classes:
            classes.append(key)
    for cls, mode in classes:
        sub = {r["tool"]: r for r in rows if r["class"] == cls and r["mode"] == mode}
        any_row = next(iter(sub.values()))
        print(f"**{LABEL.get(cls, cls)}**, `{mode}`: {any_row['files']} file(s), {int(any_row['raw_bytes']):,} raw bytes\n")
        print("| tool | packed bytes | ratio | compress MB/s | decompress MB/s |")
        print("|---|---:|---:|---:|---:|")
        best = max(float(r["ratio"]) for r in sub.values())
        for tool in ORDER + sorted(set(sub) - set(ORDER)):
            r = sub.get(tool)
            if not r:
                continue
            name = f"**{tool}**" if tool == "kzx" else tool
            ratio = float(r["ratio"])
            mark = " ★" if ratio == best else ""
            print(f"| {name} | {int(r['packed_bytes']):,} | {ratio:.2f}x{mark} | {float(r['c_mbps']):.2f} | {float(r['d_mbps']):.2f} |")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
