#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Markdown for the generated block of KERNEL-PERF.md. Pure: model in, text out.

Rows that share a file, kind, hardware set and family set and carry no measurement are merged into one row whose first cell lists every entry
point (common prefix factored out); a measured entry point always gets its own row
so its % of floor is never averaged into a group.
"""
from __future__ import annotations

import os
import re

REPO = "https://github.com/Metrale/metrale-inference"
NOT_MEASURED = "not measured"
REGISTER = "docs/kernel-perf/TRADEOFFS.md"
MEASUREMENTS = "docs/kernel-perf/MEASUREMENTS.md"
HW_SHORT = {"gb10": "gb10", "hopper": "hop", "b200": "b200", "b300": "b300", "strix": "strix", "strix-hip": "hip", "metal": "metal"}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _md(s: str) -> str:
    """Plain text safe in Markdown: `<` would open an HTML tag, `*` an emphasis run."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace("*", "\\*")


def _compress(names: list[str]) -> str:
    names = sorted(set(names))
    if len(names) == 1:
        return f"`{names[0]}`"
    prefix = os.path.commonprefix(names)
    cut = prefix.rfind("_") + 1 if "_" in prefix else 0
    prefix = prefix[:cut]
    if len(prefix) < 4:
        return ", ".join(f"`{n}`" for n in names)
    rest = ", ".join(n[len(prefix):] or "∅" for n in names)
    return f"`{prefix}{{{rest}}}` ({len(names)})"


def _regime(m: dict) -> str:
    return m["regime"].split(" · ")[0]


class Renderer:
    def __init__(self, model):
        self.m = model
        self.fam = {f["id"]: f for f in model.families}
        self.comp = {c["id"]: c for c in model.components}
        self.decoder = {f["id"] for f in model.families if f["shared_engine"]}
        self.refs: dict[str, str] = {}  # reference-style link id -> target, emitted once at the end
        self.used_names = {r.k.name for r in model.rows if r.families}
        self.file_ids = {f: f"f{i}" for i, f in enumerate(sorted({r.k.file for r in model.rows}), 1)}

    def ref(self, rid: str, target: str) -> str:
        self.refs[rid] = target
        return rid

    def pr(self, n: int) -> str:
        return f"[#{n}][{self.ref(f'pr{n}', f'{REPO}/pull/{n}')}]"

    # ---- cells ----------------------------------------------------------------------------------
    def families_cell(self, fams: set[str]) -> str:
        if not fams:
            return "—"
        n = sum(len(self.fam[f]["models"]) for f in fams)
        if self.decoder <= fams:
            extra = sorted(fams - self.decoder)
            head = f"all {len(self.decoder)} decoder families"
            return head + (" + " + ", ".join(self.fam[f]["label"] for f in extra) if extra else "") + f" ({n} ckpts)"
        return ", ".join(self.fam[f]["label"] for f in sorted(fams, key=lambda f: self.fam[f]["label"])) + f" ({n} ckpts)"

    def pct_cell(self, row) -> str:
        """Lowest–highest % over the kernel's rows, and the regime of the lowest."""
        ms = row.measures
        if not ms:
            return NOT_MEASURED
        worst = min(ms, key=lambda m: m["pct_of_floor"])
        lo, hi = worst["pct_of_floor"], max(m["pct_of_floor"] for m in ms)
        span = f"{lo:.0f}%" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f}%"
        rid = self.ref("m" + self.file_ids[row.k.file][1:] + "." + row.k.name, f"{MEASUREMENTS}#m-{_slug(row.k.key)}")
        return f"[{span}][{rid}] ({_cell(_regime(worst))})"

    def orphan_cell(self, rows) -> str:
        """Launched by engine code, yet no family both compiles this file and reaches a call site."""
        if all(r.k.name in self.used_names for r in rows):
            return "none — its callers' targets compile another copy"
        return "none — no family both compiles it and reaches its call sites"

    def file_cell(self, rows) -> str:
        k = rows[0].k
        line = min(r.k.line for r in rows)
        short = k.file.removeprefix("kernels/")
        return f"[{short}:{line}][{self.ref(self.file_ids[k.file], k.file)}]"

    def tradeoff_cell(self, rows) -> str:
        notes = []
        for r in rows:
            for t in r.tradeoffs:
                if t not in notes:
                    notes.append(t)
        if not notes:
            return "—"
        prs = sorted({p for t in notes for p in t.get("prs", [])})
        f = rows[0].k.file
        rid = self.ref("t" + self.file_ids[f][1:], f"{REGISTER}#to-{_slug(f)}")
        link = f"[{len(notes)} note{'s' if len(notes) > 1 else ''}][{rid}]"
        return link + ("" if not prs else " · " + " ".join(self.pr(p) for p in prs))

    # ---- grouping -------------------------------------------------------------------------------
    def groups(self, rows):
        out: dict[tuple, list] = {}
        for r in sorted(rows, key=lambda r: (r.k.file, r.k.name)):
            if r.measures:
                key = ("m", r.k.key)
            else:
                key = (r.k.file, r.kind, tuple(r.hardware), frozenset(r.families))
            out.setdefault(key, []).append(r)
        return list(out.values())

    def table(self, rows, *, component_col: bool = False) -> list[str]:
        head = "| Kernel (module::function) | File | " + ("Component · kind" if component_col else "Kind") + \
            " | HW | LLMs | Trade-offs · PRs | % of floor |"
        lines = [head, "|---|---|---|---|---|---|---|"]
        for g in self.groups(rows):
            r0 = g[0]
            module = sorted(r0.k.modules)[0]
            kernel = f"{module}::" + _compress([r.k.name for r in g])
            kind = f"{self.comp[r0.component]['name'].split(' (')[0]} · {r0.kind}" if component_col else r0.kind
            lines.append("| " + " | ".join([
                kernel, self.file_cell(g), _cell(kind), " ".join(HW_SHORT.get(h, h) for h in r0.hardware),
                self.families_cell(r0.families) if r0.families or not r0.engine_sites else self.orphan_cell(g),
                self.tradeoff_cell(g), self.pct_cell(r0),
            ]) + " |")
        return lines

    def borrowed_line(self, rows) -> str:
        by_comp: dict[str, list] = {}
        for r in rows:
            by_comp.setdefault(r.component, []).append(r)
        parts = []
        for cid in sorted(by_comp, key=lambda c: self.comp[c]["name"]):
            anchor = _slug(self.comp[cid]["name"])
            names = ", ".join(_compress([r.k.name for r in g]) for g in self.groups(by_comp[cid]))
            parts.append(f"[{self.comp[cid]['name'].split(' (')[0]}](#{anchor}): {names}")
        return "; ".join(parts) + "."

    # ---- sections -------------------------------------------------------------------------------
    def render(self) -> str:
        m = self.m
        live = [r for r in m.rows if r.engine_sites]
        idle = [r for r in m.rows if not r.engine_sites]
        measured = [r for r in m.rows if r.measures]
        shared_n = len(self.decoder)
        shared = [r for r in live if self.decoder <= r.families]
        out: list[str] = []
        out += ["## Inventory at a glance", ""]
        out += [
            f"- **{len(m.rows)} kernel entry points** in **{len({r.k.file for r in m.rows})} source files** "
            f"across {len({h for r in m.rows for h in r.hardware})} hardware trees "
            f"({', '.join(sorted({h for r in m.rows for h in r.hardware}))}), compiled into "
            f"{len({t for r in m.rows for t in r.k.targets})} (hardware, model, quant) targets.",
            f"- **{len(live)}** have at least one engine call site; **{len(idle)}** are compiled but launched only "
            "from tests, examples or not at all (see [Compiled but not launched](#compiled-but-not-launched)).",
            f"- **{len(m.families)} architecture families**, **{len(m.components)} components**.",
            f"- **{len(measured)}** entry points have a measured % of floor; every other row reads "
            f"“{NOT_MEASURED}”.",
            "",
        ]
        out += self.families_section(live)
        out += self.components_section(live, idle)
        out += [f"## Shared by all LLM architectures", "",
                f"Entry points used by **every one of the {shared_n} decoder families** (every family with "
                f"`shared_engine = true`; NLLB, the self-contained encoder-decoder, is excluded from the "
                f"quorum and named when it also uses the kernel). {len(shared)} entry points qualify; "
                f"{sum(1 for r in shared if r.families >= set(self.fam))} of them are used by all "
                f"{len(self.fam)} families.", ""]
        out += self.table(shared, component_col=True) + [""]
        out += ["## Kernels by component", "",
                "Every entry point with an engine call site, under each component that launches it: a full "
                "row under its primary component, and a name under every other component whose code "
                "launches it.", ""]
        for c in m.components:
            prim = [r for r in live if r.component == c["id"]]
            borrowed = [r for r in live if c["id"] in r.used_by and r.component != c["id"]]
            if not prim and not borrowed:
                continue
            out += [f"### {c['name']}", "", f"{len(prim) + len(borrowed)} entry points: {len(prim)} primary here "
                    f"(full rows), {len(borrowed)} of other components launched from this component's code "
                    "(listed after the table; their full rows are under their primary component).", ""]
            if prim:
                out += self.table(prim) + [""]
            if borrowed:
                out += ["Also launched here: " + self.borrowed_line(borrowed), ""]
        out += ["## Unique kernels by component", "",
                "Entry points whose every engine call site belongs to one component.", ""]
        for c in m.components:
            rows = [r for r in live if r.site_components == {c["id"]}]
            if not rows:
                continue
            out += [f"### Unique to {c['name']}", "", f"{len(rows)} entry points.", ""]
            out += self.table(rows) + [""]
        out += ["## Compiled but not launched", "",
                "No engine call site names these entry points: they are reached only from tests or "
                "`examples/` (microbenchmarks, the Metal Qwen3.5 driver), or not at all. They cost build "
                "time and are candidates for removal or for wiring up.", ""]
        out += self.table(idle, component_col=True) + [""]
        out += self.measurement_summary(measured)
        out += [f"[{rid}]: {target}" for rid, target in sorted(self.refs.items(), key=lambda x: (x[0][0], len(x[0]), x[0]))]
        return "\n".join(out).rstrip() + "\n"

    def families_section(self, live) -> list[str]:
        out = ["## Architecture families", "",
               "| Label | Family | Checkpoints (model directory → checkpoint) | Components | Entry points used |",
               "|---|---|---|---|---|"]
        for f in self.m.families:
            ckpts = "<br>".join(
                f"`{md}` → {self.m.checkpoints.get(md, md)}" for md in f["models"])
            comps = ", ".join(self.comp[c]["name"].split(" (")[0] for c in f["components"])
            n = sum(1 for r in live if f["id"] in r.families)
            out.append(f"| {f['label']} | {_cell(f['name'])} | {ckpts} | {comps} | {n} |")
        return out + [""]

    def components_section(self, live, idle) -> list[str]:
        out = ["## Components", "",
               "| Component | Scope | Primary entry points | Launched from it (incl. other primaries) | "
               "Unique to it | Not launched | Measured |", "|---|---|---|---|---|---|---|"]
        for c in self.m.components:
            prim = [r for r in live if r.component == c["id"]]
            used = [r for r in live if c["id"] in r.used_by]
            uniq = [r for r in live if r.site_components == {c["id"]}]
            dead = [r for r in idle if r.component == c["id"]]
            meas = [r for r in used if r.measures]
            scope = "every family" if c["generic"] else "families listing it"
            out.append(f"| {_cell(c['name'])} | {scope} | {len(prim)} | {len(used)} | {len(uniq)} | "
                       f"{len(dead)} | {len(meas)} |")
        return out + [""]

    def render_register(self) -> str:
        """The whole of docs/kernel-perf/TRADEOFFS.md (generated; links relative to that file)."""
        out = ["<!-- Generated by scripts/kernel_perf.py from docs/kernel-perf/tradeoffs.toml. Do not edit. -->",
               "", "# Kernel trade-off register", "",
               "Known trade-offs, limits and dated measurements per kernel source, curated in "
               "[`tradeoffs.toml`](tradeoffs.toml) and linked from every table of "
               "[KERNEL-PERF.md](../../KERNEL-PERF.md). *whole file* entries apply to every entry point "
               "the file defines or compiles.", ""]
        by_file: dict[str, list[tuple[str, dict]]] = {}
        for r in self.m.rows:
            for t in r.tradeoffs:
                scope = r.k.name if "::" in t["match"] else "whole file"
                lst = by_file.setdefault(r.k.file, [])
                if (scope, t) not in lst:
                    lst.append((scope, t))
        for f in sorted(by_file):
            out += [f'<a id="to-{_slug(f)}"></a>', "", f"### [{f}](../../{f})", ""]
            for scope, t in by_file[f]:
                prs = " ".join(f"[#{p}]({REPO}/pull/{p})" for p in t.get("prs", []))
                out.append(f"- *{scope}*: {_md(t['text'].strip())}" + (f" ({prs})" if prs else "") +
                           f" — source: {_md(t.get('source', ''))}")
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def measurement_summary(self, measured) -> list[str]:
        out = ["## Measurements", ""]
        rows = [x for r in measured for x in r.measures]
        if not rows:
            return out + [f"No measurement rows yet: every % of floor reads “{NOT_MEASURED}”.", ""]
        out += [f"{len(rows)} rows over {len(measured)} entry points; every row, with its shape, time, floor, "
                f"source and notes, is in [`{MEASUREMENTS}`]({MEASUREMENTS}). Per regime (median over the "
                "regime's rows, unweighted):", "",
                "| Hardware | Model | Regime | Rows | Entry points | Median % of floor |", "|---|---|---|---|---|---|"]
        groups: dict[tuple, list] = {}
        for r in measured:
            for x in r.measures:
                groups.setdefault((x["hardware"], x["model"], _regime(x)), []).append((r.k.key, x["pct_of_floor"]))
        for (hw, model, regime), xs in sorted(groups.items()):
            pcts = sorted(p for _k, p in xs)
            mid = len(pcts) // 2
            med = pcts[mid] if len(pcts) % 2 else (pcts[mid - 1] + pcts[mid]) / 2
            out.append(f"| {hw} | {_cell(model)} | {_cell(regime)} | {len(xs)} | {len({k for k, _p in xs})} | {med:.0f}% |")
        return out + [""]

    def render_measurements(self) -> str:
        """The whole of docs/kernel-perf/MEASUREMENTS.md (generated; links relative to that file)."""
        out = ["<!-- Generated by scripts/kernel_perf.py from docs/kernel-perf/measurements.toml. Do not edit. -->",
               "", "# Kernel measurements", "",
               "Every measured row behind the “% of floor” cells of [KERNEL-PERF.md](../../KERNEL-PERF.md), "
               "grouped by entry point. Method, peaks and byte/FLOP models: KERNEL-PERF.md, Methodology §6. "
               "Source data: [`measurements.toml`](measurements.toml).", ""]
        for r in sorted((r for r in self.m.rows if r.measures), key=lambda r: r.k.key):
            out += [f'<a id="m-{_slug(r.k.key)}"></a>', "",
                    f"### `{r.k.name}` — [{r.k.file}](../../{r.k.file}#L{r.k.line})", "",
                    "| HW | Model | Regime | time µs | floor µs | bound | % of floor | Source · notes |",
                    "|---|---|---|---|---|---|---|---|"]
            for x in sorted(r.measures, key=lambda x: (x["hardware"], x["model"], x["regime"])):
                out.append("| " + " | ".join([
                    x["hardware"], _cell(x["model"]), _cell(x["regime"]), f"{x['time_us']:.2f}",
                    f"{x['floor_us']:.2f}", x["bound"], f"{x['pct_of_floor']:.1f}%",
                    _cell(_md(x["source"])) + (f" — {_cell(_md(x['notes']))}" if x["notes"] else ""),
                ]) + " |")
            out.append("")
        return "\n".join(out).rstrip() + "\n"
