#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Join the tree-derived inventory with the curated taxonomy, trade-offs and measurements.

Pure: takes the parsed TOML documents and the inventory, returns rows, and raises
`ModelError` naming every inconsistency it finds (the `--check` gate). Rules:

* every entry point matches a taxonomy rule; every rule, family model, component
  site prefix and family site prefix matches something that exists;
* every trade-off `match` names an existing file or `file::function`, carries text
  of at most TEXT_MAX characters, and cites PRs only as positive integers;
* every measurement row names an existing kernel, and its stored pct_of_floor
  equals floor_us / time_us * 100 to within PCT_TOL points.

Family attribution (the "LLMs" column), per engine call site of a kernel: a family
F is admitted when (1) a target of F compiles the kernel's file, (2) the kernel's
primary component is generic or one of F's components, (3) the component that owns
the call-site path is generic or one of F's components, and (4) when the path is
family-exclusive, F is one of those families. A family with shared_engine = false
is admitted only through its exclusive paths. Test and example call sites never
admit.

A kernel is UNIQUE to component C when every engine call site belongs to C: the
owner of the call-site path (component `sites`), or the kernel's primary component
when no component owns the path.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

TEXT_MAX = 400
PCT_TOL = 0.5
MEASURE_KEYS = {
    "kernel": str, "file": str, "hardware": str, "model": str, "regime": str,
    "time_us": (int, float), "bytes": int, "flops": int, "bound": str,
    "floor_us": (int, float), "pct_of_floor": (int, float), "source": str, "notes": str,
}


class ModelError(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass
class Row:
    k: object  # kernel_perf_inventory.Kernel
    component: str
    kind: str
    used_by: set[str] = field(default_factory=set)  # primary + site_components
    site_components: set[str] = field(default_factory=set)  # owner of each engine call-site path
    families: set[str] = field(default_factory=set)
    engine_sites: list[tuple[str, int]] = field(default_factory=list)
    other_sites: int = 0
    tradeoffs: list[dict] = field(default_factory=list)
    measures: list[dict] = field(default_factory=list)

    @property
    def hardware(self) -> list[str]:
        return sorted({t[0] for t in self.k.targets})


@dataclass
class Model:
    rows: list[Row]
    families: list[dict]
    components: list[dict]
    family_of_model: dict[str, str]
    checkpoints: dict[str, str]  # model directory -> checkpoint id (MODEL.toml hf_id)


def _globs(pats: list[str], s: str) -> bool:
    return any(fnmatch.fnmatchcase(s, p) for p in pats)


def _rule_for(rules: list[dict], k) -> int | None:
    for i, r in enumerate(rules):
        if "files" in r and not _globs(r["files"], k.file):
            continue
        if "names" in r and not _globs(r["names"], k.name):
            continue
        return i
    return None


def _longest(prefixes: dict[str, object], path: str):
    best = None
    for p, v in prefixes.items():
        if path.startswith(p) and (best is None or len(p) > len(best[0])):
            best = (p, v)
    return best[1] if best else None


def build(inv, tax: dict, tradeoffs: dict, measurements: dict, existing_paths: set[str]) -> Model:
    problems: list[str] = []
    fams = tax.get("family", [])
    comps = tax.get("component", [])
    rules = tax.get("rule", [])
    comp_by_id = {c["id"]: c for c in comps}
    fam_by_id = {f["id"]: f for f in fams}
    family_of_model: dict[str, str] = {}
    for f in fams:
        for c in f["components"]:
            if c not in comp_by_id:
                problems.append(f"taxonomy: family {f['id']} lists unknown component {c!r}")
        for m in f["models"]:
            if m in family_of_model:
                problems.append(f"taxonomy: model {m} is in two families")
            family_of_model[m] = f["id"]
    tree_models = {t[1] for t in inv.targets}
    for m in sorted(tree_models - set(family_of_model)):
        problems.append(f"taxonomy: model directory {m} belongs to no family")
    for m in sorted(set(family_of_model) - tree_models):
        problems.append(f"taxonomy: family model {m} is not a model directory in kernels/")
    site_comp: dict[str, str] = {}
    for c in comps:
        for p in c.get("sites", []):
            site_comp[p] = c["id"]
    site_fams: dict[str, list[str]] = {}
    for fs in tax.get("family_site", []):
        for p in fs["paths"]:
            site_fams[p] = fs["families"]
        for f in fs["families"]:
            if f not in fam_by_id:
                problems.append(f"taxonomy: family_site names unknown family {f!r}")
    for p in list(site_comp) + list(site_fams):
        if not any(e.startswith(p) for e in existing_paths):
            problems.append(f"taxonomy: site prefix {p!r} matches no file under crates/")
    for r in rules:
        if r["component"] not in comp_by_id:
            problems.append(f"taxonomy: rule for unknown component {r['component']!r}")

    rows: list[Row] = []
    used_rules: set[int] = set()
    for key in sorted(inv.kernels):
        k = inv.kernels[key]
        ri = _rule_for(rules, k)
        if ri is None:
            problems.append(f"taxonomy: no rule classifies {k.key}")
            continue
        used_rules.add(ri)
        row = Row(k, rules[ri]["component"], rules[ri]["kind"])
        row.used_by.add(row.component)
        primary = comp_by_id[row.component]
        compiled = {family_of_model.get(t[1]) for t in k.targets} - {None}
        for path, line, kind, _exact in k.sites:
            if kind != "engine":
                row.other_sites += 1
                continue
            row.engine_sites.append((path, line))
            sc = _longest(site_comp, path)
            row.site_components.add(sc or row.component)
            row.used_by.add(sc or row.component)
            excl = _longest(site_fams, path)
            for fid in compiled:
                f = fam_by_id[fid]
                if excl is not None and fid not in excl:
                    continue
                if excl is None and not f["shared_engine"]:
                    continue
                if not primary["generic"] and row.component not in f["components"]:
                    continue
                if sc and not comp_by_id[sc]["generic"] and sc not in f["components"]:
                    continue
                row.families.add(fid)
        rows.append(row)
    for i, r in enumerate(rules):
        if i not in used_rules:
            problems.append(f"taxonomy: rule {i} ({r['component']}: {r.get('names') or r.get('files')}) classifies nothing")

    by_file: dict[str, list[Row]] = {}
    by_key: dict[str, Row] = {}
    for row in rows:
        by_file.setdefault(row.k.file, []).append(row)
        by_key[row.k.key] = row
    for t in tradeoffs.get("t", []):
        where = f"tradeoffs: {t.get('match')!r}"
        text, prs = t.get("text", ""), t.get("prs", [])
        if not isinstance(text, str) or not text.strip() or len(text) > TEXT_MAX:
            problems.append(f"{where}: text must be 1..{TEXT_MAX} characters")
        if not isinstance(prs, list) or not all(isinstance(p, int) and p > 0 for p in prs):
            problems.append(f"{where}: prs must be a list of positive integers")
        m = t.get("match", "")
        targets = [by_key[m]] if m in by_key else by_file.get(m, [])
        if not targets and "::" not in m:
            targets = [r for r in rows if m in r.k.sources]
        if not targets:
            problems.append(f"{where}: names no kernel file or file::function in the inventory")
        for row in targets:
            row.tradeoffs.append(t)

    for m in measurements.get("m", []):
        where = f"measurements: {m.get('kernel')!r} @ {m.get('file')!r} ({m.get('regime')!r})"
        bad = [k for k, ty in MEASURE_KEYS.items() if not isinstance(m.get(k), ty)]
        if bad:
            problems.append(f"{where}: missing or mistyped {', '.join(bad)}")
            continue
        if m["time_us"] <= 0 or m["floor_us"] <= 0:
            problems.append(f"{where}: time_us and floor_us must be positive")
            continue
        if m["bound"] not in ("memory", "compute"):
            problems.append(f"{where}: bound must be 'memory' or 'compute'")
        derived = m["floor_us"] / m["time_us"] * 100.0
        if abs(derived - m["pct_of_floor"]) > PCT_TOL:
            problems.append(f"{where}: pct_of_floor {m['pct_of_floor']} != floor_us/time_us*100 = {derived:.2f}")
        func = m["kernel"].rsplit("::", 1)[-1]
        row = by_key.get(f"{m['file']}::{func}")
        if row is None:
            hits = [r for r in rows if r.k.name == func and m["file"] in r.k.sources]
            row = hits[0] if len(hits) == 1 else None
        if row is None:
            problems.append(f"{where}: no kernel {func} defined in or compiled from {m['file']}")
            continue
        row.measures.append(m)
    if problems:
        raise ModelError(problems)
    checkpoints = {model: hf for (_hw, model), hf in sorted(inv.hf_ids.items())}
    return Model(rows, fams, comps, family_of_model, checkpoints)
