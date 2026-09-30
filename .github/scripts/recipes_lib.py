# SPDX-License-Identifier: MIT OR Apache-2.0
"""The launch-recipe checks, as pure functions over plain data.

Nothing here performs I/O. `recipes.py` reads the tree, runs `met` and writes
the bundle; `recipes-selftest.py` drives these functions from fixtures, with a
negative control for every check and a mutation of every check.

What a recipe must satisfy, and where each rule comes from in the engine:

  * `parse`        crates/server/src/recipe/mod.rs `Recipe::parse`: a mapping with
                   `recipe_version`, `model`, `container`, a `defaults:` mapping of
                   scalars, an optional `env:` mapping of scalars, an integer
                   `min_nodes`. The id is the path under recipes/ without `.yaml`
                   (fetch_github.rs `recipe_id`).
  * `render_argv`  `Recipe::argv_edited` + schema.rs: keys in byte order, each key
                   mapped to its flag (the manifest's `recipe_aliases` are the
                   engine's RENAMES), a presence-only flag rendered bare for `true`
                   and refused for anything else, `--world-size` from `min_nodes`.
  * `met_verdict`  what `met serve <argv without MODEL> --no-tui` answers. clap,
                   the METRALE_* lever check and `validate_serve_args` all run
                   before the model is looked at, and a modelless plain-mode boot
                   then stops with SENTINEL. Any other outcome is a refusal.

Finding kinds are stable strings: the self-test asserts on them.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import yaml

# crates/server/src/main_modules/serve.rs: the error a plain-mode `met serve`
# with no MODEL stops with, after the whole command line has been validated.
SENTINEL = "no model given, and no dashboard to choose one from"

# The GPU memory fraction a box class can be driven to. GB10 froze at 0.90 and
# needed a power cycle; 0.85 is the measured safe ceiling. Not declared in
# kernels/<hw>/HARDWARE.toml yet, so it lives here and only ever WARNS.
UTIL_CEILING = {"gb10": 0.85}

REQUIRED = ("recipe_version", "model", "container")

_DOCTOR_LINE = re.compile(r"^\s*ok\s+recipes\s+([0-9]+) cached in ", re.M)


@dataclass
class Finding:
    kind: str
    subject: str
    message: str
    error: bool = True

    def render(self) -> str:
        return f"{'ERROR' if self.error else 'WARN '} [{self.kind}] {self.subject}: {self.message}"


@dataclass
class Recipe:
    id: str
    path: str
    text: str
    version: str
    model: str
    container: str
    runtime: str | None
    min_nodes: int
    defaults: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    @property
    def is_metrale(self) -> bool:
        return self.runtime == "metrale"


class RecipeError(ValueError):
    pass


def recipe_id(rel_path: str) -> str:
    """`qwen3.6/foo.yaml` (relative to recipes/) -> `qwen3.6/foo`."""
    if not rel_path.endswith(".yaml"):
        raise RecipeError(f"{rel_path}: a recipe file ends in .yaml")
    return rel_path[: -len(".yaml")]


def _scalar_map(rid: str, block: object, name: str) -> dict[str, str]:
    if not isinstance(block, dict):
        raise RecipeError(f"{rid}: `{name}:` must be a mapping")
    out: dict[str, str] = {}
    for key, value in block.items():
        if not isinstance(value, str):
            raise RecipeError(f"{rid}: {name}.{key} is not a scalar")
        out[str(key)] = value
    return out


def parse(rel_path: str, text: str) -> Recipe:
    """One recipe file. BaseLoader keeps every scalar as its text, as the
    engine's reader does: `0.90` stays `0.90` and `true` stays `true`."""
    rid = recipe_id(rel_path)
    try:
        doc = yaml.load(text, Loader=yaml.BaseLoader)
    except yaml.YAMLError as e:
        raise RecipeError(f"{rid}: not YAML: {e}") from None
    if not isinstance(doc, dict):
        raise RecipeError(f"{rid}: the document must be a mapping")
    for key in REQUIRED:
        if not isinstance(doc.get(key), str) or not doc.get(key):
            raise RecipeError(f"{rid}: missing required key {key!r}")
    if "defaults" not in doc:
        raise RecipeError(f"{rid}: `defaults:` must be a mapping")
    defaults = _scalar_map(rid, doc["defaults"], "defaults")
    env = _scalar_map(rid, doc["env"], "env") if "env" in doc else {}
    nodes = doc.get("min_nodes", "1")
    if not isinstance(nodes, str) or not nodes.isdigit():
        raise RecipeError(f"{rid}: min_nodes {nodes!r} is not a number")
    runtime = doc.get("runtime")
    return Recipe(id=rid, path=f"recipes/{rel_path}", text=text, version=doc["recipe_version"],
                  model=doc["model"], container=doc["container"],
                  runtime=runtime if isinstance(runtime, str) else None,
                  min_nodes=int(nodes), defaults=defaults, env=env)


# ── the flag surface (`met dump-serve-options`) ──────────────────────────────


def flag_table(manifest: dict) -> dict[str, dict]:
    """Recipe key -> manifest flag entry, under the key and every recipe alias."""
    if manifest.get("schema_version") != 2:
        raise RecipeError(f"serve-options schema_version {manifest.get('schema_version')!r}; this reader knows 2")
    table: dict[str, dict] = {}
    for flag in manifest["flags"]:
        for key in [flag["key"], *flag.get("recipe_aliases", [])]:
            table[key] = flag
    return table


def lever_names(manifest: dict) -> set[str]:
    return {lever["env"] for lever in manifest["levers"]}


def flag_for(key: str, table: dict[str, dict]) -> str:
    """schema.rs `flag_for`: the aliased flag, else the key with `_` -> `-`.
    An unknown key still renders, so `met` refuses it by name."""
    return table[key]["flag"] if key in table else key.replace("_", "-")


def check_surface(recipe: Recipe, table: dict[str, dict], levers: set[str]) -> list[Finding]:
    """What the manifest alone can refuse. `met` refuses the same things; these
    say it without a binary and name the key rather than the flag."""
    out: list[Finding] = []
    if not recipe.is_metrale:
        return out
    for key, value in sorted(recipe.defaults.items()):
        entry = table.get(key)
        if entry is None:
            out.append(Finding("unknown-key", recipe.id,
                               f"defaults.{key} maps to --{flag_for(key, table)}, which `met serve` does not have"))
            continue
        if entry["presence_only"] and value != "true":
            out.append(Finding("presence-only", recipe.id,
                               f"defaults.{key}: --{entry['flag']} takes no value; write `{key}: true` "
                               f"or delete the line (got {value!r})"))
        elif entry.get("options") and value not in entry["options"]:
            out.append(Finding("bad-value", recipe.id,
                               f"defaults.{key}: {value!r} is not one of {', '.join(entry['options'])}"))
    for name in sorted(recipe.env):
        if name.startswith("METRALE_") and name not in levers:
            out.append(Finding("unknown-lever", recipe.id, f"env.{name} is not a declared METRALE_* lever"))
    return out


def render_argv(recipe: Recipe, table: dict[str, dict]) -> list[str]:
    """`Recipe::argv` with no overrides: `["met", "serve", MODEL, ...]`."""
    argv = ["met", "serve", recipe.model]
    for key in sorted(recipe.defaults, key=lambda k: k.encode()):
        value = recipe.defaults[key]
        flag = f"--{flag_for(key, table)}"
        presence = table.get(key, {}).get("presence_only", False)
        if presence and value == "true":
            argv.append(flag)
        elif presence and value == "false":
            continue
        else:
            argv += [flag, value]
    if recipe.min_nodes > 1:
        argv += ["--world-size", str(recipe.min_nodes)]
    return argv


def met_command(met: str, argv: list[str]) -> list[str]:
    """The argv `met` is run with: the recipe's own flags, without MODEL, in
    plain mode so the modelless boot stops instead of opening a dashboard."""
    flags = argv[3:]
    return [met, "serve", *flags] + ([] if "--no-tui" in flags else ["--no-tui"])


def met_verdict(returncode: int, output: str) -> tuple[bool, str]:
    """Accepted only on the one outcome that proves every check ran: exit 1
    carrying SENTINEL. Exit 0, a signal, a clap error (2), or exit 1 with any
    other message is a refusal, so a changed engine fails closed."""
    if returncode == 1 and SENTINEL in output:
        return True, "accepted"
    lines = [ln for ln in output.strip().splitlines() if ln.strip() and " INFO " not in ln]
    return False, f"exit {returncode}: " + (" | ".join(lines[-6:]) or "<no output>")


# ── BENCH.toml references and the advisory ceiling ────────────────────────────


def bench_refs(doc: object, found: list[str] | None = None) -> list[str]:
    """Every string value of a `recipe` key anywhere in a parsed BENCH.toml."""
    found = [] if found is None else found
    if isinstance(doc, dict):
        for key, value in doc.items():
            if key == "recipe" and isinstance(value, str):
                found.append(value)
            else:
                bench_refs(value, found)
    elif isinstance(doc, list):
        for item in doc:
            bench_refs(item, found)
    return found


def check_bench_refs(refs: dict[str, list[str]], ids: set[str]) -> list[Finding]:
    return [Finding("bench-missing", path, f"recipe = {rid!r} is not a file under recipes/")
            for path, rids in sorted(refs.items()) for rid in rids if rid not in ids]


def hardware_class(container: str) -> str | None:
    """`metrale/metrale-inference-gb10:latest` -> `gb10`; None when the image
    name ends in no class this file has a ceiling for."""
    name = container.split("@")[0].rsplit(":", 1)[0].rsplit("/", 1)[-1]
    suffix = name.rsplit("-", 1)[-1]
    return suffix if suffix in UTIL_CEILING and "-" in name else None


def check_util(recipe: Recipe) -> list[Finding]:
    hw = hardware_class(recipe.container)
    raw = recipe.defaults.get("gpu_memory_utilization")
    if hw is None or raw is None:
        return []
    try:
        util = float(raw)
    except ValueError:
        return [Finding("util-ceiling", recipe.id, f"gpu_memory_utilization {raw!r} is not a number")]
    if util > UTIL_CEILING[hw]:
        return [Finding("util-ceiling", recipe.id,
                        f"gpu_memory_utilization {raw} exceeds the {hw} ceiling {UTIL_CEILING[hw]}", error=False)]
    return []


# ── the bundle ────────────────────────────────────────────────────────────────


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sidecar(name: str, data: bytes) -> str:
    """`sha256sum` format, bare file name: checked with `sha256sum -c` from the
    directory holding the asset, like every other release asset."""
    return f"{sha256(data)}  {name}\n"


def index_document(recipes: list[Recipe], commit: str, committed_at: int,
                   assets: dict[str, bytes], engine_version: str) -> dict:
    """The release's `index.json`. `tree_sha`, `fetched_at` and `files` are the
    engine's recipe-cache shape (recipe/fetch.rs `parse_cache`), so the file is
    also a drop-in `~/.metrale/metrale-recipes/index.json` pinned to `commit`."""
    ordered = sorted(recipes, key=lambda r: r.id)
    return {
        "schema_version": 1,
        "commit": commit,
        "engine_version": engine_version,
        "assets": {name: sha256(data) for name, data in sorted(assets.items())},
        "recipes": [{"id": r.id, "path": r.path, "sha256": sha256(r.text.encode()), "bytes": len(r.text.encode()),
                     "model": r.model, "runtime": r.runtime, "container": r.container, "min_nodes": r.min_nodes}
                    for r in ordered],
        "tree_sha": commit,
        "fetched_at": committed_at,
        "files": {r.id: r.text for r in ordered},
    }


def doctor_verdict(stdout: str, expected: int) -> tuple[bool, str]:
    """`met doctor`'s recipes line must count every file: the engine's cache
    reader drops a recipe it cannot parse instead of failing."""
    m = _DOCTOR_LINE.search(stdout)
    if not m:
        return False, "met doctor printed no `ok recipes N cached` line: " + stdout.strip().replace("\n", " | ")
    got = int(m.group(1))
    if got != expected:
        return False, f"the engine's reader kept {got} of {expected} recipes; the rest do not parse"
    return True, f"the engine's reader kept all {got}"


def dumps(doc: dict) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=False, ensure_ascii=False) + "\n").encode()


# ── the metralectl mirror ─────────────────────────────────────────────────────


def mirror_verdict(changed_recipes: list[str] | None, pinned_serve_options: bytes | None,
                   release_serve_options: bytes) -> tuple[bool, str]:
    """Whether a release must be mirrored into metralectl, which ships the recipes to users.

    Two things are mirrored (metralectl `scripts/engine-recipes.py`): `recipes/` and the
    release's `serve-options.json` (as `vendor/serve-options.v2.json`, byte for byte).
    `changed_recipes` is the list of recipe paths that differ between the commit the mirror is
    pinned to and this release, or None when that diff could not be computed;
    `pinned_serve_options` is the mirror's snapshot, or None when it could not be read.
    An unknown answers True: a needless mirror PR is closed in a click, a missed one leaves
    every user on stale recipes."""
    if changed_recipes is None:
        return True, "the recipes could not be compared with the mirrored commit"
    if pinned_serve_options is None:
        return True, "the mirror's serve-options snapshot could not be read"
    if changed_recipes:
        return True, f"{len(changed_recipes)} recipe file(s) changed since the mirrored commit"
    if pinned_serve_options != release_serve_options:
        return True, "serve-options.json differs from the mirror's snapshot"
    return False, "the mirror already carries these recipes and serve options"
