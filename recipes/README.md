# Launch recipes

Each file here is one validated way to serve one checkpoint: the container
image, the checkpoint, and the `met serve` settings it was measured under.
`metralectl run <recipe>` launches one; a benchmark gate names one by id in
`kernels/<hw>/<model>/BENCH.toml` (`recipe = "qwen3.8/qwen3.8-27b-nvfp4-throughput"`).

## Layout

`recipes/<family>/<name>.yaml`. A recipe's id is its path under `recipes/`
without `.yaml`, so `recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml` is
`qwen3.6/qwen3.6-35b-a3b-fp8-mtp`.

A `runtime: metrale` recipe's `defaults:` keys are `met serve` flags with `_` for
`-` (`max_model_len`, `tensor_parallel` and `host` are spelled differently; see
`crates/server/src/recipe/schema.rs`). A key set to `true` on a boolean flag
passes the bare flag; writing `false` is refused, because leaving the line out
means the same. `env:` holds the `METRALE_*` levers the recipe was measured
under.

## Checks

The `recipes` job in `.github/workflows/ci.yml`, a required check, runs
`.github/scripts/recipes.py check` against the `met` built from the same commit:

- every key and value is on the `met dump-serve-options` flag surface, and every
  `env:` name is a declared lever;
- `met serve` accepts each recipe's command line (the flag parser, the lever
  check and `validate_serve_args`, all of which run before a model is loaded);
- the engine's own recipe reader keeps every file (`met doctor`);
- every `recipe = "..."` id in `kernels/**/BENCH.toml` is a file here;
- `gpu_memory_utilization` above the image's hardware class ceiling
  (`kernels/<hw>/HARDWARE.toml [memory] util_ceiling`; GB10 0.85) is refused.

Run it locally with a built `met`:

```sh
python3 .github/scripts/recipes.py check --met target/release/met --out /tmp/recipes-dist
```

## Releases

Each dev release (`bNNNN`) attaches `recipes.tar.gz` (this directory),
`index.json` (every recipe's id, path and sha256) and `serve-options.json` (the
flag surface they were checked against), each with a `.sha256`. `index.json` is
also in the shape of `met`'s recipe cache: saved as
`~/.metrale/metrale-recipes/index.json`, it is what `met`'s recipe library and launches
read until the next `met sync-recipes`.

A benchmark gate never reads that cache. It serves the recipe its BENCH entry names
from `recipes/` in the tree under test, refuses a recipe the tree does not have, and
records the recipe's canonical content hash; a later commit keeps the record only
while its recipe hashes the same (`crates/bench/src/gate/recipe_closure.rs`).

## Provenance

2026-10-02: 19 GB10 recipes had `gpu_memory_utilization` above 0.85 and now
read 0.85. Apart from that, the 32 recipe files are byte-identical to `recipes/` in
[Metrale/metralectl](https://github.com/Metrale/metralectl) at
`e89a8d1b7dd46cbabdf454ac3cde7a012197e3ba`.
`qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head-experts-nvfp4.yaml` and
`qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head-experts-nvfp4-gate-up.yaml` were added here since.
