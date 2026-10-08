# Archived midpoint-counter experiment

The rejected midpoint-retry full-model experiment remains reproducible at
[`9484665ee9a7b2445358163ec0e1e33f8c16d166`](https://github.com/Metrale/metrale-inference/commit/9484665ee9a7b2445358163ec0e1e33f8c16d166).
Use that revision's full-forward example, override packager and recorded source
identities together. Its raw results and failures remain retained; the experiment
did not qualify a production policy.

Current full-forward code refuses manifests containing `midpoint_counts` before
device setup. The experiment-only counter reader was removed because its kernel
exists only in the rejected override, outside registered production targets.
No placeholder kernel or kernel-lookup guard exception was added. Normal forward
execution and other bounded diagnostic manifests retain their existing behavior.
The primitive probe (`scripts/gpt_oss_midpoint_probe.py`), its CUDA source
(`crates/model-layers/tests/cuda/gpt_oss_mxfp4_midpoint.cu`) and the override
packager (`scripts/gpt_oss_midpoint_override.py`) were also removed from the
current tree; all three remain at the revision above.

The experiment improved some local dots but reduced full-model next-token
agreement from 244/251 to 243/251. Its wide-exponent limitation and numerical
failures are not superseded by subsequent performance improvements.

Tested checkpoint: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
at [6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee).
