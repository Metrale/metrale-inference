// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Module targets of a declared precision plan and how they match.
//!
//! Owner: config (quantization).
//! Invariants: none beyond the types.

use anyhow::{Context, Result};

/// 2026-09-28: One target or ignore entry.
#[derive(Debug, Clone)]
pub enum Target {
    /// 2026-09-28: An exact module path.
    Exact(String),
    /// 2026-09-28: A compressed-tensors `re:` pattern, anchored at both ends as Python's
    /// `re.match` plus the checkpoint's `$` do.
    Regex(String, regex::Regex),
    /// 2026-09-28: A ModelOpt glob (`*`, `?`).
    Glob(String),
    /// 2026-09-30: An HF `fp8` `modules_to_not_convert` / `ignored_layers` entry: a module
    /// path matched on whole dotted segments. It names the module itself, a parent of it
    /// (`model.visual` excludes `model.visual.blocks.0.attn.proj`), or its trailing segments
    /// (`lm_head`); `mlp.gate` names the MoE router `...mlp.gate`, never the dense
    /// `...mlp.gate_proj`. These are the three rules of transformers `should_convert_module`
    /// (quantizers/quantizers_utils.py: `re.match(f"{key}\\.")`, `re.match(key)`,
    /// `name.endswith(key)`) restricted to segment boundaries, and include vLLM's exact
    /// `prefix in ignored_layers` (`is_layer_skipped`, quantization/utils/quant_utils.py). An
    /// entry with regex syntax beyond `.` is that regex, anchored at the start (`re.match`) and
    /// ending on a segment boundary.
    HfModule(String, Option<regex::Regex>),
    /// 2026-09-28: A module class (`Linear`). Every module the plan is asked about is a
    /// linear layer, so `Linear` matches all of them and any other class none.
    Class(String),
}

/// 2026-09-30: Which format's matching rules an ignore / exclude entry follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IgnoreDialect {
    /// 2026-09-30: compressed-tensors `ignore` ([`Target::compressed_tensors`]).
    CompressedTensors,
    /// 2026-09-30: ModelOpt `exclude_modules` ([`Target::modelopt`]).
    ModelOpt,
    /// 2026-09-30: HF `fp8` `modules_to_not_convert` / `ignored_layers` ([`Target::hf_module`]).
    HfFp8,
}

impl Target {
    /// 2026-09-30: An ignore entry of `dialect`, the one matcher the precision plan and the
    /// weight loaders share. A malformed entry is refused.
    pub fn ignore_entry(dialect: IgnoreDialect, s: &str) -> Result<Self> {
        match dialect {
            IgnoreDialect::CompressedTensors => Self::compressed_tensors(s),
            IgnoreDialect::ModelOpt => Ok(Self::modelopt(s)),
            IgnoreDialect::HfFp8 => Self::hf_module(s),
        }
    }

    /// 2026-09-28: A compressed-tensors target or ignore entry: `re:<pattern>`, a class
    /// name (`Linear`), or an exact path.
    /// 2026-09-30: As compressed-tensors matches them (0.12.2, utils/match.py `_match_name`):
    /// a `re:` pattern by `re.match` (anchored at the start only), anything else by equality.
    pub fn compressed_tensors(s: &str) -> Result<Self> {
        if let Some(p) = s.strip_prefix("re:") {
            let re = regex::Regex::new(&format!("^(?:{p})"))
                .with_context(|| format!("quantization_config: bad target pattern {s:?}"))?;
            return Ok(Target::Regex(s.to_string(), re));
        }
        if !s.contains('.') && s.chars().next().is_some_and(char::is_uppercase) {
            return Ok(Target::Class(s.to_string()));
        }
        Ok(Target::Exact(s.to_string()))
    }

    /// 2026-09-30: An HF `fp8` `modules_to_not_convert` / `ignored_layers` entry
    /// ([`Target::HfModule`]). An empty entry, or one whose regex does not compile, is refused.
    pub fn hf_module(s: &str) -> Result<Self> {
        anyhow::ensure!(
            !s.trim().is_empty(),
            "quantization_config: an empty modules_to_not_convert entry"
        );
        let regex_syntax =
            s.contains(['*', '?', '[', ']', '(', ')', '+', '^', '$', '|', '\\', '{']);
        let re = if regex_syntax {
            Some(
                regex::Regex::new(&format!("^(?:{s})(?:\\.|$)"))
                    .with_context(|| format!("quantization_config: bad module pattern {s:?}"))?,
            )
        } else {
            None
        };
        Ok(Target::HfModule(s.to_string(), re))
    }

    /// 2026-09-28: A ModelOpt entry: a glob when it has a wildcard, else exact.
    /// 2026-09-30: ModelOpt writes exact module names or fnmatch globs; vLLM's
    /// `ModelOptFp8Config.is_layer_excluded` (quantization/modelopt.py) adds a substring
    /// fallback, which is not followed here: it has the `mlp.gate` / `mlp.gate_proj` hazard.
    pub fn modelopt(s: &str) -> Self {
        if s.contains(['*', '?']) {
            Target::Glob(s.to_string())
        } else if s == "Linear" {
            Target::Class(s.to_string())
        } else {
            Target::Exact(s.to_string())
        }
    }

    /// 2026-09-28: The entry as the checkpoint wrote it.
    pub fn text(&self) -> &str {
        match self {
            Target::Exact(s)
            | Target::Regex(s, _)
            | Target::Glob(s)
            | Target::HfModule(s, _)
            | Target::Class(s) => s,
        }
    }

    /// 2026-09-28: Does this entry name `module` (classes excluded)?
    pub fn matches_name(&self, module: &str) -> bool {
        match self {
            Target::Exact(s) => s == module,
            Target::Regex(_, re) => re.is_match(module),
            Target::Glob(g) => glob_match(g.as_bytes(), module.as_bytes()),
            Target::HfModule(_, Some(re)) => re.is_match(module),
            Target::HfModule(s, None) => {
                module == s
                    || module
                        .strip_prefix(s.as_str())
                        .is_some_and(|rest| rest.starts_with('.'))
                    || module
                        .strip_suffix(s.as_str())
                        .is_some_and(|head| head.ends_with('.'))
            }
            Target::Class(c) => c == "Linear",
        }
    }

    /// 2026-09-28: Precedence tier when this entry matches `module`: 0 exact name, 1
    /// pattern, 2 class. `None` when it does not match.
    pub fn tier_for(&self, module: &str) -> Option<u8> {
        if !self.matches_name(module) {
            return None;
        }
        Some(match self {
            Target::Exact(_) => 0,
            Target::HfModule(s, None) if s == module => 0,
            Target::Regex(..) | Target::Glob(_) | Target::HfModule(..) => 1,
            Target::Class(_) => 2,
        })
    }
}

/// 2026-09-28: `*` matches any run, `?` one byte; everything else literally.
fn glob_match(p: &[u8], s: &[u8]) -> bool {
    let (mut pi, mut si) = (0usize, 0usize);
    let (mut star, mut mark) = (None, 0usize);
    while si < s.len() {
        if pi < p.len() && (p[pi] == b'?' || p[pi] == s[si]) {
            pi += 1;
            si += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star = Some(pi);
            mark = si;
            pi += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            mark += 1;
            si = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == b'*')
}
