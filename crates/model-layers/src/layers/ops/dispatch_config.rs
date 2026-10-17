// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GEMM-path selection: [`GemmDispatch`], resolved from `METRALE_*`
//! variables and carried to the dispatch sites on
//! [`crate::layer::ForwardContext`].
//!
//! Owner: model-layers ops.
//! Invariants:
//! - A switch reads under the settings Switch rule (2026-10-17; it was exactly `1`);
//!   `METRALE_CUBLAS_GEMM` has its own grammar ([`CublasScope`]) and
//!   `METRALE_W4A16_VARIANT` accepts `v1`, `v2` and `v3`.
//! - Unknown `METRALE_CUBLAS_GEMM` tokens never add a family.

/// 2026-09-25: Which projection families take a cuBLASLt GEMM arm, so one
/// family can be armed without the others.
///
/// Grammar: `METRALE_CUBLAS_GEMM=<token>[,<token>]*`, ASCII-case-insensitive,
/// whitespace around a token ignored:
///
/// | token | meaning |
/// |---|---|
/// | `ffn` | dense-FFN and MoE shared-expert projections |
/// | `attn` | attention Q/K/V, O and the output gate |
/// | `ssm` | GDN `in_proj_qkvz` and `out_proj` |
/// | `head` | LM / MTP head (see [`CublasScope::head`]) |
/// | `all`, `1`, `true` | every family |
/// | `off`, `0`, `false`, empty | the empty set |
///
/// The result is the union of the tokens, so `off` clears nothing (`ffn,off`
/// is `ffn`). Unknown tokens are dropped and reported, and never widen the
/// set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CublasScope {
    /// 2026-09-25: Dense-FFN gate/up/down and the MoE shared expert.
    pub ffn: bool,
    /// 2026-09-25: Attention Q/K/V, O projection and the output gate.
    pub attn: bool,
    /// 2026-09-25: GDN `in_proj_qkvz` and `out_proj`, in prefill and decode.
    pub ssm: bool,
    /// 2026-09-25: LM / MTP head. Parsed and set by `all`, but no dispatch site
    /// reads it, so setting it changes nothing.
    pub head: bool,
}

impl CublasScope {
    /// 2026-09-25: No family armed: what an absent, empty or `off`
    /// `METRALE_CUBLAS_GEMM` resolves to, and the value in
    /// [`GemmDispatch::defaults`].
    pub const OFF: Self = Self {
        ffn: false,
        attn: false,
        ssm: false,
        head: false,
    };

    /// 2026-09-25: Every family: what `all`, `1` and `true` resolve to.
    pub const ALL: Self = Self {
        ffn: true,
        attn: true,
        ssm: true,
        head: true,
    };

    /// 2026-09-25: Whether any family is armed.
    pub fn any(&self) -> bool {
        self.ffn || self.attn || self.ssm || self.head
    }
}

/// 2026-09-25: Parse the [`CublasScope`] grammar. Returns the resolved set and
/// the lowercased tokens that matched nothing. Pure: the environment read and
/// the logging are in [`GemmDispatch::from_settings`].
pub fn parse_cublas_scope(raw: Option<&str>) -> (CublasScope, Vec<String>) {
    let mut scope = CublasScope::OFF;
    let mut unknown = Vec::new();
    let Some(raw) = raw else {
        return (scope, unknown);
    };
    for token in raw.split(',') {
        match token.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "off" => {}
            "1" | "true" | "all" => scope = CublasScope::ALL,
            "ffn" => scope.ffn = true,
            "attn" => scope.attn = true,
            "ssm" => scope.ssm = true,
            "head" => scope.head = true,
            other => unknown.push(other.to_owned()),
        }
    }
    (scope, unknown)
}

/// 2026-09-25: Which GEMM implementation each projection takes, resolved by
/// [`GemmDispatch::from_settings`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmDispatch {
    /// 2026-09-25: Block-scaled FP8 prefill (per-128-block weight scales and
    /// per-token activation scales). On unless `METRALE_FP8_SINGLE_SCALE=1`.
    pub fp8_blockscaled_prefill: bool,
    /// 2026-09-25: Which projection families take a cuBLASLt GEMM arm
    /// (`METRALE_CUBLAS_GEMM`).
    pub cublas: CublasScope,
    /// 2026-09-25: Row-wise FP8 cuBLASLt GEMM for the GDN `in_proj_qkvz`
    /// prefill (`METRALE_CUBLAS_FP8=1`).
    pub cublas_fp8: bool,
    /// 2026-09-25: CUTLASS BF16 GEMM for the GDN `in_proj_qkvz` prefill, on a
    /// memoized FP8→BF16 dequant of the weight (`METRALE_CUTLASS_GEMM=1`).
    pub cutlass_gemm: bool,
    /// 2026-09-25: CUTLASS NVFP4 GEMM (`METRALE_CUTLASS_NVFP4_GEMM=1`). Implies
    /// the per-projection NVFP4 flags below except `cutlass_nvfp4_ssm_out`.
    pub cutlass_nvfp4_gemm: bool,
    pub cutlass_nvfp4_qkvz: bool,
    pub cutlass_nvfp4_attn_q: bool,
    pub cutlass_nvfp4_attn_kv: bool,
    pub cutlass_nvfp4_attn_o: bool,
    pub cutlass_nvfp4_ssm_out: bool,
    /// 2026-09-25: `METRALE_W4A16_VARIANT`: `v1`, `v2` and `v3` give 1, 2 and 3;
    /// any other value, or none, gives 0, which the attention prefill treats
    /// like 2.
    pub w4a16_variant: u8,
}

impl GemmDispatch {
    /// 2026-10-17: Resolve from the settings (`kernels.*` and the two `cache.cutlass_nvfp4_*`
    /// rows), and log the resolved `METRALE_CUBLAS_GEMM` scope when that lever is set.
    pub fn from_settings(s: &metrale_settings::InferenceServerSettings) -> Self {
        let (k, c) = (&s.kernels, &s.cache);
        let all_nvfp4 = k.cutlass_nvfp4_gemm;
        let resolved = GemmDispatch {
            w4a16_variant: match k.w4a16_variant.as_deref() {
                Some("v1") => 1,
                Some("v2") => 2,
                Some("v3") => 3,
                _ => 0,
            },
            fp8_blockscaled_prefill: !k.fp8_single_scale,
            cublas: parse_cublas_scope(k.cublas_gemm.as_deref()).0,
            cublas_fp8: k.cublas_fp8,
            cutlass_gemm: k.cutlass_gemm,
            cutlass_nvfp4_gemm: all_nvfp4,
            cutlass_nvfp4_qkvz: all_nvfp4 || c.cutlass_nvfp4_qkvz,
            cutlass_nvfp4_attn_q: all_nvfp4 || k.cutlass_nvfp4_attn_q,
            cutlass_nvfp4_attn_kv: all_nvfp4 || c.cutlass_nvfp4_attn_kv,
            cutlass_nvfp4_attn_o: all_nvfp4 || k.cutlass_nvfp4_attn_o,
            // 2026-09-25: Not implied by `METRALE_CUTLASS_NVFP4_GEMM`.
            cutlass_nvfp4_ssm_out: k.cutlass_nvfp4_ssm_out,
        };
        log_cublas_scope(k.cublas_gemm.as_deref(), resolved.cublas);
        resolved
    }

    /// 2026-09-25: Everything off except block-scaled FP8 prefill: what an
    /// environment with none of these variables resolves to.
    pub fn defaults() -> Self {
        Self {
            w4a16_variant: 0,
            fp8_blockscaled_prefill: true,
            cublas: CublasScope::OFF,
            cublas_fp8: false,
            cutlass_gemm: false,
            cutlass_nvfp4_gemm: false,
            cutlass_nvfp4_qkvz: false,
            cutlass_nvfp4_attn_q: false,
            cutlass_nvfp4_attn_kv: false,
            cutlass_nvfp4_attn_o: false,
            cutlass_nvfp4_ssm_out: false,
        }
    }

    /// 2026-09-25: Whether CUTLASS NVFP4 is on for an attention projection:
    /// `q_proj` reads the Q flag, `k_proj` and `v_proj` the KV flag, and any
    /// other label the umbrella flag.
    pub fn cutlass_nvfp4_attn_qkv(&self, label: &str) -> bool {
        match label {
            "q_proj" => self.cutlass_nvfp4_attn_q,
            "k_proj" | "v_proj" => self.cutlass_nvfp4_attn_kv,
            _ => self.cutlass_nvfp4_gemm,
        }
    }
}

/// 2026-09-25: Log the cuBLASLt families a set `METRALE_CUBLAS_GEMM` resolved
/// to, and warn about its unknown tokens. Silent when the variable is unset.
fn log_cublas_scope(raw: Option<&str>, scope: CublasScope) {
    let Some(raw) = raw else {
        return;
    };
    let (_, unknown) = parse_cublas_scope(Some(raw));
    if !unknown.is_empty() {
        tracing::warn!(
            "METRALE_CUBLAS_GEMM={raw:?}: ignoring unknown families [{}]. The grammar is a \
             comma-separated subset of all|ffn|attn|ssm|head|off (1/true = all).",
            unknown.join(", ")
        );
    }
    tracing::info!(
        "[metrale] METRALE_CUBLAS_GEMM={raw:?} -> cuBLASLt arms ffn={} attn={} ssm={} head={} \
         (head has no consumer yet){}",
        scope.ffn,
        scope.attn,
        scope.ssm,
        scope.head,
        if scope.any() {
            ""
        } else {
            " — no arm enabled"
        }
    );
}

impl Default for GemmDispatch {
    fn default() -> Self {
        Self::defaults()
    }
}

#[cfg(test)]
#[path = "dispatch_config_tests.rs"]
mod tests;
