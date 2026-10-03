// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The runtime routes the executor runs. A FUSIONS.toml `[[runtime]]` route
//! (`metrale_circuit::runtime`) is compiled into its own program beside the primary one, at
//! every mode and row count it applies to. Each step evaluates the route's condition exactly as
//! the legacy dispatch does and runs the matching program.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every route id has an evaluator here; a rule set naming another is refused at build, never
//!   run on its primary arm alone.
//! - `gdn_state_slots_fragmented` holds when some row's GDN state is not at row 0's plus `i`
//!   slots, the check `ssm_batched_recurrent.rs` makes per layer. Layers that disagree are an
//!   error: the pools share one slot index per sequence, so they cannot.

use anyhow::{Result, bail, ensure};
use metrale_circuit::{FusionPlan, RuntimeRoute};

use super::program::{GdnState, Program};

/// 2026-09-30: The routes this executor can evaluate.
pub const KNOWN: [&str; 2] = ["gdn_state_slots_fragmented", super::prefill::PREFIX_RESTORED];

/// 2026-09-30: A route's arm, compiled.
pub struct RoutedProgram {
    /// 2026-09-30: The route.
    pub route: RuntimeRoute,
    /// 2026-09-30: The arm's program.
    pub program: Program,
    /// 2026-09-30: The plan it was compiled from.
    pub plan: FusionPlan,
}

/// 2026-09-30: Refuse a rule set with a route no evaluator here serves.
pub fn check_known(runtime: &[RuntimeRoute]) -> Result<()> {
    if let Some(r) = runtime.iter().find(|r| !KNOWN.contains(&r.id.as_str())) {
        bail!(
            "runtime route `{}` has no evaluator in the executor (it knows {KNOWN:?})",
            r.id
        );
    }
    Ok(())
}

/// 2026-09-30: Whether `route` holds for a step whose rows read `gdn` (per layer, per row).
/// `pitch` is each layer's `(h, conv)` slot pitch, `None` for a layer without GDN state.
pub fn holds(
    route: &RuntimeRoute,
    pitch: &[Option<(usize, usize)>],
    gdn: &[Vec<GdnState>],
) -> Result<bool> {
    match route.id.as_str() {
        "gdn_state_slots_fragmented" => fragmented(pitch, gdn),
        // 2026-10-03: The prefill driver takes this arm itself (`prefill::PrefillPrograms`).
        super::prefill::PREFIX_RESTORED => bail!(
            "`{}` is a prefill route, which the prefill driver selects per pass",
            route.id
        ),
        other => bail!("runtime route `{other}` has no evaluator"),
    }
}

fn fragmented(pitch: &[Option<(usize, usize)>], gdn: &[Vec<GdnState>]) -> Result<bool> {
    let mut verdict: Option<(usize, bool)> = None;
    for (layer, (p, rows)) in pitch.iter().zip(gdn).enumerate() {
        let (Some((h, conv)), Some(base)) = (p, rows.first()) else {
            continue;
        };
        let contiguous = rows
            .iter()
            .enumerate()
            .all(|(i, s)| s.h == base.h.offset(i * h) && s.conv == base.conv.offset(i * conv));
        match verdict {
            None => verdict = Some((layer, contiguous)),
            Some((first, c)) => ensure!(
                c == contiguous,
                "GDN layers {first} and {layer} disagree on slot contiguity"
            ),
        }
    }
    Ok(verdict.is_some_and(|(_, contiguous)| !contiguous))
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_gpu_runtime::gpu::DevicePtr;

    fn state(h: u64, conv: u64) -> GdnState {
        GdnState {
            h: DevicePtr(h),
            conv: DevicePtr(conv),
            h_steps: [DevicePtr::NULL; super::super::program::MAX_VERIFY_STEPS],
            conv_steps: [DevicePtr::NULL; super::super::program::MAX_VERIFY_STEPS],
        }
    }

    fn route(id: &str) -> RuntimeRoute {
        RuntimeRoute {
            id: id.into(),
            when: Default::default(),
            modes: Default::default(),
            rows: (2, 128),
            plans_as: Default::default(),
            why: "test".into(),
            cite: "test".into(),
        }
    }

    // 2026-09-30: The check is legacy's: each row at row 0 plus `i` pitches, on both the h and
    // the conv pool. Mutations: comparing only h, or only the first two rows, misses a case.
    #[test]
    fn fragmented_holds_exactly_when_a_row_is_out_of_place() {
        let r = route("gdn_state_slots_fragmented");
        let pitch = [Some((0x100, 0x10)), None, Some((0x100, 0x10))];
        let layer = |rows: &[(u64, u64)]| -> Vec<GdnState> {
            rows.iter().map(|&(h, c)| state(h, c)).collect()
        };
        let ok: Vec<GdnState> = layer(&[(0x1000, 0x50), (0x1100, 0x60), (0x1200, 0x70)]);
        let gdn = vec![ok.clone(), Vec::new(), ok.clone()];
        assert!(!holds(&r, &pitch, &gdn).unwrap());
        for bad in [
            layer(&[(0x1000, 0x50), (0x1100, 0x60), (0x1300, 0x70)]),
            layer(&[(0x1000, 0x50), (0x1100, 0x60), (0x1200, 0x80)]),
            layer(&[(0x1000, 0x50), (0x1200, 0x60), (0x1300, 0x70)]),
        ] {
            let gdn = vec![bad.clone(), Vec::new(), bad];
            assert!(holds(&r, &pitch, &gdn).unwrap());
            let mixed = vec![ok.clone(), Vec::new(), gdn[0].clone()];
            assert!(
                holds(&r, &pitch, &mixed)
                    .unwrap_err()
                    .to_string()
                    .contains("disagree")
            );
        }
    }

    #[test]
    fn an_unknown_route_is_refused_at_build() {
        assert!(check_known(&[route("gdn_state_slots_fragmented")]).is_ok());
        let e = check_known(&[route("mystery")]).unwrap_err().to_string();
        assert!(e.contains("mystery") && e.contains("no evaluator"), "{e}");
    }
}
