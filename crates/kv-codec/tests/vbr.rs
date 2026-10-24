// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The VBR ladder accepts exactly the downward, in-scope, above-floor moves, and
//! refuses to be built over rungs that do not shrink.

use metrale_error::ErrorCode;
use metrale_kv_codec::{
    BlockGeometry, KvCodecId, KvSide, KvTransition, LayerIx, PressureTrigger, RungIx, VbrLadder,
};

const G: BlockGeometry = BlockGeometry {
    block_size: 16,
    num_kv_heads: 4,
    head_dim: 128,
};

fn ladder() -> VbrLadder {
    use KvCodecId::*;
    VbrLadder::new(
        Box::new([Turbo8, Turbo4, Turbo3, Turbo2]),
        Box::new([(LayerIx(0), KvSide::V), (LayerIx(1), KvSide::K)]),
        Box::new([RungIx(3), RungIx(1)]),
        &[G],
    )
    .unwrap()
}

fn t(layer: u32, side: KvSide, from: u8, to: u8) -> KvTransition {
    KvTransition {
        layer: LayerIx(layer),
        side,
        from: RungIx(from),
        to: RungIx(to),
        trigger: PressureTrigger::Scripted { step: 0 },
    }
}

#[test]
fn permits_only_downward_in_scope_moves_above_the_floor() {
    let l = ladder();
    assert!(l.permits(&t(0, KvSide::V, 0, 1)).is_ok());
    assert!(l.permits(&t(0, KvSide::V, 0, 3)).is_ok());
    assert!(l.permits(&t(1, KvSide::K, 0, 1)).is_ok());
    for bad in [
        t(1, KvSide::K, 0, 2), // below floor
        t(0, KvSide::V, 2, 1), // upward
        t(0, KvSide::V, 1, 1), // no move
        t(0, KvSide::K, 0, 1), // out of scope
        t(2, KvSide::V, 0, 1), // out of scope
        t(0, KvSide::V, 0, 4), // past the ladder
    ] {
        let e = l.permits(&bad).unwrap_err();
        assert_eq!(e.code(), ErrorCode::KvTransitionNotPermitted, "{bad:?}");
    }
}

#[test]
fn a_ladder_must_shrink_at_every_rung() {
    use KvCodecId::*;
    // 2026-10-18: nvfp4 and turbo4 are the same size, so the second rung saves nothing.
    for rungs in [
        vec![Turbo4, Nvfp4],
        vec![Turbo3, Turbo4],
        vec![Bf16],
        vec![Fp8, Turbo8],
    ] {
        let n = rungs.len();
        let e =
            VbrLadder::new(rungs.into_boxed_slice(), Box::new([]), Box::new([]), &[G]).unwrap_err();
        assert_eq!(e.code(), ErrorCode::KvLadderInvalid, "{n} rungs");
    }
    let e = VbrLadder::new(
        Box::new([Bf16, Fp8]),
        Box::new([(LayerIx(0), KvSide::K)]),
        Box::new([RungIx(2)]),
        &[G],
    )
    .unwrap_err();
    assert_eq!(e.code(), ErrorCode::KvLadderInvalid);
    let e = VbrLadder::new(
        Box::new([Bf16, Fp8]),
        Box::new([(LayerIx(0), KvSide::K)]),
        Box::new([]),
        &[G],
    )
    .unwrap_err();
    assert_eq!(e.code(), ErrorCode::KvLadderInvalid);
}
