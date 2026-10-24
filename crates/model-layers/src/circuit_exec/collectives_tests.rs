// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The one-shot bound a rank's plans fuse under. The path check over real rank
//! circuits is model-arch's (`glm5next_plan::world_tests`), which builds them.

use super::with_comm_bound;
use metrale_circuit::Policy;
use metrale_circuit::collective::ONESHOT_MAX_KB;
use metrale_error::ErrorCode;

fn stating(kb: Option<&str>) -> Policy {
    let mut p = Policy::default();
    p.settings.insert("other".into(), "on".into());
    if let Some(kb) = kb {
        p.settings.insert(ONESHOT_MAX_KB.into(), kb.into());
    }
    p
}

/// 2026-10-23: The communicator's bound replaces the stated one (a serve at 512 KiB, or with the
/// one-shot off, plans under its own), the other settings untouched; an equal one changes nothing.
#[test]
fn the_communicators_bound_replaces_the_stated_one() {
    for (bytes, want) in [(512 * 1024, "512"), (0, "0"), (1024 * 1024, "1024")] {
        let p = with_comm_bound(&stating(Some("1024")), bytes).unwrap();
        assert_eq!(p.settings[ONESHOT_MAX_KB], want, "{bytes} B");
        assert_eq!(p.settings["other"], "on");
        assert_eq!(p.settings.len(), 2);
    }
}

/// 2026-10-23: An instance that states no bound keeps none (its one-shot rules are refused by
/// the fuser, PCND); a bound that is not whole KiB is refused, never rounded.
#[test]
fn no_stated_bound_stays_absent_and_a_partial_kib_is_refused() {
    let p = with_comm_bound(&stating(None), 1024 * 1024).unwrap();
    assert!(!p.settings.contains_key(ONESHOT_MAX_KB));
    let e = with_comm_bound(&stating(Some("1024")), 1024 * 1024 + 2).unwrap_err();
    assert_eq!(e.code(), ErrorCode::CollectivePathRefused);
    assert!(e.to_string().contains("1048578"), "{e}");
}
