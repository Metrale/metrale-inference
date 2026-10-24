// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn test_empty_layer_state_downcast() {
    let state: Box<dyn LayerState> = Box::new(EmptyLayerState);
    assert!(state.as_any().downcast_ref::<EmptyLayerState>().is_some());
    assert!(state.as_any().downcast_ref::<SsmLayerState>().is_none());
}

#[test]
fn test_ssm_layer_state_downcast() {
    let state: Box<dyn LayerState> = Box::new(SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        replay_ring: None,
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    });
    let ssm = state.as_any().downcast_ref::<SsmLayerState>().unwrap();
    assert_eq!(ssm.h_state.0, 0x1000);
    assert_eq!(ssm.conv_state.0, 0x2000);
}

#[test]
fn test_ssm_layer_state_mut() {
    let mut state: Box<dyn LayerState> = Box::new(SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        replay_ring: None,
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    });
    let ssm = state.as_any_mut().downcast_mut::<SsmLayerState>().unwrap();
    ssm.h_state = DevicePtr(0x3000);
    assert_eq!(ssm.h_state.0, 0x3000);
}

/// 2026-10-21: The swap exchanges live h/conv with intermediate 0 and leaves intermediate 1 and
/// the checkpoint alone; a second swap restores the home layout.
#[test]
fn swap_live_with_inter0_exchanges_only_the_live_and_first_intermediate() {
    let mut s = SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: Some(DevicePtr(0x7000)),
        conv_state_checkpoint: None,
        h_state_intermediates: vec![DevicePtr(0x3000), DevicePtr(0x3100)],
        conv_state_intermediates: vec![DevicePtr(0x4000), DevicePtr(0x4100)],
        replay_ring: None,
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    };
    s.swap_live_with_inter0().unwrap();
    assert_eq!(
        (s.h_state, s.conv_state),
        (DevicePtr(0x3000), DevicePtr(0x4000))
    );
    assert_eq!(
        s.h_state_intermediates,
        vec![DevicePtr(0x1000), DevicePtr(0x3100)]
    );
    assert_eq!(
        s.conv_state_intermediates,
        vec![DevicePtr(0x2000), DevicePtr(0x4100)]
    );
    assert_eq!(s.h_state_checkpoint, Some(DevicePtr(0x7000)));
    s.swap_live_with_inter0().unwrap();
    assert_eq!(
        (s.h_state, s.conv_state),
        (DevicePtr(0x1000), DevicePtr(0x2000))
    );
}

/// 2026-10-21: Without a conv intermediate 0 (or an h one) the swap refuses and moves nothing.
#[test]
fn swap_live_with_inter0_refuses_without_both_intermediates() {
    let mut s = SsmLayerState {
        h_state: DevicePtr(0x1000),
        conv_state: DevicePtr(0x2000),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: vec![DevicePtr(0x3000)],
        conv_state_intermediates: Vec::new(),
        replay_ring: None,
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    };
    assert!(s.swap_live_with_inter0().is_err());
    assert_eq!(s.h_state, DevicePtr(0x1000));
    assert_eq!(s.h_state_intermediates, vec![DevicePtr(0x3000)]);
}
