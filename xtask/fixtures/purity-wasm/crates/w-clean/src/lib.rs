// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_std]

pub fn add(a: u32, b: u32) -> u32 {
    a.wrapping_add(b)
}
