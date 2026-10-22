// SPDX-License-Identifier: MIT OR Apache-2.0

//! Context hashing per profile. One `Contexts` tracks the byte history cheaply (last eight
//! bytes, the current token and the two before it, the line's opcode and field, the record
//! column) and emits up to [`MAX_CTX`] hashes per byte boundary that select the statistics
//! buckets.
//!
//! Profiles share the five order-N and word contexts; `Ptx` adds a token-in-field context and
//! a previous-line column context, `Elf` adds 16-byte record contexts (SASS instruction words).

use crate::cast::ix;
use crate::format::Profile;

/// The most contexts any profile emits.
pub const MAX_CTX: usize = 8;
const FIELDS: usize = 8;

#[inline(always)]
fn mix(a: u32, b: u32) -> u32 {
    let h = (a ^ b.wrapping_mul(0x9E37_79B1)).wrapping_mul(0x85EB_CA6B);
    h ^ (h >> 15)
}

#[inline(always)]
fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'$' | b'%' | b'.' | crate::ptx::MARKER)
}

/// Running history state and the current context hashes.
pub struct Contexts {
    profile: Profile,
    c4: u32,
    c8: u32,
    word: u32,
    prev_word: u32,
    prev_word2: u32,
    opcode: u32,
    field: u32,
    after_opcode: bool,
    prev_line: [u32; FIELDS],
    cur_line: [u32; FIELDS],
    line: u32,
    pos: u32,
    /// The hashes for the next byte, `h[..n]` valid.
    pub h: [u32; MAX_CTX],
    /// How many contexts the profile emits.
    pub n: usize,
}

impl Contexts {
    /// Fresh state for `profile`.
    pub fn new(profile: Profile) -> Self {
        let n = match profile {
            Profile::Stored | Profile::Generic => 6,
            Profile::Ptx | Profile::Elf => 8,
        };
        let mut c = Self {
            profile,
            c4: 0,
            c8: 0,
            word: 0,
            prev_word: 0,
            prev_word2: 0,
            opcode: 0,
            field: 0,
            after_opcode: false,
            prev_line: [0; FIELDS],
            cur_line: [0; FIELDS],
            line: 0,
            pos: 0,
            h: [0; MAX_CTX],
            n,
        };
        c.emit(&[]);
        c
    }

    /// The field index within the current line (PTX), for mixer selection.
    #[inline(always)]
    pub fn field(&self) -> u32 {
        self.field.min(7)
    }

    /// Hash of the current token so far (the identifier class at a PTX marker).
    #[inline(always)]
    pub fn word(&self) -> u32 {
        self.word
    }

    /// Hash of the current line so far (the skeleton context of a PTX number).
    #[inline(always)]
    pub fn line(&self) -> u32 {
        self.line
    }

    /// Absorb `byte`; `hist` is the history including it.
    #[inline]
    pub fn update(&mut self, byte: u8, hist: &[u8]) {
        self.c8 = (self.c8 << 8) | (self.c4 >> 24);
        self.c4 = (self.c4 << 8) | u32::from(byte);
        self.pos = self.pos.wrapping_add(1);
        self.line = if byte == b'\n' { 0 } else { mix(self.line, u32::from(byte)) };
        if is_token_byte(byte) {
            self.word = mix(self.word, u32::from(byte));
        } else {
            if self.word != 0 {
                if !self.after_opcode {
                    self.opcode = self.word;
                }
                let f = ix(self.field.min(7));
                self.cur_line[f] = mix(self.cur_line[f], self.word);
                self.prev_word2 = self.prev_word;
                self.prev_word = self.word;
                self.word = 0;
            }
            match byte {
                b'\n' => {
                    self.prev_line = self.cur_line;
                    self.cur_line = [0; FIELDS];
                    self.field = 0;
                    self.after_opcode = false;
                }
                b',' => self.field = self.field.saturating_add(1),
                b' ' | b'\t' => {
                    if self.opcode != 0 && !self.after_opcode {
                        self.after_opcode = true;
                        self.field = 1;
                    }
                }
                _ => {}
            }
        }
        self.emit(hist);
    }

    fn emit(&mut self, hist: &[u8]) {
        let c4 = self.c4;
        self.h[0] = mix(1, c4 & 0xff);
        self.h[1] = mix(2, c4 & 0xffff);
        self.h[2] = mix(3, c4 & 0xff_ffff);
        self.h[3] = mix(4, c4);
        self.h[4] = mix(mix(5, c4), self.c8 & 0xffff);
        self.h[5] = mix(mix(6, self.word), mix(self.prev_word, c4 & 0xff));
        match self.profile {
            Profile::Stored | Profile::Generic => {}
            Profile::Ptx => {
                self.h[6] = mix(mix(7, self.opcode), mix(self.field, self.word));
                let f = ix(self.field.min(7));
                self.h[7] = mix(
                    mix(8, self.prev_line[f]),
                    mix(self.word, self.prev_word2 ^ self.field),
                );
            }
            Profile::Elf => {
                let n = hist.len();
                let b16 = u32::from(n.checked_sub(16).and_then(|i| hist.get(i)).copied().unwrap_or(0));
                let b32 = u32::from(n.checked_sub(32).and_then(|i| hist.get(i)).copied().unwrap_or(0));
                let col = self.pos & 15;
                self.h[6] = mix(mix(7, col), mix(b16, c4 & 0xff));
                self.h[7] = mix(mix(8, col), mix(b16, b32 << 8));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_fields_follow_commas_and_reset_on_newline() {
        let mut c = Contexts::new(Profile::Ptx);
        let line = b"\tadd.s32 \t%r1, %r2, %r3;\n";
        let mut hist = alloc::vec::Vec::new();
        let mut max_field = 0;
        for &b in line.iter() {
            hist.push(b);
            c.update(b, &hist);
            max_field = max_field.max(c.field());
        }
        assert_eq!(max_field, 3);
        assert_eq!(c.field(), 0);
        assert_ne!(c.prev_line[1], 0);
    }

    #[test]
    fn same_history_gives_same_hashes() {
        let mut a = Contexts::new(Profile::Elf);
        let mut b = Contexts::new(Profile::Elf);
        let data: alloc::vec::Vec<u8> = (0..200u32).map(|i| (i.wrapping_mul(31) & 0xff) as u8).collect();
        for i in 1..=data.len() {
            a.update(data[i - 1], &data[..i]);
            b.update(data[i - 1], &data[..i]);
        }
        assert_eq!(a.h, b.h);
    }
}
