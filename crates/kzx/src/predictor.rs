// SPDX-License-Identifier: MIT OR Apache-2.0

//! The bit predictor: per-profile context bit histories mapped to probabilities, the match
//! model, a gated mixer and two SSE stages; plus the byte-level "match continues" flag model
//! that lets the engine code a confidently predicted byte as one binary decision instead of
//! eight.

use crate::cast::ix;
use crate::context::{Contexts, MAX_CTX};
use crate::format::Profile;
use crate::matcher::MatchModel;
use crate::mixer::{Apm, Mixer};
use crate::slots::{COUNTER_INIT, Rates, Slots, StateMap, predict};
use crate::states::StateTable;
use crate::tables::Stretch;
use alloc::vec::Vec;

/// Smallest `mem_log2` a block may name.
pub const MIN_MEM_LOG2: u8 = 10;
/// Largest `mem_log2` a block may name (one table is then 1 GiB).
pub const MAX_MEM_LOG2: u8 = 26;

/// Tunables that differ per profile.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Match length from which a byte is coded as a single flag.
    pub lzp_min: u32,
    /// Bytes a match candidate must agree on.
    pub match_min: u32,
    /// Mixer learning rate.
    pub mixer_lr: i32,
}

impl Params {
    /// The parameters for `profile`.
    pub const fn for_profile(profile: Profile) -> Self {
        match profile {
            Profile::Stored | Profile::Generic => Params {
                lzp_min: 24,
                match_min: 6,
                mixer_lr: 6,
            },
            Profile::Ptx => Params {
                lzp_min: 24,
                match_min: 7,
                mixer_lr: 6,
            },
            Profile::Elf => Params {
                lzp_min: 24,
                match_min: 6,
                mixer_lr: 6,
            },
        }
    }
}

/// Bucket-count log2 of context table `i` at `mem_log2`: order 1 and 2 have few distinct
/// contexts and take small tables, the rest take the full size.
fn table_log2(i: usize, mem_log2: u8) -> u8 {
    match i {
        0 => mem_log2.min(13),
        1 => mem_log2.min(21),
        _ => mem_log2,
    }
}

/// Model memory in bytes for `profile` at `mem_log2` (what the decoder will allocate).
pub fn memory_bytes(profile: Profile, mem_log2: u8) -> u64 {
    let n = Contexts::new(profile).n;
    let tables: u64 = (0..n).map(|i| Slots::bytes(table_log2(i, mem_log2))).sum();
    let matcher = MatchModel::bytes(mem_log2.saturating_add(2));
    let mixers = 4096u64 * 24 * 4 + 16 * 4 * 4;
    let apms = (256u64 + 65_536) * 24 * 2;
    tables + matcher + mixers + apms + 4096 * 2 + 4096 + 256 * 4 * 8
}

/// The bit predictor for one block.
pub struct Predictor {
    params: Params,
    stretch: Stretch,
    rates: Rates,
    states: StateTable,
    slots: Vec<Slots>,
    maps: Vec<StateMap>,
    ctx: Contexts,
    bases: [u32; MAX_CTX],
    mixer: Mixer,
    apm1: Apm,
    apm2: Apm,
    mm: MatchModel,
    c0: u32,
    nibble: u32,
    bitpos: u32,
    excluded: Option<u8>,
    pr: i32,
    lzp_a: Vec<u32>,
    lzp_b: Vec<u32>,
    lzp_hist: u32,
    lzp_ctx_a: u32,
    lzp_ctx_b: u32,
    lzp_mixer: Mixer,
}

impl Predictor {
    /// A predictor for `profile` with `2^mem_log2` buckets per high-order context.
    pub fn new(profile: Profile, mem_log2: u8) -> Self {
        let params = Params::for_profile(profile);
        let ctx = Contexts::new(profile);
        let n = ctx.n;
        let states = StateTable::new();
        let slots = (0..n).map(|i| Slots::new(table_log2(i, mem_log2))).collect();
        let maps = (0..n).map(|_| StateMap::new(&states)).collect();
        Self {
            params,
            stretch: Stretch::new(),
            rates: Rates::new(),
            states,
            slots,
            maps,
            ctx,
            bases: [0; MAX_CTX],
            mixer: Mixer::new(2 * n + 4, 4096, params.mixer_lr, 12_000),
            apm1: Apm::new(256),
            apm2: Apm::new(65_536),
            mm: MatchModel::new(mem_log2.saturating_add(2), params.match_min),
            c0: 1,
            nibble: 1,
            bitpos: 0,
            excluded: None,
            pr: 2048,
            lzp_a: alloc::vec![COUNTER_INIT; 16 * 16],
            lzp_b: alloc::vec![COUNTER_INIT; 256 * 16],
            lzp_hist: 0,
            lzp_ctx_a: 0,
            lzp_ctx_b: 0,
            lzp_mixer: Mixer::new(3, 16, 4, 20_000),
        }
    }

    /// The byte the engine may code as one flag, if the match is long enough.
    #[inline(always)]
    pub fn lzp_candidate(&self) -> Option<u8> {
        if self.mm.len() >= self.params.lzp_min {
            self.mm.expected()
        } else {
            None
        }
    }

    /// P(the candidate byte is next), 12 bits.
    #[inline]
    pub fn lzp_p(&mut self) -> i32 {
        let bucket = MatchModel::len_bucket(self.mm.len());
        let expected = u32::from(self.mm.expected().unwrap_or(0));
        self.lzp_ctx_a = bucket * 16 + (self.lzp_hist & 15);
        self.lzp_ctx_b = expected * 16 + bucket;
        let pa = predict(self.lzp_a[ix(self.lzp_ctx_a)]);
        let pb = predict(self.lzp_b[ix(self.lzp_ctx_b)]);
        self.lzp_mixer.set_context(bucket);
        self.lzp_mixer.add(self.stretch.stretch(pa));
        self.lzp_mixer.add(self.stretch.stretch(pb));
        self.lzp_mixer.add(256);
        self.lzp_mixer.mix()
    }

    /// Learn the flag outcome.
    #[inline]
    pub fn lzp_update(&mut self, hit: bool) {
        let bit = u32::from(hit);
        self.rates.update(&mut self.lzp_a[ix(self.lzp_ctx_a)], bit, 1023);
        self.rates.update(&mut self.lzp_b[ix(self.lzp_ctx_b)], bit, 1023);
        self.lzp_mixer.update(bit);
        self.lzp_hist = (self.lzp_hist << 1) | bit;
    }

    /// Tell the bit models that the next byte is known not to be `byte`.
    #[inline(always)]
    pub fn set_excluded(&mut self, byte: u8) {
        self.excluded = Some(byte);
    }

    /// P(next bit = 1), 12 bits.
    #[inline]
    pub fn p(&mut self) -> i32 {
        let n = self.ctx.n;
        if self.bitpos == 0 || self.bitpos == 4 {
            for i in 0..n {
                let h = self.ctx.h[i].wrapping_add(self.c0.wrapping_mul(0x5bd1_e995));
                self.bases[i] = self.slots[i].find(h, &self.states);
            }
            self.nibble = 1;
        }
        for i in 0..n {
            let s = self.slots[i].get(self.bases[i] + self.nibble);
            let p = self.maps[i].predict(s);
            self.mixer.add(self.stretch.stretch(p));
            let (n0, n1) = self.states.counts(s);
            let det = if n0 == 0 && n1 > 0 {
                i32::try_from(n1.min(31)).unwrap_or(0) * 64
            } else if n1 == 0 && n0 > 0 {
                -i32::try_from(n0.min(31)).unwrap_or(0) * 64
            } else {
                0
            };
            self.mixer.add(det);
        }
        self.mm.add_inputs(self.c0, self.bitpos, &mut self.mixer, &self.stretch);
        let excl = match self.excluded {
            Some(e) if (u32::from(e) | 0x100) >> (8 - self.bitpos) == self.c0 => {
                let bit = (u32::from(e) >> (7 - self.bitpos)) & 1;
                let strength = 2047 >> (7 - self.bitpos) >> 1;
                if bit == 1 { -strength } else { strength }
            }
            _ => 0,
        };
        self.mixer.add(excl);
        self.mixer.add(256);
        let sel = self.c0 | (self.mm.mixer_context() << 8);
        self.mixer.set_context(sel);
        let p0 = self.mixer.mix();
        let p1 = self.apm1.refine(p0, self.c0, &self.stretch);
        let p = (p0 + 3 * p1) >> 2;
        let o1 = self.ctx.h[0] & 0xff;
        let p2 = self.apm2.refine(p, (self.c0 | (o1 << 8)) & 0xffff, &self.stretch);
        self.pr = ((p + p2) >> 1).clamp(1, 4095);
        self.pr
    }

    /// Learn `bit` and advance within the byte.
    #[inline]
    pub fn update(&mut self, bit: u32) {
        let n = self.ctx.n;
        for i in 0..n {
            let at = self.bases[i] + self.nibble;
            let s = self.slots[i].get(at);
            self.maps[i].update(s, bit, &self.rates);
            self.slots[i].set(at, self.states.next(s, bit));
        }
        self.mm.update_bit(bit, &self.rates);
        self.mixer.update(bit);
        self.apm1.update(bit, 7);
        self.apm2.update(bit, 7);
        self.c0 = (self.c0 << 1) | bit;
        self.nibble = (self.nibble << 1) | bit;
        self.bitpos += 1;
    }

    /// The byte is complete (coded or flagged): `hist` ends with it.
    #[inline]
    pub fn byte_done(&mut self, hist: &[u8]) {
        let Some(&byte) = hist.last() else {
            return;
        };
        self.c0 = 1;
        self.nibble = 1;
        self.bitpos = 0;
        self.excluded = None;
        self.ctx.update(byte, hist);
        self.mm.update_byte(hist);
    }
}
