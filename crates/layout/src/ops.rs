// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Operations on layouts: coalescing, composition, complement, and the tiling built
//! from them (divide: split a layout into a tile and the tiles' index; product: repeat a tile).
//! Each computes a closed form and then checks it against the function it stands for by
//! evaluation; a closed form that disagrees is an error.
//!
//! Owner: metrale-layout.
//! Invariants: see [`crate`].

use crate::layout::{Digit, Layout, LayoutError};

/// 2026-10-05: The largest domain an operation verifies by evaluation (a 128 x 256 tile is
/// 32768 indices).
const MAX_CHECKED: u64 = 1 << 22;

fn err(s: String) -> LayoutError {
    LayoutError(s)
}

/// 2026-10-05: One mode of merged digits: size-1 digits dropped, and a digit merged into the one
/// before it when it continues it (`stride = previous size x previous stride`).
pub fn coalesce(l: &Layout) -> Layout {
    let mut out: Vec<Digit> = Vec::new();
    for &d in l.digits().filter(|d| d.size > 1) {
        match out.last_mut() {
            Some(p) if d.stride == p.size as i64 * p.stride => p.size *= d.size,
            _ => out.push(d),
        }
    }
    if out.is_empty() {
        out.push(Digit { size: 1, stride: 0 });
    }
    Layout { modes: vec![out] }
}

/// 2026-10-05: The digits of `a` that index `j * d` for `j` in `[0, s)`: skip the digits `d`
/// steps over, then take `s` values.
fn compose_digit(a: &[Digit], s: u64, d: i64) -> Result<Vec<Digit>, LayoutError> {
    if s == 1 {
        return Ok(vec![Digit { size: 1, stride: 0 }]);
    }
    if d == 0 {
        return Ok(vec![Digit { size: s, stride: 0 }]);
    }
    if d < 0 {
        return Err(err(format!(
            "stride {d}: an index into a layout is never negative"
        )));
    }
    let mut rest = d as u64;
    let mut digits: Vec<Digit> = a.to_vec();
    let mut k = 0;
    while rest > 1 {
        let Some(&x) = digits.get(k) else {
            return Err(err(format!("stride {d} steps past the layout")));
        };
        if rest.is_multiple_of(x.size) {
            rest /= x.size;
            k += 1;
        } else if x.size.is_multiple_of(rest) {
            digits[k] = Digit {
                size: x.size / rest,
                stride: x.stride * rest as i64,
            };
            rest = 1;
        } else {
            return Err(err(format!(
                "stride {d} does not divide the layout's digit sizes"
            )));
        }
    }
    let mut out = Vec::new();
    let mut need = s;
    for x in &digits[k..] {
        if need == 1 {
            break;
        }
        if need.is_multiple_of(x.size) {
            out.push(*x);
            need /= x.size;
        } else if x.size.is_multiple_of(need) {
            out.push(Digit {
                size: need,
                stride: x.stride,
            });
            need = 1;
        } else {
            return Err(err(format!(
                "size {s} does not divide the layout's digit sizes"
            )));
        }
    }
    if need > 1 {
        return Err(err(format!("{s} x {d} indexes past the layout")));
    }
    Ok(out)
}

fn verify(name: &str, got: &Layout, want: impl Fn(u64) -> i64) -> Result<(), LayoutError> {
    let n = got.size();
    if n > MAX_CHECKED {
        return Err(err(format!(
            "{name}: {n} indices is more than this crate verifies"
        )));
    }
    match (0..n).find(|&i| got.eval(i) != want(i)) {
        Some(i) => Err(err(format!(
            "{name}: no layout equals it (index {i}: {} != {})",
            got.eval(i),
            want(i)
        ))),
        None => Ok(()),
    }
}

/// 2026-10-05: The layout of `a ∘ b`: index `i` maps to `a(b(i))`. `b` must map into
/// `[0, size(a))`; the result has `b`'s modes.
pub fn compose(a: &Layout, b: &Layout) -> Result<Layout, LayoutError> {
    let flat: Vec<Digit> = coalesce(a).modes.remove(0);
    let mut modes = Vec::with_capacity(b.modes.len());
    for m in &b.modes {
        let mut mode = Vec::new();
        for d in m {
            mode.extend(compose_digit(&flat, d.size, d.stride)?);
        }
        modes.push(mode);
    }
    let out = Layout::from_modes(modes)?;
    let size = a.size();
    verify("compose", &out, |i| {
        let j = b.eval(i);
        if j < 0 || j as u64 >= size {
            i64::MIN
        } else {
            a.eval(j as u64)
        }
    })?;
    Ok(out)
}

/// 2026-10-05: The layout `c` such that `(l, c)` maps `[0, m)` one to one onto `[0, m)`: the
/// offsets `l` leaves out, in order. `l` must be injective with positive strides, each
/// stride a multiple of the extent of the smaller digits.
pub fn complement(l: &Layout, m: u64) -> Result<Layout, LayoutError> {
    let mut ds: Vec<Digit> = l.digits().copied().filter(|d| d.size > 1).collect();
    if ds.iter().any(|d| d.stride <= 0) {
        return Err(err("complement: strides must be positive".into()));
    }
    ds.sort_by_key(|d| d.stride);
    let mut cur: u64 = 1;
    let mut out = Vec::new();
    for d in ds {
        let st = d.stride as u64;
        if !st.is_multiple_of(cur) {
            return Err(err(format!(
                "complement: stride {st} is not a multiple of {cur}"
            )));
        }
        if st / cur > 1 {
            out.push(Digit {
                size: st / cur,
                stride: cur as i64,
            });
        }
        cur = d.size * st;
    }
    if !m.is_multiple_of(cur) {
        return Err(err(format!(
            "complement: {m} is not a multiple of the extent {cur}"
        )));
    }
    if m / cur > 1 {
        out.push(Digit {
            size: m / cur,
            stride: cur as i64,
        });
    }
    if out.is_empty() {
        out.push(Digit { size: 1, stride: 0 });
    }
    let c = Layout { modes: vec![out] };
    let mut both = l.modes.clone();
    both.extend(c.modes.clone());
    let both = Layout::from_modes(both)?;
    if both.size() != m || !both.is_bijective() {
        return Err(err(format!(
            "complement: the layout and its complement do not tile [0, {m})"
        )));
    }
    Ok(c)
}

/// 2026-10-05: Split `l` by the tile `t`: mode 0 indexes inside a tile, mode 1 indexes the
/// tiles (`l ∘ (t, complement(t, size(l)))`).
pub fn divide(l: &Layout, t: &Layout) -> Result<Layout, LayoutError> {
    let c = complement(t, l.size())?;
    let inner: Vec<Digit> = t.digits().copied().collect();
    let b = Layout::from_modes(vec![inner, c.modes[0].clone()])?;
    compose(l, &b)
}

/// 2026-10-05: Repeat `l` by the layout `t` of repetitions: mode 0 is `l`, mode 1 places the
/// copies after `l`'s extent (`(l, complement(l, size(l) x cosize(t)) ∘ t)`).
pub fn product(l: &Layout, t: &Layout) -> Result<Layout, LayoutError> {
    let c = complement(l, l.size() * t.cosize())?;
    let rep = compose(&c, t)?;
    let a: Vec<Digit> = l.digits().copied().collect();
    let b: Vec<Digit> = rep.digits().copied().collect();
    Layout::from_modes(vec![a, b])
}
