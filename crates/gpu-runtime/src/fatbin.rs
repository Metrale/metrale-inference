// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: The PTX text inside an NVIDIA fatbin kernel image (`METRALE_KERNEL_IMAGE=fatbin`,
//! the default). The driver loads the fatbin's SASS; the registry reads the PTX beside it for
//! what only the text says: the module's globals (`ptx_globals`) and its waiting entries
//! (`ptx_pdl`). A fatbin without readable PTX is an error, never "no metadata": a module that
//! silently lists no waiting entry would lose its PDL launches.
//!
//! Layout read here (the `fatbinary` container nvcc writes): a 16-byte header, magic
//! `0xBA55ED50` (u32 LE), version u16, header size u16, then a u64 byte count of the entries
//! that follow. Each entry begins with kind u16 (1 = PTX, 2 = ELF), version u16, header size
//! u32 (the payload starts there), payload size u64, and flags u64 at byte 40, where `0x2000`
//! (LZ4) or `0x8000` (zstd, nvcc 13's default) marks a compressed payload. A PTX payload is
//! NUL-padded text; nvcc stores it with comments and indentation removed.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Pure: no driver calls, no allocation beyond the returned list.
//! - Every offset is bounds-checked; a malformed container is an error, not a panic.

/// 2026-10-24: The fatbin magic, little-endian on disk.
const MAGIC: u32 = 0xBA55_ED50;
const KIND_PTX: u16 = 1;
const FLAG_COMPRESSED: u64 = 0x2000 | 0x8000;
const ENTRY_FLAGS_AT: usize = 40;

/// 2026-10-24: Whether `blob` is a fatbin container (its first four bytes are the magic).
pub fn is_fatbin(blob: &[u8]) -> bool {
    read_u32(blob, 0) == Some(MAGIC)
}

/// 2026-10-24: Every PTX payload in the fatbin `blob`, as text, in container order. `Err` names
/// what is wrong: not a fatbin, a truncated header or entry, a compressed PTX payload, a PTX
/// payload that is not UTF-8, or no PTX payload at all.
pub fn ptx_texts(blob: &[u8]) -> Result<Vec<&str>, String> {
    if !is_fatbin(blob) {
        return Err("not a fatbin (magic 0xBA55ED50 absent)".into());
    }
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < blob.len() && read_u32(blob, at) == Some(MAGIC) {
        let header = usize::from(read_u16(blob, at + 6).ok_or("truncated fatbin header")?);
        let body = to_usize(read_u64(blob, at + 8).ok_or("truncated fatbin header")?)?;
        let start = at.checked_add(header).ok_or("fatbin header overflows")?;
        let end = start.checked_add(body).ok_or("fatbin size overflows")?;
        if end > blob.len() {
            return Err(format!(
                "fatbin claims {body} bytes past offset {start}, blob has {}",
                blob.len()
            ));
        }
        let mut e = start;
        while e < end {
            let kind = read_u16(blob, e).ok_or("truncated fatbin entry")?;
            let eh = to_usize(u64::from(
                read_u32(blob, e + 4).ok_or("truncated fatbin entry")?,
            ))?;
            let size = to_usize(read_u64(blob, e + 8).ok_or("truncated fatbin entry")?)?;
            let flags = read_u64(blob, e + ENTRY_FLAGS_AT).ok_or("truncated fatbin entry")?;
            let p0 = e.checked_add(eh).ok_or("fatbin entry overflows")?;
            let p1 = p0.checked_add(size).ok_or("fatbin entry overflows")?;
            if eh < ENTRY_FLAGS_AT + 8 || p1 > end {
                return Err(format!(
                    "fatbin entry at {e} (header {eh}, payload {size}) exceeds its container"
                ));
            }
            if kind == KIND_PTX {
                if flags & FLAG_COMPRESSED != 0 {
                    return Err(format!(
                        "fatbin PTX entry at {e} is compressed; the kernel build passes \
                         --no-compress so the PTX stays readable"
                    ));
                }
                let raw = &blob[p0..p1];
                let text = &raw[..raw.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1)];
                out.push(
                    std::str::from_utf8(text).map_err(|err| format!("fatbin PTX at {e}: {err}"))?,
                );
            }
            e = p1;
        }
        at = end;
    }
    if out.is_empty() {
        return Err("fatbin carries no PTX entry".into());
    }
    Ok(out)
}

fn to_usize(v: u64) -> Result<usize, String> {
    usize::try_from(v).map_err(|_| format!("fatbin size {v} does not fit usize"))
}

fn read_u16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        b.get(at..at.checked_add(2)?)?.try_into().ok()?,
    ))
}

fn read_u32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        b.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn read_u64(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        b.get(at..at.checked_add(8)?)?.try_into().ok()?,
    ))
}

#[cfg(test)]
#[path = "fatbin_tests.rs"]
mod tests;
