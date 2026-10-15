// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: table b-tree traversal (file-format sections 1.5 and 1.6).
//! Interior pages (0x05) are walked left to right, leaf pages (0x0d) yield
//! (rowid, payload); payloads that spill are reassembled from their overflow
//! chain. Every page is visited at most once per scan, so a corrupt file
//! with a page cycle is an error, not an endless loop.

use std::borrow::Cow;

use crate::adapters::sqlite::record::varint;
use crate::core::error::Result;
use crate::fail;

const INTERIOR: u8 = 0x05;
const LEAF: u8 = 0x0d;

/// The page geometry the traversal needs, over the whole file image.
pub struct Pages<'a> {
    pub data: &'a [u8],
    pub page_size: usize,
    pub usable: usize,
    pub count: u32,
}

impl<'a> Pages<'a> {
    fn page(&self, n: u32) -> Result<&'a [u8]> {
        if n == 0 || n > self.count {
            fail!("page {n} is outside the database (1..={})", self.count);
        }
        let start = (n as usize - 1) * self.page_size;
        match self.data.get(start..start + self.page_size) {
            Some(p) => Ok(p),
            None => fail!("page {n} runs past the end of the file"),
        }
    }

    /// Payload bytes of a table-leaf cell kept on the leaf page itself.
    fn local_len(&self, payload: usize) -> usize {
        let u = self.usable;
        let x = u - 35;
        if payload <= x {
            return payload;
        }
        let m = ((u - 12) * 32 / 255) - 23;
        let k = m + (payload - m) % (u - 4);
        if k <= x { k } else { m }
    }

    fn overflow(
        &self,
        mut next: u32,
        mut need: usize,
        out: &mut Vec<u8>,
        seen: &mut [bool],
    ) -> Result<()> {
        while need > 0 {
            if next == 0 {
                fail!("overflow chain ends with {need} payload bytes missing");
            }
            mark(seen, next)?;
            let p = self.page(next)?;
            let take = need.min(self.usable - 4);
            out.extend_from_slice(&p[4..4 + take]);
            need -= take;
            next = u32::from_be_bytes([p[0], p[1], p[2], p[3]]);
        }
        Ok(())
    }

    /// Visit every row of the table rooted at `root`, in rowid order.
    pub fn scan(&self, root: u32, f: &mut dyn FnMut(i64, &[u8]) -> Result<()>) -> Result<()> {
        let mut seen = vec![false; self.count as usize + 1];
        let mut stack = vec![root];
        while let Some(n) = stack.pop() {
            mark(&mut seen, n)?;
            let page = self.page(n)?;
            let h = if n == 1 { 100 } else { 0 };
            let Some(&kind) = page.get(h) else {
                fail!("page {n} has no header")
            };
            let cells = usize::from(be16(page, h + 3)?);
            match kind {
                INTERIOR => {
                    let right = be32(page, h + 8)?;
                    let mut children = Vec::with_capacity(cells + 1);
                    for i in 0..cells {
                        let at = usize::from(be16(page, h + 12 + 2 * i)?);
                        children.push(be32(page, at)?);
                    }
                    children.push(right);
                    stack.extend(children.into_iter().rev());
                }
                LEAF => {
                    for i in 0..cells {
                        let at = usize::from(be16(page, h + 8 + 2 * i)?);
                        let (rowid, payload) = self.leaf_cell(page, at, &mut seen)?;
                        f(rowid, &payload)?;
                    }
                }
                k => fail!("page {n} has type {k:#04x}, expected a table b-tree page"),
            }
        }
        Ok(())
    }

    fn leaf_cell(
        &self,
        page: &'a [u8],
        at: usize,
        seen: &mut [bool],
    ) -> Result<(i64, Cow<'a, [u8]>)> {
        let (size, a) = varint(page, at)?;
        let (rowid, b) = varint(page, at + a)?;
        let Ok(size) = usize::try_from(size) else {
            fail!("payload size overflows")
        };
        if size > (self.count as usize).saturating_mul(self.usable) {
            fail!("cell payload of {size} bytes is larger than the database");
        }
        let start = at + a + b;
        let local = self.local_len(size);
        let Some(here) = page.get(start..start + local) else {
            fail!("cell payload runs past its page")
        };
        if local == size {
            return Ok((rowid as i64, Cow::Borrowed(here)));
        }
        let next = be32(page, start + local)?;
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(here);
        self.overflow(next, size - local, &mut out, seen)?;
        Ok((rowid as i64, Cow::Owned(out)))
    }
}

fn mark(seen: &mut [bool], n: u32) -> Result<()> {
    match seen.get_mut(n as usize) {
        Some(true) => fail!("page {n} is reached twice: the b-tree is corrupt"),
        Some(s) => {
            *s = true;
            Ok(())
        }
        None => fail!("page {n} is outside the database"),
    }
}

pub fn be16(b: &[u8], at: usize) -> Result<u16> {
    match b.get(at..at + 2) {
        Some(s) => Ok(u16::from_be_bytes([s[0], s[1]])),
        None => fail!("read past the end of a page"),
    }
}

pub fn be32(b: &[u8], at: usize) -> Result<u32> {
    match b.get(at..at + 4) {
        Some(s) => Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]])),
        None => fail!("read past the end of a page"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_payload_follows_the_spec_formula() {
        let p = Pages {
            data: &[],
            page_size: 512,
            usable: 512,
            count: 0,
        };
        assert_eq!(p.local_len(477), 477, "X = U - 35 fits on the page");
        // M = (500 * 32 / 255) - 23 = 39; K = M + (P - M) % 508.
        assert_eq!(
            p.local_len(478),
            39,
            "K = 478 exceeds X, so only M stays local"
        );
        assert_eq!(
            p.local_len(3000),
            460,
            "K = 39 + 2961 % 508 = 460 fits under X"
        );
    }
}
