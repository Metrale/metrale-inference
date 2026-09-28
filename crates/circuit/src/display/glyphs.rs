// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The two glyph sets of `met circuit display`: Unicode box drawing, and a pure
//! ASCII fallback for dumb terminals.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every Unicode glyph here is one terminal column wide, so a line's width is its char count
//!   (`display_tests.rs` checks this against `unicode-width`).
//! - The ASCII set is 7-bit only. `+`, `#` and `=` are its structural characters; the Unicode
//!   set never emits them.

/// 2026-09-28: Which glyph set to draw with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glyphs {
    /// 2026-09-28: Box drawing and symbols.
    Unicode,
    /// 2026-09-28: 7-bit ASCII only.
    Ascii,
}

/// 2026-09-28: One box style: corners top-left, top-right, bottom-left, bottom-right, then the
/// horizontal and vertical strokes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Frame {
    pub tl: char,
    pub tr: char,
    pub bl: char,
    pub br: char,
    pub h: char,
    pub v: char,
}

/// 2026-09-28: Every glyph the renderer draws.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Set {
    /// 2026-09-28: Light ops and the header card.
    pub light: Frame,
    /// 2026-09-28: Heavy (opaque) ops.
    pub heavy: Frame,
    /// 2026-09-28: A fused group's frame.
    pub fused: Frame,
    /// 2026-09-28: A materialised edge's stroke.
    pub wire: char,
    /// 2026-09-28: A fused (on-chip) edge's stroke.
    pub dotted: char,
    /// 2026-09-28: Arrow head.
    pub arrow: char,
    /// 2026-09-28: The branch off the main path, the corner turning down into the side path,
    /// and the corner turning back in.
    pub fan: [char; 3],
    /// 2026-09-28: Layer strip: GatedDeltaNet, attention, dense FFN, MoE FFN.
    pub layer: [char; 4],
    /// 2026-09-28: Numerics badges: bit-identical, reference, differs.
    pub badge: [&'static str; 3],
    /// 2026-09-28: Separator between facts.
    pub dot: &'static str,
    /// 2026-09-28: Truncation marker.
    pub ellipsis: &'static str,
    /// 2026-09-28: Multiplication sign in shapes and counts.
    pub times: &'static str,
    /// 2026-09-28: Kernel sequence arrow.
    pub then: &'static str,
    /// 2026-09-28: Marks an input read from a node other than the one above.
    pub reads: &'static str,
}

impl Glyphs {
    pub(crate) fn set(self) -> Set {
        match self {
            Glyphs::Unicode => Set {
                light: Frame {
                    tl: '╭',
                    tr: '╮',
                    bl: '╰',
                    br: '╯',
                    h: '─',
                    v: '│',
                },
                heavy: Frame {
                    tl: '╔',
                    tr: '╗',
                    bl: '╚',
                    br: '╝',
                    h: '═',
                    v: '║',
                },
                fused: Frame {
                    tl: '┏',
                    tr: '┓',
                    bl: '┗',
                    br: '┛',
                    h: '━',
                    v: '┃',
                },
                wire: '│',
                dotted: '┊',
                arrow: '▼',
                fan: ['├', '┐', '┘'],
                layer: ['▰', '◆', '▪', '✱'],
                badge: ["✓ bit-identical", "≈ reference", "⚠ differs"],
                dot: " · ",
                ellipsis: "…",
                times: "×",
                then: " → ",
                reads: "◂ ",
            },
            Glyphs::Ascii => Set {
                light: Frame {
                    tl: '+',
                    tr: '+',
                    bl: '+',
                    br: '+',
                    h: '-',
                    v: '|',
                },
                heavy: Frame {
                    tl: '#',
                    tr: '#',
                    bl: '#',
                    br: '#',
                    h: '=',
                    v: 'H',
                },
                fused: Frame {
                    tl: '#',
                    tr: '#',
                    bl: '#',
                    br: '#',
                    h: '#',
                    v: '#',
                },
                wire: '|',
                dotted: ':',
                arrow: 'v',
                fan: ['+', '+', '+'],
                layer: ['G', 'A', 'd', 'm'],
                badge: ["[ok] bit-identical", "[~] reference", "[!] differs"],
                dot: " . ",
                ellipsis: "...",
                times: "x",
                then: " -> ",
                reads: "< ",
            },
        }
    }
}
