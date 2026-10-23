// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The kernel-math table of `kbp-graph/v2`: the explanation of every kernel family
//! and tensor format the graph names, once each. Nodes refer to it by `family`, edges by
//! `format`; the text is the engine's (`metrale_circuit::explain`), never the store's.
//!
//! Owner: blueprint.
//! Invariants: a mirror of [`metrale_circuit::explain::KernelMath`] with a JSON Schema; the
//! table comes from the exporting engine's KERNEL_FAMILIES.toml (`exported_by` names it), so it
//! is presentation, never part of the signed body.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use metrale_circuit::explain::KernelMath;

use super::schema::json_struct;

json_struct! {
    /// 2026-10-22: One kernel family.
    pub struct GraphFamilyMath {
        /// 2026-10-22: One line.
        pub description: String,
        /// 2026-10-22: What its kernels compute (prose, inline `$...$` math).
        pub about: String,
        /// 2026-10-22: Formats, accumulation and rounding (prose, inline math).
        pub numerics: String,
        /// 2026-10-22: Display equations (KaTeX).
        pub math: Vec<String>,
    }
}

json_struct! {
    /// 2026-10-22: One tensor format.
    pub struct GraphFormatMath {
        /// 2026-10-22: Its name in words.
        pub title: String,
        /// 2026-10-22: Layout and scales.
        pub description: String,
        /// 2026-10-22: Display equations: the decode.
        pub math: Vec<String>,
        /// 2026-10-22: The quantization into it, where the engine quantizes into it.
        pub encode: Option<String>,
    }
}

json_struct! {
    /// 2026-10-22: The table.
    pub struct GraphExplain {
        /// 2026-10-22: By family id (a node's `family`).
        pub families: BTreeMap<String, GraphFamilyMath>,
        /// 2026-10-22: By format spelling (an edge's `format`).
        pub formats: BTreeMap<String, GraphFormatMath>,
    }
}

json_struct! {
    /// 2026-10-22: An edge's requantisation: its producer read `from` and stored `to`.
    pub struct GraphRequant {
        /// 2026-10-22: The producer's input format.
        pub from: String,
        /// 2026-10-22: The stored, quantized format.
        pub to: String,
    }
}

impl From<KernelMath> for GraphExplain {
    fn from(t: KernelMath) -> Self {
        GraphExplain {
            families: t
                .families
                .into_iter()
                .map(|(k, f)| {
                    let m = GraphFamilyMath {
                        description: f.description,
                        about: f.about,
                        numerics: f.numerics,
                        math: f.math,
                    };
                    (k, m)
                })
                .collect(),
            formats: t
                .formats
                .into_iter()
                .map(|(k, f)| {
                    let m = GraphFormatMath {
                        title: f.title,
                        description: f.description,
                        math: f.math,
                        encode: f.encode,
                    };
                    (k, m)
                })
                .collect(),
        }
    }
}
