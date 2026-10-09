// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: What the accuracy integration tests share: the checked-out tree as a [`Repo`].
//!
//! Owner: metrale-accuracy tests.
//! Invariants: none beyond the types.

#![allow(dead_code)]

use std::path::PathBuf;

use metrale_circuit::venn::Repo;

/// 2026-10-09: The repository root (two levels above this crate).
pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// 2026-10-09: The checked-out tree, as the CLI's file-system repo reads it.
pub struct Tree;

impl Repo for Tree {
    fn read(&self, rel: &str) -> Result<String, String> {
        std::fs::read_to_string(root().join(rel)).map_err(|e| format!("{rel}: {e}"))
    }

    fn list(&self, rel: &str) -> Result<Vec<String>, String> {
        let base = root();
        let mut out = Vec::new();
        let mut stack = vec![base.join(rel)];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir)
                .map_err(|e| e.to_string())?
                .flatten()
            {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if let Ok(r) = path.strip_prefix(&base) {
                    out.push(r.to_string_lossy().into_owned());
                }
            }
        }
        out.sort();
        Ok(out)
    }
}

use metrale_accuracy::case::Case;
use metrale_accuracy::contract::{Contract, parse_contracts};
use metrale_accuracy::elem::Elem;
use metrale_accuracy::plan;
use metrale_accuracy::refs::Reference;
use metrale_accuracy::runner::{KernelRunner, SENTINEL};
use metrale_circuit::venn::families::{Families, Family};

/// 2026-10-09: The gb10 family manifest.
pub fn families() -> Families {
    metrale_circuit::venn::parse_families(
        &Tree
            .read("kernels/gb10/common/KERNEL_FAMILIES.toml")
            .unwrap(),
    )
    .unwrap()
}

/// 2026-10-09: One contract from a TOML snippet (the file header is added).
pub fn contract(body: &str) -> Contract {
    let text = format!("schema = 1\nhardware = \"gb10\"\nseed = 20261009\n{body}");
    parse_contracts(&text).unwrap().contracts.remove(0)
}

/// 2026-10-09: How the CPU "kernel" behaves.
#[derive(Clone)]
pub enum Behaviour {
    /// 2026-10-09: The conforming emulation.
    Conforming,
    /// 2026-10-09: The emulation with this accumulator (a kernel that breaks its declaration).
    Accumulator(Elem),
}

/// 2026-10-09: A CPU runner: the conforming emulation of the case's contract, honouring shards
/// and leaving unwritten columns at the sentinel. `wrong` names an entry point that runs the
/// emulation with the odd scale groups reading the even groups' scales (a wrong-symbol stand-in).
pub struct Emu {
    pub contract: Contract,
    pub family: Family,
    pub behaviour: Behaviour,
    pub wrong: String,
}

impl KernelRunner for Emu {
    fn run(&mut self, case: &Case) -> Result<Vec<u8>, String> {
        let reference = Reference::parse(&self.contract.reference).unwrap();
        let mut case = case.clone();
        if case.kernel == self.wrong {
            let mut r = metrale_accuracy::inputs::SplitMix64::new(0);
            reference.mutate(
                &mut case,
                &metrale_accuracy::mutation::Mutation::SwapScaleGranularity,
                &mut r,
            )?;
        }
        let kernel = self.contract.kernels[0].clone();
        let pipeline = plan::declared(
            &self.family,
            &kernel,
            &self.contract.op,
            &Default::default(),
        )?;
        let shape = metrale_accuracy::points::Shape {
            op: self.contract.op.clone(),
            weight: None,
            activation: None,
            output: None,
            in_dim: case.tensor("x")?.dims[1] as u64,
            out_dim: case.out.0[1] as u64,
            rows: case.out.0[0] as u64,
            runtime: Default::default(),
        };
        let p = plan::plan(
            &self.contract,
            pipeline.clone(),
            &reference.lens(&shape, &pipeline),
        )?;
        let (rows, cols) = (case.out.0[0], case.out.0[1]);
        let acc = match &self.behaviour {
            Behaviour::Conforming => None,
            Behaviour::Accumulator(e) => Some(*e),
        };
        let shards = if case.split.is_empty() {
            vec![metrale_accuracy::case::Shard {
                lo: 0,
                hi: cols,
                out_at: 0,
            }]
        } else {
            case.split.clone()
        };
        let mut vals = vec![f64::NAN; rows * cols];
        for s in &shards {
            let idx: Vec<usize> = (0..rows)
                .flat_map(|r| (s.lo..s.hi).map(move |c| r * cols + c))
                .collect();
            let got = reference.emulate(&case, &p, acc, 0, &idx)?;
            for (&i, v) in idx.iter().zip(got) {
                let (r, c) = (i / cols, i % cols);
                let at = s.out_at + (c - s.lo);
                if at < cols {
                    vals[r * cols + at] = v;
                }
            }
        }
        let enc = case.out.1;
        let mut out = vec![SENTINEL; enc.bytes_for(rows * cols)];
        let width = enc.bytes_for(1);
        for (i, v) in vals.iter().enumerate() {
            if v.is_nan() {
                continue;
            }
            let t = metrale_accuracy::case::Tensor::encode(enc, vec![1], &[*v])?;
            out[i * width..(i + 1) * width].copy_from_slice(&t.bytes);
        }
        Ok(out)
    }

    fn closure(&self) -> String {
        "cpu-emulation".into()
    }

    fn device(&self) -> String {
        "cpu".into()
    }
}
