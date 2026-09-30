// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Which kernels a device can run: a rule's kernel is available when the class
//! compiles its module and names the function, no MODEL.toml declares it absent, the class's
//! build defines do not compile its region out, and the device runs natively the instruction
//! kind its source guard stands for (`kernels/DEVICES.toml` `[[guard]]`).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - A kernel's requirement is read from the source it compiles from: the preprocessor region
//!   holding its `__global__` definition. No kernel is marked by hand.
//! - The build and the device must agree: a class that compiles a guarded region out while one
//!   of its devices claims the instruction, or keeps it while a device lacks it, is an error
//!   ([`check_build`]), never resolved in either's favour.

use std::collections::BTreeMap;

use super::HwError;
use super::class::ClassInfo;
use super::device::{Device, Guard, Instr, Polarity};
use super::sources::ClassSources;
use crate::fuser::AvailableKernels;
use crate::rules::{KernelId, Rule};

/// 2026-09-30: Why a kernel is not available on the device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Absence {
    /// 2026-09-30: The class does not compile the module, or the module does not name it.
    NotCompiled,
    /// 2026-09-30: A MODEL.toml `[expected_absent]` entry, with its reason.
    ExpectedAbsent(String),
    /// 2026-09-30: The class's build flags define the macro that compiles its region out.
    CompiledOut {
        /// 2026-09-30: The macro.
        macro_name: String,
        /// 2026-09-30: What the region needs.
        requires: Instr,
    },
    /// 2026-09-30: The device does not run the region's instruction kind.
    NotNative(Instr),
}

impl Absence {
    /// 2026-09-30: One line for the report.
    pub fn describe(&self) -> String {
        match self {
            Absence::NotCompiled => "not compiled for this class".into(),
            Absence::ExpectedAbsent(why) => format!("expected absent: {why}"),
            Absence::CompiledOut {
                macro_name,
                requires,
            } => format!(
                "compiled out by -D{macro_name} (needs {})",
                requires.name()
            ),
            Absence::NotNative(i) => format!("needs {}, which the device lacks", i.name()),
        }
    }
}

/// 2026-09-30: The kernels of a rule set on one device.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Availability {
    /// 2026-09-30: What the fuser may select.
    pub kernels: AvailableKernels,
    /// 2026-09-30: Every rule kernel that is not available, and why.
    pub absent: BTreeMap<KernelId, Absence>,
    /// 2026-09-30: The instruction kind each guarded kernel needs.
    pub requires: BTreeMap<KernelId, Instr>,
}

/// 2026-09-30: Fail when the class's build defines contradict the device's instruction kinds.
pub fn check_build(device: &Device, class: &ClassInfo, guards: &[Guard]) -> Result<(), HwError> {
    for g in guards.iter().filter(|g| g.polarity == Polarity::Ifndef) {
        let defined = class.defines.contains(&g.macro_name);
        let native = device.native.contains(&g.requires);
        if defined == native {
            return Err(HwError::BuildContradictsDevice {
                device: device.id.clone(),
                class: class.name.clone(),
                macro_name: g.macro_name.clone(),
                requires: g.requires.name(),
                defined,
            });
        }
    }
    Ok(())
}

/// 2026-09-30: `word` appears in `text` as a whole identifier.
pub fn names_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(at, _)| {
        let ok = |c: Option<char>| c.is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
        ok(text[..at].chars().next_back()) && ok(text[at + word.len()..].chars().next())
    })
}

/// 2026-09-30: The guard whose region holds the `__global__` definition of `func` in `text`,
/// when every definition sits in one.
pub fn guard_of<'g>(text: &str, func: &str, guards: &'g [Guard]) -> Option<&'g Guard> {
    let lines: Vec<&str> = text.lines().collect();
    let regions = regions(&lines, guards);
    let mut found: Option<&Guard> = None;
    let mut any = false;
    for (i, line) in lines.iter().enumerate() {
        if !defines_here(line, func) {
            continue;
        }
        let head = lines[i.saturating_sub(3)..=i].join(" ");
        if !head.contains("__global__") {
            continue;
        }
        any = true;
        match regions[i] {
            Some(g) => found = found.or(Some(&guards[g])),
            None => return None,
        }
    }
    if any { found } else { None }
}

fn defines_here(line: &str, func: &str) -> bool {
    line.match_indices(func).any(|(at, _)| {
        let before = line[..at].chars().next_back();
        let after = line[at + func.len()..].trim_start();
        before.is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_')) && after.starts_with('(')
    })
}

/// 2026-09-30: Per line, the index of the guard whose requiring branch holds it.
fn regions(lines: &[&str], guards: &[Guard]) -> Vec<Option<usize>> {
    // 2026-09-30: One frame per open `#if`: the guard it tests and whether the current branch
    // is the one that needs the guard's instruction.
    let mut stack: Vec<Option<(usize, bool)>> = Vec::new();
    let mut out = Vec::with_capacity(lines.len());
    for line in lines {
        let d = line.trim_start();
        let directive = d.strip_prefix('#').map(str::trim_start);
        match directive {
            Some(rest) if rest.starts_with("if") => stack.push(opening(rest, guards)),
            Some(rest) if rest.starts_with("elif") || rest.starts_with("else") => {
                if let Some(Some((_, needing))) = stack.last_mut() {
                    *needing = !*needing && rest.starts_with("else");
                }
            }
            Some(rest) if rest.starts_with("endif") => {
                stack.pop();
            }
            _ => {}
        }
        out.push(
            stack
                .iter()
                .rev()
                .find_map(|f| f.and_then(|(g, needing)| needing.then_some(g))),
        );
    }
    out
}

fn opening(rest: &str, guards: &[Guard]) -> Option<(usize, bool)> {
    let (neg, name) = if let Some(n) = rest.strip_prefix("ifndef") {
        (true, n.trim())
    } else if let Some(n) = rest.strip_prefix("ifdef") {
        (false, n.trim())
    } else {
        let e = rest.strip_prefix("if")?.trim();
        let (neg, e) = match e.strip_prefix('!') {
            Some(x) => (true, x.trim_start()),
            None => (false, e),
        };
        let e = e.strip_prefix("defined")?.trim();
        let e = e.trim_start_matches('(').trim_end_matches(')').trim();
        (neg, e)
    };
    let name: String = name
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    let g = guards.iter().position(|g| g.macro_name == name)?;
    let needing = match guards[g].polarity {
        Polarity::Ifndef => neg,
        Polarity::Ifdef => !neg,
    };
    Some((g, needing))
}

/// 2026-09-30: Whether `device` can run kernel `k` from `sources` of `class`, and the
/// instruction kind its source guard needs (`None`: unguarded or not compiled).
pub fn kernel_status(
    device: &Device,
    class: &ClassInfo,
    sources: &ClassSources,
    guards: &[Guard],
    k: &KernelId,
) -> (Result<(), Absence>, Option<Instr>) {
    let text = sources.modules.get(&k.module).map(|m| m.text.as_str());
    let Some(text) = text.filter(|t| names_word(t, &k.func)) else {
        return (Err(Absence::NotCompiled), None);
    };
    if let Some(why) = sources
        .expected_absent
        .get(&k.module)
        .and_then(|m| m.get(&k.func))
    {
        return (Err(Absence::ExpectedAbsent(why.clone())), None);
    }
    let Some(g) = guard_of(text, &k.func, guards) else {
        return (Ok(()), None);
    };
    let status = if g.polarity == Polarity::Ifndef && class.defines.contains(&g.macro_name) {
        Err(Absence::CompiledOut {
            macro_name: g.macro_name.clone(),
            requires: g.requires,
        })
    } else if !device.native.contains(&g.requires) {
        Err(Absence::NotNative(g.requires))
    } else {
        Ok(())
    };
    (status, Some(g.requires))
}

/// 2026-09-30: The rule kernels `device` can run from `sources` of `class`.
pub fn availability(
    device: &Device,
    class: &ClassInfo,
    sources: &ClassSources,
    rules: &[Rule],
    guards: &[Guard],
) -> Availability {
    let mut out = Availability::default();
    for r in rules {
        out.kernels.caps.extend(r.requires.iter().cloned());
        for k in &r.kernels {
            if out.kernels.kernels.contains(k) || out.absent.contains_key(k) {
                continue;
            }
            let (status, req) = kernel_status(device, class, sources, guards, k);
            if let Some(i) = req {
                out.requires.insert(k.clone(), i);
            }
            match status {
                Ok(()) => {
                    out.kernels.kernels.insert(k.clone());
                }
                Err(a) => {
                    out.absent.insert(k.clone(), a);
                }
            }
        }
    }
    out
}
