// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The TOML schema of KERNEL_FAMILIES.toml and its validation into
//! [`super::Families`].
//!
//! Owner: metrale-circuit (venn).
//! Invariants: see [`super`]; every check runs at load, in file order, and the first failure is
//! returned.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use super::ComputeUnit;
use super::compute_file::ComputeFile;
use super::op_file::{OpFile, op_spec};
use super::{
    Discover, Evidence, EvidenceSource, Extract, Families, Family, FamilyError, How, Param,
    ParamKind, Point, Roofline, Values,
};
use crate::rules::KernelId;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    schema: u32,
    hardware: String,
    roofline: RooflineFile,
    family: Vec<FamilyFile>,
    #[serde(default)]
    legacy_path: Vec<super::legacy_file::LegacyFile>,
    #[serde(default)]
    reduction: Vec<super::reduction::ReductionFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RooflineFile {
    dram_gbps: f64,
    bf16_tflops: f64,
    fp8_tflops: f64,
    nvfp4_tflops: f64,
    context_tokens: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyFile {
    id: String,
    description: String,
    #[serde(default)]
    kernels: Vec<String>,
    #[serde(default)]
    emitters: Vec<String>,
    rows: [u64; 2],
    /// 2026-10-03: The plan modes it runs in (absent: every mode).
    #[serde(default)]
    modes: Vec<String>,
    op: Vec<OpFile>,
    #[serde(default)]
    param: Vec<ParamFile>,
    point: Vec<PointFile>,
    #[serde(default)]
    evidence: Vec<EvidenceFile>,
    #[serde(default)]
    discover: Vec<DiscoverFile>,
    compute: String,
    mma: Option<String>,
    #[serde(default)]
    kernel_compute: BTreeMap<String, ComputeFile>,
    #[serde(default)]
    workspace: Vec<super::workspace_file::WorkspaceFile>,
    pipeline: BTreeMap<String, toml::Value>,
    #[serde(default)]
    kernel_pipeline: BTreeMap<String, BTreeMap<String, toml::Value>>,
    #[serde(default)]
    reduction: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum FromFile {
    All(String),
    ByOp(BTreeMap<String, String>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ParamFile {
    name: String,
    kind: String,
    from: Option<FromFile>,
    absent: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PointFile {
    values: Values,
    how: String,
    files: Vec<String>,
    compute: Option<String>,
    mma: Option<String>,
    #[serde(default)]
    pipeline: BTreeMap<String, toml::Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceFile {
    point: Values,
    rows: Vec<u64>,
    measurement: Option<String>,
    microbench: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DiscoverFile {
    kind: String,
    glob: Option<String>,
    values: Option<Values>,
    file: Option<String>,
    name: Option<String>,
    args: Option<BTreeMap<String, usize>>,
    #[serde(default)]
    map: BTreeMap<String, BTreeMap<String, String>>,
}

pub(super) fn parse(text: &str) -> Result<Families, FamilyError> {
    let file: File = toml::from_str(text).map_err(|e| FamilyError::Parse(e.to_string()))?;
    if file.schema != 1 {
        return Err(FamilyError::Parse(format!(
            "schema {} (this build reads 1)",
            file.schema
        )));
    }
    let r = &file.roofline;
    let positive = [r.dram_gbps, r.bf16_tflops, r.fp8_tflops, r.nvfp4_tflops];
    if positive.iter().any(|v| !v.is_finite() || *v <= 0.0) || r.context_tokens == 0 {
        return Err(FamilyError::Parse(
            "every [roofline] value must be positive".into(),
        ));
    }
    let roofline = Roofline {
        dram_gbps: r.dram_gbps,
        bf16_tflops: r.bf16_tflops,
        fp8_tflops: r.fp8_tflops,
        nvfp4_tflops: r.nvfp4_tflops,
        context_tokens: r.context_tokens,
    };
    let mut ids = BTreeSet::new();
    let mut owners: BTreeMap<String, String> = BTreeMap::new();
    let mut families = Vec::with_capacity(file.family.len());
    for f in file.family {
        if !ids.insert(f.id.clone()) {
            return Err(FamilyError::Parse(format!(
                "family `{}` is listed twice",
                f.id
            )));
        }
        let fam = family(f)?;
        let names = fam
            .kernels
            .iter()
            .map(|k| k.to_string())
            .chain(fam.emitters.iter().map(|e| format!("emitter {e}")));
        for name in names {
            if let Some(other) = owners.insert(name.clone(), fam.id.clone()) {
                return Err(FamilyError::Parse(format!(
                    "`{name}` is in families `{other}` and `{}`",
                    fam.id
                )));
            }
        }
        families.push(fam);
    }
    let legacy = file
        .legacy_path
        .into_iter()
        .map(super::legacy_file::legacy)
        .collect::<Result<_, _>>()?;
    let reductions = super::reduction::reductions(file.reduction).map_err(FamilyError::Parse)?;
    super::reduction::check_names(&families, &reductions).map_err(FamilyError::Parse)?;
    Ok(Families {
        hardware: file.hardware,
        roofline,
        families,
        legacy,
        reductions,
    })
}

fn family(f: FamilyFile) -> Result<Family, FamilyError> {
    let field = |detail: String| FamilyError::Field {
        family: f.id.clone(),
        detail,
    };
    if f.rows[0] == 0 || f.rows[0] > f.rows[1] {
        return Err(field(format!("rows {:?} is not a range from 1", f.rows)));
    }
    if f.op.is_empty() || f.point.is_empty() {
        return Err(field("needs at least one op and one point".into()));
    }
    if f.kernels.is_empty() == f.emitters.is_empty() {
        return Err(field(
            "lists kernels or emitters, exactly one of the two".into(),
        ));
    }
    let mut kernels = Vec::with_capacity(f.kernels.len());
    for k in &f.kernels {
        let (module, func) = k
            .split_once("::")
            .filter(|(m, n)| !m.is_empty() && !n.is_empty())
            .ok_or_else(|| field(format!("kernel `{k}` is not `module::function`")))?;
        kernels.push(KernelId {
            module: module.to_string(),
            func: func.to_string(),
        });
    }
    let modes = f
        .modes
        .iter()
        .map(|m| {
            crate::rules::Mode::parse(m).ok_or_else(|| field(format!("mode `{m}` is no plan mode")))
        })
        .collect::<Result<std::collections::BTreeSet<_>, _>>()?;
    let ops =
        f.op.iter()
            .map(|o| op_spec(&f.id, o))
            .collect::<Result<Vec<_>, _>>()?;
    let op_names: BTreeSet<&str> = ops.iter().map(|o| o.op.as_str()).collect();
    let mut params: Vec<Param> = Vec::with_capacity(f.param.len());
    for p in f.param {
        if params.iter().any(|q| q.name == p.name) {
            return Err(field(format!("parameter `{}` is declared twice", p.name)));
        }
        let kind = ParamKind::parse(&p.kind).ok_or_else(|| {
            field(format!(
                "parameter kind `{}` (runtime | compile | policy | numerics)",
                p.kind
            ))
        })?;
        // 2026-10-05: A numerics parameter is chosen by the kernel's point, not read from the
        // circuit; every other kind is read from it.
        let by_op: BTreeMap<String, String> = match (p.from, kind) {
            (Some(_), ParamKind::Numerics) => {
                return Err(field(format!(
                    "numerics parameter `{}` reads nothing: drop `from`",
                    p.name
                )));
            }
            (None, ParamKind::Numerics) => BTreeMap::new(),
            (None, _) => return Err(field(format!("parameter `{}` needs `from`", p.name))),
            (Some(FromFile::All(e)), _) => op_names
                .iter()
                .map(|o| (o.to_string(), e.clone()))
                .collect(),
            (Some(FromFile::ByOp(m)), _) => m,
        };
        let mut from = BTreeMap::new();
        for (op, e) in by_op {
            if !op_names.contains(op.as_str()) {
                return Err(field(format!(
                    "parameter `{}` reads op `{op}`, which the family does not implement",
                    p.name
                )));
            }
            let ex = Extract::parse(&e)
                .ok_or_else(|| field(format!("parameter `{}`: unknown extractor `{e}`", p.name)))?;
            from.insert(op, ex);
        }
        params.push(Param {
            name: p.name,
            kind,
            from,
            absent: p.absent,
        });
    }
    let pointed: BTreeSet<&str> = params
        .iter()
        .filter(|p| p.kind != ParamKind::Runtime)
        .map(|p| p.name.as_str())
        .collect();
    let check_keys = |keys: &mut dyn Iterator<Item = &String>| {
        for k in keys {
            if !pointed.contains(k.as_str()) {
                return Err(FamilyError::UnknownParam {
                    family: f.id.clone(),
                    param: k.clone(),
                });
            }
        }
        Ok(())
    };
    let mut points = Vec::with_capacity(f.point.len());
    for p in f.point {
        check_keys(&mut p.values.keys())?;
        if p.values.len() != pointed.len() {
            return Err(field(format!(
                "point {:?} does not state every compile-time, policy and numerics parameter",
                p.values
            )));
        }
        let how = match p.how.as_str() {
            "instantiation" => How::Instantiation,
            "copy" => How::Copy,
            "branch" => How::Branch,
            other => {
                return Err(field(format!(
                    "how `{other}` (instantiation | copy | branch)"
                )));
            }
        };
        if p.files.is_empty() || points.iter().any(|q: &Point| q.values == p.values) {
            return Err(field(format!(
                "point {:?} lists no files or is listed twice",
                p.values
            )));
        }
        let compute = p
            .compute
            .as_deref()
            .map(|c| ComputeUnit::parse(c, p.mma.as_deref()))
            .transpose()
            .map_err(|e| field(format!("point {:?}: {e}", p.values)))?;
        if p.compute.is_none() && p.mma.is_some() {
            return Err(field(format!(
                "point {:?} names an `mma` without `compute`",
                p.values
            )));
        }
        let pipeline = crate::pipeline::declare::parse_by_op(&p.pipeline)
            .map_err(|e| field(format!("point {:?}: {e}", p.values)))?;
        points.push(Point {
            values: p.values,
            how,
            files: p.files,
            compute,
            pipeline,
        });
    }
    let mut evidence = Vec::with_capacity(f.evidence.len());
    for e in f.evidence {
        check_keys(&mut e.point.keys())?;
        if !points.iter().any(|p| p.values == e.point) {
            return Err(field(format!(
                "evidence at {:?}, which is not an instantiated point",
                e.point
            )));
        }
        if e.rows.is_empty() || e.rows.contains(&0) {
            return Err(field("evidence needs row counts of at least 1".into()));
        }
        let source = match (e.measurement, e.microbench) {
            (Some(m), None) => EvidenceSource::Measurement(m),
            (None, Some(b)) => EvidenceSource::Microbench(b),
            _ => {
                return Err(field(
                    "evidence names exactly one of `measurement` / `microbench`".into(),
                ));
            }
        };
        evidence.push(Evidence {
            point: e.point,
            rows: e.rows.into_iter().collect(),
            source,
        });
    }
    let mut discover = Vec::with_capacity(f.discover.len());
    for d in f.discover {
        discover.push(
            match (d.kind.as_str(), d.glob, d.values, d.file, d.name, d.args) {
                ("file", Some(glob), Some(values), None, None, None) if d.map.is_empty() => {
                    check_keys(&mut values.keys())?;
                    Discover::File { glob, values }
                }
                ("macro", None, None, Some(file), Some(name), Some(args)) => {
                    check_keys(&mut args.keys())?;
                    check_keys(&mut d.map.keys())?;
                    if let Some(k) = d.map.keys().find(|k| !args.contains_key(*k)) {
                        return Err(field(format!("discover map `{k}` has no argument index")));
                    }
                    Discover::Macro {
                        file,
                        name,
                        args,
                        map: d.map,
                    }
                }
                _ => {
                    return Err(field(
                        "discover is `file` (glob, values) or `macro` (file, name, args, map)"
                            .into(),
                    ));
                }
            },
        );
    }
    let compute = super::compute_file::family_compute(
        &f.compute,
        f.mma.as_deref(),
        &f.kernel_compute,
        &kernels,
        &points,
    )
    .map_err(field)?;
    let workspace = super::workspace_file::workspaces(f.workspace).map_err(field)?;
    let pipeline = super::compute_file::family_pipelines(
        &f.pipeline,
        &f.kernel_pipeline,
        &kernels,
        &ops,
        &points,
    )
    .map_err(field)?;
    let reduction = super::reduction::kernel_reductions(&f.reduction, &kernels).map_err(field)?;
    Ok(Family {
        id: f.id,
        description: f.description,
        kernels,
        emitters: f.emitters,
        rows: (f.rows[0], f.rows[1]),
        modes,
        ops,
        params,
        points,
        evidence,
        discover,
        compute,
        workspace,
        pipeline,
        reduction,
    })
}
