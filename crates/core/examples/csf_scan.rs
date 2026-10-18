// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Decode every `csf_32x2_v1` scale plane of a safetensors checkpoint on the host and
//! check it against the format's invariants, with the exception arena consumed completely.
//!
//! Usage: `csf_scan <checkpoint-dir> <nvfp4|mxfp4> [--reference <dir>] [--report <prefix>]...`
//!
//! - Every header is validated first: a known dtype, `shape x elem_size == span`, every span
//!   inside the data region, no two spans of a file overlapping.
//! - Every plane is assembled from its five components (dtypes and extents checked, the logical
//!   `[R, C]` taken from the packed weight's shape), decoded with `decode_csf_32x2_v1`, and
//!   summarised. A component without a `.csf32_codes` sibling is an error.
//! - `--report <prefix>` prints per-plane scale statistics for each plane whose weight name
//!   starts with the prefix; `--reference <dir>` adds the same statistics for the same-named
//!   native `weight_scale` of a second checkpoint, plus both `weight_scale_2` and how many scale
//!   bytes the two hold identically.
//!
//! The checkpoint is only ever opened for reading. Exit status is non-zero on any error.
//!
//! Owner: metrale-core.

use anyhow::{Context, Result, bail, ensure};
use metrale_core::csf_blocks::{
    CsfGeometry, CsfOffsetWidth, CsfPlaneRef, SUFFIX_BASES, SUFFIX_CODES, SUFFIX_EXCEPTIONS,
    SUFFIX_OFFSETS, SUFFIX_PAGE_BASES, decode_csf_32x2_v1_to_vec,
};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const HEADER_CAP: u64 = 256 << 20;

struct Entry {
    file: usize,
    dtype: String,
    shape: Vec<usize>,
    begin: u64,
    end: u64,
}

struct Checkpoint {
    files: Vec<File>,
    data_start: Vec<u64>,
    tensors: BTreeMap<String, Entry>,
}

fn elem_size(dtype: &str) -> Result<u64> {
    Ok(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        other => bail!("unknown dtype {other}"),
    })
}

/// 2026-10-17: Parse and validate one header against its data region of `data_len` bytes.
fn parse_header(json: &[u8], file: usize, data_len: u64) -> Result<Vec<(String, Entry)>> {
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(json)?;
    let mut out = Vec::with_capacity(map.len());
    for (name, v) in map.into_iter().filter(|(k, _)| k != "__metadata__") {
        let dtype = v["dtype"].as_str().context("dtype")?.to_string();
        let shape = v["shape"]
            .as_array()
            .context("shape")?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).context("dim"))
            .collect::<Result<Vec<_>>>()?;
        let off = v["data_offsets"].as_array().context("data_offsets")?;
        ensure!(off.len() == 2, "{name}: data_offsets");
        let (begin, end) = (
            off[0].as_u64().context("begin")?,
            off[1].as_u64().context("end")?,
        );
        let want = shape
            .iter()
            .try_fold(elem_size(&dtype)?, |a, &d| a.checked_mul(d as u64))
            .with_context(|| format!("{name}: shape overflows"))?;
        ensure!(
            begin <= end && end <= data_len && end - begin == want,
            "{name}: span [{begin}, {end}) of a {data_len}-byte region, {want} bytes needed"
        );
        out.push((
            name,
            Entry {
                file,
                dtype,
                shape,
                begin,
                end,
            },
        ));
    }
    let mut spans: Vec<(u64, u64, &str)> = out
        .iter()
        .map(|(n, e)| (e.begin, e.end, n.as_str()))
        .collect();
    spans.sort();
    for w in spans.windows(2) {
        ensure!(w[1].0 >= w[0].1, "{} overlaps {}", w[1].2, w[0].2);
    }
    Ok(out)
}

impl Checkpoint {
    fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<_>>()?;
        paths.retain(|p| p.extension().is_some_and(|x| x == "safetensors"));
        paths.sort();
        ensure!(
            !paths.is_empty(),
            "no safetensors files in {}",
            dir.display()
        );
        let (mut files, mut data_start, mut tensors) = (vec![], vec![], BTreeMap::new());
        for (i, p) in paths.iter().enumerate() {
            let mut f = File::open(p)?;
            let mut len = [0u8; 8];
            f.read_exact(&mut len)?;
            let hlen = u64::from_le_bytes(len);
            ensure!(
                hlen <= HEADER_CAP,
                "{}: header of {hlen} bytes",
                p.display()
            );
            let mut json = vec![0u8; hlen as usize];
            f.read_exact(&mut json)?;
            let size = f.metadata()?.len();
            let data_len = size.checked_sub(8 + hlen).context("truncated")?;
            for (name, e) in parse_header(&json, i, data_len)
                .with_context(|| format!("header of {}", p.display()))?
            {
                ensure!(tensors.insert(name.clone(), e).is_none(), "{name} twice");
            }
            files.push(f);
            data_start.push(8 + hlen);
        }
        Ok(Self {
            files,
            data_start,
            tensors,
        })
    }

    fn entry(&self, name: &str) -> Result<&Entry> {
        self.tensors
            .get(name)
            .with_context(|| format!("{name} absent"))
    }

    fn read(&mut self, name: &str) -> Result<Vec<u8>> {
        let e = self.entry(name)?;
        let (file, at, len) = (e.file, self.data_start[e.file] + e.begin, e.end - e.begin);
        let f = &mut self.files[file];
        f.seek(SeekFrom::Start(at))?;
        let mut out = vec![0u8; len as usize];
        f.read_exact(&mut out)?;
        Ok(out)
    }

    fn scalar_f32(&mut self, name: &str) -> Result<f32> {
        let b = self.read(name)?;
        ensure!(b.len() == 4, "{name}: not one f32");
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// 2026-10-17: Value of one scale byte, `None` for the NaN encodings.
fn scale_value(b: u8, width: CsfOffsetWidth) -> Option<f32> {
    match width {
        CsfOffsetWidth::Nvfp4 => {
            let (e, m) = ((b >> 3) & 0xF, f32::from(b & 7));
            let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
            (b & 0x7F != 0x7F).then(|| match e {
                0 => sign * m / 8.0 * 2f32.powi(-6),
                _ => sign * (1.0 + m / 8.0) * 2f32.powi(i32::from(e) - 7),
            })
        }
        CsfOffsetWidth::Mxfp4 => (b != 0xFF).then(|| 2f32.powi(i32::from(b) - 127)),
    }
}

#[derive(Clone, Copy)]
struct Stats {
    n: usize,
    zeros: usize,
    nan: usize,
    negative: usize,
    min_nonzero: f32,
    max: f32,
}

impl Stats {
    fn new() -> Self {
        Self {
            n: 0,
            zeros: 0,
            nan: 0,
            negative: 0,
            min_nonzero: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }

    fn of(bytes: &[u8], width: CsfOffsetWidth) -> Self {
        let mut s = Self::new();
        for &b in bytes {
            s.n += 1;
            match scale_value(b, width) {
                None => s.nan += 1,
                Some(0.0) => s.zeros += 1,
                Some(v) => {
                    s.negative += usize::from(v < 0.0);
                    s.min_nonzero = s.min_nonzero.min(v.abs());
                    s.max = s.max.max(v);
                }
            }
        }
        s
    }

    fn merge(&mut self, o: &Self) {
        self.n += o.n;
        self.zeros += o.zeros;
        self.nan += o.nan;
        self.negative += o.negative;
        self.min_nonzero = self.min_nonzero.min(o.min_nonzero);
        self.max = self.max.max(o.max);
    }

    fn line(&self, ws2: Option<f32>) -> String {
        let eff = ws2.map_or(String::new(), |g| {
            format!(
                " ws2={g:.6e} eff[min_nz={:.4e} max={:.4e}]",
                self.min_nonzero * g,
                self.max * g
            )
        });
        format!(
            "n={} zeros={} nan={} neg={} min_nz={} max={}{eff}",
            self.n, self.zeros, self.nan, self.negative, self.min_nonzero, self.max
        )
    }
}

/// 2026-10-17: The component of `scale` with `suffix`, with its dtype and element count checked.
fn component(ck: &mut Checkpoint, scale: &str, suffix: &str, dtype: &str) -> Result<Vec<u8>> {
    let name = format!("{scale}{suffix}");
    let e = ck.entry(&name)?;
    ensure!(
        e.dtype == dtype && e.shape.len() == 1,
        "{name}: {} {:?}",
        e.dtype,
        e.shape
    );
    ck.read(&name)
}

/// 2026-10-17: Decode the plane stored for `scale` (`<proj>.weight_scale`).
fn decode_plane(
    ck: &mut Checkpoint,
    scale: &str,
    width: CsfOffsetWidth,
) -> Result<(Vec<u8>, usize)> {
    let weight = scale.strip_suffix("_scale").context("not a weight_scale")?;
    let w = ck.entry(weight)?;
    ensure!(
        w.dtype == "U8" && w.shape.len() == 2,
        "{weight}: {} {:?}",
        w.dtype,
        w.shape
    );
    let group = match width {
        CsfOffsetWidth::Nvfp4 => 16,
        CsfOffsetWidth::Mxfp4 => 32,
    };
    let (rows, packed) = (w.shape[0], w.shape[1]);
    ensure!(
        (2 * packed).is_multiple_of(group),
        "{weight}: K {} not whole groups",
        2 * packed
    );
    let geometry = CsfGeometry::new(rows, 2 * packed / group, width)?;
    let bases = component(ck, scale, SUFFIX_BASES, "U8")?;
    let codes = component(ck, scale, SUFFIX_CODES, "U8")?;
    let page_bases = component(ck, scale, SUFFIX_PAGE_BASES, "U32")?;
    let offsets = component(ck, scale, SUFFIX_OFFSETS, "U16")?;
    let exceptions = component(ck, scale, SUFFIX_EXCEPTIONS, "U16")?;
    let plane = CsfPlaneRef {
        geometry,
        bases: &bases,
        codes: &codes,
        page_bases: &page_bases,
        offsets: &offsets,
        exceptions: &exceptions,
    };
    let out = decode_csf_32x2_v1_to_vec(&plane).with_context(|| format!("decoding {scale}"))?;
    ensure!(
        out.len() == rows * geometry.columns,
        "{scale}: {} decoded bytes",
        out.len()
    );
    Ok((out, exceptions.len() / 2))
}

const SUFFIXES: [&str; 5] = [
    SUFFIX_BASES,
    SUFFIX_CODES,
    SUFFIX_PAGE_BASES,
    SUFFIX_OFFSETS,
    SUFFIX_EXCEPTIONS,
];

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage =
        "csf_scan <checkpoint-dir> <nvfp4|mxfp4> [--reference <dir>] [--report <prefix>]...";
    ensure!(args.len() >= 2, "{usage}");
    let width = match args[1].as_str() {
        "nvfp4" => CsfOffsetWidth::Nvfp4,
        "mxfp4" => CsfOffsetWidth::Mxfp4,
        w => bail!("unknown width {w}: {usage}"),
    };
    let (mut reference, mut report) = (None, vec![]);
    let mut rest = args[2..].iter();
    while let Some(flag) = rest.next() {
        let value = rest
            .next()
            .with_context(|| format!("{flag} needs a value: {usage}"))?;
        match flag.as_str() {
            "--reference" => reference = Some(Checkpoint::open(Path::new(value))?),
            "--report" => report.push(value.clone()),
            f => bail!("unknown flag {f}: {usage}"),
        }
    }
    let started = std::time::Instant::now();
    let mut ck = Checkpoint::open(Path::new(&args[0]))?;
    eprintln!(
        "headers valid: {} tensors in {} files",
        ck.tensors.len(),
        ck.files.len()
    );
    let mut planes: Vec<(usize, u64, String)> = ck
        .tensors
        .iter()
        .filter_map(|(n, e)| {
            n.strip_suffix(SUFFIX_CODES)
                .map(|s| (e.file, e.begin, s.to_string()))
        })
        .collect();
    planes.sort();
    let components = ck
        .tensors
        .keys()
        .filter(|n| SUFFIXES.iter().any(|s| n.ends_with(s)))
        .count();
    ensure!(
        components == 5 * planes.len(),
        "{components} csf components for {} planes",
        planes.len()
    );
    let (mut global, mut records, mut decoded) = (Stats::new(), 0usize, 0usize);
    for (i, (_, _, scale)) in planes.iter().enumerate() {
        let (scales, n) = decode_plane(&mut ck, scale, width)?;
        let s = Stats::of(&scales, width);
        global.merge(&s);
        records += n;
        decoded += scales.len();
        if i % 2000 == 0 {
            eprintln!("{i}/{} planes", planes.len());
        }
        if !report.iter().any(|p| scale.starts_with(p.as_str())) {
            continue;
        }
        let ws2 = ck.scalar_f32(&format!("{scale}_2")).ok();
        println!("{scale}\n  csf    {} exceptions={n}", s.line(ws2));
        if let Some(r) = reference.as_mut() {
            let native = r.read(scale)?;
            ensure!(
                native.len() == scales.len(),
                "{scale}: reference holds {} bytes",
                native.len()
            );
            let rws2 = r.scalar_f32(&format!("{scale}_2")).ok();
            let same = native.iter().zip(&scales).filter(|(a, b)| a == b).count();
            println!(
                "  native {} identical_bytes={same}",
                Stats::of(&native, width).line(rws2)
            );
        }
    }
    ensure!(global.nan == 0, "{} NaN scale bytes decoded", global.nan);
    println!(
        "OK: {} planes decoded, every invariant held, arenas fully consumed; {records} exception \
         records, {decoded} scale bytes; all planes: {}; {:.1} s",
        planes.len(),
        global.line(None),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
