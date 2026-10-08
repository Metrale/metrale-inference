// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Checks that every kernel a tiled launcher runs publishes its N tile, in every
//! target tree, and that the published value is the tile the kernel body uses.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! The launchers in `model-layers` `ops` (`n_tile_blocks`) size grid.x as
//! `ceil(n / t)`, `t` being the `extern "C" __device__ unsigned int <entry>_n_tile` of the
//! resolved module (`GpuBackend::kernel_n_tile`). One entry name carries different tiles in
//! different trees (`moe_w4a16_grouped_gemm_ptrtable_t` is 64 wide in `gb10/common`, 128 in
//! the Qwen trees), which is how a grid sized for 128 left half of Nemotron-3-Nano's
//! routed-expert columns unwritten. A symbol that disagrees with `blockIdx.x * TILE` in the
//! body is the same bug moved into the kernel file, so this test compares the two.

use std::collections::BTreeMap;

use metrale_closure::layout::{discover, walk};

#[path = "support/cu_source.rs"]
mod cu_source;
use cu_source::{block_at, entry_start, int_defines, is_ident, workspace_root};

/// 2026-09-29: The entry points the tiled launchers run. A kernel added to a tiled launcher
/// without a symbol fails at its first launch; listing it here moves that failure to CI.
const TILED_ENTRIES: &[&str] = &[
    "w4a16_gemm",
    "w4a16_gemm_t",
    "w4a16_gemm_t_p3",
    "w4a16_gemm_t_k64",
    "w4a16_gemm_t_k64_p3",
    "w4a16_gemm_t_k64_n64_p3",
    "w4a16_gemm_t_m128",
    "w4a16_gemm_t_m128_v2",
    "w4a16_gemm_t_m128_v3",
    "w4a16_gemm_t_m128_bf16",
    "w4a16_gemm_t_m128_bf16_v2",
    "w4a16_gemm_t_m64_bf16",
    "moe_w4a16_grouped_gemm_ptrtable",
    "moe_w4a16_grouped_gemm_ptrtable_k32",
    "moe_w4a16_grouped_gemm_ptrtable_m256",
    "moe_w4a16_grouped_gemm_ptrtable_relu2",
    "moe_w4a16_grouped_gemm_ptrtable_e8m0",
    "moe_w4a16_grouped_gemm_ptrtable_t",
    "moe_w4a16_grouped_gemm_ptrtable_t_e8m0",
    "moe_w4a16_grouped_gemm_ptrtable_t_k64",
    "moe_w4a16_grouped_gemm_ptrtable_t_k64_e8m0",
    "moe_w4a16_fused_gate_up_t",
    "moe_w4a16_fused_gate_up_t_e8m0",
    "moe_w4a16_fused_gate_up_t_k64",
    "moe_w4a16_fused_gate_up_t_k64_e8m0",
    "moe_w4a16_fused_gate_up_t_k64_m128",
    "moe_w4a16_fused_gate_up_t_k64_fp4",
    "moe_fp8_grouped_gemm_ptrtable_t",
    "moe_w4a4_grouped_gemm_relu2",
];

/// 2026-09-29: The identifiers `blockIdx.x` is multiplied by in `body`.
fn block_x_factors(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (i, _) in body.match_indices("blockIdx.x") {
        let rest = body[i + "blockIdx.x".len()..].trim_start();
        let Some(rest) = rest.strip_prefix('*') else {
            continue;
        };
        let rest = rest.trim_start().trim_start_matches('(').trim_start();
        let ident: String = rest.chars().take_while(|c| is_ident(*c)).collect();
        if !ident.is_empty() {
            out.push(ident);
        }
    }
    out
}

/// 2026-10-07: The tiles of each `callee<args>(` call in `body` whose definition is a
/// `template<...>` helper multiplying `blockIdx.x` by one of its template parameters. The
/// parameter takes the call's argument at its position, else its default; a value counts
/// when it is an integer literal or a `#define`d integer.
fn template_call_tiles(text: &str, body: &str, defines: &BTreeMap<String, u32>) -> Vec<u32> {
    let value = |v: &str| {
        v.trim()
            .parse::<u32>()
            .ok()
            .or_else(|| defines.get(v.trim()).copied())
    };
    let mut out = Vec::new();
    for (lt, _) in body.match_indices('<') {
        let name_start = body[..lt]
            .rfind(|c: char| !is_ident(c))
            .map_or(0, |p| p + 1);
        let callee = &body[name_start..lt];
        let Some(gt) = body[lt..].find('>').map(|p| lt + p) else {
            continue;
        };
        if callee.is_empty() || !body[gt + 1..].trim_start().starts_with('(') {
            continue;
        }
        let args: Vec<&str> = body[lt + 1..gt].split(',').map(str::trim).collect();
        let Some(def) = text.find(&format!("void {callee}(")) else {
            continue;
        };
        // 2026-10-07: The `template<...>` must directly precede the definition.
        let Some(tpl) = text[..def].rfind("template<") else {
            continue;
        };
        let Some(tpl_end) = text[tpl..def].find('>').map(|p| tpl + p) else {
            continue;
        };
        if text[tpl_end..def].contains([';', '}']) {
            continue;
        }
        let params: Vec<(&str, Option<&str>)> = text[tpl + "template<".len()..tpl_end]
            .split(',')
            .map(|p| {
                let (decl, default) = p.split_once('=').map_or((p, None), |(d, v)| (d, Some(v)));
                (decl.split_whitespace().last().unwrap_or(""), default)
            })
            .collect();
        let Some(helper) = block_at(text, def) else {
            continue;
        };
        for ident in block_x_factors(helper) {
            let Some(pos) = params.iter().position(|(name, _)| *name == ident) else {
                continue;
            };
            let v = args.get(pos).copied().or(params[pos].1);
            out.extend(v.and_then(value));
        }
    }
    out
}

/// 2026-10-07: The N tile(s) the entry's body uses: `blockIdx.x * TILE` in the entry's own
/// body, else in the body of each `*_impl` helper it calls, else through the template
/// parameters of each templated helper it calls (`template_call_tiles`). Only `#define`d
/// integers count, and template arguments that are integers or `#define`d integers.
/// Bodies are brace-matched: a helper defined between two entry points belongs to neither.
fn body_tiles(text: &str, entry: &str, defines: &BTreeMap<String, u32>) -> Vec<u32> {
    let start = entry_start(text, entry).expect("caller checked the entry exists");
    let body = block_at(text, start).unwrap_or("");
    let mut idents = block_x_factors(body);
    if idents.iter().all(|i| !defines.contains_key(i)) {
        for (i, _) in body.match_indices("_impl") {
            let name_start = body[..i].rfind(|c: char| !is_ident(c)).map_or(0, |p| p + 1);
            let callee = &body[name_start..i + "_impl".len()];
            let Some(def) = text.find(&format!("void {callee}(")) else {
                continue;
            };
            if let Some(helper) = block_at(text, def) {
                idents.extend(block_x_factors(helper));
            }
        }
    }
    let mut tiles: Vec<u32> = idents
        .iter()
        .filter_map(|i| defines.get(i).copied())
        .collect();
    if tiles.is_empty() {
        tiles = template_call_tiles(text, body, defines);
    }
    tiles.sort_unstable();
    tiles.dedup();
    tiles
}

/// 2026-09-29: The value of `extern "C" __device__ unsigned int <entry>_n_tile = X;`.
fn published_tile(text: &str, entry: &str, defines: &BTreeMap<String, u32>) -> Option<u32> {
    let decl = format!("unsigned int {entry}_n_tile =");
    let at = text.find(&decl)? + decl.len();
    let value = text[at..].split(';').next()?.trim();
    value
        .parse::<u32>()
        .ok()
        .or_else(|| defines.get(value).copied())
}

#[test]
fn every_tiled_entry_publishes_the_tile_its_body_uses() {
    let root = workspace_root();
    let mut checked = 0usize;
    let mut failures = Vec::new();
    for target in walk(&root).expect("the tree resolves") {
        let layout = discover(&root, &target).unwrap_or_else(|e| panic!("{target}: {e}"));
        for (_stem, entry) in layout.modules() {
            let Ok(text) = std::fs::read_to_string(&entry.source) else {
                continue;
            };
            let defines = int_defines(&text);
            for &name in TILED_ENTRIES {
                if entry_start(&text, name).is_none() {
                    continue;
                }
                checked += 1;
                let file = entry.source.display();
                let body = body_tiles(&text, name, &defines);
                match published_tile(&text, name, &defines) {
                    None => failures.push(format!("{target}: {file}: {name} publishes no N tile")),
                    Some(p) if body.is_empty() => failures.push(format!(
                        "{target}: {file}: {name} publishes {p} but no `blockIdx.x * TILE` was found"
                    )),
                    Some(p) if body.iter().any(|&b| b != p) => failures.push(format!(
                        "{target}: {file}: {name} publishes {p}, its body tiles N by {body:?}"
                    )),
                    Some(_) => {}
                }
            }
        }
    }
    assert!(checked > 100, "only {checked} (target, entry) pairs found");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// 2026-09-29: The scanner reads the tile through the entry's own `*_impl` helper, ignores a
/// helper defined after the entry, and sees a published value the body does not use.
#[test]
fn a_published_tile_that_differs_from_the_body_is_found() {
    let text = "#define N_TILE_LG 128\n#define N_TILE_SM 64\n\
        __device__ void k_impl(int n) { unsigned cta_n = blockIdx.x * N_TILE_LG; }\n\
        extern \"C\" __global__ void k(int n) { k_impl(n); }\n\
        __device__ void other_impl(int n) { unsigned cta_n = blockIdx.x * N_TILE_SM; }\n\
        extern \"C\" __device__ unsigned int k_n_tile = N_TILE_SM;\n";
    let defines = int_defines(text);
    assert_eq!(
        body_tiles(text, "k", &defines),
        vec![128],
        "only k's own helper counts"
    );
    assert_eq!(published_tile(text, "k", &defines), Some(64));
}

/// 2026-10-07: A templated body resolves its tile from the call's template argument, else
/// the parameter default; a published value that differs from either is found.
#[test]
fn a_templated_body_tiles_by_its_template_argument() {
    let text = "#define N_TILE_SM 64\n\
        template<int MRows = 64, int NCols = 64>\n\
        __device__ __forceinline__ void k_body(int n) { unsigned cta_n = blockIdx.x * NCols; }\n\
        extern \"C\" __global__ void k(int n) { k_body<64>(n); }\n\
        extern \"C\" __global__ void k32(int n) { k_body<16, 32>(n); }\n\
        extern \"C\" __global__ void ksm(int n) { k_body<16, N_TILE_SM>(n); }\n\
        extern \"C\" __device__ unsigned int k_n_tile = 32;\n";
    let defines = int_defines(text);
    assert_eq!(body_tiles(text, "k", &defines), vec![64], "the default");
    assert_eq!(
        body_tiles(text, "k32", &defines),
        vec![32],
        "a literal argument"
    );
    assert_eq!(
        body_tiles(text, "ksm", &defines),
        vec![64],
        "a #define argument"
    );
    // 2026-10-07: Known-bad control: k publishes 32 but its body tiles by 64.
    assert_eq!(published_tile(text, "k", &defines), Some(32));
}

/// 2026-09-29: An entry declared with `__launch_bounds__` between `__global__` and its name
/// is found, not skipped.
#[test]
fn an_entry_with_launch_bounds_is_found() {
    let text = "#define N_TILE_LG 128\n\
        extern \"C\" __global__\n__launch_bounds__(128, 3)\nvoid k(int n) {\n\
        unsigned cta_n = blockIdx.x * N_TILE_LG; }\n";
    let defines = int_defines(text);
    assert!(entry_start(text, "k").is_some());
    assert_eq!(body_tiles(text, "k", &defines), vec![128]);
}
