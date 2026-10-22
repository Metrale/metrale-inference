// SPDX-License-Identifier: MIT OR Apache-2.0

//! A file CLI: `file c <in> <out> [profile] [mem_log2]` compresses, `file d <in> <out>`
//! decompresses, both printing size and throughput. `profile` is `auto`, `generic`, `ptx` or
//! `elf`.

use std::time::Instant;

fn profile(name: &str) -> Option<kzx::Profile> {
    match name {
        "generic" => Some(kzx::Profile::Generic),
        "ptx" => Some(kzx::Profile::Ptx),
        "elf" => Some(kzx::Profile::Elf),
        _ => None,
    }
}

fn mbps(bytes: usize, secs: f64) -> f64 {
    if secs > 0.0 { bytes as f64 / 1e6 / secs } else { 0.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, input, output, rest @ ..] = args.as_slice() else {
        eprintln!("usage: file c|d <in> <out> [profile] [mem_log2]");
        std::process::exit(2);
    };
    let data = std::fs::read(input).expect("read input");
    let t0 = Instant::now();
    let out = match mode.as_str() {
        "c" => {
            let p = rest.first().and_then(|s| profile(s));
            let m = rest.get(1).and_then(|s| s.parse::<u8>().ok());
            match (p, m) {
                (Some(p), Some(m)) => kzx::compress_with(&data, p, m).expect("parameter"),
                (Some(p), None) => kzx::compress_with(&data, p, kzx::default_mem_log2(data.len())).expect("parameter"),
                _ => kzx::compress(&data),
            }
        }
        "d" => {
            let limits = kzx::Limits { max_output: 1 << 32, max_memory: 4 << 30 };
            kzx::decompress(&data, &limits).expect("decompress")
        }
        _ => {
            eprintln!("mode must be c or d");
            std::process::exit(2);
        }
    };
    let secs = t0.elapsed().as_secs_f64();
    std::fs::write(output, &out).expect("write output");
    let (raw, packed) = if mode == "c" { (data.len(), out.len()) } else { (out.len(), data.len()) };
    let info = kzx::inspect(if mode == "c" { &out } else { &data }).expect("inspect");
    let names: Vec<&str> = info.blocks.iter().map(|b| b.profile.name()).collect();
    println!(
        "{mode} {raw} -> {packed} ratio {:.3} {:.2}s {:.2} MB/s codecs {:?} mem_log2 {:?}",
        raw as f64 / packed.max(1) as f64,
        secs,
        mbps(raw, secs),
        names,
        info.blocks.iter().map(|b| b.mem_log2).collect::<Vec<_>>()
    );
}
