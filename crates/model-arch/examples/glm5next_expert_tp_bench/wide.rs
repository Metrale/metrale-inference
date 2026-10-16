// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `GLM_BENCH_WIDE=1`: the A/B of `METRALE_GLM_MOE_W4A4_WIDE_SWEEP`, `forward_moe`
//! at 8 to 32 rows on a tp 704 site under the serve's grouped W4A4 plan
//! (`METRALE_GLM_MOE_W4A4_GROUPED=1`).
//! 2026-10-15: One process, both call sites, the lever `1`: `forward_moe_at` at
//! `MoeCallSite::General` (17 rows and up run the grouped W4A4 GEMM, as every prefill launch
//! does whatever the lever says) and at `MoeCallSite::BatchedDecode` (up to 32 rows run the
//! wide union sweeps). Per routing, row count and draw of `x` it prints a `DIG` line, one
//! FNV-1a digest of each output row, after checking the two sites' outputs are the same bytes.
//! Then a `TIME` line: the mean over the draws of each site's CUDA-graph replay median and 10th
//! percentile, in us, all candidates interleaved per replay round.
//!
//! Two routings: `uniform` (the random router) and `skew` (a +8 selection bias on 12 experts,
//! so every row picks 8 of the 12 and each union expert carries about `rows * 2 / 3` rows: the
//! sweep's 3- and 4-tile items, which uniform routing at these widths almost never reaches).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: refuses to run without both levers (the wide rows need the grouped scratch);
//! exits with an error on the first byte difference between the call sites.

use metrale_model_arch::glm5next_mlp::forward::{
    MoeCallSite, check_wide_sweep_at_load, forward_moe_at, moe_w4a4_grouped, moe_w4a4_wide_sweep,
};

use super::*;

const ROWS: [usize; 6] = [8, 16, 17, 24, 25, 32];
const WIDE_ROWS: usize = 32;
/// 2026-10-14: Experts the `skew` routing favours, one every `EXPERTS / SKEW_POOL` ids.
const SKEW_POOL: usize = 12;
const SKEW_BIAS: f32 = 8.0;
const WIDTH: usize = 704;

pub fn run(g: &dyn GpuBackend, k: &Glm5NextMlpKernels, stream: u64) -> Result<()> {
    ensure!(
        moe_w4a4_grouped()? && moe_w4a4_wide_sweep()?,
        "GLM_BENCH_WIDE=1 needs METRALE_GLM_MOE_W4A4_GROUPED=1 and \
         METRALE_GLM_MOE_W4A4_WIDE_SWEEP=1"
    );
    check_wide_sweep_at_load(k)?;
    let mut rng = Rng(0x6a09_e667_f3bc_c908);
    let pool = [
        Pool::new(g, &mut rng)?,
        Pool::new(g, &mut rng)?,
        Pool::new(g, &mut rng)?,
    ];
    let router: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| rng.unit() * 0.05).collect();
    let shared: Vec<f32> = (0..HIDDEN * SHARED * 3)
        .map(|_| rng.unit() * 0.02)
        .collect();
    let xs: Vec<DevicePtr> = (0..env_or("GLM_BENCH_DRAWS", 3))
        .map(|_| {
            up_bf16(
                g,
                &(0..WIDE_ROWS * HIDDEN)
                    .map(|_| rng.unit() * 3.0)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Result<_>>()?;
    let out = g.alloc(WIDE_ROWS * HIDDEN * 2)?;
    let c = cfg(&WIDTH.to_string(), 0)?;
    let expert = |id: usize| -> Result<Glm5NextExpertWeights> {
        Ok(Glm5NextExpertWeights {
            gate_proj: pool[0].view(id, WIDTH, HIDDEN, GS_GATE_UP),
            up_proj: pool[1].view(id, WIDTH, HIDDEN, GS_GATE_UP),
            down_proj: pool[2].view(id, HIDDEN, WIDTH, GS_DOWN),
        })
    };
    let prec = |has| {
        GroupPrecision::resolve(
            MlpGroup::RoutedExperts,
            ActivationQuantization::default()
                .ladder(ProjFamily::Moe)
                .clone(),
            Nvfp4Act::A4,
            k.w4a4_expert_rows_grouped(),
            has,
        )
    };
    let mut sites = Vec::new();
    for (name, skew) in [("uniform", false), ("skew", true)] {
        let bias: Vec<f32> = (0..EXPERTS)
            .map(|e| {
                if skew && e % (EXPERTS / SKEW_POOL) == 0 {
                    SKEW_BIAS
                } else {
                    0.0
                }
            })
            .collect();
        let load = |n: &str| -> Result<Vec<f32>> {
            Ok(match n {
                "mlp.gate.weight" => router.clone(),
                "mlp.gate.e_score_correction_bias" => bias.clone(),
                _ => shared.clone(),
            })
        };
        sites.push(Site {
            name: name.to_string(),
            w: build_moe(g, &c, SHARED * 3, &load, &expert, &prec, WIDE_ROWS)?,
            ws: Glm5NextMlpWorkspace::new_with(g, &c, WIDE_ROWS, true)?,
            c,
        });
    }
    let out_b = g.alloc(WIDE_ROWS * HIDDEN * 2)?;
    let call = |s: &Site, x, out, rows, at, capturing| {
        forward_moe_at(
            g,
            k,
            &s.c,
            &s.w,
            x,
            out,
            rows,
            &[rows],
            &s.ws,
            at,
            capturing,
            stream,
        )
    };
    let sites_at = [
        ("general", MoeCallSite::General, out),
        ("decode", MoeCallSite::BatchedDecode, out_b),
    ];
    for s in &sites {
        for rows in ROWS {
            for (d, &x) in xs.iter().enumerate() {
                let mut got = Vec::new();
                for (_, at, o) in sites_at {
                    call(s, x, o, rows, at, false)?;
                    g.synchronize(stream)?;
                    got.push(read(g, o, rows * HIDDEN * 2)?);
                }
                ensure!(
                    got[0] == got[1],
                    "{} rows={rows} draw={d}: the batched decode site differs from the general one",
                    s.name
                );
                let digs: Vec<String> = got[0]
                    .chunks(HIDDEN * 2)
                    .map(|r| format!("{:016x}", fnv1a(r)))
                    .collect();
                println!("DIG {} rows={rows} draw={d} {}", s.name, digs.join(" "));
            }
            let mut fs: Vec<Box<dyn Fn() -> Result<()> + '_>> = Vec::new();
            for (_, at, o) in sites_at {
                for &x in &xs {
                    fs.push(Box::new(move || call(s, x, o, rows, at, true)));
                }
            }
            let refs: Vec<&dyn Fn() -> Result<()>> = fs.iter().map(|f| f.as_ref()).collect();
            let t = time_set(g, stream, &refs)?;
            let n = xs.len();
            let mean = |i: usize, j: usize| {
                t[i * n..(i + 1) * n].iter().map(|t| t[j]).sum::<f64>() / n as f64
            };
            println!(
                "TIME {} rows={rows} {} median_us {:.1} p10_us {:.1} | {} median_us {:.1} \
                 p10_us {:.1} | decode/general {:.3}",
                s.name,
                sites_at[0].0,
                mean(0, 0),
                mean(0, 1),
                sites_at[1].0,
                mean(1, 0),
                mean(1, 1),
                mean(1, 0) / mean(0, 0)
            );
        }
    }
    Ok(())
}
