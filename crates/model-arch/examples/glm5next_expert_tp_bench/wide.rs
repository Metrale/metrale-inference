// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `GLM_BENCH_WIDE=1`: the A/B of `METRALE_GLM_MOE_W4A4_WIDE_SWEEP`, `forward_moe`
//! at 8 to 32 rows on a tp 704 site under the serve's grouped W4A4 plan
//! (`METRALE_GLM_MOE_W4A4_GROUPED=1`). Run it twice on the same seeds, in separate processes,
//! with the lever `0` (17 rows and up run the grouped W4A4 GEMM) and `1` (up to 32 rows run the
//! wide union sweeps). Per routing, row count and draw of `x` it prints a `DIG` line, one FNV-1a
//! digest of each output row: the two runs' `DIG` lines must be identical. Then a `TIME` line:
//! the mean over the draws of the CUDA-graph replay median and 10th percentile, in us.
//!
//! Two routings: `uniform` (the random router) and `skew` (a +8 selection bias on 12 experts,
//! so every row picks 8 of the 12 and each union expert carries about `rows * 2 / 3` rows: the
//! sweep's 3- and 4-tile items, which uniform routing at these widths almost never reaches).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: refuses to run without the grouped lever (the wide rows need its scratch).

use metrale_model_arch::glm5next_mlp::forward::{
    check_wide_sweep_at_load, moe_w4a4_grouped, moe_w4a4_wide_sweep,
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
        moe_w4a4_grouped()?,
        "GLM_BENCH_WIDE=1 needs METRALE_GLM_MOE_W4A4_GROUPED=1"
    );
    check_wide_sweep_at_load(k)?;
    println!(
        "wide bench: METRALE_GLM_MOE_W4A4_WIDE_SWEEP={}",
        u8::from(moe_w4a4_wide_sweep()?)
    );
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
    for s in &sites {
        for rows in ROWS {
            for (d, &x) in xs.iter().enumerate() {
                run_site(g, k, s, x, out, rows, false, stream)?;
                g.synchronize(stream)?;
                let b = read(g, out, rows * HIDDEN * 2)?;
                let digs: Vec<String> = b
                    .chunks(HIDDEN * 2)
                    .map(|r| format!("{:016x}", fnv1a(r)))
                    .collect();
                println!("DIG {} rows={rows} draw={d} {}", s.name, digs.join(" "));
            }
            let fs: Vec<Box<dyn Fn() -> Result<()> + '_>> = xs
                .iter()
                .map(|&x| {
                    Box::new(move || run_site(g, k, s, x, out, rows, true, stream))
                        as Box<dyn Fn() -> Result<()>>
                })
                .collect();
            let refs: Vec<&dyn Fn() -> Result<()>> = fs.iter().map(|f| f.as_ref()).collect();
            let t = time_set(g, stream, &refs)?;
            let mean = |j: usize| t.iter().map(|t| t[j]).sum::<f64>() / t.len() as f64;
            println!(
                "TIME {} rows={rows} median_us {:.1} p10_us {:.1}",
                s.name,
                mean(0),
                mean(1)
            );
        }
    }
    Ok(())
}
