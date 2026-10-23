// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A small in-memory source tree (the in-memory "port": the derive takes an
//! already-read `SourceSet`), shaped like the real campaign files.

#![allow(dead_code)]

use metrale_kg_derive::{Config, SourceSet};

pub const CONFIG: &str = r#"
schema = "kg-config/v1"
[sources]
root = "/r"
principles = ["CLAUDE.md", "decoupling.md"]
designs = ["bp/DESIGN-x.md"]
graph_theory_map = "bp/GTM.md"
mathbomb = "bp/MB.md"
memory_dir = "mem"
lkb_repo = "repo"
[[sources.campaigns]]
id = "glm/mk"
ledger = "c/LEDGER.tsv"
tooling = "c/TOOLING.md"
charter = "c/CHARTER.md"
ledger_check = "c/tools/ledger.py"
[embedder]
mode = "unavailable"
reason = "test"
[limits]
max_rows = 100
cypher_timeout_ms = 1000
token_budget = 1200
[store]
data_dir = "/tmp/none"
scratch_dir = "/tmp/none-scratch"
[validator]
python = "/usr/bin/python3"
[search]
w_cosine = 0.6
w_hops = 0.25
w_kind = 0.15
"#;

pub const LEDGER: &str = "ts\tphase\tid\thypothesis\tpredicted_us_saved\tmeasured_delta\tbitwise\tverdict\tprofile\tcost\tcommit\twhy_differed\tdepends_on\tretracts
2026-10-23T01:00Z\tP0\tp0-x\tplan it (ROTI)\t0 us/step\tnot measured\tn/a\tBLOCKED\t-\t1 h\t-\tblocked on decision:2026-10-22/p0-split, see DESIGN-x.md §6\t-\t-
2026-10-23T02:00Z\tP1\tl2-pf\tMETRALE_GLM_AR_L2_PREFETCH=6 warms the next layer's weights during the all-reduce\t<=315 us/step\t1k -0.53% [-2.72,+1.65]; 16k -0.22% [-2.32,+1.88]\tIDENTICAL\tNULL (no win)\tglm53-tp3-512k-promo1 + env\t2 h\tabc1234\tthe read is at 89% of floor\tp0-x\t-
2026-10-23T03:00Z\tP3\tseg\tdense segment\t<=100 us\t1k -1.7% [-2.0,-1.4]\tIDENTICAL\tLOSS\tmkp3\t3 h\tec5326efb\tpersistent grid starves the side stream\tl2-pf\t-
";

pub const TOOLING: &str =
    "# TOOLING frictions (phase | friction | cost_min | proposed_change | done)

| phase | friction | cost_min | proposed_change | done |
|---|---|---:|---|---|
| P0 | BENCH_ACTIVE blocks cargo (SBIO) | 30 | `tools/cargo-guard.sh` | yes (a708b140a) |
| P1 | pkill -f matched my shell | 5 | kill by PID: `ps -eo pid,args | grep [p]attern` | no |
2026-10-23\ta tab-separated line that is not a table row
";

pub const CHARTER: &str = "# Charter

Owner decisions (2026-10-22, owner answered):
- P0 first (GLM decode onto the circuit executor).
- Kernel generation: CUDA templates are the default,
  enforced by a gate.

## Re-plan 2026-10-22 (main session): P0 split
P0 needs prerequisites; p0-x stays BLOCKED.

## Decisions 2026-10-23
- Allowlist growth approved.
";

pub const DESIGN: &str = "# Design X

Written 2026-10-23.

## 6. Phases
text

```bash
# not a heading
```

## Owner decisions (2026-10-23): accepted
1. A new server.
2. The embedder is OpenRouter.
   - model from config
";

pub const CLAUDE: &str = "| Principle | Rule | Violation Response |
|---|---|---|
| **SSOT** | Every data item has one source | Derive |
| **ROTI** | Resolve Once, Then Immutable | x |
";

pub const DECOUPLING: &str = "## ROTI: Resolve Once, Then Immutable
**Rule.** Every decision input is resolved once.
**Enforcement.** An env-ban gate.
";

pub const GTM: &str = "# map

## Ranked experiments (all CPU-only unless noted)

| # | Experiment | Theory it rests on | Lands as | Gate / acceptance | Effort |
|---|---|---|---|---|---|
| 3 | **Critical-pair report** | Newman 2109.06049 | `met circuit rules --critical-pairs` | zero overlaps | 1 d |

## Deferred, with the reason

- **E-graph / equality saturation** (egglog). Wait for #3.

## Sources

- **Rewriting theory:** BSZ III 2109.06049
";

pub const MATHBOMB: &str = "# mb

## Results relevant to compute graphs

| # | Result | Lean | Relevance to us | Verdict |
|---|---|---|---|---|
| 124 | P3 scheduling | yes | DAG on 3 machines | maybe |

## Process ideas worth copying (stronger than any single theorem)

1. **Comparator statements.** Each claim ships as a statement.
";

pub const MEM_FEEDBACK: &str = "---
name: feedback_tests_prove_the_bug
description: \"★5 POSITIVE + NEGATIVE tests, proven RED by sabotage\"
metadata:
  type: feedback
  modified: 2026-08-10T00:00:00Z
---

Body text with [[project_dense_seg]] and [[missing_one]].
";

pub const MEM_PROJECT: &str = "---
name: project_dense_seg
description: \"dense segment LOSS: persistent grid starves the side stream\"
metadata:
  type: project
---

See [[feedback_tests_prove_the_bug]].
";

pub fn config() -> Config {
    Config::parse(CONFIG).unwrap()
}

/// 2026-10-23: The files, added in `order` (a permutation of the indices) to prove the derive
/// does not depend on read order.
pub fn sources_in(order: &[usize]) -> SourceSet {
    let files: [(&str, &str); 12] = [
        ("c/LEDGER.tsv", LEDGER),
        ("c/TOOLING.md", TOOLING),
        ("c/CHARTER.md", CHARTER),
        ("bp/DESIGN-x.md", DESIGN),
        ("CLAUDE.md", CLAUDE),
        ("decoupling.md", DECOUPLING),
        ("bp/GTM.md", GTM),
        ("bp/MB.md", MATHBOMB),
        ("mem/feedback_tests_prove_the_bug.md", MEM_FEEDBACK),
        ("mem/project_dense_seg.md", MEM_PROJECT),
        (
            "mem/reference_cloudflare_tokens_metrale.md",
            "---\nname: x\ndescription: cfat_SECRETVALUE\n---\n",
        ),
        (
            "mem/MEMORY.md",
            "# index\n- [dense](project_dense_seg.md)\n",
        ),
    ];
    let mut s = SourceSet::default();
    for &i in order {
        let (p, t) = files[i];
        s.add(p, t.to_string());
    }
    s
}

pub fn sources() -> SourceSet {
    sources_in(&(0..12).collect::<Vec<_>>())
}
