# Laguna dense BF16 small-row experiment

The default-off `METRALE_DENSE_GEMV_SMALL_ROWS=1` switch selects the capacity-4
member of the existing dense BF16 batch family for eligible groups of one to
four rows. Larger row groups retain the original capacity-16 entry. Split
launches select by rows per block, after existing admission checks. Kernel
handles are cached per backend, not globally across contexts. Missing opted-in
kernels return an error. No model default or supported-hardware claim changes.

Related checkpoint and measured serving workload:
[poolside/Laguna-XS-2.1-NVFP4 at d32afde](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4/tree/d32afde8b09af1539b49ff96ff5551c674485f8e).

The prototype (an overlay, not this tree) used source base `020057a974f1da08a8bef4fd8a1974edf524b0dd`
plus a recorded three-file overlay. It reduced total request latency on the
fixed-count decode workload by 6.66% / 6.69% at C2 and 4.67% / 6.91% at C4 in
opposite-order comparisons. C1 improved 1.18% / 1.57%; prefill remained within
approximately ±0.4%. These are whole-request measurements, not isolated decode
kernel throughput. Four fresh servers ran A/B/B2/A2; all 144 cohorts passed
input/output-count admission. Loading and profiling were excluded.

Prototype executable SHA-256:
`de25b10a089ff8ac48a522c69d0394923bf456c8bee1bbf819e2fcb2409672d0`.
Baseline: `13732afc36bec64b878157c2bf2f5608a1913bf086901d9e0954dc1eac45b989`.
The first prototype startup refused an unregistered option; its failure is
retained, and registration preceded the successful rebuild.

Prototype correctness evidence covers 63 constructed shape/split/scalar cases,
27 legacy family cases, geometry refusals and an independent integer oracle.
Live checks passed six structured outputs, four tool/stream regressions and six
client-C4 structured responses. Bounded isolated coding remains 9/12, with the
same three retry-task failures. No broad quality pass is claimed.

The public refactor shares one arithmetic body between both capacities and adds
fresh-process host selection/refusal tests. The public compiled family passes
legacy instruction comparison and compilation
for GB10, B200, B300 and Hopper. The actual GB10 library passes the same
63 shape/split/scalar and 27 legacy-family controls. The shared
source reaches GB10, B200, B300 and Hopper; existing entries and defaults must
remain unchanged. Combined prefill-plus-dense improvements require a separate
measurement. Full model and benchmark certification remain open.

The first public compile gate refused the refactor even though numerical tests
passed: missing exported pointer `__restrict__` qualifiers changed two legacy
`ld.global.nc` instructions to `ld.global`. Restoring the original aliasing
contract made all four strict comparisons pass. The failed attempt and source
identities are retained. Only compiler-generated labels/shared symbols,
comments and whitespace are normalized; instructions are not discarded.

## Public-source serving gates

The public M4 and M16 overlays were frozen over source
`5b838b040e6cc8b88b5d02f565e84e0e17502595`. The resulting executable is
`5784c0fe1064b2dd7ab55dca39823d7ec3fef0c1ea205f6943edc783b63451b5`.
Every staged source hash was rechecked after the build, and the complete dense
and expert PTX byte strings were verified inside that executable. A later
comment-date-only change does not alter the tested runtime.

Dense-only and combined configurations each passed six structured outputs, four
tool/stream checks and six client-C4 structured responses. All twelve generated
coding candidates are byte-identical between these two configurations; isolated
semantic grading remains 9/12, including all three retry-task failures. These
are bounded regression gates, not broad coding qualification. The separate public-source timing campaign passed all 144 fixed-count cohorts
across fresh servers A/B/B2/A2. On 64-input/64-output requests, median total
latency improved 6.99% / 7.67% at C2 and 7.00% / 4.72% at C4 in opposite orders.
C1 improved 0.88% / 1.34%. Prefill observations range from -0.42% to +1.26%;
no prefill improvement is claimed. These unprofiled whole-request results
remain distinct from isolated steady-state throughput and the prototype
measurements above. The separate combined-option gate also passed all 144 cohorts. Short-prefill
latency improved 9.70–10.07%, long-prefill 4.05–4.43%, and 64-output total
latency improved 8.85% / 9.60% at C2 and 8.59% / 8.53% at C4 in opposite
orders (C1: 2.98% / 3.24%). These compare both flags to the frozen baseline,
not additive sums of the isolated improvements. Both flags remain default-off.
