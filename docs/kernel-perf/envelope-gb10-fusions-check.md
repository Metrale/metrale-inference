# FUSIONS row ranges against SCHEDULES (gb10)

Rules checked: 49. Disagreements: 72. Proposed row changes: 12.

| rule | kernels | rows | shapes | disagreements | proposal |
|---|---|---|---|---|---|
| w4a16_gemv_sw_1row | w4a16_gemv::w4a16_gemv_sw | [1, 1] | 3 | 0 | [1, 1] (unchanged) |
| w4a16_tc8_multi_seq | w4a16_gemv_tc::w4a16_gemv_tc8 | [2, 8] | 6 | 8 | none: the shapes disagree at rows [4, 8] (one launches the rule's kernel, another does not) |
| w4a16_tc8_verify_k4 | w4a16_gemv_tc::w4a16_gemv_tc8 | [4, 4] | 6 | 2 | none: the shapes disagree at rows [4, 8] (one launches the rule's kernel, another does not) |
| w4a16_tc8_verify_batch | w4a16_gemv_tc::w4a16_gemv_tc8 | [4, 8] | 6 | 2 | none: the shapes disagree at rows [4, 8] (one launches the rule's kernel, another does not) |
| w4a16_tc8_gate_up_km | w4a16_gemv_tc::w4a16_gemv_tc8 | [4, 8] | 6 | 2 | none: the shapes disagree at rows [4, 8] (one launches the rule's kernel, another does not) |
| w4a16_batch2_verify | w4a16_gemv::w4a16_gemv_batch2 | [2, 2] | 1 | 0 | [2, 2] (unchanged) |
| w4a16_batch2_ms_o_down | w4a16_gemv::w4a16_gemv_batch2 | [2, 2] | 1 | 0 | [2, 2] (unchanged) |
| w4a16_batch3_verify | w4a16_gemv::w4a16_gemv_batch3 | [3, 3] | 3 | 0 | [3, 3] (unchanged) |
| w8a16_m32 | w8a16_gemm_pipelined_m32::w8a16_gemm_pipelined_m32 | [1, 32] | 4 | 0 | none: the shapes disagree at rows [64, 64] (one launches the rule's kernel, another does not) |
| w8a16_m64 | w8a16_gemm_pipelined_m32::w8a16_gemm_pipelined_m64 | [33, 64] | 4 | 1 | none: the shapes disagree at rows [128, 128] (one launches the rule's kernel, another does not) |
| w8a16_m64_attn_chunked | w8a16_gemm_pipelined_m32::w8a16_gemm_pipelined_m64 | [65, 128] | 4 | 5 | none: the shapes disagree at rows [128, 128] (one launches the rule's kernel, another does not) |
| w8a16_full_gdn_wide | w8a16_gemm_pipelined::w8a16_gemm_pipelined | [65, 128] | 1 | 0 | [96, 128] (change) |
| moe_router_gemv | gemv::dense_gemv_bf16 | [1, 1] | 8 | 0 | [1, 1] (unchanged) |
| moe_grouped_router_rows | dense_gemv_bf16_batchm::dense_gemv_bf16_batchm | [2, 64] | 3 | 0 | [2, 8] (change) |
| moe_grouped_nvfp4_tc_wide | dense_gemv_bf16_batchm::dense_gemv_bf16_batchm, moe_fp8_grouped_blend::moe_weighted_sum_blend_fp8_grouped, moe_fp8_grouped_sort::moe_fp8_grouped_sort, moe_nvfp4_grouped_tc::moe_expert_down_act_nvfp4_grouped_tc, moe_nvfp4_grouped_tc::moe_expert_gate_up_act_nvfp4_grouped_tc, moe_topk::moe_topk_softmax_rows | [65, 256] | 0 | 0 | none: no swept shape routes the rule's kernels today |
| lm_head_nvfp4_w4a16_rows | w4a16_tc_rows::w4a16_tc_rows_64 | [1, 256] | 1 | 3 | none: the agreeing rows are not one range: [3, 3], [8, 8], [12, 128] |
| lm_head_bf16_gemv | gemv::dense_gemv_bf16 | [1, 1] | 8 | 0 | [1, 1] (unchanged) |
| lm_head_bf16_batchm | dense_gemv_bf16_batchm::dense_gemv_bf16_batchm | [2, 8] | 3 | 0 | [2, 8] (unchanged) |
| draft_lm_head_nvfp4 | w4a16_gemv::w4a16_gemv_sw | [1, 1] | 3 | 0 | [1, 1] (unchanged) |
| draft_dense_gemv_tc8 | dense_gemv_bf16_tc::dense_gemv_bf16_tc8 | [2, 8] | 5 | 8 | none: the agreeing rows are not one range: [2, 2], [8, 8] |
| draft_dense_gate_up_tc8 | dense_gemv_bf16_tc::dense_gemv_bf16_tc8 | [2, 8] | 5 | 8 | none: the agreeing rows are not one range: [2, 2], [8, 8] |
| draft_dense_gemv_tc16 | dense_gemv_bf16_tc::dense_gemv_bf16_tc16 | [9, 16] | 5 | 6 | none: the shapes disagree at rows [16, 16] (one launches the rule's kernel, another does not) |
| draft_dense_gate_up_tc16 | dense_gemv_bf16_tc::dense_gemv_bf16_tc16 | [9, 16] | 5 | 6 | none: the shapes disagree at rows [16, 16] (one launches the rule's kernel, another does not) |
| draft_dense_gemv_tc32 | dense_gemv_bf16_tc::dense_gemv_bf16_tc32 | [17, 32] | 5 | 5 | none: the shapes disagree at rows [16, 16] (one launches the rule's kernel, another does not) |
| draft_dense_gate_up_tc32 | dense_gemv_bf16_tc::dense_gemv_bf16_tc32 | [17, 32] | 5 | 5 | none: the shapes disagree at rows [16, 16] (one launches the rule's kernel, another does not) |
| draft_lm_head_tc8 | w4a16_gemv_tc::w4a16_gemv_tc8 | [2, 8] | 6 | 8 | none: the shapes disagree at rows [4, 8] (one launches the rule's kernel, another does not) |
| draft_lm_head_tc16 | w4a16_gemv_tc::w4a16_gemv_tc16 | [9, 16] | 1 | 1 | [16, 16] (change) |
| draft_dense_gemv_bf16 | gemv::dense_gemv_bf16 | [1, 1] | 8 | 0 | [1, 1] (unchanged) |
| draft_dense_gate_up_bf16 | gemv::dense_gemv_bf16 | [1, 1] | 8 | 0 | [1, 1] (unchanged) |
| w8a8_gemv_blk128_mb1_ku8 | w8a8_gemv::w8a8_gemv_blk128_mb1_ku8 | [1, 8] | 4 | 1 | none: the shapes disagree at rows [8, 8] (one launches the rule's kernel, another does not) |
| w8a8_gemv_blk128_mb2 | w8a8_gemv::w8a8_gemv_blk128_mb2 | [9, 16] | 4 | 1 | none: the shapes disagree at rows [12, 12] (one launches the rule's kernel, another does not) |
| w8a8_gemv_blk128_mb4 | w8a8_gemv::w8a8_gemv_blk128_mb4 | [17, 32] | 4 | 0 | [24, 32] (change) |
| w8a8_gemv_blk128_mb8 | w8a8_gemv::w8a8_gemv_blk128_mb8 | [33, 64] | 4 | 0 | none: the shapes disagree at rows [8, 8] (one launches the rule's kernel, another does not) |
| w8a8_gemv_blk128_mb16 | w8a8_gemv::w8a8_gemv_blk128_mb16 | [65, 128] | 4 | 0 | none: the shapes disagree at rows [12, 12] (one launches the rule's kernel, another does not) |
| w8a8_gemv_mb1_ku8 | w8a8_gemv::w8a8_gemv_rowscale_mb1_ku8 | [1, 8] | 5 | 0 | [1, 8] (unchanged) |
| w8a8_gemv_gate_up_mb1_ku8 | w8a8_gemv::w8a8_gemv_rowscale_mb1_ku8 | [1, 8] | 5 | 0 | [1, 8] (unchanged) |
| w8a8_gemv_mb2 | w8a8_gemv::w8a8_gemv_rowscale_mb2 | [9, 16] | 5 | 0 | [12, 16] (change) |
| w8a8_gemv_gate_up_mb2 | w8a8_gemv::w8a8_gemv_rowscale_mb2 | [9, 16] | 5 | 0 | [12, 16] (change) |
| w8a8_gemv_mb4 | w8a8_gemv::w8a8_gemv_rowscale_mb4 | [17, 32] | 5 | 0 | [24, 32] (change) |
| w8a8_gemv_gate_up_mb4 | w8a8_gemv::w8a8_gemv_rowscale_mb4 | [17, 32] | 5 | 0 | [24, 32] (change) |
| w8a8_gemv_mb8 | w8a8_gemv::w8a8_gemv_rowscale_mb8 | [33, 64] | 5 | 0 | [48, 64] (change) |
| w8a8_gemv_gate_up_mb8 | w8a8_gemv::w8a8_gemv_rowscale_mb8 | [33, 64] | 5 | 0 | [48, 64] (change) |
| w8a8_gemv_mb16 | w8a8_gemv::w8a8_gemv_rowscale_mb16 | [65, 128] | 5 | 0 | [96, 128] (change) |
| w8a8_gemv_gate_up_mb16 | w8a8_gemv::w8a8_gemv_rowscale_mb16 | [65, 128] | 5 | 0 | [96, 128] (change) |
| glm_bf16_gemv_kda | gemv::dense_gemv_bf16 | [1, 1] | 8 | 0 | [1, 1] (unchanged) |
| glm_bf16_gemv_dsa | gemv::dense_gemv_bf16 | [1, 1] | 8 | 0 | [1, 1] (unchanged) |
| glm_shared_gate_up_kda | gemv::dense_gemv_bf16 | [1, 1] | 8 | 0 | [1, 1] (unchanged) |
| glm_shared_gate_up_dsa | gemv::dense_gemv_bf16 | [1, 1] | 8 | 0 | [1, 1] (unchanged) |
| glm_mla_absorbed | gemv::dense_gemv_bf16, glm5next_dsa_mla_decode::glm5next_dsa_mla_decode_fp8 | [1, 1] | 8 | 0 | [1, 1] (unchanged) |

## w4a16_tc8_multi_seq

- linear nvfp4/g16 x bf16 k=5120 n=1024 rows [4, 8]: bit-identical faster sibling `w4a16_gemv_tc::w4a16_gemv_tc16` is the default choice
- linear nvfp4/g16 x bf16 k=5120 n=16384 rows [2, 2]: no served plan routes it (`new`)
- linear nvfp4/g16 x bf16 k=5120 n=16384 rows [3, 3]: a default build launches `w4a16_gemv::w4a16_gemv_batch3`
- linear nvfp4/g16 x bf16 k=6144 n=5120 rows [2, 2]: no served plan routes it (`new`)
- linear nvfp4/g16 x bf16 k=6144 n=5120 rows [3, 3]: a default build launches `w4a16_gemv::w4a16_gemv_batch3`
- linear nvfp4/g16 x bf16 k=17408 n=5120 rows [2, 2]: a default build launches `w4a16_gemv::w4a16_gemv_batch2`
- linear nvfp4/g16 x bf16 k=17408 n=5120 rows [3, 3]: a default build launches `w4a16_gemv::w4a16_gemv_batch3`
- lm_head nvfp4/g16 x bf16 k=5120 n=248320 rows [3, 4]: no served plan routes it (`new`)

## w4a16_tc8_verify_k4

- linear nvfp4/g16 x bf16 k=5120 n=1024 rows [4, 4]: bit-identical faster sibling `w4a16_gemv_tc::w4a16_gemv_tc16` is the default choice
- lm_head nvfp4/g16 x bf16 k=5120 n=248320 rows [4, 4]: no served plan routes it (`new`)

## w4a16_tc8_verify_batch

- linear nvfp4/g16 x bf16 k=5120 n=1024 rows [4, 8]: bit-identical faster sibling `w4a16_gemv_tc::w4a16_gemv_tc16` is the default choice
- lm_head nvfp4/g16 x bf16 k=5120 n=248320 rows [4, 4]: no served plan routes it (`new`)

## w4a16_tc8_gate_up_km

- linear nvfp4/g16 x bf16 k=5120 n=1024 rows [4, 8]: bit-identical faster sibling `w4a16_gemv_tc::w4a16_gemv_tc16` is the default choice
- lm_head nvfp4/g16 x bf16 k=5120 n=248320 rows [4, 4]: no served plan routes it (`new`)

## w8a16_m64

- linear fp8/block128x128 x bf16 k=2048 n=512 rows [64, 64]: bit-identical faster sibling `w8a16_gemm_pipelined_m32::w8a16_gemm_pipelined_m32` is the default choice

## w8a16_m64_attn_chunked

- linear fp8/block128x128 x bf16 k=2048 n=512 rows [65, 96]: bit-identical faster sibling `w8a16_gemm_pipelined_m32::w8a16_gemm_pipelined_m32` is the default choice
- linear fp8/block128x128 x bf16 k=2048 n=8192 rows [96, 128]: bit-identical faster sibling `w8a16_gemm_pipelined::w8a16_gemm_pipelined` is the default choice
- linear fp8/block128x128 x bf16 k=2048 n=12288 rows [96, 128]: a default build launches `w8a16_gemm_pipelined::w8a16_gemm_pipelined`
- linear fp8/block128x128 x bf16 k=4096 n=2048 rows [96, 96]: no served plan routes it (`new`)
- linear fp8/block128x128 x bf16 k=4096 n=2048 rows [128, 128]: no served plan routes it (`new`)

## lm_head_nvfp4_w4a16_rows

- lm_head nvfp4/g16 x bf16 k=2048 n=248320 rows [1, 1]: no served plan routes it (`new`)
- lm_head nvfp4/g16 x bf16 k=2048 n=248320 rows [2, 2]: bit-identical faster sibling `w4a16_gemv_tc::w4a16_gemv_tc8` is the default choice
- lm_head nvfp4/g16 x bf16 k=2048 n=248320 rows [4, 4]: bit-identical faster sibling `w4a16_gemv_tc::w4a16_gemv_tc16` is the default choice

## draft_dense_gemv_tc8

- linear bf16 x bf16 k=5120 n=1024 rows [3, 3]: no served plan routes it (`new`)
- linear bf16 x bf16 k=5120 n=1024 rows [4, 4]: no served plan routes it (`new`)
- linear bf16 x bf16 k=5120 n=12288 rows [3, 4]: no served plan routes it (`new`)
- linear bf16 x bf16 k=6144 n=5120 rows [3, 3]: no served plan routes it (`new`)
- linear bf16 x bf16 k=6144 n=5120 rows [4, 4]: no served plan routes it (`new`)
- linear bf16 x bf16 k=10240 n=5120 rows [3, 4]: no served plan routes it (`new`)
- linear bf16 x bf16 k=17408 n=5120 rows [3, 3]: no served plan routes it (`new`)
- linear bf16 x bf16 k=17408 n=5120 rows [4, 4]: no served plan routes it (`new`)

## draft_dense_gate_up_tc8

- linear bf16 x bf16 k=5120 n=1024 rows [3, 3]: no served plan routes it (`new`)
- linear bf16 x bf16 k=5120 n=1024 rows [4, 4]: no served plan routes it (`new`)
- linear bf16 x bf16 k=5120 n=12288 rows [3, 4]: no served plan routes it (`new`)
- linear bf16 x bf16 k=6144 n=5120 rows [3, 3]: no served plan routes it (`new`)
- linear bf16 x bf16 k=6144 n=5120 rows [4, 4]: no served plan routes it (`new`)
- linear bf16 x bf16 k=10240 n=5120 rows [3, 4]: no served plan routes it (`new`)
- linear bf16 x bf16 k=17408 n=5120 rows [3, 3]: no served plan routes it (`new`)
- linear bf16 x bf16 k=17408 n=5120 rows [4, 4]: no served plan routes it (`new`)

## draft_dense_gemv_tc16

- linear bf16 x bf16 k=5120 n=1024 rows [12, 12]: no served plan routes it (`new`)
- linear bf16 x bf16 k=5120 n=1024 rows [16, 16]: bit-identical faster sibling `dense_gemv_bf16_tc::dense_gemv_bf16_tc32` is the default choice
- linear bf16 x bf16 k=5120 n=12288 rows [12, 12]: no served plan routes it (`new`)
- linear bf16 x bf16 k=6144 n=5120 rows [12, 12]: no served plan routes it (`new`)
- linear bf16 x bf16 k=10240 n=5120 rows [12, 12]: no served plan routes it (`new`)
- linear bf16 x bf16 k=17408 n=5120 rows [12, 12]: no served plan routes it (`new`)

## draft_dense_gate_up_tc16

- linear bf16 x bf16 k=5120 n=1024 rows [12, 12]: no served plan routes it (`new`)
- linear bf16 x bf16 k=5120 n=1024 rows [16, 16]: bit-identical faster sibling `dense_gemv_bf16_tc::dense_gemv_bf16_tc32` is the default choice
- linear bf16 x bf16 k=5120 n=12288 rows [12, 12]: no served plan routes it (`new`)
- linear bf16 x bf16 k=6144 n=5120 rows [12, 12]: no served plan routes it (`new`)
- linear bf16 x bf16 k=10240 n=5120 rows [12, 12]: no served plan routes it (`new`)
- linear bf16 x bf16 k=17408 n=5120 rows [12, 12]: no served plan routes it (`new`)

## draft_dense_gemv_tc32

- linear bf16 x bf16 k=5120 n=1024 rows [24, 24]: no served plan routes it (`new`)
- linear bf16 x bf16 k=5120 n=12288 rows [24, 24]: no served plan routes it (`new`)
- linear bf16 x bf16 k=6144 n=5120 rows [24, 24]: no served plan routes it (`new`)
- linear bf16 x bf16 k=10240 n=5120 rows [24, 24]: no served plan routes it (`new`)
- linear bf16 x bf16 k=17408 n=5120 rows [24, 24]: no served plan routes it (`new`)

## draft_dense_gate_up_tc32

- linear bf16 x bf16 k=5120 n=1024 rows [24, 24]: no served plan routes it (`new`)
- linear bf16 x bf16 k=5120 n=12288 rows [24, 24]: no served plan routes it (`new`)
- linear bf16 x bf16 k=6144 n=5120 rows [24, 24]: no served plan routes it (`new`)
- linear bf16 x bf16 k=10240 n=5120 rows [24, 24]: no served plan routes it (`new`)
- linear bf16 x bf16 k=17408 n=5120 rows [24, 24]: no served plan routes it (`new`)

## draft_lm_head_tc8

- linear nvfp4/g16 x bf16 k=5120 n=1024 rows [4, 8]: bit-identical faster sibling `w4a16_gemv_tc::w4a16_gemv_tc16` is the default choice
- linear nvfp4/g16 x bf16 k=5120 n=16384 rows [2, 2]: no served plan routes it (`new`)
- linear nvfp4/g16 x bf16 k=5120 n=16384 rows [3, 3]: a default build launches `w4a16_gemv::w4a16_gemv_batch3`
- linear nvfp4/g16 x bf16 k=6144 n=5120 rows [2, 2]: no served plan routes it (`new`)
- linear nvfp4/g16 x bf16 k=6144 n=5120 rows [3, 3]: a default build launches `w4a16_gemv::w4a16_gemv_batch3`
- linear nvfp4/g16 x bf16 k=17408 n=5120 rows [2, 2]: a default build launches `w4a16_gemv::w4a16_gemv_batch2`
- linear nvfp4/g16 x bf16 k=17408 n=5120 rows [3, 3]: a default build launches `w4a16_gemv::w4a16_gemv_batch3`
- lm_head nvfp4/g16 x bf16 k=5120 n=248320 rows [3, 4]: no served plan routes it (`new`)

## draft_lm_head_tc16

- lm_head nvfp4/g16 x bf16 k=5120 n=248320 rows [12, 12]: no served plan routes it (`new`)

## w8a8_gemv_blk128_mb1_ku8

- linear fp8/block128x128 x fp8/g128 k=4096 n=2048 rows [8, 8]: bit-identical faster sibling `w8a8_gemv::w8a8_gemv_blk128_mb8` is the default choice

## w8a8_gemv_blk128_mb2

- linear fp8/block128x128 x fp8/g128 k=2048 n=12288 rows [12, 12]: bit-identical faster sibling `w8a8_gemv::w8a8_gemv_blk128_mb16` is the default choice
