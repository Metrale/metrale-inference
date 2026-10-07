// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: Compile with -I kernels/gb10/common, --fmad=false, -shared -Xcompiler -fPIC.
#include "argmax_feed.cu"
extern "C" void run(const __nv_bfloat16*x,const unsigned*m,unsigned*out,unsigned v,unsigned rows,unsigned stride){argmax_bf16_batch_masked_host<<<rows,256>>>(x,m,out,v,stride);}
