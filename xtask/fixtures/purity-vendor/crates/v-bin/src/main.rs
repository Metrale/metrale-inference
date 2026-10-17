// SPDX-License-Identifier: MIT OR Apache-2.0
fn main() {
    let ctx = cudarc::driver::CudaContext::new(0);
    drop(ctx);
}
