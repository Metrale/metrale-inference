// SPDX-License-Identifier: MIT OR Apache-2.0
pub fn reduce() {
    nccl_all_reduce();
    nvmlInit();
}
