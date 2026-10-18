// A fixture for `cargo xtask env-ban`; never compiled.
use std::env as process;
use std::env::var_os as raw;
use metrale_gpu_sys::env::first_set;

pub fn reads() {
    let _ = process::var("A");
    let _ = process::args();
    let _ = metrale_gpu_sys::env::flag("B");
}
