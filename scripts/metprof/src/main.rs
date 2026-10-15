// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the `metprof` binary: argv in, the real adapters, the result
//! files a command asked for, stdout, stderr and the exit status out.

use std::io::Write;

use metprof::adapters::files::{FsFiles, write};
use metprof::adapters::nsys::NsysSource;
use metprof::cli::{Io, run};

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let out = run(
        &argv,
        &Io {
            traces: &NsysSource,
            files: &FsFiles,
        },
    );
    for (path, bytes) in &out.files {
        if let Err(e) = write(path, bytes) {
            eprintln!("metprof: error: cannot write {path}: {e}");
            std::process::exit(1);
        }
    }
    let written = std::io::stdout()
        .write_all(out.stdout.as_bytes())
        .and_then(|()| std::io::stdout().flush());
    if let Err(e) = written {
        eprintln!("metprof: error: cannot write the result: {e}");
        std::process::exit(1);
    }
    eprint!("{}", out.stderr);
    std::process::exit(out.code);
}
