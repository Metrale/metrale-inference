// A crate-root build script may read the environment.
fn main() {
    let _ = std::env::var("OUT_DIR");
}
