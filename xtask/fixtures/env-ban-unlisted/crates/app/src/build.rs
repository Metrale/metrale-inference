// A module named build.rs inside src/ is not a build script.

pub fn not_a_build_script() {
    let _ = std::env::var("A");
}
