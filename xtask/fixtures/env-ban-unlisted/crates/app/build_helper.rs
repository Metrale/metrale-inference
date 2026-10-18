// A build-script helper beside the manifest.
pub fn profile() -> Option<String> {
    std::env::var("PROFILE").ok()
}
