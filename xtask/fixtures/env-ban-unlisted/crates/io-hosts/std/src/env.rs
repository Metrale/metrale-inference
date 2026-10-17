// The ProcessEnv host is the sanctioned reader.
pub fn var(name: &str) -> Option<String> {
    std::env::var(name).ok()
}
