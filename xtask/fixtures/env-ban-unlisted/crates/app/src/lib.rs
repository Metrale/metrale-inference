// A fixture for `cargo xtask env-ban`; never compiled.
// std::env::var("COMMENT") is a comment.
/* env::var /* nested */ env::var */
use std::env;

pub fn reads<'a>(x: &'a str) -> &'a str {
    let _ = std::env::var("A");
    let _ = env :: var_os("B");
    let q = '"'; let _ = env::vars();
    let _ = "std::env::var(\"STRING\")";
    let _ = r#"env::var("RAW")"#;
    let _ = (Env::var, my_env::var, env!("CARGO"));
    let _ = metrale_config::levers::var("C");
    let _ = self.env.var("D");
    for _ in std::env::vars_os() {}
    x
}
