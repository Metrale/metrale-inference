// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: how values are quoted inside error messages and sort keys, so a
//! message names exactly the offending value: strings in single quotes (double
//! when the string holds a single quote), lists in brackets.

pub fn quote(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(q);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

pub fn quote_list<S: AsRef<str>>(items: &[S]) -> String {
    let inner: Vec<String> = items.iter().map(|s| quote(s.as_ref())).collect();
    format!("[{}]", inner.join(", "))
}

/// A TOML value as it would be written back into a message.
pub fn toml_value(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) => quote(s),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Float(f) => float(*f),
        toml::Value::Boolean(b) => if *b { "True" } else { "False" }.to_string(),
        toml::Value::Datetime(d) => d.to_string(),
        toml::Value::Array(a) => format!(
            "[{}]",
            a.iter().map(toml_value).collect::<Vec<_>>().join(", ")
        ),
        toml::Value::Table(t) => {
            let inner: Vec<String> = t
                .iter()
                .map(|(k, v)| format!("{}: {}", quote(k), toml_value(v)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

/// Shortest round-trip float text with a decimal point kept (`1.0`, `0.25`).
pub fn float(f: f64) -> String {
    if f.is_finite() && f == f.trunc() && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoting_matches_the_message_convention() {
        assert_eq!(quote("gemm"), "'gemm'");
        assert_eq!(quote("it's"), "\"it's\"");
        assert_eq!(quote_list(&["a", "b"]), "['a', 'b']");
        assert_eq!(quote_list::<&str>(&[]), "[]");
        assert_eq!(float(1.0), "1.0");
        assert_eq!(float(0.25), "0.25");
        assert_eq!(toml_value(&toml::Value::Boolean(true)), "True");
    }
}
