// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The secrets denylist (owner decision 7; design §4.4). Two checks: a text check
//! every string that enters the bundle passes ([`secret_pattern`]), and a name check for source
//! files whose contents are never read ([`name_is_secret`]).
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - A match is reported by pattern *name* only; the matched text is never copied anywhere.
//! - The bundle builder applies [`secret_pattern`] to every payload string, text and id, so no
//!   adapter can bypass it.

/// 2026-10-23: `(name, needle)`: a field containing the needle is dropped.
const TEXT_PATTERNS: &[(&str, &str)] = &[
    ("openrouter-key", "sk-or-"),
    ("anthropic-key", "sk-ant-"),
    ("cloudflare-token", "cfat_"),
    ("github-token", "ghp_"),
    ("github-pat", "github_pat_"),
    ("github-oauth", "gho_"),
    ("pem-block", "-----BEGIN"),
    ("private-key", "PRIVATE KEY"),
];

/// 2026-10-23: The name of the first denylisted pattern `text` contains.
pub fn secret_pattern(text: &str) -> Option<&'static str> {
    TEXT_PATTERNS
        .iter()
        .find(|(_, needle)| text.contains(needle))
        .map(|(name, _)| *name)
}

/// 2026-10-23: The file-name segments (split on `_`, `-`, `.`) that mark a credential file:
/// its contents are never read (`reference_cloudflare_tokens_metrale.md`, `*.pat.txt`, `*.pem`).
const NAME_SEGMENTS: &[&str] = &[
    "token",
    "tokens",
    "credential",
    "credentials",
    "secret",
    "secrets",
    "ssh",
    "key",
    "keys",
    "pat",
    "pem",
    "apikey",
    "password",
    "passwords",
];

/// 2026-10-23: Whether the file name (the last path component) names a credential file.
pub fn name_is_secret(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    name.split(['_', '-', '.'])
        .any(|seg| NAME_SEGMENTS.contains(&seg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_denylisted_token_shape_is_caught_and_named() {
        let fake = [
            "sk-or-v1-000",
            "cfat_000",
            "ghp_000",
            "github_pat_000",
            "-----BEGIN RSA",
        ];
        for f in fake {
            assert!(
                secret_pattern(&format!("prefix {f} suffix")).is_some(),
                "{f}"
            );
        }
        assert_eq!(secret_pattern("key sk-or-v1-abc"), Some("openrouter-key"));
    }

    #[test]
    fn ordinary_campaign_text_is_not_a_secret() {
        for t in [
            "BF16 keys for DSA select",
            "skip-or-not",
            "the ghost path",
            "PEM-like",
        ] {
            assert_eq!(secret_pattern(t), None, "{t}");
        }
    }

    #[test]
    fn credential_file_names_are_recognised_by_segment() {
        for p in [
            "memory/reference_cloudflare_tokens_metrale.md",
            "/workspace/metrale/openrouter.embedder.token.txt",
            "gh.pat.txt",
            "certification.private-key.pem",
            "reference_strix_ssh.md",
        ] {
            assert!(name_is_secret(p), "{p}");
        }
        for p in [
            "feedback_no_mcp_this_machine.md",
            "project_tokenweave.md",
            "LEDGER.tsv",
        ] {
            assert!(!name_is_secret(p), "{p}");
        }
    }
}
