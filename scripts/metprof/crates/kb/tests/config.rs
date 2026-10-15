// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `embedding.toml` (ADDENDUM-EMBEDDINGS E2) and the reranker it
//! selects. Every key below is a synthetic reference, never a key.

use metprof_kb::config::{
    EmbedderConfig, Lowered, OPENROUTER_BASE_URL, RerankerConfig, SecretRef, parse,
};
use metprof_kb::rerank::{Candidate, select};

const OPENROUTER: &str = r#"
[embedder]
kind = "openrouter"
model = "vendor/embed-a:free"
api_key = { file = "keys/openrouter.txt" }
dims = 2048
precision = 6
provider = { data_collection = "deny", order = ["a", "b"] }

[reranker]
kind = "none"
"#;

const LOCAL: &str = r#"
[embedder]
kind = "openai_compatible"
endpoint = "http://127.0.0.1:8890/v1"
model = "local-embed"
dims = 1024
precision = 6

[reranker]
kind = "openai_compatible"
endpoint = "https://Rerank.Example.test/v1"
model = "local-rerank"
api_key = { env = "KKB_RERANK_KEY" }
"#;

fn err(text: &str) -> String {
    parse(text).expect_err("must be refused").0
}

#[test]
fn a_literal_key_is_refused_and_never_echoed() {
    let literal = "sk-or-v1-0123456789abcdef";
    for (table, needle) in [
        ("[embedder]", "api_key = { file = \"keys/openrouter.txt\" }"),
        ("[reranker]", "kind = \"none\""),
    ] {
        let text = if table == "[embedder]" {
            OPENROUTER.replace(needle, &format!("api_key = \"{literal}\""))
        } else {
            LOCAL.replace(
                "api_key = { env = \"KKB_RERANK_KEY\" }",
                &format!("api_key = \"{literal}\""),
            )
        };
        assert!(text.contains(literal), "anchor for {table} missing");
        let msg = err(&text);
        assert!(msg.contains("literal key is refused"), "{msg}");
        assert!(
            !msg.contains(literal) && !msg.contains("0123456789"),
            "{msg}"
        );
    }
    // A table that is not exactly one env/file reference is refused too.
    for bad in [
        "{ value = \"x\" }",
        "{ env = \"A\", file = \"B\" }",
        "{ env = \"\" }",
        "{ env = 7 }",
    ] {
        let text = OPENROUTER.replace("{ file = \"keys/openrouter.txt\" }", bad);
        assert!(text.contains(bad));
        assert!(err(&text).contains("literal key is refused"), "{bad}");
    }
}

#[test]
fn a_missing_required_field_or_table_fails_fast() {
    let cases = [
        (
            OPENROUTER.replace("api_key = { file = \"keys/openrouter.txt\" }\n", ""),
            "requires `api_key`",
        ),
        (OPENROUTER.replace("dims = 2048\n", ""), "requires `dims`"),
        (
            LOCAL.replace("endpoint = \"http://127.0.0.1:8890/v1\"\n", ""),
            "requires `endpoint`",
        ),
        (LOCAL.replace("precision = 6\n", ""), "requires `precision`"),
        (
            OPENROUTER.replace("[reranker]\nkind = \"none\"\n", ""),
            "reranker",
        ),
        (OPENROUTER.replace("kind = \"none\"\n", ""), "kind"),
        (String::new(), "embedder"),
    ];
    for (text, needle) in cases {
        let msg = err(&text);
        assert!(msg.contains(needle), "want {needle:?} in {msg}");
    }
    // `err` panics on a file that parses, so a no-op replace (a missing
    // anchor) cannot pass; the controls themselves parse.
    assert!(parse(OPENROUTER).is_ok() && parse(LOCAL).is_ok());
}

#[test]
fn a_field_the_kind_does_not_take_or_an_unknown_kind_is_refused() {
    let hashed_with_model =
        "[embedder]\nkind = \"hashed\"\nmodel = \"m\"\n[reranker]\nkind = \"none\"\n";
    assert!(err(hashed_with_model).contains("does not take `model`"));
    let provider_on_local =
        LOCAL.replace("precision = 6\n", "precision = 6\nprovider = { x = 1 }\n");
    assert!(err(&provider_on_local).contains("does not take `provider`"));
    assert!(
        err(&LOCAL.replace(
            "\"openai_compatible\"\nendpoint = \"http",
            "\"cohere\"\nendpoint = \"http"
        ))
        .contains("\"cohere\"")
    );
    assert!(err(&format!("{LOCAL}\nextra = 1\n")).contains("extra"));
    let creds = LOCAL.replace(
        "http://127.0.0.1:8890/v1",
        "http://user:pw-secret@127.0.0.1:8890/v1",
    );
    let msg = err(&creds);
    assert!(
        msg.contains("credentials") && !msg.contains("pw-secret"),
        "{msg}"
    );
}

#[test]
fn openrouter_lowers_to_openai_compatible_at_the_fixed_base_url() {
    let cfg = parse(OPENROUTER).expect("valid");
    let Lowered::Dense(d) = cfg.embedder.lower().expect("lowers") else {
        panic!("openrouter must lower to a dense embedder");
    };
    assert_eq!(d.endpoint, OPENROUTER_BASE_URL);
    assert_eq!(d.host, "openrouter");
    assert_eq!(d.model, "vendor/embed-a:free");
    assert_eq!(
        d.api_key,
        Some(SecretRef::File("keys/openrouter.txt".into()))
    );
    assert_eq!((d.dims, d.precision), (2048, 6));
    let provider = serde_json::Value::Object(d.provider.expect("passed through"));
    assert_eq!(
        provider,
        serde_json::json!({"data_collection": "deny", "order": ["a", "b"]})
    );
    assert_eq!(cfg.reranker, RerankerConfig::None);

    let local = parse(LOCAL).expect("valid");
    let Lowered::Dense(l) = local.embedder.lower().expect("lowers") else {
        panic!("openai_compatible is dense");
    };
    assert_eq!(
        (l.host.as_str(), l.api_key, l.provider),
        ("127.0.0.1", None, None)
    );
    let RerankerConfig::OpenAiCompatible { api_key, .. } = local.reranker else {
        panic!("reranker kind lost");
    };
    assert_eq!(api_key, Some(SecretRef::Env("KKB_RERANK_KEY".into())));
}

#[test]
fn space_ids_are_stable_and_follow_model_host_and_dims() {
    let space = |text: &str| parse(text).expect("valid").embedder.space().expect("space");
    assert_eq!(
        space(OPENROUTER).id(),
        "openai_compatible:openrouter:vendor/embed-a:free@2048"
    );
    assert_eq!(
        space(LOCAL).id(),
        "openai_compatible:127.0.0.1:local-embed@1024"
    );
    // Layout, key order, the key reference, precision and routing preferences
    // are not the space.
    let reordered = "[reranker]\nkind = \"none\"\n\n[embedder]\nprecision = 3\ndims = 2048\n\
        api_key = { env = \"OTHER\" }\nmodel = \"vendor/embed-a:free\"\nkind = \"openrouter\"\n";
    assert_eq!(space(reordered), space(OPENROUTER));
    let upper = LOCAL.replace("127.0.0.1:8890", "Embed.Local.Test");
    assert_eq!(
        space(&upper).id(),
        "openai_compatible:embed.local.test:local-embed@1024"
    );

    for (changed, what) in [
        (
            OPENROUTER.replace("vendor/embed-a:free", "vendor/embed-b:free"),
            "model",
        ),
        (OPENROUTER.replace("dims = 2048", "dims = 1024"), "dims"),
        (LOCAL.replace("127.0.0.1:8890", "10.0.0.1:8890"), "host"),
    ] {
        let base = if what == "host" { LOCAL } else { OPENROUTER };
        assert_ne!(space(&changed), space(base), "{what}");
        assert!(space(&changed).require(&space(base)).is_err(), "{what}");
    }
    let hashed = "[embedder]\nkind = \"hashed\"\n[reranker]\nkind = \"none\"\n";
    assert_eq!(space(hashed).id(), "kkb-embed/1");
    assert_eq!(
        parse(hashed).expect("valid").embedder,
        EmbedderConfig::Hashed
    );
}

#[test]
fn the_reranker_is_selected_by_config_and_none_keeps_first_stage_order() {
    let none = select(&parse(OPENROUTER).expect("valid").reranker).expect("none");
    assert_eq!(none.name(), "none");
    let c = |id: u64, score: f32| Candidate {
        id,
        text: String::new(),
        score,
    };
    let out = none
        .rerank("q", vec![c(1, 0.2), c(2, 0.9), c(3, 0.2), c(4, 0.5)])
        .expect("reranks");
    let ids: Vec<u64> = out.iter().map(|c| c.id).collect();
    assert_eq!(ids, [2, 4, 1, 3], "best first, ties in input order");
    // This crate has no transport, so a configured endpoint is not silently
    // replaced by `none`.
    assert!(select(&parse(LOCAL).expect("valid").reranker).is_err());
}
