// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Where the chat template comes from (`TemplateSource`) on both apply
//! paths, and the template variables GLM-5.3's served template reads: `thinking`
//! (passed through only when the client sends it), `enable_thinking` and
//! `reasoning_effort`.
//!
//! Owner: server (tokenizer) tests.
//! Invariants: none beyond the types.

use super::super::chat_render::{RenderFlags, render_chat};
use super::super::jinja_helpers;
use super::super::{ChatTokenizer, TemplateSource};
use serde_json::json;

/// 2026-10-08: A model dir whose tokenizer maps each source's marker word to its own
/// id, and whose checkpoint template renders `CKPT`; a repo dir with an override
/// (`OVERRIDE`) and an OpenAI variant (`OPENAI`); and a template file (`FILE`).
struct Fixture {
    _root: tempfile::TempDir,
    model: std::path::PathBuf,
    repo: std::path::PathBuf,
    file: std::path::PathBuf,
}

const KIND: &str = "glm5_next";

fn fixture() -> Fixture {
    use tokenizers::models::wordlevel::WordLevel;
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model");
    let repo = root.path().join("repo");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::create_dir_all(repo.join("jinja-templates/openai")).unwrap();
    let vocab = [
        ("[UNK]", 0),
        ("CKPT", 1),
        ("OVERRIDE", 2),
        ("OPENAI", 3),
        ("FILE", 4),
    ]
    .into_iter()
    .map(|(w, i)| (w.to_string(), i))
    .collect();
    let word_level = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tok = tokenizers::Tokenizer::new(word_level);
    tok.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace {}));
    tok.save(model.join("tokenizer.json"), false).unwrap();
    std::fs::write(
        model.join("tokenizer_config.json"),
        json!({"chat_template": "CKPT"}).to_string(),
    )
    .unwrap();
    std::fs::write(
        repo.join(format!("jinja-templates/{KIND}.jinja")),
        "OVERRIDE",
    )
    .unwrap();
    std::fs::write(
        repo.join(format!("jinja-templates/openai/{KIND}.jinja")),
        "OPENAI",
    )
    .unwrap();
    let file = root.path().join("served.jinja");
    std::fs::write(&file, "FILE").unwrap();
    Fixture {
        _root: root,
        model,
        repo,
        file,
    }
}

/// 2026-10-08: The ids the Jinja and the OpenAI apply paths render to.
fn rendered(f: &Fixture, source: TemplateSource<'_>) -> (Vec<u32>, Vec<u32>) {
    let t = ChatTokenizer::from_model_dir(&f.model, 0, true, KIND, Some(&f.repo), source)
        .expect("tokenizer loads");
    let messages = [json!({"role": "user", "content": "hi"})];
    (
        t.apply_chat_template_jinja(&messages, None, true, false)
            .unwrap(),
        t.apply_chat_template_openai(&messages, None, true, false)
            .unwrap(),
    )
}

#[test]
fn each_source_selects_its_template_on_both_apply_paths() {
    let f = fixture();
    assert_eq!(
        rendered(&f, TemplateSource::OverrideDir),
        (vec![2], vec![3])
    );
    // 2026-10-08: `--disable-template-overrides` leaves the OpenAI variant in place,
    // as it always has.
    assert_eq!(rendered(&f, TemplateSource::Checkpoint), (vec![1], vec![3]));
    assert_eq!(
        rendered(&f, TemplateSource::File(&f.file)),
        (vec![4], vec![4]),
        "--chat-template outranks the override directory and its OpenAI variant"
    );
}

#[test]
fn an_unreadable_template_file_fails_the_load() {
    let f = fixture();
    let missing = f.file.with_file_name("missing.jinja");
    let err = ChatTokenizer::from_model_dir(
        &f.model,
        0,
        true,
        KIND,
        Some(&f.repo),
        TemplateSource::File(&missing),
    )
    .err()
    .expect("a missing --chat-template file must fail the load");
    assert!(format!("{err:#}").contains("missing.jinja"), "{err:#}");
}

/// 2026-10-08: The first lines of GLM-5.3's served template: the effort it writes
/// into the system turn from `thinking`, `enable_thinking` and `reasoning_effort`.
const GLM_EFFORT_HEADER: &str = "\
{%- set thinking_off = (thinking is defined and not thinking) or (enable_thinking is defined and not enable_thinking) -%}
{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else ('low' if thinking_off else 'max') -%}
{{ effective_reasoning_effort | capitalize }}";

fn effort(flags: RenderFlags<'_>) -> String {
    let converted = jinja_helpers::convert_python_jinja_to_minijinja(GLM_EFFORT_HEADER);
    let env = jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    render_chat(
        &env,
        &[json!({"role": "user", "content": "hi"})],
        None,
        flags,
    )
    .unwrap()
}

#[test]
fn the_thinking_kwarg_reaches_the_template_only_when_sent() {
    let base = RenderFlags {
        enable_thinking: true,
        reasoning_effort: Some("medium"),
        ..Default::default()
    };
    assert_eq!(effort(base), "Max", "medium with thinking on is max effort");
    assert_eq!(
        effort(RenderFlags {
            thinking: Some(false),
            ..base
        }),
        "Low",
        "thinking off maps to low effort even beside a medium effort"
    );
    assert_eq!(
        effort(RenderFlags {
            thinking: Some(true),
            ..base
        }),
        "Max"
    );
    assert_eq!(
        effort(RenderFlags {
            enable_thinking: true,
            reasoning_effort: Some("low"),
            ..Default::default()
        }),
        "Low",
        "the served default effort"
    );
    assert_eq!(
        effort(RenderFlags {
            enable_thinking: false,
            ..Default::default()
        }),
        "Low",
        "thinking off with no effort is low effort"
    );
}
