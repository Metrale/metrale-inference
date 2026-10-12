// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The two QCI recipes, compiled in from `recipes/`.
//!
//! The files use the same required keys as `Recipe::parse` (`recipe_version`, `model`,
//! `container`, a `defaults` map of scalars). `runtime` is not `metrale`, so the recipes
//! job parses them and does not launch them. The `qci` map is the pin block.

use std::collections::BTreeMap;

const GEMMA: &str = include_str!("../../../recipes/gemma4/gemma-4-26b-a4b-it-gguf-q4km.yaml");
const NEMOTRON: &str =
    include_str!("../../../recipes/nemotron-3.5/nemotron-3.5-lightning-30b-a3b-gguf-q4_0.yaml");

/// 2026-10-06: One pinned campaign, read from its recipe file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipe {
    pub id: String,
    pub model: String,
    pub case_id: String,
    pub issue_url: String,
    pub priority: String,
    pub blocks_p0_exit: String,
    pub p0_exit_member: String,
    pub original_repository: String,
    pub repository: String,
    pub revision: String,
    pub artifact: String,
    pub artifact_bytes: String,
    pub vision_artifact: String,
    pub vision_artifact_bytes: String,
    pub quantization: String,
    pub license_tag: String,
    pub license_link: String,
    pub license_status: String,
    pub concurrency: String,
    pub context: String,
    pub max_images: u64,
    pub max_image_bytes: u64,
    pub max_text_chars: u64,
    pub supplemental_hardware: String,
    pub target_hardware: String,
    pub reference_runtime: String,
    pub native_limitations: String,
    pub distinct_from: String,
    pub mtp: String,
    pub tokenizer_bos: String,
    pub tokenizer_eos: String,
    pub tokenizer_json_sha256: String,
    pub chat_template_blob: String,
    pub template_status: String,
    pub pipeline: String,
    pub required_contracts: String,
    pub extra_blockers: String,
}

pub fn for_model(model: &str) -> Option<Recipe> {
    embedded()
        .into_iter()
        .find(|r| r.model == model || r.repository == model || r.original_repository == model)
}

pub fn for_case(case_id: &str) -> Option<Recipe> {
    embedded().into_iter().find(|r| r.case_id == case_id)
}

fn embedded() -> Vec<Recipe> {
    vec![
        parse("gemma4/gemma-4-26b-a4b-it-gguf-q4km", GEMMA).expect("gemma recipe"),
        parse(
            "nemotron-3.5/nemotron-3.5-lightning-30b-a3b-gguf-q4_0",
            NEMOTRON,
        )
        .expect("nemotron recipe"),
    ]
}

fn parse(id: &str, text: &str) -> Result<Recipe, String> {
    let root = mapping(text)?;
    require(&root, "recipe_version")?;
    let model = require(&root, "model")?;
    require(&root, "container")?;
    let defaults = root
        .get("defaults")
        .and_then(Value::as_map)
        .ok_or_else(|| format!("{id}: defaults must be a mapping"))?;
    if defaults.values().any(|v| !matches!(v, Value::Scalar(_))) {
        return Err(format!("{id}: defaults values must be scalars"));
    }
    let qci = root
        .get("qci")
        .and_then(Value::as_map)
        .ok_or_else(|| format!("{id}: qci must be a mapping"))?;
    let scalar = |key: &str| -> Result<String, String> {
        match qci.get(key) {
            Some(Value::Scalar(s)) if !s.is_empty() => Ok(s.clone()),
            _ => Err(format!("{id}: qci.{key} must be a non-empty scalar")),
        }
    };
    let optional = |key: &str| -> String {
        match qci.get(key) {
            Some(Value::Scalar(s)) => s.clone(),
            _ => String::new(),
        }
    };
    let number = |key: &str| -> Result<u64, String> {
        scalar(key)?
            .parse()
            .map_err(|_| format!("{id}: qci.{key} is not an integer"))
    };
    Ok(Recipe {
        id: id.to_string(),
        model,
        case_id: scalar("case")?,
        issue_url: scalar("issue_url")?,
        priority: scalar("priority")?,
        blocks_p0_exit: scalar("blocks_four_model_p0_exit")?,
        p0_exit_member: scalar("p0_exit_member")?,
        original_repository: scalar("original_repository")?,
        repository: scalar("repository")?,
        revision: scalar("revision")?,
        artifact: scalar("artifact")?,
        artifact_bytes: scalar("artifact_bytes")?,
        vision_artifact: optional("vision_artifact"),
        vision_artifact_bytes: optional("vision_artifact_bytes"),
        quantization: scalar("quantization")?,
        license_tag: scalar("license_tag")?,
        license_link: scalar("license_link")?,
        license_status: scalar("license_status")?,
        concurrency: scalar("concurrency")?,
        context: scalar("context")?,
        max_images: number("max_images")?,
        max_image_bytes: number("max_image_bytes")?,
        max_text_chars: number("max_text_chars")?,
        supplemental_hardware: scalar("supplemental_hardware")?,
        target_hardware: scalar("target_hardware")?,
        reference_runtime: scalar("reference_runtime")?,
        native_limitations: scalar("native_limitations")?,
        distinct_from: optional("distinct_from"),
        mtp: scalar("mtp")?,
        tokenizer_bos: optional("tokenizer_bos"),
        tokenizer_eos: optional("tokenizer_eos"),
        tokenizer_json_sha256: optional("tokenizer_json_sha256"),
        chat_template_blob: optional("chat_template_blob"),
        template_status: optional("template_status"),
        pipeline: scalar("pipeline")?,
        required_contracts: scalar("required_contracts")?,
        extra_blockers: scalar("extra_blockers")?,
    })
}

#[derive(Debug)]
enum Value {
    Scalar(String),
    Map(BTreeMap<String, Value>),
}

impl Value {
    fn as_map(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Map(m) => Some(m),
            Value::Scalar(_) => None,
        }
    }
}

fn require(map: &BTreeMap<String, Value>, key: &str) -> Result<String, String> {
    match map.get(key) {
        Some(Value::Scalar(s)) if !s.is_empty() => Ok(s.clone()),
        _ => Err(format!("missing required key {key}")),
    }
}

/// 2026-10-06: A subset of the engine recipe reader: nested maps of scalars, quotes stripped.
fn mapping(text: &str) -> Result<BTreeMap<String, Value>, String> {
    let lines = significant(text)?;
    let (value, end) = block(&lines, 0, 0)?;
    if end != lines.len() {
        return Err(format!("line {}: unexpected indent", lines[end].0));
    }
    match value {
        Value::Map(m) => Ok(m),
        Value::Scalar(_) => Err("document must be a mapping".to_string()),
    }
}

struct Line(usize, usize, String);

fn significant(text: &str) -> Result<Vec<Line>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let indent = raw.len() - raw.trim_start().len();
        let trimmed = raw.trim_start();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let text = strip_comment(trimmed);
        if text.is_empty() {
            continue;
        }
        out.push(Line(i + 1, indent, text));
    }
    Ok(out)
}

fn strip_comment(s: &str) -> String {
    let mut in_quotes = false;
    for (i, c) in s.char_indices() {
        match c {
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes && (i == 0 || s.as_bytes()[i - 1] == b' ') => {
                return s[..i].trim_end().to_string();
            }
            _ => {}
        }
    }
    s.trim_end().to_string()
}

fn unquote(s: &str) -> String {
    let b = s.as_bytes();
    if b.len() >= 2
        && ((b[0] == b'"' && b[b.len() - 1] == b'"') || (b[0] == b'\'' && b[b.len() - 1] == b'\''))
    {
        return s[1..s.len() - 1].to_string();
    }
    s.to_string()
}

fn block(lines: &[Line], start: usize, indent: usize) -> Result<(Value, usize), String> {
    let mut map = BTreeMap::new();
    let mut i = start;
    while i < lines.len() {
        let Line(no, ind, text) = &lines[i];
        if *ind < indent {
            break;
        }
        if *ind > indent {
            return Err(format!("line {no}: unexpected indent"));
        }
        let Some((key, rest)) = text.split_once(':') else {
            return Err(format!("line {no}: expected key: value"));
        };
        let key = key.trim();
        if key.is_empty() {
            return Err(format!("line {no}: empty key"));
        }
        let rest = rest.trim();
        if !rest.is_empty() {
            map.insert(key.to_string(), Value::Scalar(unquote(rest)));
            i += 1;
            continue;
        }
        let next_indent = lines.get(i + 1).map(|l| l.1).unwrap_or(0);
        if next_indent <= indent {
            return Err(format!("line {no}: {key} has no value"));
        }
        let (value, next) = block(lines, i + 1, next_indent)?;
        map.insert(key.to_string(), value);
        i = next;
    }
    Ok((Value::Map(map), i))
}
