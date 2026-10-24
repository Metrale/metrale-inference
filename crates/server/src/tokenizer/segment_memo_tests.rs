// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The piece memo returns `Tokenizer::encode`'s ids exactly, cold, warm, on a grown
//! conversation and under eviction, and a warm encode reads the memo.

use std::str::FromStr;

use tokenizers::Tokenizer;

use super::{MIN_PIECE_BYTES, MIN_TEXT_BYTES, SegmentMemo};

/// 2026-10-22: GLM-5.3-Flash's pre-tokenizer split.
const GLM_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// 2026-10-22: A byte-level BPE over the 256 byte symbols with `merges`, GLM's pre-tokenizer
/// and added tokens that stress the split: `<|u` is a prefix of `<|user|>` (leftmost-longest),
/// `<|observation|>` strips whitespace on both sides, `hello` is a non-special added word.
fn fixture(merges: &[(&str, &str)], normalizer: serde_json::Value) -> Tokenizer {
    let mut vocab = serde_json::Map::new();
    for c in tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet() {
        let id = vocab.len();
        vocab.insert(c.to_string(), id.into());
    }
    for (a, b) in merges {
        let id = vocab.len();
        vocab.insert(format!("{a}{b}"), id.into());
    }
    let base = vocab.len();
    let added = [
        ("<|user|>", true, false),
        ("<|assistant|>", true, false),
        ("<|u", false, false),
        ("<think>", true, false),
        ("<|observation|>", true, true),
        ("hello", false, false),
    ];
    let added_tokens: Vec<_> = added
        .iter()
        .enumerate()
        .map(|(i, (content, special, strip))| {
            serde_json::json!({"id": base + i, "content": content, "single_word": false,
                "lstrip": strip, "rstrip": strip, "normalized": false, "special": special})
        })
        .collect();
    let merges: Vec<String> = merges.iter().map(|(a, b)| format!("{a} {b}")).collect();
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added_tokens,
        "normalizer": normalizer,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": GLM_SPLIT}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
        "post_processor": null, "decoder": null,
        "model": {"type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null, "fuse_unk": false,
            "byte_fallback": false, "ignore_merges": false, "vocab": vocab, "merges": merges},
    });
    Tokenizer::from_str(&json.to_string()).unwrap()
}

/// 2026-10-22: Merges across every character a piece boundary touches (a word's first
/// letters, space, tab and newline runs, a digit pair), so a misplaced cut changes the ids.
const MERGES_A: &[(&str, &str)] = &[
    ("Ġ", "t"), ("h", "e"), ("Ġt", "he"), ("i", "n"), ("Ġ", "a"), ("e", "r"), ("P", "a"),
    ("Pa", "r"), ("Ġ", "Ġ"), ("ĠĠ", "Ġ"), ("Ċ", "Ċ"), ("ĉ", "Ġ"), ("1", "2"), ("x", "<"),
    ("Ġ", "x"), ("Ġ", "s"), ("s", "h"),
];
const MERGES_B: &[(&str, &str)] = &[("Ġ", "a"), ("n", "d"), ("Ġa", "nd")];

/// 2026-10-22: A text of `n` paragraphs, each at least `MIN_PIECE_BYTES`, ending in what a
/// regex lookahead sees at a piece boundary: trailing spaces, newlines, a digit run, an accent.
fn paragraph(i: usize) -> String {
    let tails = ["   ", "\n\n", " 12345", "e\u{301}", "\t \n", "'s", ""];
    // 2026-10-22: The memoized piece starts after the added word `hello`, inside the merged
    // `Par`, so a cut one byte off changes the ids.
    let mut p = format!("{i} helloParagraph: the winter in Kraków — naïve café, 日本語 😀. ");
    while p.len() < MIN_PIECE_BYTES + 37 * i {
        p.push_str("The other inn and the rather tall tower; and there, tallthere. ");
    }
    p + tails[i % tails.len()]
}

/// 2026-10-22: A conversation of `turns` turns, with adjacent added tokens, a near-miss
/// `<|user`, the prefix token `<|u` alone, whitespace around a stripping token and short runs.
fn conversation(turns: usize) -> String {
    let mut s = String::from("[gMASK]<sop><|u <|user");
    for i in 0..turns {
        s.push_str(&format!("<|user|>{}<|assistant|><think>", paragraph(i)));
        s.push_str(&format!("  short {i} <|observation|>  \n{}  ", paragraph(i + 100)));
        s.push_str("<|user|><|assistant|> x<|u");
    }
    s
}

fn assert_exact(memo: &SegmentMemo, tok: &Tokenizer, text: &str, what: &str) {
    let want = tok.encode(text, false).unwrap().get_ids().to_vec();
    let got = memo.encode(tok, text).unwrap();
    assert_eq!(got.len(), want.len(), "{what}: length");
    assert_eq!(got, want, "{what}: ids");
}

/// 2026-10-22: Cold, warm and grown conversations give `encode`'s ids, with and without an NFC
/// normalizer; texts below the threshold and a stripping normalizer take the whole encode.
#[test]
fn memo_encode_equals_whole_encode() {
    for normalizer in [serde_json::Value::Null, serde_json::json!({"type": "NFC"})] {
        let tok = fixture(MERGES_A, normalizer.clone());
        let memo = SegmentMemo::new(64 << 20);
        let three = conversation(3);
        assert!(three.len() >= MIN_TEXT_BYTES, "the fixture must take the memo path");
        assert_exact(&memo, &tok, &three, "cold");
        assert!(memo.held_bytes() > 0, "{normalizer}: no piece was memoized");
        assert_exact(&memo, &tok, &three, "warm");
        assert_exact(&memo, &tok, &conversation(5), "grown");
        assert_exact(&memo, &tok, &paragraph(1), "short text");
    }
    let tok = fixture(MERGES_A, serde_json::json!({"type": "Strip", "strip_left": true, "strip_right": true}));
    assert_exact(&SegmentMemo::new(64 << 20), &tok, &conversation(3), "strip normalizer");
}

/// 2026-10-22: A memo smaller than the conversation evicts and stays exact; it never holds
/// more than its budget, and a budget of 0 memoizes nothing.
#[test]
fn eviction_keeps_ids_exact_and_bytes_bounded() {
    let tok = fixture(MERGES_A, serde_json::Value::Null);
    let budget = 3 * MIN_PIECE_BYTES * 5;
    let memo = SegmentMemo::new(budget);
    for turns in [4, 6, 2, 6] {
        assert_exact(&memo, &tok, &conversation(turns), "evicting");
        assert!(memo.held_bytes() <= budget, "{} > {budget}", memo.held_bytes());
    }
    let off = SegmentMemo::new(0);
    assert_exact(&off, &tok, &conversation(3), "off");
    assert_eq!(off.held_bytes(), 0);
}

/// 2026-10-22: Positive control: a warm encode takes memoized pieces instead of encoding them.
/// The memo filled under merges A, read through a tokenizer with merges B, keeps A's ids for the
/// memoized pieces; encoding every unit through B would give B's whole encode exactly (the
/// first test), so a difference proves the lookup produced them.
#[test]
fn a_warm_encode_reads_the_memo() {
    let (a, b) = (fixture(MERGES_A, serde_json::Value::Null), fixture(MERGES_B, serde_json::Value::Null));
    let text = conversation(3);
    let memo = SegmentMemo::new(64 << 20);
    let want_a = a.encode(text.as_str(), false).unwrap().get_ids().to_vec();
    let want_b = b.encode(text.as_str(), false).unwrap().get_ids().to_vec();
    assert_ne!(want_a, want_b, "the two fixtures must disagree");
    memo.encode(&a, &text).unwrap();
    let via_memo = memo.encode(&b, &text).unwrap();
    assert_ne!(via_memo, want_b, "the warm encode re-encoded every piece");
}


/// 2026-10-22: A real checkpoint tokenizer on a real prompt: exact cold and warm, and the warm
/// encode's time beside the whole encode's (printed, not judged: timing is the host's).
#[test]
#[ignore = "needs METRALE_MEMO_TOKENIZER (a tokenizer.json) and METRALE_MEMO_TEXT (a prompt file)"]
fn a_checkpoint_tokenizer_is_exact_on_a_real_prompt() {
    use metrale_abstract_io::Env as _;
    let var = |k: &str| {
        metrale_io_host_std::ProcessEnv
            .var(k)
            .expect("unicode")
            .unwrap_or_else(|| panic!("{k} is unset"))
    };
    let tok = Tokenizer::from_file(var("METRALE_MEMO_TOKENIZER")).unwrap();
    let text = std::fs::read_to_string(var("METRALE_MEMO_TEXT")).unwrap();
    let memo = SegmentMemo::new(64 << 20);
    let t0 = std::time::Instant::now();
    let want = tok.encode(text.as_str(), false).unwrap().get_ids().to_vec();
    let t1 = std::time::Instant::now();
    assert_eq!(memo.encode(&tok, &text).unwrap(), want, "cold");
    let t2 = std::time::Instant::now();
    assert_eq!(memo.encode(&tok, &text).unwrap(), want, "warm");
    let t3 = std::time::Instant::now();
    eprintln!(
        "{} tokens: whole {:?}, memo cold {:?}, memo warm {:?}",
        want.len(),
        t1 - t0,
        t2 - t1,
        t3 - t2
    );
}
