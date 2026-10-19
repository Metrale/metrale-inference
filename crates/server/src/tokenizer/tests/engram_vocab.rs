// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CompressedTokenMap;
use tokenizers::Tokenizer;
use tokenizers::decoders::byte_level::ByteLevel;
use tokenizers::models::wordlevel::WordLevel;

/// 2026-10-19: Byte-level token strings and the compressed id each must get. `Ġ` is a space
/// and `ĉ` a tab in the byte-level alphabet; `Ã`/`Ä` are lone UTF-8 lead bytes (they decode to
/// U+FFFD); a bare combining accent normalizes to nothing.
const CASES: [(&str, u32); 12] = [
    ("ĠThe", 0),
    ("the", 0),
    ("THE", 0),
    ("Ġ", 1),
    ("ĉ", 1),
    ("Ã", 2),
    ("Ä", 3),
    ("\u{301}", 4),
    ("\u{300}", 5),
    ("Ġthe", 0),
    ("cafÃ©", 6),
    ("cafe", 6),
];

fn fixture() -> Tokenizer {
    let vocab = CASES
        .iter()
        .zip(0u32..)
        .map(|((t, _), id)| (t.to_string(), id))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("the".into())
        .build()
        .unwrap();
    let mut t = Tokenizer::new(model);
    t.with_decoder(Some(ByteLevel::default()));
    t
}

/// 2026-10-19: Case, accents and leading whitespace collapse; a lone space or tab survives as
/// one space instead of emptying; partial-byte tokens and accents that normalize to nothing
/// stay distinct (each keyed by its raw form). A port that dropped any one rule merges two of
/// these ids or splits a collapsed pair.
#[test]
fn the_publisher_rules_collapse_exactly_the_alike_tokens() {
    let map = CompressedTokenMap::build(&fixture(), 7).unwrap();
    let want: Vec<u32> = CASES.iter().map(|(_, c)| *c).collect();
    assert_eq!(map.lookup(), want.as_slice());
    assert_eq!(map.size(), 7);
    assert_eq!(map.get(9).unwrap(), 0);
}

/// 2026-10-19: Every hash multiplier derives from the size, so any other declared size is
/// refused, and an id past the vocab is an error rather than a wrapped lookup.
#[test]
fn a_size_mismatch_and_an_out_of_vocab_id_are_refused() {
    let err = CompressedTokenMap::build(&fixture(), 8).unwrap_err();
    assert!(
        err.to_string().contains("is 7 but the config declares 8"),
        "{err}"
    );
    let map = CompressedTokenMap::build(&fixture(), 7).unwrap();
    assert!(map.get(12).is_err());
}

/// 2026-10-19: The real checkpoint: the map built from its `tokenizer.json` has exactly the
/// size its `config.json` declares (the publisher asserts the same before hashing), " The"
/// and "the" share an id, and one more than the declared size is refused.
#[test]
#[ignore = "needs METRALE_DSV41_CKPT: a DeepSeek-V4.1-Flash snapshot directory"]
fn the_checkpoint_tokenizer_gives_the_declared_compressed_size() {
    let snap = std::path::PathBuf::from(
        std::env::var("METRALE_DSV41_CKPT")
            .expect("METRALE_DSV41_CKPT names the snapshot directory"),
    );
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(snap.join("config.json")).unwrap()).unwrap();
    let declared = config["text_config"]["engram_compressed_vocab_size"]
        .as_u64()
        .expect("engram_compressed_vocab_size");
    let tok = Tokenizer::from_file(snap.join("tokenizer.json")).unwrap();
    let map = CompressedTokenMap::build(&tok, declared).unwrap();
    assert_eq!(u64::from(map.size()), declared);
    assert_eq!(map.lookup().len(), tok.get_vocab_size(true));
    let id = |s: &str| {
        tok.token_to_id(s)
            .unwrap_or_else(|| panic!("no token {s:?}"))
    };
    assert_eq!(map.get(id("ĠThe")).unwrap(), map.get(id("the")).unwrap());
    assert!(CompressedTokenMap::build(&tok, declared + 1).is_err());
}
