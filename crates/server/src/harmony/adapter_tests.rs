// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Identity-based framing and corrupted metadata controls.
use super::adapter::{TokenClass, TokenMap};
use super::{Decoder, Ending, Token};

const METADATA: &str = include_str!("fixtures/gpt-oss-token-metadata.json");
fn changed(f: impl FnOnce(&mut serde_json::Value)) -> String {
    let mut data: serde_json::Value = serde_json::from_str(METADATA).unwrap();
    f(&mut data);
    data.to_string()
}

#[test]
fn checkpoint_ids_keep_three_ending_meanings_distinct() {
    let map = TokenMap::from_tokenizer_json(METADATA).unwrap();
    for (id, token) in [
        (200006, Token::Start),
        (200005, Token::Channel),
        (200008, Token::Separator),
        (200007, Token::End),
        (200002, Token::Finish),
        (200012, Token::Handoff),
    ] {
        assert_eq!(map.classify(id).unwrap(), TokenClass::Framing(token));
    }
    assert_eq!(map.classify(0).unwrap(), TokenClass::Ordinary);
    assert_eq!(map.classify(199997).unwrap(), TokenClass::Ordinary);
}

#[test]
fn padding_reserved_and_unassigned_logits_are_not_text_or_eos() {
    let map = TokenMap::from_tokenizer_json(METADATA).unwrap();
    for id in [
        199998,
        199999,
        200000,
        200003,
        200018,
        200019,
        201087,
        u32::MAX,
    ] {
        assert!(map.classify(id).is_err(), "accepted unsupported ID {id}");
    }
}

#[test]
fn renamed_ids_follow_metadata_instead_of_assuming_numeric_constants() {
    let data = changed(|d| d["added_tokens"][8]["id"] = 123456.into());
    let map = TokenMap::from_tokenizer_json(&data).unwrap();
    assert_eq!(
        map.classify(123456).unwrap(),
        TokenClass::Framing(Token::Start)
    );
    assert!(map.classify(200006).is_err());
}

#[test]
fn corrupt_or_incomplete_metadata_is_refused() {
    let mutations: &[fn(&mut serde_json::Value)] = &[
        |d| {
            d["added_tokens"].as_array_mut().unwrap().remove(8);
        },
        |d| d["added_tokens"][8]["special"] = false.into(),
        |d| d["added_tokens"][8]["normalized"] = true.into(),
        |d| d["added_tokens"][8]["id"] = 0.into(),
        |d| d["added_tokens"][8]["id"] = (-1).into(),
        |d| d["added_tokens"][8]["id"] = 4294967296_u64.into(),
        |d| d["added_tokens"][8]["content"] = "<|im_start|>".into(),
        |d| {
            let copy = d["added_tokens"][8].clone();
            d["added_tokens"].as_array_mut().unwrap().push(copy);
        },
        |d| d["model"]["vocab"]["second spelling"] = 0.into(),
        |d| {
            d["added_tokens"][8]
                .as_object_mut()
                .unwrap()
                .remove("special");
        },
    ];
    for mutate in mutations {
        assert!(TokenMap::from_tokenizer_json(&changed(mutate)).is_err());
    }
    assert!(TokenMap::from_tokenizer_json("{}").is_err());
    assert!(TokenMap::from_tokenizer_json("not JSON").is_err());
}

#[test]
fn decoded_literal_delimiters_do_not_become_control_events() {
    let map = TokenMap::from_tokenizer_json(METADATA).unwrap();
    let mut decoder = Decoder::new([], 100, 100);
    let TokenClass::Framing(token) = map.classify(200006).unwrap() else {
        panic!()
    };
    decoder.push(token).unwrap();
    decoder.push(Token::Text("assistant")).unwrap();
    let TokenClass::Framing(separator) = map.classify(200008).unwrap() else {
        panic!()
    };
    decoder.push(separator).unwrap();
    // 2026-10-07: The ordinary token decoder owns byte/Unicode assembly. Its output is never reparsed.
    assert_eq!(map.classify(0).unwrap(), TokenClass::Ordinary);
    decoder
        .push(Token::Text("<|return|><|call|>analysis"))
        .unwrap();
    let TokenClass::Framing(finish) = map.classify(200002).unwrap() else {
        panic!()
    };
    let message = decoder.push(finish).unwrap().unwrap();
    assert_eq!(message.body, "<|return|><|call|>analysis");
    assert_eq!(message.ending, Ending::Turn);
    decoder.finish().unwrap();
}

#[test]
fn consistent_base_vocab_overlap_is_valid_but_alias_collision_is_not() {
    let data = changed(|d| d["model"]["vocab"]["<|start|>"] = 200006.into());
    let map = TokenMap::from_tokenizer_json(&data).unwrap();
    assert_eq!(
        map.classify(200006).unwrap(),
        TokenClass::Framing(Token::Start)
    );
    let data = changed(|d| d["model"]["vocab"]["<|start|>"] = 19.into());
    assert!(TokenMap::from_tokenizer_json(&data).is_err());
}

#[test]
fn non_special_added_tokens_remain_ordinary() {
    let data = changed(|d| {
        d["added_tokens"][0]["special"] = false.into();
    });
    let map = TokenMap::from_tokenizer_json(&data).unwrap();
    assert_eq!(map.classify(199998).unwrap(), TokenClass::Ordinary);
}
