// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Actual checkpoint byte vocabulary, strict and lossy controls.
use super::stream::{ByteTokenizer, Stream};
use super::{Decoder, Ending};

const FIXTURE: &str = include_str!("fixtures/gpt-oss-byte-vocab.json");

fn ids(text: &str) -> Vec<u32> {
    let tokenizer = tokenizers::Tokenizer::from_bytes(FIXTURE).unwrap();
    let alphabet: Vec<u8> = (33..=126)
        .chain(161..=172)
        .chain(174..=255)
        .chain((0..=255).filter(|b| !matches!(b, 33..=126 | 161..=172 | 174..=255)))
        .collect();
    let ids: Vec<_> = text
        .as_bytes()
        .iter()
        .map(|b| alphabet.iter().position(|v| v == b).unwrap() as u32)
        .collect();
    assert_eq!(tokenizer.decode(&ids, false).unwrap(), text);
    ids
}

fn stream(tokenizer: &ByteTokenizer) -> Stream<'_> {
    let mut decoder = Decoder::new(["functions.lookup".into()], 128, 1024);
    decoder.seed_assistant_header("assistant").unwrap();
    let mut stream = Stream::new(tokenizer, decoder);
    stream.push(200008).unwrap();
    stream
}

#[test]
fn unicode_splits_and_literal_delimiters_round_trip_without_replacement() {
    let tokenizer = ByteTokenizer::from_tokenizer_json(FIXTURE).unwrap();
    let mut stream = stream(&tokenizer);
    let text = "café 日本 😀 �\0\n<|return|><|call|>";
    for id in ids(text) {
        assert!(stream.push(id).unwrap().is_none());
    }
    let message = stream.push(200002).unwrap().unwrap();
    assert_eq!(message.body, text);
    assert_eq!(message.ending, Ending::Turn);
    stream.finish().unwrap();
    assert!(stream.push(ids("é")[0]).is_err());
    assert!(stream.finish().is_err());
}

#[test]
fn invalid_and_truncated_bytes_are_not_silently_replaced() {
    let tokenizer = ByteTokenizer::from_tokenizer_json(FIXTURE).unwrap();
    let lossy = tokenizers::Tokenizer::from_bytes(FIXTURE).unwrap();
    let bytes = ids("é");
    assert_eq!(lossy.decode(&bytes[..1], false).unwrap(), "�");
    let mut a = stream(&tokenizer);
    a.push(bytes[0]).unwrap();
    assert!(a.finish().is_err());
    assert!(a.push(200002).is_err());
    assert!(a.push(bytes[1]).is_err());
    let mut b = stream(&tokenizer);
    assert!(b.push(bytes[1]).is_err());
    let mut c = stream(&tokenizer);
    c.push(bytes[0]).unwrap();
    assert!(c.push(ids("x")[0]).is_err());
}

#[test]
fn message_boundaries_reset_unicode_but_do_not_end_turn() {
    let tokenizer = ByteTokenizer::from_tokenizer_json(FIXTURE).unwrap();
    let mut s = Stream::new(&tokenizer, Decoder::new([], 128, 100));
    for (channel, text, ending) in [("analysis", "理由", 200007), ("final", "答え", 200002)] {
        s.push(200006).unwrap();
        for id in ids("assistant") {
            s.push(id).unwrap();
        }
        s.push(200005).unwrap();
        for id in ids(channel) {
            s.push(id).unwrap();
        }
        s.push(200008).unwrap();
        for id in ids(text) {
            s.push(id).unwrap();
        }
        let message = s.push(ending).unwrap().unwrap();
        assert_eq!(message.channel.as_deref(), Some(channel));
        assert_eq!(message.body, text);
        assert_eq!(s.finish().is_ok(), ending == 200002);
    }
}

#[test]
fn unsupported_ids_poison_stream_and_limits_still_apply() {
    let tokenizer = ByteTokenizer::from_tokenizer_json(FIXTURE).unwrap();
    for bad in [200003, 201087, u32::MAX] {
        let mut s = stream(&tokenizer);
        assert!(s.push(bad).is_err());
        assert!(s.push(200002).is_err());
    }
    let mut decoder = Decoder::new([], 128, 1);
    decoder.seed_assistant_header("assistant").unwrap();
    let mut s = Stream::new(&tokenizer, decoder);
    s.push(200008).unwrap();
    let bytes = ids("é");
    s.push(bytes[0]).unwrap();
    assert!(s.push(bytes[1]).is_err());
}

#[test]
fn wrong_decoder_and_invalid_byte_vocabulary_are_refused() {
    for (key, value) in [
        ("decoder", serde_json::json!({"type":"WordPiece"})),
        ("model", serde_json::json!({"type":"WordLevel","vocab":{}})),
    ] {
        let mut data: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
        data[key] = value;
        assert!(ByteTokenizer::from_tokenizer_json(&data.to_string()).is_err());
    }
    let mut data: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    data["model"]["vocab"]["😀"] = 10000.into();
    assert!(ByteTokenizer::from_tokenizer_json(&data.to_string()).is_err());
}

#[test]
fn tool_handoff_preserves_unicode_json_and_requires_explicit_end() {
    let tokenizer = ByteTokenizer::from_tokenizer_json(FIXTURE).unwrap();
    let mut s = Stream::new(
        &tokenizer,
        Decoder::new(["functions.lookup".into()], 128, 1024),
    );
    s.push(200006).unwrap();
    for id in ids("assistant to=functions.lookup") {
        s.push(id).unwrap();
    }
    s.push(200005).unwrap();
    for id in ids("commentary json") {
        s.push(id).unwrap();
    }
    s.push(200008).unwrap();
    let body = "{\"query\":\"東京\"}";
    for id in ids(body) {
        s.push(id).unwrap();
    }
    assert!(s.finish().is_err());
    let message = s.push(200012).unwrap().unwrap();
    assert_eq!(message.body, body);
    assert_eq!(message.recipient.as_deref(), Some("functions.lookup"));
    assert_eq!(message.content_type.as_deref(), Some("json"));
    assert_eq!(message.ending, Ending::Tool);
    s.finish().unwrap();
}

// 2026-10-07: Blocking serving must consume terminal IDs without exposing private analysis.
#[test]
fn blocking_final_is_separated_from_analysis_and_requires_terminal() {
    let tokenizer = ByteTokenizer::from_tokenizer_json(FIXTURE).unwrap();
    assert_eq!(tokenizer.stop_ids(), vec![200002, 200012]);
    let prompt = [vec![200006], ids("assistant")].concat();
    let output = [
        vec![200005],
        ids("analysis"),
        vec![200008],
        ids("private"),
        vec![200007, 200006],
        ids("assistant"),
        vec![200005],
        ids("final"),
        vec![200008],
        ids("4 <|return|>"),
        vec![200002],
    ]
    .concat();
    assert_eq!(
        super::api::text_choice(&tokenizer, &prompt, &output).unwrap(),
        "4 <|return|>"
    );
    assert!(super::api::text_choice(&tokenizer, &prompt, &output[..output.len() - 1]).is_err());
    let mut trailing = output.clone();
    trailing.extend(ids("unexpected"));
    assert!(super::api::text_choice(&tokenizer, &prompt, &trailing).is_err());
}

#[test]
fn blocking_refuses_tool_handoff_and_invalid_prompt_prefix() {
    let tokenizer = ByteTokenizer::from_tokenizer_json(FIXTURE).unwrap();
    let prompt = [vec![200006], ids("assistant")].concat();
    let tool = [
        ids(" to=functions.lookup"),
        vec![200008],
        ids("{}"),
        vec![200012],
    ]
    .concat();
    assert!(super::api::text_choice(&tokenizer, &prompt, &tool).is_err());
    let output = [vec![200008], ids("ok"), vec![200002]].concat();
    let user_prefix = [vec![200006], ids("user")].concat();
    assert!(super::api::text_choice(&tokenizer, &user_prefix, &output).is_err());
    let body_prefix = [prompt.clone(), vec![200008]].concat();
    assert!(super::api::text_choice(&tokenizer, &body_prefix, &output).is_err());
    let final_prefix = [prompt, vec![200005], ids("final")].concat();
    assert_eq!(
        super::api::text_choice(&tokenizer, &final_prefix, &output).unwrap(),
        "ok"
    );
}
