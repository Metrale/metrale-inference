// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Harmony generation boundaries, malformed streams and chunk invariance.
use super::*;

fn decoder() -> Decoder {
    Decoder::new(["functions.lookup".into()], 4096, 8192)
}
fn message(d: &mut Decoder, header: &str, body: &str, end: Token<'_>) -> Message {
    d.push(Token::Start).unwrap();
    d.push(Token::Text(header)).unwrap();
    d.push(Token::Separator).unwrap();
    d.push(Token::Text(body)).unwrap();
    d.push(end).unwrap().unwrap()
}
#[test]
fn analysis_boundary_is_not_turn_completion() {
    let mut d = decoder();
    d.push(Token::Start).unwrap();
    d.push(Token::Text("assistant")).unwrap();
    d.push(Token::Channel).unwrap();
    d.push(Token::Text("analysis")).unwrap();
    d.push(Token::Separator).unwrap();
    d.push(Token::Text("reasoning")).unwrap();
    let m = d.push(Token::End).unwrap().unwrap();
    assert_eq!(m.channel, Some("analysis".into()));
    assert_eq!(m.ending, Ending::Message);
    assert_eq!(
        message(&mut d, "assistant", "42", Token::Finish).ending,
        Ending::Turn
    );
    assert!(d.finish().is_ok());
}
#[test]
fn recipient_and_channel_orders_are_equivalent() {
    let mut results = vec![];
    for (a, b) in [
        ("assistant to=functions.lookup", "commentary"),
        ("assistant", "commentary to=functions.lookup"),
    ] {
        let mut d = decoder();
        d.push(Token::Start).unwrap();
        d.push(Token::Text(a)).unwrap();
        d.push(Token::Channel).unwrap();
        d.push(Token::Text(b)).unwrap();
        d.push(Token::Separator).unwrap();
        d.push(Token::Text("{\"id\":17}")).unwrap();
        results.push(d.push(Token::Handoff).unwrap().unwrap());
        assert!(d.finish().is_ok());
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0].ending, Ending::Tool);
}
#[test]
fn chunks_and_literal_protocol_words_do_not_change_body() {
    let body = "analysis to=functions.lookup café 東京";
    for split in (0..=body.len()).filter(|i| body.is_char_boundary(*i)) {
        let mut d = decoder();
        d.seed_assistant_header("assistant").unwrap();
        d.push(Token::Separator).unwrap();
        d.push(Token::Text(&body[..split])).unwrap();
        d.push(Token::Text(&body[split..])).unwrap();
        assert_eq!(d.push(Token::Finish).unwrap().unwrap().body, body);
    }
}
#[test]
fn truncated_stream_never_fabricates_success() {
    for tokens in [
        vec![Token::Start],
        vec![Token::Start, Token::Text("assistant"), Token::Separator],
        vec![
            Token::Start,
            Token::Text("assistant"),
            Token::Separator,
            Token::Text("partial"),
        ],
    ] {
        let mut d = decoder();
        for t in tokens {
            d.push(t).unwrap();
        }
        assert!(d.finish().is_err());
    }
    let mut d = decoder();
    message(&mut d, "assistant", "thinking", Token::End);
    assert!(d.finish().is_err());
}
#[test]
fn unknown_recipient_and_illegal_endings_fail_closed() {
    for (header, end) in [
        ("assistant to=functions.unknown", Token::Handoff),
        ("assistant", Token::Handoff),
        ("assistant to=functions.lookup", Token::Finish),
        ("system", Token::Finish),
        (
            "assistant to=functions.lookup to=functions.lookup",
            Token::Handoff,
        ),
    ] {
        let mut d = decoder();
        d.push(Token::Start).unwrap();
        d.push(Token::Text(header)).unwrap();
        if d.push(Token::Separator).is_ok() {
            assert!(d.push(end).is_err());
        }
        assert!(d.push(Token::Text("must not recover")).is_err());
    }
}
#[test]
fn bounds_and_duplicate_terminals_fail() {
    let mut d = Decoder::new([], 3, 3);
    d.push(Token::Start).unwrap();
    assert!(d.push(Token::Text("assistant")).is_err());
    let mut d = Decoder::new([], 100, 3);
    d.seed_assistant_header("assistant").unwrap();
    d.push(Token::Separator).unwrap();
    assert!(d.push(Token::Text("four")).is_err());
    let mut d = decoder();
    message(&mut d, "assistant", "ok", Token::Finish);
    assert!(d.push(Token::Finish).is_err());
}
#[test]
fn header_text_can_arrive_one_character_at_a_time() {
    let mut d = decoder();
    d.push(Token::Start).unwrap();
    for c in "assistant to=functions.lookup".chars() {
        d.push(Token::Text(&c.to_string())).unwrap();
    }
    d.push(Token::Channel).unwrap();
    d.push(Token::Text("commentary")).unwrap();
    d.push(Token::Separator).unwrap();
    assert_eq!(
        d.push(Token::Handoff)
            .unwrap()
            .unwrap()
            .recipient
            .as_deref(),
        Some("functions.lookup")
    );
}
