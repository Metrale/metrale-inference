// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: CPU replay of completed and incomplete JSON with the real pinned
//! Laguna tokenizer. This diagnoses masks, not the live scheduler's token path.

use super::*;

#[test]
#[ignore = "requires LAGUNA_TOKENIZER_JSON pointing to the pinned existing tokenizer"]
fn pinned_laguna_json_completion_stop_masks() {
    let path = std::env::var("LAGUNA_TOKENIZER_JSON").expect("explicit local tokenizer");
    let tokenizer = tokenizers::Tokenizer::from_file(path).unwrap();
    assert_eq!(tokenizer.token_to_id("〈|EOS|〉"), Some(2));
    assert_eq!(tokenizer.token_to_id("</assistant>"), Some(24));
    let mut engine = GrammarEngine::from_tokenizer(&tokenizer, Some(100352), &[2, 24])
        .expect("actual engine adapter");
    let schema = serde_json::json!({
        "type":"object", "properties":{"answer":{"type":"integer"}},
        "required":["answer"], "additionalProperties":false
    });
    let compiled = engine.compile_json_schema(&schema.to_string()).unwrap();
    for (text, complete) in [
        ("{\"answer\": 4}", true),
        ("{\"answer\": 26}", true),
        ("{\"answer\": -12}", true),
        ("{\"answer\": 1000}", true),
        ("{\"answer\":", false),
        ("{\"answer\": 4", false),
    ] {
        let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        let mut state = GrammarState::new(&compiled, 100352)
            .unwrap()
            .with_stop_tokens(&[2, 24]);
        for &id in &ids {
            state.fill_bitmask();
            assert!(state.is_token_allowed(id), "{text:?}: token {id} refused");
            assert!(
                state.accept_token(id),
                "{text:?}: token {id} did not advance"
            );
        }
        let legal = state.stop_legal(&[2, 24]);
        assert_eq!(legal, complete, "{text:?}");
        assert_eq!(grammar_blocks_stop(Some(&mut state), &[2, 24]), !complete);
    }
}
