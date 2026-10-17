// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The per-request media caps (`--limit-images-per-prompt`,
//! `--limit-videos-per-prompt`): a chat request carrying more images or videos, counted
//! over all its messages, is refused with a 400 before any media is fetched or decoded.
//!
//! Owner: server chat API.
//! Invariants:
//! - A limit of `None` refuses nothing; `Some(0)` refuses any item of that kind.

use crate::ir::{MediaKind, Message};

/// 2026-10-08: The caps, from the serve flags; `None` is no cap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaLimits {
    pub images: Option<usize>,
    pub videos: Option<usize>,
}

/// 2026-10-08: A media kind over its cap: the request parameter it concerns and the
/// message for the 400.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OverLimit {
    pub(crate) param: &'static str,
    pub(crate) message: String,
}

/// 2026-10-08: Check `messages` against `limits`, images first.
pub(crate) fn check_media_limits(
    messages: &[Message],
    limits: MediaLimits,
) -> Result<(), OverLimit> {
    let count = |kind: MediaKind| -> usize {
        messages
            .iter()
            .map(|m| m.media_kinds().into_iter().filter(|k| *k == kind).count())
            .sum()
    };
    for (kind, limit, param) in [
        (MediaKind::Image, limits.images, "image"),
        (MediaKind::Video, limits.videos, "video"),
    ] {
        if let Some(limit) = limit
            && count(kind) > limit
        {
            return Err(OverLimit {
                param,
                message: format!("At most {limit} {param}(s) may be provided in one prompt."),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::message::ImageSource;
    use crate::ir::{ContentPart, ImageData, Role, VideoSource};

    fn user(images: usize, videos: usize) -> Message {
        let mut content = vec![ContentPart::Text("look".into())];
        for _ in 0..images {
            content.push(ContentPart::Image(ImageSource {
                data: ImageData::Base64("x".into()),
            }));
        }
        for _ in 0..videos {
            content.push(ContentPart::Video(VideoSource {
                data: ImageData::Base64("v".into()),
            }));
        }
        Message {
            role: Role::User,
            content,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
            tool_error: false,
        }
    }

    const SERVED: MediaLimits = MediaLimits {
        images: Some(16),
        videos: Some(0),
    };

    #[test]
    fn images_are_counted_across_every_message_of_the_request() {
        let at_cap = [user(10, 0), user(6, 0)];
        assert_eq!(check_media_limits(&at_cap, SERVED), Ok(()));
        let over = [user(10, 0), user(7, 0)];
        assert_eq!(
            check_media_limits(&over, SERVED),
            Err(OverLimit {
                param: "image",
                message: "At most 16 image(s) may be provided in one prompt.".into(),
            })
        );
    }

    #[test]
    fn a_zero_cap_refuses_any_item_and_no_cap_refuses_none() {
        let one_video = [user(0, 1)];
        assert_eq!(
            check_media_limits(&one_video, SERVED).map_err(|e| e.param),
            Err("video")
        );
        let unlimited = MediaLimits {
            images: None,
            videos: None,
        };
        assert_eq!(check_media_limits(&[user(500, 9)], unlimited), Ok(()));
        assert_eq!(check_media_limits(&[user(0, 0)], SERVED), Ok(()));
    }
}
