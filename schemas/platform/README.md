# Metrale platform contracts, version 1

Every file in this directory (`schemas/platform/`) is licensed under the Apache License 2.0
(`LICENSE`), separately from the rest of this repository. The schemas say so in their
`$comment` (`SPDX-License-Identifier: Apache-2.0`).

This directory is the single source of the documents that the runtime, the node agent and the
platform exchange. The repositories that implement them copy it unchanged (vendor it) and prove
that the copy is unchanged.

## Contents

| file | what it is |
|---|---|
| `v1/common.v1.schema.json` | definitions shared by every document (ids, digests, scope, `measured`/`derived`/`absent` values, number formats, topology) |
| `v1/<kind>.v1.schema.json` | one JSON Schema (draft 2020-12) per document kind; `v1/INDEX.json` `schemas` lists the kinds |
| `v1/fixtures/<kind>/valid/*.json` | documents every implementation must accept |
| `v1/fixtures/<kind>/invalid/*.json` | documents every implementation must refuse; each one is a valid fixture with one change |
| `v1/INDEX.json` | every fixture with its kind and expected verdict; for an invalid fixture, the rule it breaks, its base fixture and the one change (`op`, `pointer`, `value`); for a valid fixture, the SHA-256 of its canonical JSON |
| `v1/vectors/release-signer.json` | the public keys of a published **test** release signer (its seed is public: 32 bytes of `0x07`); the `document-signature` fixture is its real signature over the `release-manifest` fixture |
| `SHA256SUMS` | the SHA-256 of every other file here, in `sha256sum` format |

The document kinds are `model-recipe`, `hardware-fingerprint`, `model-deployment`,
`deployment-action`, `deployment-action-result`, `runtime-health`, `inference-usage`,
`audit-event`, `release-manifest`, `compatibility-receipt`, `document-signature`,
`revocation-list` and `error`. Every value in the fixtures is synthetic: digests, commits and ids
are made up and refer to nothing. The one exception is the signature fixture, which is a real
signature by the published test signer, so a consumer's verifier can be checked against it
(`docs/contracts/PLATFORM.md`, "Release signatures and revocation lists").

## Rules every document follows

- `version` is an integer. A reader refuses a version it does not implement.
- Every object is closed (no unknown keys) and every key is present. A key that may have no value
  is written as `null`; nothing is defaulted.
- A quantity that may be unmeasured is `{"measured": n}`, `{"derived": n}` or
  `{"absent": "<reason>"}`. Absent is never written as zero.
- Numbers are integers. The unit is part of the field name (`_unix_ms`, `_ms`, `_us`, `_bytes`,
  `_permille`, `millijoules`). Timestamps are UTC Unix milliseconds and are never negative.
- Ids and digests are hex. Writers emit lower case; readers accept either case. A `trace_id` is
  16 bytes (the W3C trace-id width) and is never all zero.
- A model revision or runtime build is a full git commit. An image is a repository plus a
  `sha256:` digest. A branch, a tag or `latest` never identifies anything.
- A tenant, project, deployment name or release name is a DNS label.
- The digest of a document is the SHA-256 of its canonical JSON (RFC 8785). Because no document
  holds a floating-point number, canonical JSON here means: object keys sorted by UTF-16 code
  units, no whitespace, integers in plain decimal, and strings escaping only `"`, `\` and
  control characters (`\b \t \n \f \r` in short form, the rest as `\u00xx`).

## How a consuming repository uses this directory

1. Copy `schemas/platform/` from a Metrale/mfs commit into the repository, without changes.
2. Commit a pin file next to the copy that records that mfs commit and the SHA-256 of
   `SHA256SUMS`.
3. Add a CI check that:
   - runs `sha256sum -c SHA256SUMS` in the copy (`shasum -a 256 -c` on macOS);
   - compares the SHA-256 of `SHA256SUMS` with the pin;
   - checks that no file in the copy is missing from `SHA256SUMS`;
   - runs the repository's validator over every entry of `v1/INDEX.json`. The validator must
     accept every `valid` entry and refuse every `invalid` one. Where the repository can
     compute canonical JSON, it must also reproduce each `jcs_sha256`.

   Any JSON Schema 2020-12 validator works. The schemas reference `common.v1.schema.json` by
   relative `$ref`, so register every file under its `$id` before validating. A repository that
   has its own types for these documents also runs its parser over the fixtures. The verdicts
   must match the index.
4. To take a newer version, copy the directory again and update the pin in one commit. Never
   edit the copy by hand.

## Changing a contract

Changes happen in Metrale/mfs only. That repository's own types are checked against these
schemas over the same fixtures (`crates/contracts/tests/platform`).

A change that every existing reader can still parse goes into the current version. Because
every object is closed, adding a field is never such a change. Any other change creates the
next version (`v2/`), next to `v1/`, which stays unchanged. Writers move to the new version
only after every reader implements it.
