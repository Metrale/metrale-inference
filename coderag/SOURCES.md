# CodeRAG corpora: sources and credits

Built by `.github/workflows/coderag.yml`. Each corpus is a self-verifying
lattice-jsonl file; its `.meta.json` names the repository, commit and
embedding model it was built from.

| corpus | repository | commit | model | points | built |
|---|---|---|---|---|---|
| books-rust | rust-lang/book | 1500248d8f230566e4ec9f27fcbb8fe9e2898ab1 | nvidia/llama-nemotron-embed-vl-1b-v2:free | 828 | 2026-10-11T21:54:58.867Z |
| metrale-code-crates-model | Metrale/metrale-inference | a012b442d769f88503a711fbf2da819d72c7828f | nvidia/llama-nemotron-embed-vl-1b-v2:free | 2712 | 2026-10-12T00:03:56.710Z |
| metrale-code-crates | Metrale/metrale-inference | a012b442d769f88503a711fbf2da819d72c7828f | nvidia/llama-nemotron-embed-vl-1b-v2:free | 2536 | 2026-10-11T23:59:04.203Z |
| metrale-code-kernels | Metrale/metrale-inference | a012b442d769f88503a711fbf2da819d72c7828f | nvidia/llama-nemotron-embed-vl-1b-v2:free | 1376 | 2026-10-12T00:06:35.640Z |
| metrale-coderag | Metrale/metrale-inference | 246f71bf5da3a0a984d2ab9273ab22c0abdd371e | nvidia/llama-nemotron-embed-vl-1b-v2:free | 5997 | 2026-09-27T21:12:12.711Z |
| metrale-docs | Metrale/metrale-inference | a012b442d769f88503a711fbf2da819d72c7828f | nvidia/llama-nemotron-embed-vl-1b-v2:free | 1013 | 2026-10-12T00:06:56.569Z |

## Third-party content

- **books-rust** is chunked from *The Rust Programming Language* (https://github.com/rust-lang/book), pinned at commit `1500248d8f230566e4ec9f27fcbb8fe9e2898ab1`, licensed MIT OR Apache-2.0. Copyright the Rust Project Developers; the licence texts are LICENSE-MIT and LICENSE-APACHE in that repository at the pinned commit. The text is embedded unmodified, and every chunk names its source file and line range.
