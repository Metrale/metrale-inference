// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met blueprint`: the I/O side of metrale-blueprint. Files in and out, the trust
//! policy (compiled anchors, or a declared dev anchor), the clock, and the embedded kernels.
//!
//! Owner: server CLI (blueprint).
//! Invariants:
//! - Every command that reads a blueprint opens it through `frame::open`, so nothing is decoded
//!   before it verifies.
//! - A dev anchor is announced on stderr as the declared mode `trust = dev-anchor`.
//! - Keys come from files only (metrale-blueprint-sign).

use std::path::Path;

use anyhow::{Context, Result};
use metrale_abstract_io::{Clock as _, UnixMs};
use metrale_blueprint::export::{self, ExportContext};
use metrale_blueprint::frame::{self, Opened};
use metrale_blueprint::types::Mode;
use metrale_blueprint::{recipe_view, resolve, transport};
use metrale_blueprint_sign::{read_key, read_public, write_public};
use metrale_blueprint_trust::{
    Anchor, KeyCert, Kid, Revocations, Revoked, Scope, SignedRevocations, TrustAnchors,
    TrustPolicy, certify, sign_revocations, verify_revocations,
};

use super::blueprint_args::{
    BlueprintAction, BlueprintArgs, ExportArgs, KeyAction, OutArgs, TrustArgs,
};
use super::blueprint_build as build;

fn now() -> UnixMs {
    metrale_io_host_std::SystemClock::new().now_unix_ms()
}

/// 2026-10-22: A blueprint file, binary or `kbp1.` text.
fn read_kbp(path: &Path) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.starts_with(transport::PREFIX.as_bytes()) {
        return Ok(transport::from_text(std::str::from_utf8(&bytes)?)?);
    }
    Ok(bytes)
}

fn write_kbp(out: &OutArgs, file: &[u8]) -> Result<()> {
    let bytes = if out.text {
        let mut t = transport::to_text(file);
        t.push('\n');
        t.into_bytes()
    } else {
        file.to_vec()
    };
    std::fs::write(&out.output, &bytes)
        .with_context(|| format!("writing {}", out.output.display()))?;
    eprintln!("wrote {} ({} bytes)", out.output.display(), bytes.len());
    Ok(())
}

/// 2026-10-22: The anchors and revocation state `t` names.
fn trust(t: &TrustArgs) -> Result<(TrustAnchors, Revoked)> {
    let anchors = match &t.trust_anchor {
        Some(p) => {
            let a: Anchor = read_public(p)?;
            eprintln!(
                "trust = dev-anchor: {} from {} (declared mode; not a compiled anchor)",
                a.kid,
                p.display()
            );
            TrustAnchors::new(vec![a])?
        }
        None => TrustAnchors::compiled()?,
    };
    let revoked = match &t.revocations {
        Some(p) => {
            let signed: SignedRevocations = read_public(p)?;
            verify_revocations(
                &anchors,
                &signed,
                metrale_blueprint_trust::REVOCATIONS_FLOOR,
            )?
        }
        None => Revoked::embedded()?,
    };
    Ok((anchors, revoked))
}

fn open_with(path: &Path, t: &TrustArgs) -> Result<(Vec<u8>, Opened, TrustAnchors, Revoked)> {
    let file = read_kbp(path)?;
    let (anchors, revoked) = trust(t)?;
    let policy = TrustPolicy {
        anchors: &anchors,
        revoked: &revoked,
        now: now(),
    };
    let opened = frame::open(&file, &policy)?;
    Ok((file, opened, anchors, revoked))
}

fn internal_of(o: &Opened) -> Result<impl Fn(&str) -> Option<&'static [u8]>> {
    let modules = build::embedded(&o.blueprint.kernels.target)?;
    Ok(move |name: &str| modules.iter().find(|(n, _)| *n == name).map(|(_, b)| *b))
}

fn summary(o: &Opened, file_len: usize) -> String {
    let bp = &o.blueprint;
    let full = o.header.flags & frame::FLAG_FULL != 0;
    let mut s = format!(
        "blueprint {}\n  name      {} (revision {}, lineage {})\n  form      {} ({} bytes, body {} bytes)\n  signed    {} under {} ({})\n",
        o.id,
        bp.name,
        bp.revision,
        bp.lineage.0.hex(),
        if full { "full" } else { "lite" },
        file_len,
        o.header.body_len,
        o.signer.kid,
        o.signer.anchor,
        o.signer.cert.keys.suite.name(),
    );
    s += &format!(
        "  model     {}@{} ({}, {} layers)\n  hardware  {} {} x{} (>= {} MiB)\n  execution {}\n",
        bp.checkpoint.repo,
        bp.checkpoint.revision,
        bp.circuit.arch,
        bp.circuit.layer_kinds.len(),
        bp.hardware.class,
        bp.hardware.ptx_arch,
        bp.hardware.world_size,
        bp.hardware.min_visible_mib,
        bp.execution.label(),
    );
    s += &format!(
        "  kernels   {} modules ({} bytes), {} planned entries, target {}/{}/{}\n  plans     {}\n  levers    {} set, {} passthrough; {} flags\n",
        bp.kernels.modules.len(),
        bp.kernels.modules.iter().map(|m| m.len).sum::<u64>(),
        bp.kernels.specs.len(),
        bp.kernels.target.arch,
        bp.kernels.target.model,
        bp.kernels.target.quant,
        bp.plans
            .iter()
            .map(|p| format!("{}:{}", p.mode.name(), p.rows))
            .collect::<Vec<_>>()
            .join(" "),
        bp.settings.levers.len(),
        bp.settings.passthrough.len(),
        bp.policy.flags.len(),
    );
    if !bp.plan_gaps.is_empty() {
        s += &format!(
            "  gaps      {} (listed arms the rules do not cover)\n",
            bp.plan_gaps
                .iter()
                .map(|g| format!("{}:{}", g.mode.name(), g.rows))
                .collect::<Vec<_>>()
                .join(" "),
        );
    }
    s
}

/// 2026-10-22: The recipe YAML a blueprint projects to (JSON-quoted scalars are valid YAML).
fn recipe_yaml(o: &Opened) -> Result<(String, Vec<String>)> {
    let parts = recipe_view::recipe_parts(&o.blueprint, &o.unsigned.resources);
    let q = |s: &str| serde_json::to_string(s).expect("a string serializes");
    let mut y = format!(
        "# projected from blueprint {}\nrecipe_version: {}\nmodel: {}\n",
        o.id,
        q(&parts.version),
        q(&parts.model)
    );
    if let Some(r) = &parts.runtime {
        y += &format!("runtime: {}\n", q(r));
    }
    y += &format!(
        "container: {}\nmin_nodes: {}\ndefaults:\n",
        q(&parts.container),
        parts.min_nodes
    );
    for (k, v) in &parts.defaults {
        y += &format!("  {k}: {}\n", q(v));
    }
    y += "env:\n";
    for (k, v) in &parts.env {
        y += &format!("  {k}: {}\n", q(v));
    }
    let recipe = crate::recipe::Recipe::parse(parts.id.clone(), &y)?;
    let argv = recipe.argv(&Default::default())?;
    Ok((y, argv))
}

fn parse_arm(s: &str) -> Result<(Mode, u64)> {
    let (m, r) = s.split_once(':').context("--arm is mode:rows")?;
    Ok((
        Mode::parse(m).with_context(|| format!("unknown mode `{m}`"))?,
        r.parse()?,
    ))
}

fn export_cmd(a: &ExportArgs) -> Result<()> {
    let value = if let Some(which) = &a.schema {
        match which.as_str() {
            "card" => export::card_schema(),
            _ => export::graph_schema(),
        }
    } else {
        let input = a.input.as_deref().context("an input blueprint")?;
        let (file, o, _, _) = open_with(input, &a.trust)?;
        let ctx = ExportContext {
            exported_by: super::METRALE_VERSION.to_string(),
            verified_at_ms: now().0,
            file_len: file.len() as u64,
        };
        if a.card {
            serde_json::to_value(export::card::card(&o, &ctx))?
        } else {
            let arm = parse_arm(a.arm.as_deref().context("--graph needs --arm")?)?;
            serde_json::to_value(export::graph::graph(&o, arm, &ctx)?)?
        }
    };
    let mut text = serde_json::to_string_pretty(&value)?;
    text.push('\n');
    std::fs::write(&a.output, &text).with_context(|| format!("writing {}", a.output.display()))?;
    eprintln!("wrote {} ({} bytes)", a.output.display(), text.len());
    Ok(())
}

fn key_cmd(k: KeyAction) -> Result<()> {
    match k {
        KeyAction::Anchor { key, output } => {
            let s = read_key(&key)?;
            write_public(
                &output,
                &Anchor {
                    kid: s.kid().clone(),
                    keys: s.public(),
                },
            )?;
        }
        KeyAction::Certify {
            root,
            key,
            scope,
            not_before_ms,
            not_after_ms,
            serial,
            output,
        } => {
            let leaf = read_key(&key)?;
            let cert = KeyCert {
                kid: leaf.kid().clone(),
                keys: leaf.public(),
                scope: Scope::parse(&scope).context("scope")?,
                not_before_ms,
                not_after_ms,
                serial,
            };
            write_public(&output, &certify(&read_key(&root)?, &cert)?)?;
        }
        KeyAction::Revoke {
            root,
            serial,
            kid,
            output,
        } => {
            let revoked = kid
                .iter()
                .map(|k| Kid::new(k))
                .collect::<Result<Vec<_>, _>>()?;
            write_public(
                &output,
                &sign_revocations(&read_key(&root)?, &Revocations { serial, revoked })?,
            )?;
        }
    }
    Ok(())
}

/// 2026-10-22: Run `met blueprint`.
pub(crate) fn dispatch(args: BlueprintArgs) -> Result<()> {
    match args.action {
        BlueprintAction::FromRecipe(b) => {
            let (bp, unsigned) = build::build(&b)?;
            write_kbp(&b.out, &frame::encode_unsigned(&bp, &unsigned, &[])?)
        }
        BlueprintAction::Build { build: b, full } => {
            let (bp, unsigned) = build::build(&b)?;
            if !full {
                return write_kbp(&b.out, &frame::encode_unsigned(&bp, &unsigned, &[])?);
            }
            let modules = build::embedded(&bp.kernels.target)?;
            let bytes = bp
                .kernels
                .modules
                .iter()
                .map(|m| {
                    modules
                        .iter()
                        .find(|(n, _)| *n == m.name)
                        .map(|(_, b)| *b)
                        .context("embedded module")
                })
                .collect::<Result<Vec<_>>>()?;
            let (sources, payload) = resolve::inline_all(&bytes)?;
            let unsigned = frame::Unsigned {
                sources,
                resources: unsigned.resources,
            };
            write_kbp(&b.out, &frame::encode_unsigned(&bp, &unsigned, &payload)?)
        }
        BlueprintAction::Sign {
            input,
            key,
            cert,
            out,
        } => write_kbp(
            &out,
            &metrale_blueprint_sign::sign_kbp(&read_kbp(&input)?, &key, &cert)?,
        ),
        BlueprintAction::Verify { input, trust } => {
            let (file, o, _, _) = open_with(&input, &trust)?;
            let internal = internal_of(&o)?;
            let modules = resolve::resolve_modules(&o, internal)?;
            print!("{}", summary(&o, file.len()));
            println!(
                "verified: header, body hash, chain, ML-DSA + Ed25519, decode, {} kernel modules by hash",
                modules.len()
            );
            Ok(())
        }
        BlueprintAction::Show {
            input,
            trust,
            format,
        } => {
            let (file, o, _, _) = open_with(&input, &trust)?;
            match format.as_str() {
                "text" => print!("{}", summary(&o, file.len())),
                "json" => {
                    let ctx = ExportContext {
                        exported_by: super::METRALE_VERSION.to_string(),
                        verified_at_ms: now().0,
                        file_len: file.len() as u64,
                    };
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&export::card::card(&o, &ctx))?
                    );
                }
                _ => print!("{}", recipe_yaml(&o)?.0),
            }
            Ok(())
        }
        BlueprintAction::ToRecipe { input, trust } => {
            let (_, o, _, _) = open_with(&input, &trust)?;
            let (yaml, argv) = recipe_yaml(&o)?;
            print!("{yaml}");
            println!("# argv: {}", argv.join(" "));
            Ok(())
        }
        BlueprintAction::Lite { input, trust, out } => {
            let (file, o, anchors, revoked) = open_with(&input, &trust)?;
            let (u, p) = resolve::lite(&o);
            let policy = TrustPolicy {
                anchors: &anchors,
                revoked: &revoked,
                now: now(),
            };
            write_kbp(&out, &frame::rewrite(&file, &policy, &u, &p)?)
        }
        BlueprintAction::Full { input, trust, out } => {
            let (file, o, anchors, revoked) = open_with(&input, &trust)?;
            let (u, p) = resolve::full(&o, internal_of(&o)?)?;
            let policy = TrustPolicy {
                anchors: &anchors,
                revoked: &revoked,
                now: now(),
            };
            write_kbp(&out, &frame::rewrite(&file, &policy, &u, &p)?)
        }
        BlueprintAction::Export(a) => export_cmd(&a),
        BlueprintAction::Key(k) => key_cmd(k),
    }
}
