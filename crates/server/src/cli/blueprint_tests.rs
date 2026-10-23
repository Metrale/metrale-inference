// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met blueprint` against the real tree: every recipe projects and renders back to
//! its own argv; the golden instances' plans reproduce the committed plan digests; and the GLM
//! recipe builds, signs, verifies, resolves its embedded kernels and converts lite ↔ full with
//! one id. Keys are generated here from fixed test seeds.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::cli::blueprint::DECODE_MEMORY;
use metrale_abstract_io::UnixMs;
use metrale_blueprint::frame::{self, KernelSource};
use metrale_blueprint::{recipe_view, resolve};
use metrale_blueprint_trust::{
    Anchor, KeyCert, Kid, Revoked, Scope, Signer, Suite, TrustAnchors, TrustPolicy, certify,
};

use super::*;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn recipes() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for fam in std::fs::read_dir(repo().join("recipes")).unwrap() {
        let fam = fam.unwrap().path();
        if fam.is_dir() {
            for f in std::fs::read_dir(&fam).unwrap() {
                let f = f.unwrap().path();
                if f.extension().is_some_and(|e| e == "yaml") {
                    out.push(f);
                }
            }
        }
    }
    out.sort();
    out
}

#[test]
fn every_recipe_projects_and_renders_back_to_its_own_argv() {
    let all = recipes();
    assert!(all.len() >= 36, "found {} recipes", all.len());
    let mut metrale = 0;
    for path in all {
        let r = recipe_at(&path).unwrap();
        let proj = recipe_view::project(&parts_of(&r).unwrap())
            .unwrap_or_else(|e| panic!("{}: {e}", r.id));
        let back = recipe_of(proj.render());
        assert_eq!(
            (&back.model, back.min_nodes, &back.defaults, &back.env),
            (&r.model, r.min_nodes, &r.defaults, &r.env),
            "{}",
            r.id
        );
        if r.is_metrale() {
            metrale += 1;
            let none = BTreeMap::new();
            assert_eq!(
                back.argv(&none).unwrap(),
                r.argv(&none).unwrap(),
                "{}",
                r.id
            );
        }
    }
    assert!(metrale > 0);
}

/// 2026-10-22: Negative control: a projection that drops one env lever renders a different
/// recipe, so the equality above can fail.
#[test]
fn a_lossy_projection_would_be_caught() {
    let r = recipe_at(&repo().join("recipes/glm-5.3/glm-5.3-flash-nvfp4-tp3-512k.yaml")).unwrap();
    let mut proj = recipe_view::project(&parts_of(&r).unwrap()).unwrap();
    proj.settings.levers.pop_first();
    assert_ne!(recipe_of(proj.render()).env, r.env);
}

fn golden_digest(file: &str) -> String {
    let text = std::fs::read_to_string(repo().join("kernels/circuits/plans").join(file)).unwrap();
    text.lines()
        .find_map(|l| l.strip_prefix("digest: "))
        .unwrap()
        .to_string()
}

#[test]
fn golden_instances_reproduce_their_committed_plan_digests() {
    let insts: Vec<_> =
        metrale_circuit::parse_instances(metrale_model_layers::circuit_exec::sources::INSTANCES)
            .unwrap()
            .into_iter()
            .filter(|i| i.golden)
            .collect();
    assert_eq!(insts.len(), 4);
    for inst in insts {
        let loaded =
            metrale_circuit::load(&inst, super::super::circuit::sources(&inst).unwrap()).unwrap();
        // 2026-10-22: The committed plans were rendered offline (`met circuit show`: every kernel
        // a rule names counts as built), so they are reproduced under that availability.
        let avail = metrale_circuit::AvailableKernels::all_named_by(&loaded.rules);
        let (plans, gaps) = fuse_plans(&inst, &loaded, &avail).unwrap();
        assert!(gaps.is_empty());
        let arms: Vec<(metrale_circuit::Mode, u64)> = inst
            .plans
            .iter()
            .flat_map(|(m, rs)| rs.iter().map(move |r| (*m, *r)))
            .collect();
        assert_eq!(plans.len(), arms.len());
        for (p, (m, r)) in plans.iter().zip(arms) {
            assert_eq!(
                p.digest,
                golden_digest(&inst.plan_file(m, r)),
                "{} {} n{r}",
                inst.recipe,
                m.name()
            );
        }
    }
}

fn keys() -> (Signer, Signer, TrustAnchors) {
    let root = Signer::from_seeds(
        Kid::new("test-root").unwrap(),
        Suite::MlDsa87Ed25519,
        &[1; 32],
        &[2; 32],
    );
    let leaf = Signer::from_seeds(
        Kid::new("test-blueprints").unwrap(),
        Suite::MlDsa65Ed25519,
        &[3; 32],
        &[4; 32],
    );
    let anchors = TrustAnchors::new(vec![Anchor {
        kid: root.kid().clone(),
        keys: root.public(),
    }])
    .unwrap();
    (root, leaf, anchors)
}

/// 2026-10-22: The GLM recipe built and signed on the kernels this binary embeds. The recipe's
/// chat template lives on the measuring box, so this serves a copy of the recipe whose template
/// is a temp file under `dir`; everything else is the recipe as committed.
pub(crate) struct Glm {
    pub args: super::super::blueprint_args::BuildArgs,
    pub recipe_path: PathBuf,
    pub bp: metrale_blueprint::Blueprint,
    pub unsigned: frame::Unsigned,
    pub lite: Vec<u8>,
    pub root: Signer,
    pub leaf: Signer,
    pub anchors: TrustAnchors,
    pub cert: metrale_blueprint_trust::SignedCert,
}

pub(crate) fn glm(dir: &Path) -> Glm {
    let tpl = dir.join("t.jinja");
    std::fs::write(&tpl, "{{ messages }}").unwrap();
    let text =
        std::fs::read_to_string(repo().join("recipes/glm-5.3/glm-5.3-flash-nvfp4-tp3-512k.yaml"))
            .unwrap();
    let text = text.replace(
        "/home/claude/glm-bench/glm53-chat-template.jinja",
        tpl.to_str().unwrap(),
    );
    let fam = dir.join("glm-5.3");
    std::fs::create_dir(&fam).unwrap();
    let recipe_path = fam.join("glm-5.3-flash-nvfp4-tp3-512k.yaml");
    std::fs::write(&recipe_path, &text).unwrap();
    let args = super::super::blueprint_args::BuildArgs {
        recipe: recipe_path.clone(),
        checkpoint: repo()
            .join("crates/circuit/tests/fixtures/checkpoints/nvidia--GLM-5.3-Flash-NVFP4"),
        checkpoint_revision: "da920bb0b9f4a06727223a349e55468e38352348".into(),
        weights_digest: "skip".into(),
        blueprint_revision: 1,
        min_visible_mib: 98304,
        out: super::super::blueprint_args::OutArgs {
            output: dir.join("x.kbp"),
            text: false,
        },
    };
    let (bp, unsigned) = build(&args).unwrap();
    let (root, leaf, anchors) = keys();
    let cert = certify(
        &root,
        &KeyCert {
            kid: leaf.kid().clone(),
            keys: leaf.public(),
            scope: Scope::Blueprints,
            not_before_ms: 0,
            not_after_ms: i64::MAX,
            serial: 1,
        },
    )
    .unwrap();
    let a = frame::encode_unsigned(&bp, &unsigned, &[]).unwrap();
    let lite = frame::sign(&a, &leaf, vec![cert.clone()], DECODE_MEMORY).unwrap();
    Glm {
        args,
        recipe_path,
        bp,
        unsigned,
        lite,
        root,
        leaf,
        anchors,
        cert,
    }
}

/// 2026-10-22: The GLM recipe end to end: builds deterministically, signs, verifies, resolves
/// its embedded kernels and converts lite ↔ full with one id.
#[test]
fn the_glm_recipe_builds_signs_verifies_and_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let Glm {
        args,
        recipe_path,
        bp,
        unsigned,
        lite,
        anchors,
        ..
    } = glm(dir.path());
    assert!(
        bp.kernels.modules.len() > 100,
        "{} modules",
        bp.kernels.modules.len()
    );
    assert!(!bp.kernels.specs.is_empty());
    assert!(bp.policy.chat_template.is_some());
    let a = frame::encode_unsigned(&bp, &unsigned, &[]).unwrap();
    assert_eq!(
        a,
        frame::encode_unsigned(&build(&args).unwrap().0, &unsigned, &[]).unwrap(),
        "deterministic build"
    );
    let revoked = Revoked::embedded().unwrap();
    let policy = TrustPolicy {
        anchors: &anchors,
        revoked: &revoked,
        now: UnixMs(1_760_000_000_000),
    };
    let o = frame::open(&lite, &policy, DECODE_MEMORY).unwrap();
    let modules = embedded(&o.blueprint.kernels.target).unwrap();
    let internal = |n: &str| modules.iter().find(|(m, _)| *m == n).map(|(_, b)| *b);
    assert_eq!(
        resolve::resolve_modules(&o, internal).unwrap().len(),
        bp.kernels.modules.len()
    );

    let (u, p) = resolve::full(&o, internal).unwrap();
    let full = frame::rewrite(&lite, &policy, DECODE_MEMORY, &u, &p).unwrap();
    let f = frame::open(&full, &policy, DECODE_MEMORY).unwrap();
    assert_eq!(f.id, o.id);
    assert!(
        f.unsigned
            .sources
            .iter()
            .all(|s| matches!(s, KernelSource::Inline { .. }))
    );
    assert!(full.len() > lite.len());
    let (u, p) = resolve::lite(&f);
    assert_eq!(
        frame::rewrite(&full, &policy, DECODE_MEMORY, &u, &p).unwrap(),
        lite
    );

    let original = recipe_at(&recipe_path).unwrap();
    let back = recipe_of(recipe_view::recipe_parts(
        &f.blueprint,
        &f.unsigned.resources,
    ));
    let none = BTreeMap::new();
    assert_eq!(back.argv(&none).unwrap(), original.argv(&none).unwrap());
    assert_eq!(back.env, original.env);
}
