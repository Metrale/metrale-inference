// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: [`waiting_entries`] on the forms nvcc emits: a waiting entry, an entry that only
//! triggers its dependents, a wait in a `.func` after an entry, and a commented-out wait.

use super::waiting_entries;

const MODULE: &str = "\
.version 9.0
.target sm_121a
.address_size 64

\t// .globl\twaits
.visible .entry waits(
\t.param .u64 waits_param_0
)
{
\tld.global.nc.u32 %r1, [%rd1];
\tgriddepcontrol.wait;
\tgriddepcontrol.launch_dependents;
\tret;
}
.visible .entry triggers_only(
\t.param .u64 triggers_only_param_0
)
{
\tgriddepcontrol.launch_dependents;
\t// griddepcontrol.wait;
\tret;
}
.func helper(
\t.param .u64 helper_param_0
)
{
\tgriddepcontrol.wait;
\tret;
}
.entry plain_after_func(
\t.param .u64 p
)
{
\tret;
}
.weak .entry also_waits(
)
{
\tgriddepcontrol.wait;
}
";

#[test]
fn only_entries_with_their_own_wait_are_listed() {
    let got = waiting_entries(MODULE);
    let mut v: Vec<&str> = got.iter().map(String::as_str).collect();
    v.sort();
    assert_eq!(v, ["also_waits", "waits"]);
}

#[test]
fn a_wait_in_a_func_does_not_mark_the_entry_before_it() {
    let got = waiting_entries(MODULE);
    assert!(!got.contains("triggers_only"), "a trigger is not a wait");
    assert!(!got.contains("helper"), "a .func is not an entry");
    assert!(!got.contains("plain_after_func"));
}

#[test]
fn a_module_without_waits_lists_nothing() {
    assert!(waiting_entries(".entry k(\n)\n{\n\tret;\n}\n").is_empty());
    assert!(waiting_entries("").is_empty());
}
