# The improvement loop: iterate until Metrale beats vLLM

The bring-up does not end at "baseline measured". After the same-box vLLM baseline exists, run
this loop. **It may not start until bit parity is achieved and recorded** (Tiers 1 and 2 on the
real model, `references/bit-parity.md`), and every iteration keeps it. One iteration is one lever; it ends with the lever kept or discarded and the evidence
recorded either way.

## One iteration

**a. Measure.** Same box, same harness, same instrument as the recorded vLLM baseline, at every
rung C1, C2, C4, C8, C16, C32, C64, C128: tok/s, J/tok (NVML energy counter, GPU rail) and TTFT
(cold and warm). The PARITY-O.R.A.C.L.E rules on the configs before the numbers count. Write
the scoreboard: one row per rung, Metrale vs vLLM, ratio, win or loss on each axis.

**b. Profile the worst losing rung.** Pick the rung with the largest loss (tok/s first, then
J/tok). nsys at that rung on a fresh serve:
- per-kernel-class time and power: which planned group (GEMV/GEMM tiers, attention, recurrent
  state, MoE experts, quantizers, norms) takes the step, and at what draw;
- the burst vs steady split: prefill bursts, graph replays, the steady decode step;
- host gaps: time between kernel launches with the device idle (scheduler, sampling, Python-free
  but still serial host work).

**c. Rank and choose the lever.** Rank the gaps by the time (or energy) they cost at that rung.
For the top gap, prefer in this order:
1. a **parameterization** of an existing kernel the Venn already lists (a new point of a
   template, a policy of the WxAy engine, a tile tier extended to these rows);
2. a **routing or policy** change that selects an existing better kernel (a rule, a
   `[defaults]` row, a `[tensor_core_policy]` backlog entry closed);
3. a **split** that lets the auto-fuser recombine pieces that already exist;
4. new code, last, and only for a gap the Venn classed `novel`.
Write down, before running anything, the two legs of the A/B and the one variable that differs.

**d. Prove identity (keep bit parity).**
- A lever that keeps precision: strict prefill bits (byte-identical logits at prefill) AND
  greedy decode transcripts over the fixed prompt set, both against a control run of the
  unchanged binary in the same session. A control that changes nothing must show zero
  differences; a lever whose arm also shows zero differences may not have been armed (check
  the runtime predicate, not the startup log line).
- A lever that changes precision: behind a flag, default off, and the accuracy bar (BFCL with
  its draw recorded, agentic-webserver with a same-night control) before it is used.
- A lever that breaks any Tier 1 or Tier 2 check and is not such a flag is discarded, whatever
  it buys.

**e. Timed A/B.** Same box, same binary build directory hygiene, n = 3 interleaved fresh
serves per arm (A B A B A B), die temperature at or below the gate before every serve, one
fresh serve per rung at C >= 64 with a watchdog. Report median and spread; a difference inside
the control spread is no difference.

**f. Keep or discard.** Keep only a lever that wins at its rung without losing another rung
outside the noise band, on both axes. Record the outcome, kept or not, with its numbers, its
profile and the commit, in the campaign PR (a table: iteration, lever, rung, before, after,
verdict). A discarded lever's evidence is as valuable as a kept one's. Update the ledger:
`iterations`, `levers_kept` or `levers_discarded`, and the lines added and removed.

**g. Repeat** from (a): the scoreboard moved, so the worst rung may be a different one.

## Exit criterion

Stop iterating when ALL of these hold on one same-box scoreboard:
- **tok/s**: Metrale is faster than vLLM at EVERY rung C1-C128;
- **J/tok**: Metrale is cheaper at every rung, except at most ONE mid-ladder rung (C4-C32)
  that loses by a small margin (within twice the measured serve-to-serve spread, and never
  more than 5 %).

These keep the loop going:
- any tok/s loss at any rung;
- a J/tok loss at an edge rung (C1/C2 or C64/C128), at any margin;
- J/tok losses at two or more rungs.

A vLLM stall, hang or crash at a rung counts as a Metrale win at that rung only with its
evidence recorded (launch command, image digest, time at zero tok/s with requests running,
logs) and only if a fresh vLLM serve at the same configuration reproduces it; one stall in two
attempts is reported as "intermittent", not as a loss for vLLM.

## Stop and report

Do not loop forever. After **three consecutive iterations without progress**, stop and report
to whoever owns the campaign, with the current scoreboard, the profile of the worst rung, the
levers tried and why each was discarded, and the next candidates. "Progress" means a kept lever,
or a losing rung's margin reduced by more than the noise band. Also stop and report when the
top gap needs a decision you cannot make alone: a precision change, a new kernel family, a
measurement-definition change (those ride their own PR), or a change to another class's
kernels.
