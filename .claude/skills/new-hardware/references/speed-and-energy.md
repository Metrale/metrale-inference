# Speed and energy are separate metrics

Finishing a request sooner does not mean using less energy: power draw can be higher. Never
infer energy from speed, or speed from energy. Measure and report both, every time: tok/s and
TTFT, and J/tok from the NVML energy counter.

Customers weigh them differently: some want only speed (the end-user experience), some want
only energy and accept a slower experience. Winning both at once is achievable and is the aim
(it is the state against vLLM on the GB10 reference box), but each is recorded on its own.

## The arithmetic

Energy is power integrated over time, `E = ∫P(t)dt`, and `J/tok = E / tokens`.

- Engine A takes `t`, engine B takes `t/k` (`k > 1`, B is faster). With `P̄` the average power
  over each run: `E_A = P̄_A·t`, `E_B = P̄_B·t/k`. **B uses less energy only if
  `P̄_B < k·P̄_A`**: the power ratio must be below the speed ratio.
- Per token: `J/tok = P̄ / (tok/s)`. So `J/tok_B < J/tok_A` exactly when
  `P̄_B / P̄_A < S_B / S_A` (S = tok/s).
- Over a shared fixed window `T ≥ t` that includes the idle tail at `P_idle`:
  `E_B − E_A = t·[(P̄_B − P_idle)/k − (P̄_A − P_idle)]`. B wins exactly when
  `(P̄_B − P_idle) < k·(P̄_A − P_idle)`: only the above-idle power has to beat the speedup. So
  the two engines' accounting windows must match, or the comparison measures the window, not
  the engine (the PARITY-O.R.A.C.L.E's energy-accounting item).
- **A measured case** (GB10 reference box, the NVFP4 MoE at C128, PR #120 against vLLM
  0.31.0): 1042.4 vs 908.9 tok/s (`k = 1.147`), 0.0546 vs 0.0535 J/tok, so average power
  ≈ 56.9 W vs 48.6 W (power ratio 1.170 > k): **14.7 % faster and 2 % more energy per token.**

**The practical rule for the loop:** when a rung is faster but loses J/tok, the lever to look for
reduces POWER (bytes moved, MMA width, switching activity), not time.

## Energy accounting must match

Both engines' J/tok must use the same integration and accounting, or the comparison is a
PARITY-BLOCK:
- **the window**: each engine's own measured run window read at its edges, or one fixed wall
  window that includes the idle tail; the same choice for both;
- **the source**: the NVML cumulative energy counter, not sampled `power.draw` (which reads a
  few percent high under load);
- **idle / baseline power**: included on both sides or subtracted on both sides;
- **tokens**: completion tokens actually produced, counted the same way; an early stop or a
  timeout truncates output and inflates J/tok.

## Victory per objective

The bring-up ledger records two milestones, not one:
- **TTPV-speed**: the first same-box scoreboard on which Metrale is faster (tok/s) at every
  rung C1-C128;
- **TTPV-energy**: the first on which Metrale is cheaper (J/tok) at every rung but at most one
  mid-ladder rung lost by a small margin;
- **TTPV** is the later of the two.

## Trade-off levers

A lever that wins speed but costs energy, or the reverse, is never kept silently and never folded
into the default. Record it with both deltas, then offer it as a flagged, recipe-level choice
(default off), disclosed on every record it produces.
