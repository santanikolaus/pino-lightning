# UNet3D TTA arm — Re100→500 target definition

Forward-only reads, no adaptation. Fixes what adaptation is aiming at before any
gradient step is taken.

## Runs

| role | ckpt | data | npz |
|------|------|------|-----|
| in-dist anchor | 3z5bxjzp (Re100, 300ep) | Re100 res128 | `msc/tta/outputs/report/unet_3z5bxjzp_re100.npz` |
| OOD start | 3z5bxjzp | Re500 res128 | `msc/tta/outputs/report/unet_3z5bxjzp_re500.npz` |
| native reference | qr6rs0jb (Re500, 300ep) | Re500 res128 | `msc/tta/outputs/report/unet_qr6rs0jb_re500.npz` |

`msc.tta.eval.report`, test [270:300] n=30, T_eff=65, default bands. Three
forward-only reads of the locked split, 2026-08-21. Model: UNet3D base64 depth3 +
`SpatialSpectralMixer` at the S/8 bottleneck, modes=8 — no spectral truncation on
the conv path, unlike the FNO's `n_modes=8`.

## The reference is not the Re100 column

Re100 in-dist overstates what adaptation could reach: Re500 is intrinsically less
predictable. Pooled k1-64, frames 1-64:

| | rel_l2 | ρ | γ | ρ<0.9 horizon |
|---|---|---|---|---|
| Re100 in-dist | 0.2200 | 0.9755 | 0.9767 | 63.2 |
| **Re500 native (reference)** | **0.3529** | **0.9357** | **0.9272** | **49.9** |
| Re500 OOD (start) | 0.4737 | 0.8849 | 0.7988 | 34.1 |

Recoverable: **0.1208 rel_l2, +15.8 frames of horizon.** One checkpoint, one seed —
an empirical reference, not a bound: the OOD model already beats the native one at
t≤5, at k≥13, and on width-corrected W1 (0.0266 vs 0.0410).

## Where in t

Amplitude-only correction bottoms out at `rel_l2 = sqrt(1-ρ²)` (γ→ρ, not γ→1).

| window | reference | OOD | gap | amp floor | amp closes |
|---|---|---|---|---|---|
| t1-5 | 0.1555 | 0.1501 | **−0.0054** | 0.1010 | no gap |
| t1-10 | 0.1662 | 0.1881 | 0.0220 | 0.1462 | 100 % |
| t1-15 | 0.1797 | 0.2245 | 0.0448 | 0.1886 | 80 % |
| t1-20 | 0.1953 | 0.2575 | 0.0622 | 0.2264 | 50 % |
| t1-30 | 0.2307 | 0.3171 | 0.0864 | 0.2939 | 27 % |
| t1-64 | 0.3529 | 0.4737 | 0.1208 | 0.4659 | **6.5 %** |

**The two ceilings are anti-aligned in t.** Where amplitude works there is nothing
to win; where there is most to win it is 93 % phase. The t≤5 inversion is real, not
noise — the native model blurs fine scales from frame 1 (k8-16 rel_l2 at t1: native
0.5448 vs OOD 0.2263).

## Where in k

Frames 1-64. amp% suppressed where the gap is negligible.

| band | reference | OOD | gap | γ ref | γ OOD | ρ ref | ρ OOD | amp% | hz ref | hz OOD |
|---|---|---|---|---|---|---|---|---|---|---|
| k1 | 0.0797 | 0.1771 | 0.0974 | 0.980 | 0.842 | 0.9970 | 0.9963 | 94 | 65.0 | 65.0 |
| k2 | 0.2681 | 0.4499 | 0.1819 | 0.947 | 0.825 | 0.9635 | 0.8959 | 3 | 58.9 | 41.3 |
| k3 | 0.3397 | 0.5498 | 0.2101 | 0.941 | 0.783 | 0.9405 | 0.8371 | 1 | 49.5 | 32.2 |
| k4 | 0.4099 | 0.5971 | 0.1872 | 0.909 | 0.784 | 0.9121 | 0.8024 | 0 | 41.3 | 26.4 |
| k5 | 0.5696 | 0.7319 | 0.1623 | 0.864 | 0.763 | 0.8229 | 0.6858 | 3 | 28.2 | 20.9 |
| k6 | 0.6775 | 0.8243 | 0.1467 | 0.768 | 0.704 | 0.7362 | 0.5796 | 6 | 21.1 | 15.0 |
| k7 | 0.7527 | 0.8360 | 0.0833 | 0.688 | 0.660 | 0.6590 | 0.5581 | 7 | 16.4 | 12.9 |
| k8 | 0.8038 | 0.8639 | 0.0601 | 0.608 | 0.606 | 0.5951 | 0.5124 | 9 | 13.3 | 11.2 |
| k9-12 | 0.8568 | 0.8903 | 0.0335 | 0.499 | 0.500 | 0.5159 | 0.4573 | 3 | 8.8 | 8.4 |
| k13-16 | 0.9161 | 0.9163 | 0.0003 | 0.353 | 0.345 | 0.4042 | 0.4048 | — | 3.9 | 6.2 |
| k17-32 | 0.9577 | 0.9534 | −0.0042 | 0.248 | 0.226 | 0.2911 | 0.3143 | — | 1.0 | 2.8 |
| k33-64 | 0.9938 | 0.9962 | 0.0024 | 0.097 | 0.074 | 0.1118 | 0.0876 | — | 0.2 | 0.0 |

- All recoverable error is in **k1-k8**; k13-16 is exhausted, k17-64 the OOD model
  already matches or beats.
- **γ inverts at k7+**: the OOD model carries MORE energy than the native one
  (t1-20: dγ +0.06 at k7 rising to +0.15 at k14). The amplitude target per band is
  the native model's γ, not 1.0 — native γ(k8-16) is 0.50. Pushing γ→1 above k6
  moves away from the reference, which is what the FNO sweep's high-lr runs did.
- ρ headroom peaks at **k5-k6** (dρ −0.137 / −0.157) on an architecture with no
  truncation at k7. Weakens the "FNO's k5-7 phase gain is its `n_modes=8` edge"
  hypothesis before any adaptation runs, though the UNet's headroom is broader
  (k2-k4 also carry −0.07..−0.11). Step 2's question narrows to whether adaptation
  can reach the headroom, not where it sits.

## Horizon

Pooled k1-64: 34.07 → 49.87, **+15.8 frames**. Per shell the win is concentrated
low: k2 +17.7, k3 +17.2, k4 +14.9, k5 +7.3, k6 +6.1, k7 +3.5, k8 +2.1. At k13-16
and k17-32 the OOD model is already longer than the native one.

## Distribution and residual

- W1 0.1366 (OOD) vs 0.0686 (native), but width-corrected **0.0266 vs 0.0410** —
  the OOD model's value distribution is already the better *shape*; the whole
  distributional deficit is scalar width, i.e. amplitude again.
- covRMSE 0.3226 vs 0.1009, both under the GT-GT floor 0.4058 — real gap,
  on-attractor at both ends.
- res_rms/|f| k1-4 aggr: native **1.045** vs OOD **0.8692**. The residual ranks the
  more accurate model as the worse one. Mechanism: blur lowers the residual (smoother
  field, smaller advection mismatch) and the OOD model is the blurrier one (γ 0.80 vs
  0.93). Two models, one dataset — the strongest evidence yet that the pde objective
  is an anti-proxy here. Reinforced by the Re-mismatch term being ~2 % of the OOD
  residual (0.8692 against Re500 vs 0.8497 against its own Re100).

## Operative target — val [240:270]

Adaptation probes heldout = val, so the gap it is scored against has to live on the
same 30 chains. Two extra forwards, `--split val`:
`unet_qr6rs0jb_re500_val.npz`, `unet_3z5bxjzp_re500_val.npz`.

**Primary readout: k2-8, frames 15-25. s0 = 0.3841, target 0.2427, gap 0.1414
(36.8 % of s0), of which amplitude can close 11 %.**

| band | window | ref | OOD (s0) | gap | gap % | amp |
|---|---|---|---|---|---|---|
| **k2-8** | **t15-25** | **0.2427** | **0.3841** | **0.1414** | **36.8** | **11 %** |
| k2-8 | t11-20 | 0.1991 | 0.3169 | 0.1178 | 37.2 | 19 % |
| k2-8 | t21-40 | 0.3526 | 0.5360 | 0.1835 | 34.2 | 2 % |
| k1 | t15-25 | 0.0399 | 0.1510 | 0.1111 | 73.6 | 95 % |
| k9-64 | t15-25 | 0.8114 | 0.8748 | 0.0634 | 7.2 | 6 % |
| k1-64 | t15-25 | 0.2438 | 0.3444 | 0.1005 | 29.2 | 23 % |

`s0` is also the load guard: an adaptation run's step-0 heldout snapshot must
reproduce 0.3841 at k2-8/t15-25, or the checkpoint or split moved.

The val gap tracks the test gap closely (k2-8 t15-25: 36.8 % vs 37.0 %), so the
test-split tables below stand as measured — the val read is for like-for-like
subtraction, not because the test numbers were in doubt.

Neither `wandb` nor `_snapshot_metrics` computes this readout — its fixed bands are
k1-4/k5-7/k8+ and its W1 frames are (4, 63). Rank offline from the run's npz, which
banks the full per-snapshot arrays. `report_tta.py --bands 2-8 --frames 15-25` reads
it directly.

### The zero-gradient competitor — pre-registered, computed before the bracket ran

Least-squares per-shell gain `g_k = Re<pred,gt>_k / P_pred,k`, recovered from the
banked powers via `2 Re<pred,gt> = P_pred + P_gt - P_err` and fitted at **t=0**,
where the target is the given IC — no labels, no gradients, legal TTA. Computed from
`unet_3z5bxjzp_re500_val.npz`.

| readout | s0 | per-shell gain | single gain | oracle (t=15, uses GT) | target | gain closes |
|---|---|---|---|---|---|---|
| **k2-8 t15-25** | 0.3841 | **0.3715** | 0.3719 | 0.3690 | 0.2427 | **8.9 %** |
| k2-8 t11-20 | 0.3169 | 0.2987 | 0.2994 | 0.2939 | 0.1991 | 15.5 % |
| k1 t15-25 | 0.1510 | 0.0994 | 0.0924 | 0.0460 | 0.0399 | 46.5 % |
| k1-64 t1-64 | 0.4641 | 0.4577 | 0.4569 | 0.4570 | 0.3508 | 5.6 % |

Fitted gains are near-uniform (k1 1.065 → k10 1.102, single scalar 1.0742), so
per-shell buys essentially nothing over one number.

**The bar is 0.3715 at k2-8/t15-25.** Any cell landing at or above it bought the
free scalar and nothing else. Note the *oracle* gain — refitted inside the target
window using GT, an illegal method — reaches only 0.3690, closing 10.7 %. So no
rescaling of any kind, at any granularity, with or without labels, can take more
than ~11 % of the k2-8 gap. The remaining 89 % is phase and is only reachable by a
method.

## Banked target gaps — test [270:300]

**Read disjoint windows, never cumulative `t1-N`.** A cumulative window pools the
easy early frames into every later readout, so the gap it reports is diluted by
exactly the frames that hold no headroom — structurally the mistake that let the FNO
sweep rank on a metric the free scalar gain already owned. Every number below is a
disjoint window.

Reference qr6rs0jb, start 3z5bxjzp, test [270:300]. `amp` = fraction of the gap that
`γ→ρ` alone can close; the rest requires phase.

**k2-8 — the method target**

| window | ref | OOD | gap | gap % | amp |
|---|---|---|---|---|---|
| t1-10 | 0.1311 | 0.1817 | 0.0507 | 27.9 | 89 % |
| **t11-20** | 0.2111 | 0.3420 | 0.1310 | **38.3** | 16 % |
| **t15-25** | 0.2592 | 0.4114 | 0.1522 | **37.0** | 8 % |
| t21-40 | 0.3738 | 0.5611 | 0.1873 | 33.4 | 1 % |
| t41-64 | 0.5748 | 0.7811 | 0.2063 | 26.4 | 3 % |

**k1 — the free channel**

| window | ref | OOD | gap | gap % | amp |
|---|---|---|---|---|---|
| t1-10 | 0.0313 | 0.1170 | 0.0857 | 73.2 | 92 % |
| t15-25 | 0.0453 | 0.1558 | 0.1105 | 70.9 | 95 % |
| t41-64 | 0.1128 | 0.2092 | 0.0964 | 46.1 | 100 % |

**k9-64 — closed**

| window | ref | OOD | gap | gap % |
|---|---|---|---|---|
| t1-10 | 0.6594 | 0.5439 | **−0.1154** | −21.2 |
| t15-25 | 0.8287 | 0.8897 | 0.0611 | 6.9 |
| t41-64 | 1.0171 | 1.0264 | 0.0092 | 0.9 |

Two separate targets, and the pooled k1-64 view mixes them: **k1 is a 71 % gap that
is 95 % free**, k2-8 is a 37 % gap that is **92 % phase**. Relative gap peaks at
t11-25 for both. Reporting k1-64 pooled is mostly reporting k1, i.e. mostly reporting
the gain.

## Target

Adaptation must buy **phase in k2-k8 over frames 15-25**. Amplitude is free and
exhausted by t~10; k1 moves for free and above k8 there is nothing left to take.

Pre-registered readout for every adaptation run on this arm:

- **primary** — heldout rel_l2, k2-8, frames 15-25. s0 (test) 0.4114, target 0.2592.
- **secondary** — ρ at k2-k8; ρ<0.9 horizon at k2-k4.
- **guards** — per-shell γ against the *native model's* γ, never 1.0. The OOD model
  already exceeds native γ above k7; any run raising γ there moves away from target.
- **reported, never ranked on** — k1 rel_l2, and any window starting at t1. Both
  improve in every run and neither distinguishes a method from a scalar.

---

# Bracket 1 — full-weight `ic`, Re100→500, n=5, 30 steps

Five lr cells, `probe_every=1`, `locus=full`, project `unet-sweep`. All 31 probes
completed. npz under `msc/tta/outputs/adapt/unet/`.

| lr | wandb | final heldout rel_l2 (pooled, DC-incl.) |
|---|---|---|
| **6e-5** | **k3rca87o** | **0.4597** |
| 3e-4 | tusq1eb7 | 0.4715 |
| 1e-3 | riqqwyq1 | 0.5696 |
| 3e-3 | wkui9iap | 0.6994 |
| 1e-2 | tlk2s1md | 0.7418 |

s0 = 0.4606 on every cell — load guard passes. Only 6e-5 is not worse than s0.

## k3rca87o (lr 6e-5) — the kill criterion fired

Primary readout k2-8 / t15-25: s0 **0.3841** → best **0.3786** at step 6 → 0.3880 at
step 30. Gain bar **0.3715**. Target **0.2427**.

**The 30-step gradient method never crosses the closed-form gain bar.** It closes
3.9 % of the gap; the zero-gradient scalar closes 8.9 %. Stronger than the FNO
result, where TTA at least tied the gain on the narrow window and beat it aggregate.

### ρ does not move — it declines

Paired per-chain bootstrap over 30 chains, t15-25. `headroom` = native reference − s0.

| shell | s0 | step 6 | dρ | 95 % CI | ref | headroom | taken |
|---|---|---|---|---|---|---|---|
| k2 | 0.9759 | 0.9756 | −0.0003 | [−0.0005, −0.0002] | 0.9945 | +0.0186 | **−1.9 %** |
| k3 | 0.9538 | 0.9528 | −0.0010 | [−0.0013, −0.0008] | 0.9892 | +0.0354 | −2.9 % |
| k4 | 0.9234 | 0.9221 | −0.0013 | [−0.0021, −0.0007] | 0.9766 | +0.0531 | −2.5 % |
| k5 | 0.8816 | 0.8792 | −0.0024 | [−0.0033, −0.0016] | 0.9406 | +0.0590 | −4.1 % |
| k6 | 0.8197 | 0.8164 | −0.0033 | [−0.0045, −0.0021] | 0.8993 | +0.0797 | −4.1 % |
| k7 | 0.7574 | 0.7530 | −0.0043 | [−0.0056, −0.0031] | 0.8435 | +0.0861 | −5.0 % |
| k8 | 0.7060 | 0.7002 | −0.0058 | [−0.0077, −0.0042] | 0.7912 | +0.0853 | −6.8 % |
| **k2-8** | 0.9295 | 0.9282 | −0.0013 | [−0.0017, −0.0010] | 0.9697 | +0.0402 | **−3.3 %** |

Every shell is significantly negative and the damage orders with k. At step 30 the
same signs, ~3× larger (k2-8 −0.0047, −11.6 % of headroom), so the conclusion is not
an artifact of selecting the rel_l2-optimal pass. **ρ falls monotonically at every
one of the 30 passes — never up once**, so no stopping rule rescues it.

### γ is the whole effect, and it overshoots

Uniform +0.03-0.04 at every shell, the global-gain signature the FNO sweep found.
Above k7 it passes the native value — the pre-registered guard, fired:

| shell | γ s0 | γ step 6 | γ native |
|---|---|---|---|
| k7 | 0.682 | **0.719** | 0.686 |
| k8 | 0.626 | **0.660** | 0.608 |
| k9-12 | 0.495 | **0.524** | 0.485 |
| k13-16 | 0.345 | **0.368** | 0.351 |

γ is also non-monotone under a constant lr — 0.8626 at step 7, 0.8423 by step 17,
0.8535 by step 26. Unexplained.

### Horizon per shell — moves only where it does not count

k2 +0.07 ns · k3 −0.17 ns · **k4 −0.13 worse** · k5 −0.10 ns · k6 −0.10 ns ·
k7 −0.07 ns · k8 −0.03 ns. The only significant gains are **k9-12 +0.13** and
**k17-32 +0.13** — bands the target marks closed, where the OOD model already
exceeds the native one.

w1wc is dead too: 0.0337 → 0.0330 (~2 %), against the FNO's −23 %.

## Verdict

Full-weight `ic` adaptation does not reach the phase headroom on this arm. It
**consumes** phase to buy amplitude, monotonically, at the safest lr in the bracket,
and the amplitude it buys is worth less than a closed-form scalar fitted for free.
Since parameter-space narrowing did not localize the output effect for the FNO, a
locus rung is not expected to change this — the next question is whether any
objective moves ρ at all, not where in the network it is applied.

## Why ρ did not move — mechanism, read off the loss

`KFLoss.__call__` (`src/pde/ns.py:326`):

```python
u_in = w[:, :, :, 0]       # prediction at t=0
u0   = target[:, :, :, 0]  # the true IC
ic   = self.lp.rel(u_in, u0)
```

**The `ic` objective is a single-frame constraint at t=0.** It carries no information
about frames 15-25. And at t=0 the prediction's phase is already essentially perfect
— ρ(k2-8, t0) = 0.9994 — so the *only* error the objective can see is the ~7 %
amplitude deficit (γ = 0.929). Through shared weights the cheapest descent direction
for a t=0 amplitude residual is a global output scale. That is exactly the uniform
+0.03-0.04 γ measured at every shell.

Phase error at t15-25 lives in the *evolution map* — the operator advances the flow
at Re100 speed. No t=0 state constraint reaches it, however well it is minimised.

The optimiser did its job: `ic` fell **0.0630 → 0.0265 (−58 %)** while the
out-of-gradient GT term rose **0.4032 → 0.4279 (+6.1 %)**, exceeding its start by
step 2. And it nearly recovered the optimal scalar — γ(k2-8, t1-10) 0.875 → 0.932
against the closed-form gain's 0.940. Nothing is broken; the objective is a t=0
amplitude probe and it was minimised.

Second architecture, second objective, same no-label-free-stopping signature as the
FNO's `hdc8jwst`: the adaptation loss and GT disagree from update one. Note the FNO's
`ic` arm (`2n8qq7m2`) did NOT show this — its GT term fell too. The divergence is new
here.

### Structural consequence for the other objectives

- `pde` is evaluated over the whole trajectory, so it *is* temporally extended — but
  every NS solution zeroes it. Phase is the coordinate along that null space, so the
  residual cannot pin which trajectory. Confirmed empirically twice: the FNO's
  pde arm improved the residual 20 % while degrading accuracy 15 %, and the Re500
  native model has a *higher* residual (1.045) than the OOD model (0.8692).
- `pde + ic` is the PINO formulation and is in principle well-posed — but the ic half
  still only pins t=0 and the pde half is soft, scored as `lp.rel(Du, forcing)`.
- What is missing is a signal that is **both** trajectory-pinning **and** temporally
  extended. Neither term is.

---

# Workshop paper — what it is

**What does test-time adaptation actually buy for neural PDE surrogates?**
A protocol and a two-architecture negative result.

- Exact amplitude/phase decomposition of the OOD gap, `rel_l2² = (1−ρ²) + (γ−ρ)²`.
- A **native-model ceiling**: the in-distribution source column overstates reachable
  performance ~2×; without it, OOD-improvement claims have no denominator.
- A **closed-form gain competitor with an oracle bound**: ~11 % is the most any
  rescaling can take at any granularity, and 30 steps of gradient TTA on 24.5 M
  parameters loses to the free scalar.
- **Mechanism**: current label-free objectives are structurally amplitude-only —
  `ic` is a t=0 state probe, `pde` has a null space containing exactly the phase
  we want. Neither is trajectory-pinning *and* temporally extended.
- The physics residual is an **anti-proxy**, measured across two models on one
  dataset (native 1.045 vs OOD 0.8692) and both directions on a third.

Open: the FNO-specific claim rests on the lr 6e-5 cell; the matched-window ρ read on
lr 3e-4 / 3e-3 (`tusq1eb7`, `wkui9iap`) is outstanding.

## Bracket 1, all five cells — corrects the FNO-specific claim

Primary readout k2-8/t15-25, heldout. Gain bar 0.3715, target 0.2427.

| lr | id | pool s0 | pool best | held s0 | held best | step | vs gain bar |
|---|---|---|---|---|---|---|---|
| 6e-5 | k3rca87o | 0.3830 | 0.3757 | 0.3841 | 0.3786 | 6 | +0.0071 short |
| **3e-4** | **tusq1eb7** | 0.3830 | **0.3747** | 0.3841 | **0.3773** | **1** | **+0.0058 short** |
| 1e-3 | riqqwyq1 | 0.3830 | 0.3830 | 0.3841 | 0.3841 | **0** | never improves |
| 3e-3 | wkui9iap | 0.3830 | 0.3830 | 0.3841 | 0.3841 | **0** | never improves |
| 1e-2 | tlk2s1md | 0.3830 | 0.3830 | 0.3841 | 0.3841 | **0** | never improves |

- **No cell beats the closed-form gain.** Best in the whole bracket is 0.3773.
- The best cell peaks at **step 1** — one update. The top three lrs are monotone
  damage from the first update.
- pool tracks heldout throughout (3e-4: 0.3747 vs 0.3773) — no transductive gap.

### The phase effect is NOT absent on the UNet — it is early-window and ~10x smaller

Per-shell dρ with paired 30-chain CIs, at each cell's best pass:

| cell | window | k2 | k4 | k6 | k7 | k8 |
|---|---|---|---|---|---|---|
| 3e-4 s1 | **t1-16** | +0.0002+ | +0.0004+ | +0.0011+ | +0.0016+ | +0.0017+ |
| 3e-4 s1 | t15-25 | −0.0001 | −0.0011− | −0.0027− | −0.0040− | −0.0048− |
| 6e-5 s6 | **t1-16** | +0.0000 | +0.0002 | +0.0005+ | +0.0009+ | +0.0009+ |
| 6e-5 s6 | t15-25 | −0.0003− | −0.0013− | −0.0033− | −0.0043− | −0.0058− |

(+ = CI above zero, − = CI below zero.)

**Correction to the earlier entry.** In the FNO's own window the UNet reproduces the
FNO's sign and its k-ordering — dρ positive, rising monotonically with k, CI-positive
at k5-k8. The effect is real on an architecture with no spectral truncation. Two
differences remain:

1. **Magnitude**: +0.0017 at k8 against the FNO's +0.016-0.018 at k6-k7 — 10x smaller.
2. **No k5-7 peak**: the UNet's dρ rises monotonically through k8 with no maximum at
   the FNO's `n_modes=8` edge. So the *localisation* looks like the truncation edge;
   the *effect* does not.
3. **It reverses in the target window**: the same shells go significantly negative at
   t15-25.

Open, and cheap — no GPU: **was the FNO's phase gain also early-window?** It was only
ever read at t1-16. Re-reading the banked FNO npz at t15-25 would say whether the one
robust positive result of the FNO sweep survives the window where the headroom is.

---

# Bracket 2 — `physics` (pde 1 + ic 5). The method gate opens.

Same axes as bracket 1: n=5, 30 steps, `probe_every=1`, `locus=full`, five lrs.

| obj | lr | id | s0 | best | step | vs s0 | vs gain bar |
|---|---|---|---|---|---|---|---|
| ic | 6e-5 | k3rca87o | 0.3841 | 0.3786 | 6 | −0.0055 | +0.0071 |
| ic | 3e-4 | tusq1eb7 | 0.3841 | 0.3773 | 1 | −0.0068 | +0.0058 |
| ic | 1e-3 / 3e-3 / 1e-2 | — | 0.3841 | 0.3841 | 0 | 0 | +0.0126 |
| physics | 6e-5 | n2j5s819 | 0.3841 | 0.3841 | 1 | −0.0000 | +0.0126 |
| **physics** | **3e-4** | **vwuc4m8c** | 0.3841 | **0.3678** | **30** | **−0.0163** | **−0.0037** |
| physics | 1e-3 / 3e-3 / 1e-2 | — | 0.3841 | 0.3841 | 0 | 0 | +0.0126 |

**vwuc4m8c is the first cell in either bracket to reach the closed-form gain bar.**
It does NOT beat it: paired per-chain, physics vs gain is **-0.0065
[-0.0142, +0.0011] — a tie**. See the verification section below.

## vwuc4m8c — ρ moves, positively, in the target window

Per-shell dρ, paired 30-chain CIs (`+` = CI above zero, `−` = below):

| window | k2 | k3 | k4 | k5 | k6 | k7 | k8 |
|---|---|---|---|---|---|---|---|
| t1-10 | +0.0008+ | +0.0019+ | +0.0022+ | +0.0037+ | +0.0064+ | +0.0082+ | +0.0122+ |
| t1-16 | +0.0018+ | +0.0037+ | +0.0045+ | +0.0069+ | +0.0120+ | +0.0167+ | +0.0236+ |
| t11-20 | +0.0038+ | +0.0070+ | +0.0088+ | +0.0121+ | +0.0223+ | +0.0333+ | +0.0468+ |
| **t15-25** | **+0.0043+** | **+0.0063+** | **+0.0079+** | **+0.0099+** | **+0.0222+** | **+0.0360+** | **+0.0511+** |
| t21-40 | +0.0007 | −0.0048 | −0.0067 | −0.0075 | +0.0106 | +0.0295+ | +0.0325+ |
| t41-64 | −0.0109− | −0.0330− | −0.0268− | −0.0230− | −0.0038 | +0.0010 | +0.0065 |

**Horizon** (full rollout, CI-tagged): k2 −0.93− · k3 −0.10 · k4 +1.23 ·
**k5 +1.03+ · k6 +1.77+ · k7 +2.03+ · k8 +2.10+**

γ **falls** at k5-k8 (0.749→0.729, 0.714→0.656, 0.682→0.595, 0.626→0.540) — the pde
term's blur pressure, as predicted. **Correction: this crosses BELOW the native
reference at k7 (0.595 vs 0.686) and k8 (0.540 vs 0.608).** The gate was coded
`gamma <= native`, so undershoot counted as a pass; it is not restraint, it is blur.
The pre-registered failure mode and the phase gain co-occur.

The phase gain survives that anyway, because ρ is provably invariant to any per-shell
rescaling: no amount of blur or gain can manufacture it. **The dρ above is genuine
phase.**

## The trajectory inverts the ic arm

| step | k2-8 t15-25 | ρ(k8) | γ(k8) | pde |
|---|---|---|---|---|
| 0 | 0.3841 | 0.7060 | 0.663 | 1.183 |
| 6 | **0.4231** | 0.7367 | 0.488 | 1.043 |
| 10 | 0.4156 | 0.7493 | 0.479 | 1.000 |
| 25 | 0.3715 | 0.7541 | 0.553 | 0.871 |
| 30 | **0.3678** | **0.7571** | 0.562 | 0.822 |

- **ρ(k8) rises monotonically at every one of the 30 passes — never down once.** The
  exact mirror of the `ic` arm, where it fell at every pass.
- rel_l2 is **U-shaped**: it degrades to 0.4231 by step 6 before recovering and
  crossing s0 near step 23. **Any early-stopping rule kills this run at step 6.**
- Still improving at step 30 on both ρ and rel_l2 — **budget-truncated**.
- Cost is real and localised: late-window low-k phase (t41-64 k3 −0.0330) and the k2
  horizon (−0.93).

## What this overturns

My pre-registered prediction was ρ flat-to-worse under `physics`. **Refuted.** The
null-space argument was too strong: the residual alone cannot pin a trajectory, but
`pde + ic` — residual over the whole rollout, IC pinning t=0 — does, and it moves
phase where nothing else has. The `ic` term is not the method; it is the constraint
that makes the residual usable.

## Verification — what survives a paired CI

Pooled identity `rel_l2² = (1−ρ²) + (γ−ρ)²` over k2-8 / t15-25:

| arm | rel_l2 | ρ | γ | phase (1−ρ²) | amp (γ−ρ)² |
|---|---|---|---|---|---|
| s0 | 0.3841 | 0.9294 | 0.8231 | 0.13625 | 0.01130 |
| gain only | 0.3715 | 0.9294 | 0.8875 | **0.13625** | 0.00175 |
| physics s30 | 0.3678 | 0.9368 | 0.8233 | **0.12240** | 0.01288 |
| **physics + gain** | **0.3607** | 0.9368 | 0.8485 | 0.12232 | 0.00781 |

The gain leaves the phase term **bit-identical** (as it must — ρ is rescaling-
invariant) and nearly eliminates the amplitude term. Physics does the opposite:
**phase term −10.2 %**, amplitude term slightly worse. They are orthogonal by
construction.

Paired per-chain rel_l2, 30 chains, 4000 bootstrap draws:

| comparison | Δ | 95 % CI | verdict |
|---|---|---|---|
| physics vs s0 | −0.0193 | [−0.0250, −0.0134] | BETTER |
| gain vs s0 | −0.0128 | [−0.0157, −0.0099] | BETTER |
| **physics vs gain** | **−0.0065** | **[−0.0142, +0.0011]** | **tie** |
| **physics + gain vs gain** | **−0.0138** | **[−0.0210, −0.0067]** | **BETTER** |
| physics + gain vs physics | −0.0073 | [−0.0084, −0.0063] | BETTER |

**The defensible claim is the composite, not the gradient method alone.** Physics-TTT
buys the phase term, which no rescaling can touch; the free scalar then collects the
amplitude term physics leaves on the table (γ 0.823 undershoots ρ 0.937). Together
they beat the gain alone with a CI that excludes zero. Alone, physics only ties it.

Outstanding before this is claimable: pool-shift replication at `[10:15]`, the
200-step curve to locate the plateau, and a persistence control on ρ(k8).
