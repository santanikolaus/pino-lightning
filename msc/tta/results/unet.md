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

---

# Bracket 3 — physics, 200 steps. Three retractions and a real result.

`{1e-4, 2e-4, 3e-4, 5e-4, 7e-4}`, n=5, `probe_every=5`, `locus=full`.

## The 30-step read caught the method mid-transit

| lr | id | ρ(k8) peak | @step | primary rel_l2 min | @step | γ(k8) s0→peak |
|---|---|---|---|---|---|---|
| 1e-4 | 665u7dbi | 0.7726 | **200** | 0.3272 | 200 | 0.626→0.617 |
| **2e-4** | **z8io8n3g** | **0.7775** | **195** | **0.3206** | **195** | 0.626→0.638 |
| 3e-4 | cxx0fkld | 0.7765 | **200** | 0.3259 | 190 | 0.626→0.631 |
| 5e-4 | p25j482q | 0.7723 | 135 | 0.3362 | 200 | 0.626→0.597 |
| 7e-4 | zsrzt2hk | 0.7675 | 115 | 0.3546 | 175 | 0.626→0.571 |

**Not peaked.** For lr ≤ 3e-4 ρ is still rising at the last probe. Only 5e-4 and
7e-4 turn over (at 135 / 115) and both to a lower maximum.

### Retracted: "the physics objective trades amplitude for phase"

False. γ(k8) went 0.663 → **0.540** at s30 → **0.629** at s200 while ρ climbed
throughout. **The blur is a transient, not the price.** γ rises at every shell by
s200 (k2 0.830→0.914, k5 0.749→0.823, k8 0.626→0.638).

### Retracted: the k2 horizon cost

−0.93 at s30 became **+3.30 [+1.80, +4.80]** at s200. Same transient.

### Downgraded: the composite

`physics + gain` vs `physics` is now −0.0039 [−0.0050, −0.0028] — a minor add-on,
not the method. At s30 the gain was collecting amplitude physics had spent; by s200
physics has collected it itself. **The method is physics-TTT run long.**

## Result — lr 2e-4, step 195/200, val heldout

ρ<0.9 horizon, paired 30-chain CIs:

| k | s0 | adapted | **gain** | native | Δ | 95 % CI | recovered |
|---|---|---|---|---|---|---|---|
| 2 | 41.27 | 44.57 | 41.27 | 60.93 | +3.30 | [+1.80, +4.80] | 16.8 % |
| 3 | 32.00 | 35.67 | 32.00 | 51.23 | +3.67 | [+2.37, +5.20] | 19.1 % |
| 4 | 26.83 | 30.40 | 26.83 | 39.93 | +3.57 | [+1.87, +5.93] | 27.2 % |
| 5 | 20.50 | 23.27 | 20.50 | 29.00 | +2.77 | [+2.10, +3.40] | 32.5 % |
| 6 | 16.47 | 19.87 | 16.47 | 22.10 | +3.40 | [+2.77, +4.07] | 60.4 % |
| 7 | 13.87 | 17.07 | 13.87 | 17.20 | +3.20 | [+2.50, +3.90] | **96.0 %** |
| 8 | 11.77 | 14.47 | 11.77 | 14.23 | +2.70 | [+2.17, +3.23] | **109.5 %** |

**+2.7 to +3.7 frames at every shell, all CIs excluding zero.** The gain column is
identical to s0 at every shell — the competitor moves the horizon by exactly zero.
At k7-k8 the adapted horizon **matches the native reference** (nominally exceeds it
at k8, 14.47 vs 14.23; one checkpoint, do not build on the exceedance — γ there also
sits marginally above native, 0.629 vs 0.608).

Pooled identity, k2-8 / t15-25:

| arm | rel_l2 | ρ | γ | phase (1−ρ²) | amp (γ−ρ)² |
|---|---|---|---|---|---|
| s0 | 0.3841 | 0.9294 | 0.8231 | 0.13625 | 0.01130 |
| gain only | 0.3715 | 0.9294 | 0.8875 | 0.13625 | 0.00175 |
| **physics s200** | **0.3215** | **0.9487** | 0.8905 | **0.09995** | 0.00339 |

**Phase term −27 %.** Gap to the native reference closed **44 %**, against the
gain's 8.9 %.

| comparison | Δ | 95 % CI | verdict |
|---|---|---|---|
| physics vs s0 | −0.0663 | [−0.0729, −0.0600] | BETTER |
| **physics vs gain** | **−0.0535** | **[−0.0616, −0.0455]** | **BETTER** |
| physics+gain vs physics | −0.0039 | [−0.0050, −0.0028] | BETTER |

The s30 tie against the gain is superseded: at 200 steps physics beats it outright.

## Band × window at the ρ peak (lr 2e-4, step 195)

dρ is positive at **every** shell and **every** window:

| window | k2 | k4 | k6 | k7 | k8 |
|---|---|---|---|---|---|
| t1-10 | +0.0008 | +0.0028 | +0.0079 | +0.0101 | +0.0142 |
| t11-20 | +0.0056 | +0.0166 | +0.0369 | +0.0475 | +0.0591 |
| **t15-25** | +0.0082 | +0.0224 | +0.0450 | **+0.0597** | **+0.0715** |
| t21-40 | +0.0135 | +0.0249 | +0.0480 | +0.0662 | +0.0721 |
| t41-64 | +0.0207 | +0.0032 | +0.0212 | +0.0346 | +0.0340 |

rel_l2 improves everywhere except **t41-64 at mid-k** (k5 **+0.0331**, k4 +0.0149)
— the one slice where the two disagree: phase improved there while the amplitude
term degraded. Late-frame γ drift, the same signature the FNO arm showed. It bounds
the claim window.

## Caveats attached to every number above

One pool (train [0:5]), one checkpoint, one seed. At 200 steps the pool is cycled
40×, so pool-overfitting is *more* plausible than at s30, not less. The `p10`
pool-shift replication is s30 — it tests the mechanism (monotone ρ rise, sign), not
these magnitudes.

## Pool-shift replication — `465im803`, train [10:15)

Identical to `vwuc4m8c` in every respect except the five adapt chains. Heldout is the
same val split, so s0 is bit-identical and the comparison is controlled.

| | pool [0:5) `vwuc4m8c` | pool [10:15) `465im803` |
|---|---|---|
| ρ(k8) t15-25 | 0.7060 → 0.7571 (**+0.0511**) | 0.7060 → 0.7713 (**+0.0654**) |
| ρ monotone | 24/30 up | 28/30 up |
| primary rel_l2 | 0.3841 → 0.3678 | 0.3841 → **0.3646** |
| γ(k8) | 0.663 → 0.562 | 0.663 → 0.537 |

**The effect is not pool-specific.** On different chains it is *larger*, more
monotone, and lands lower on the primary readout. The 30-step blur transient
reproduces too (γ 0.537), as expected at this budget.

Horizon at s30, paired 30-chain CIs:

| k | s0 | adapted | gain | native | Δ | 95 % CI | recovered |
|---|---|---|---|---|---|---|---|
| 2 | 41.27 | 41.10 | 41.27 | 60.93 | −0.17 | ns | — |
| 3 | 32.00 | 32.10 | 32.00 | 51.23 | +0.10 | ns | — |
| 4 | 26.83 | 27.30 | 26.83 | 39.93 | +0.47 | ns | — |
| 5 | 20.50 | 21.80 | 20.50 | 29.00 | +1.30 | [+0.43, +2.10] | 15.3 % |
| 6 | 16.47 | 18.67 | 16.47 | 22.10 | +2.20 | [+1.60, +2.83] | 39.1 % |
| 7 | 13.87 | 16.37 | 13.87 | 17.20 | +2.50 | [+1.97, +3.03] | 75.0 % |
| 8 | 11.77 | 14.17 | 11.77 | 14.23 | +2.40 | [+2.03, +2.77] | **97.3 %** |

Every k5-k8 CI excludes zero, same shells and same ordering as `vwuc4m8c` — and each
delta is *larger* (+1.30/+2.20/+2.50/+2.40 vs +1.03/+1.77/+2.03/+2.10). k2-k4 are ns
at 30 steps on both pools; they only become significant at 200 steps.

**Two independent pools, same sign, same shells, larger effect.** The remaining
single points of failure are the checkpoint and the seed, not the pool.

---

# Bracket 4 — 800 steps at 2e-4 (`pm1135ol`). The plateau, and a third regime.

`probe_every=20`, n=5, pool [0:5), `locus=full`, physics.

## Plateau

| step | primary k2-8/t15-25 | ρ(k8) | γ(k8) | γ(k2) | hz k7 | hz k2 | pde |
|---|---|---|---|---|---|---|---|
| 0 | 0.3841 | 0.7060 | 0.626 | 0.830 | 13.87 | 41.27 | 1.183 |
| 20 | 0.3807 | 0.7555 | **0.520** | 0.807 | 15.53 | 40.80 | 0.912 |
| 100 | 0.3351 | 0.7668 | 0.605 | 0.889 | 16.47 | 43.13 | 0.721 |
| 200 | 0.3215 | 0.7770 | 0.629 | 0.908 | 17.07 | 44.57 | 0.630 |
| 340 | 0.3113 | 0.7807 | 0.668 | 0.927 | 17.30 | 45.80 | 0.579 |
| **500** | 0.3080 | **0.7818** | 0.695 | 0.933 | 17.40 | 46.50 | 0.525 |
| **620** | **0.3063** | 0.7806 | 0.713 | 0.939 | 17.43 | 46.70 | 0.512 |
| 800 | 0.3095 | 0.7783 | **0.728** | 0.941 | 17.37 | 46.70 | 0.500 |

**Plateau band ≈ steps 500-620.** ρ(k8) peaks 0.7818 at 500; the primary readout
bottoms 0.3063 at 620. Last five probe-to-probe dρ are mixed-sign noise
(+0.0004, −0.0002, −0.0005, −0.0005, −0.0003). Resolution limited by
`probe_every=20`.

**Over-running is nearly free**: 620 → 800 costs +0.0032 on the primary. The method
needs a budget *floor*, not a precise stop — anywhere in 400-800 is within ~1 % of
optimal.

## A third regime: transit → sweet spot → overfill

γ does not stop when ρ does. After the ρ plateau it keeps climbing and **crosses
above the native reference at k6-k8**, while ρ(k8) drifts slightly down from its
step-500 peak:

| shell | γ s0 | γ s800 | γ native |
|---|---|---|---|
| k5 | 0.749 | 0.852 | 0.862 |
| k6 | 0.714 | **0.812** | 0.769 |
| k7 | 0.682 | **0.778** | 0.686 |
| k8 | 0.626 | **0.727** | 0.608 |

So the full trajectory is **blur transit (0-100) → sweet spot (500-620) → overfill
(700+)**. Per-shell ρ is rescaling-invariant, so the horizon gains are not an
artifact of the γ inflation — but the co-occurrence is banked, and the ≥100 %
recovery figures below carry it.

## State at step 800 (end of run), val heldout

Horizon, paired 30-chain CIs:

| k | s0 | s800 | gain | native | Δ | 95 % CI | recovered |
|---|---|---|---|---|---|---|---|
| 2 | 41.27 | 46.70 | 41.27 | 60.93 | **+5.43** | [+3.27, +7.60] | 27.6 % |
| 3 | 32.00 | 36.90 | 32.00 | 51.23 | +4.90 | [+3.30, +6.90] | 25.5 % |
| 4 | 26.83 | 30.77 | 26.83 | 39.93 | +3.93 | [+2.37, +5.83] | 30.0 % |
| 5 | 20.50 | 23.80 | 20.50 | 29.00 | +3.30 | [+2.53, +4.07] | 38.8 % |
| 6 | 16.47 | 20.27 | 16.47 | 22.10 | +3.80 | [+3.00, +4.60] | 67.5 % |
| 7 | 13.87 | 17.37 | 13.87 | 17.20 | +3.50 | [+2.67, +4.37] | 105.0 % |
| 8 | 11.77 | 14.30 | 11.77 | 14.23 | +2.53 | [+1.90, +3.17] | 102.7 % |

k7/k8 nominally exceed the native reference — one checkpoint, and γ there now sits
above native too; do not build on the exceedance. **Different shells plateau at
different times**: k2 was ns at s30, +3.30 at s200, +5.43 at s800 and still the
strongest mover.

Pooled identity, k2-8 / t15-25:

| arm | rel_l2 | ρ | γ | phase (1−ρ²) | amp (γ−ρ)² |
|---|---|---|---|---|---|
| s0 | 0.3841 | 0.9294 | 0.8231 | 0.13625 | 0.01130 |
| gain only | 0.3715 | 0.9294 | 0.8875 | 0.13625 | 0.00175 |
| **physics s800** | **0.3095** | **0.9511** | 0.9324 | **0.09545** | **0.00035** |

**Phase term −30 %.** Gap to the native reference closed **55.0 % at step 620**
(0.3063), **52.8 % at step 800** — against the gain's 8.9 %.

| comparison | Δ | 95 % CI | verdict |
|---|---|---|---|
| physics vs s0 | −0.0798 | [−0.0874, −0.0726] | BETTER |
| **physics vs gain** | **−0.0670** | **[−0.0759, −0.0585]** | **BETTER** |
| physics + gain vs physics | +0.0002 | [−0.0010, +0.0013] | **tie** |

**The composite is now exactly a tie.** The amplitude term is 0.00035 — physics has
collected it all; the gain has nothing left. "The method is physics-TTT run long"
is complete.

## Remaining bound

dρ is positive at every shell and every window including t41-64 (k2 +0.0302,
k7 +0.0547). But rel_l2 still degrades late at mid-k (**t41-64 k5 +0.0282**,
k6 +0.0241) — phase improves there while the amplitude term worsens. The claim
window stays bounded; this persists at every budget tested.

The pde loss falls monotonically all 800 steps (1.183 → 0.500) while the primary
readout turns at 620 — **still no label-free stopping signal**. The flat plateau is
what makes that survivable.

---

# Second checkpoint — `wobwri1s` (Re100 UNet, 150ep), 200 steps at 2e-4

`pylk2jkn`, `exp=unet150`, out_dir `adapt/unet150/`. Identical protocol to the
`3z5bxjzp` 200-step cell; only the base weights differ. The two checkpoints are
genuinely different models — `wobwri1s` starts **better** OOD (s0 primary 0.3522 vs
0.3841) despite being worse in-distribution (val_l2 0.1507, not converged).

## The method reproduces on different weights

Horizon at step 200, paired 30-chain CIs, val heldout:

| k | s0 | s200 | gain | native | Δ | 95 % CI | recovered | (3z5bxjzp Δ @200) |
|---|---|---|---|---|---|---|---|---|
| 2 | 43.70 | 45.97 | 43.70 | 60.93 | +2.27 | [+0.93, +3.67] | 13.2 % | +3.30 |
| 3 | 33.97 | 36.60 | 33.97 | 51.23 | +2.63 | [+1.47, +3.73] | 15.3 % | +3.67 |
| 4 | 30.90 | 32.57 | 30.90 | 39.93 | +1.67 | [+0.13, +3.07] | 18.5 % | +3.57 |
| 5 | 22.67 | 25.30 | 22.67 | 29.00 | +2.63 | [+1.77, +3.57] | 41.6 % | +2.77 |
| 6 | 17.70 | 20.83 | 17.70 | 22.10 | +3.13 | [+2.43, +3.87] | 71.2 % | +3.40 |
| 7 | 14.23 | 17.23 | 14.23 | 17.20 | +3.00 | [+2.40, +3.57] | 101.1 % | +3.20 |
| 8 | 12.03 | 15.00 | 12.03 | 14.23 | +2.97 | [+2.57, +3.37] | **134.8 %** | +2.70 |

**Every shell CI-positive, same ordering, comparable magnitudes.** The gain column is
again identical to s0 everywhere. Deltas run slightly smaller at k2-k4 and slightly
larger at k8 — consistent with the smaller headroom this checkpoint starts with.

Pooled identity, k2-8 / t15-25:

| arm | rel_l2 | ρ | γ | phase (1−ρ²) | amp (γ−ρ)² |
|---|---|---|---|---|---|
| s0 | 0.3522 | 0.9401 | 0.8521 | 0.11630 | 0.00773 |
| gain only | 0.3444 | 0.9401 | 0.8920 | 0.11629 | 0.00231 |
| **physics s200** | **0.2956** | **0.9577** | 0.8907 | **0.08289** | 0.00449 |

Phase term **−28.7 %** (vs −27 % on `3z5bxjzp` at the same budget). physics vs gain
**−0.0494 [−0.0574, −0.0414] BETTER**.

## Two differences worth recording

**1. It is not converged at 200 and does not plateau there.** primary rel_l2 is still
falling at the last probe (0.2956 at step 200, its minimum); the last five
probe-to-probe dρ are mixed-sign but the trend is up. The ρ(k8) "peak" at step 85 is
probe noise on a rising curve — `probe_every=5` resolves fluctuations the s800 run's
`probe_every=20` averaged out. This checkpoint needs the same long budget.

**2. Less overfill, more blur, at matched budget.** γ at s200 sits *below* native at
k5-k7 (0.796/0.735/0.680 vs 0.862/0.769/0.686) where `3z5bxjzp` had already crossed
above at k6-k8. The composite still helps here (physics+gain vs physics
−0.0034 [−0.0039, −0.0029]) where on `3z5bxjzp` at s800 it had become a tie. Both
are consistent with this run sitting earlier on the same transit → sweet spot →
overfill arc.

## What this closes

Two checkpoints, two pools, same objective, same lr: **same sign, same shells, same
ordering, comparable magnitude.** The effect is not a property of one set of weights.
The remaining unvaried factor is the pretraining seed, which would cost a 300-epoch
retrain — stated as a limitation, not spent.

Bound unchanged: at the ρ peak the late window degrades at low-k
(t41-64 dρ k3 **−0.0253**, rel_l2 k3 +0.0287) — same claim-window bound as every
other cell, here at low rather than mid k.

---

# Pool shift at 200 steps — `x55ga61f`, train [10:15)

Identical to the `3z5bxjzp` 200-step cell except the five adapt chains. Heldout is
the same val split, so s0 is bit-identical.

## The headline magnitudes replicate — and exceed

Horizon at step 200, paired 30-chain CIs, val heldout:

| k | s0 | p10 s200 | gain | native | Δ | 95 % CI | recovered | *(p0 Δ)* |
|---|---|---|---|---|---|---|---|---|
| 2 | 41.27 | 45.87 | 41.27 | 60.93 | **+4.60** | [+3.47, +5.70] | 23.4 % | *+3.30* |
| 3 | 32.00 | 36.20 | 32.00 | 51.23 | +4.20 | [+3.03, +5.43] | 21.8 % | *+3.67* |
| 4 | 26.83 | 30.00 | 26.83 | 39.93 | +3.17 | [+1.87, +4.53] | 24.2 % | *+3.57* |
| 5 | 20.50 | 23.70 | 20.50 | 29.00 | +3.20 | [+2.30, +4.10] | 37.6 % | *+2.77* |
| 6 | 16.47 | 20.80 | 16.47 | 22.10 | **+4.33** | [+3.50, +5.17] | 76.9 % | *+3.40* |
| 7 | 13.87 | 17.97 | 13.87 | 17.20 | **+4.10** | [+3.30, +4.93] | 123.0 % | *+3.20* |
| 8 | 11.77 | 15.10 | 11.77 | 14.23 | +3.33 | [+2.70, +3.97] | 135.1 % | *+2.70* |

Six of seven shells are **larger** than on pool [0:5). Same sign, same ordering,
every CI excluding zero, gain column exactly s0 throughout.

Pooled identity, k2-8 / t15-25:

| arm | rel_l2 | ρ | γ | phase (1−ρ²) | amp (γ−ρ)² |
|---|---|---|---|---|---|
| s0 | 0.3841 | 0.9294 | 0.8231 | 0.13625 | 0.01130 |
| gain only | 0.3715 | 0.9294 | 0.8875 | 0.13625 | 0.00175 |
| **p10 s200** | **0.3069** | **0.9536** | 0.8934 | **0.09057** | 0.00363 |
| *p0 s200* | *0.3215* | *0.9487* | *0.8905* | *0.09995* | *0.00339* |

Phase term **−33.5 %** (p0: −27 %). physics vs gain **−0.0628 [−0.0738, −0.0521]**.

## Two things this settles

**1. dρ is positive at every shell AND every window, including t41-64** (k2 +0.0267,
k7 +0.0549) — and here **rel_l2 also improves in t41-64 at every shell**
(k5 −0.0028, k6 −0.0119, k7 −0.0223). On pool [0:5) that window degraded at mid-k
(k5 +0.0331 at s200, +0.0282 at s800). **The late-window bound is pool-dependent,
not a property of the method.** Correct statement: the late window is where the
method is weakest and can go either way, not where it reliably fails.

**2. This run is still in the sweet spot, not overfill.** γ at s200 sits below
native at k5-k8 (0.787/0.716/0.663/0.600 vs 0.862/0.769/0.686/0.608) and the
composite still pays (+physics+gain vs physics −0.0043 [−0.0057, −0.0030]). Yet its
horizons already exceed p0's at s200 and rival p0's at s800. Different pools traverse
the arc at different rates; the shells and signs do not care.

## Replication status after this run

| axis | varied | result |
|---|---|---|
| pool chains | [0:5) vs [10:15), at s30 and s200 | same sign, same shells, larger on the shift |
| checkpoint | 3z5bxjzp (300ep) vs wobwri1s (150ep) | same sign, same shells, comparable |
| lr | basin 1e-4…3e-4 at s200 | all work; 5e-4/7e-4 peak early and lower |
| budget | 30 / 200 / 800 | monotone in ρ to a 500-620 plateau |
| pretraining seed | **not varied** | stated limitation (300-epoch retrain) |

---

# Ablation — `pde` alone, 200 steps at 2e-4 (`776pzxvr`)

`pde_weight=1, ic_weight=0`. Everything else matches the `physics` 200-step cell.

## The objective succeeds; the model is destroyed

| step | primary k2-8/t15-25 | ρ(k8) | γ(k8) | γ(k2) | hz k7 | hz k2 | **pde loss** |
|---|---|---|---|---|---|---|---|
| 0 | **0.3841** | 0.7060 | 0.626 | 0.830 | 13.87 | 41.27 | 1.183 |
| 10 | 0.4865 | 0.7142 | 0.408 | 0.653 | 13.73 | 38.47 | 0.675 |
| 100 | — | — | — | — | — | — | ~0.16 |
| 200 | **0.7038** | **0.4289** | **0.231** | **0.424** | **7.60** | **27.47** | **0.093** |

**The pde loss fell 92 % (1.183 → 0.093) while every quality metric collapsed.**
Best primary is step 0 — it never improves, not once.

Horizon, paired 30-chain CIs — every shell catastrophically worse:

| k | s0 | s200 | Δ | 95 % CI |
|---|---|---|---|---|
| 2 | 41.27 | 27.47 | **−13.80** | [−15.27, −12.37] |
| 3 | 32.00 | 21.00 | −11.00 | [−12.40, −9.73] |
| 4 | 26.83 | 10.67 | **−16.17** | [−18.73, −13.83] |
| 5 | 20.50 | 12.13 | −8.37 | [−9.40, −7.40] |
| 6 | 16.47 | 8.93 | −7.53 | [−8.37, −6.70] |
| 7 | 13.87 | 7.60 | −6.27 | [−7.07, −5.50] |
| 8 | 11.77 | 6.70 | −5.07 | [−5.63, −4.53] |

Pooled identity: phase term 0.13625 → **0.36165**, amplitude 0.01130 → **0.13370**.
physics-vs-s0 **+0.3304 [+0.3179, +0.3421] WORSE**. γ(k8) 0.626 → **0.231**: the
prediction retains ~5 % of GT energy at k8. This is the null-space collapse,
realised — blur the field toward nothing and the residual goes to zero.

## The `ic` term is load-bearing

Not a weighting detail. `ic` is the only thing standing between the residual and its
null space: blurring lowers the residual but raises `‖pred(t=0) − IC‖`, so the ic
term is an *anchor against blur*, not merely a trajectory selector. Remove it and the
optimiser walks straight down the null space.

This also settles the earlier claim that the banked "pde refuted, 25 cells" was read
inside the U-shaped valley. **It was not a valley.** For `pde` alone there is no
recovery at any budget — monotone collapse to step 200. The valley-and-recovery is a
property of `pde + ic`, not of the residual.

## Retraction — the recurrent plan needs rethinking

Earlier in this file: *"on the recurrent model `physics` reduces to `pde` alone, and
that is a feature — the ic term was scaffolding the architecture provides for free."*
**Wrong, and this run shows why.**

`ic` penalises `‖pred(t=0) − IC‖`. On a recurrent rollout frame 0 *is* the seed,
copied verbatim, so that penalty is identically zero — **satisfied, and therefore
exerting no gradient at all**. The recurrent architecture does not supply the anchor;
it removes it. Recurrent + `pde` is structurally the condition that just collapsed
here, with the residual parametrisation `u_next = u_prev + 0.104·net_out` making
"shrink the increment" the cheapest descent direction — collapse toward persistence
rather than toward zero, same category.

**The recurrent arm needs a substitute anchor before it is worth running.** Candidates:
an ic-analogue applied at a later frame, an increment-norm penalty, or the observed
early frames as genuine one-step supervision. Not a config flip.
