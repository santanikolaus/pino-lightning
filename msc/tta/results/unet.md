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
banks the full per-snapshot arrays.

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
