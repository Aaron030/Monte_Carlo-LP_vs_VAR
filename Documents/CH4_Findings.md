# Chapter 4 — Extension 1: Dynamic (VARMA) Misspecification — Findings

> Working notes synthesising the user's figure readings with the mechanism narrative from the
> `Extension_Dynamic_Misspec` notebooks, in the same style as [CH3_Findings.md](CH3_Findings.md)
> and [CH5_Findings.md](CH5_Findings.md). **[obs]** = read off the figures (user's readings);
> **[mech]** = interpretation/mechanism; **[check]** = to verify / decide before writing prose.

---

## 0. What this chapter is testing (recap)

Extension 1 stresses the LP(4)-vs-VAR(q)-sweep comparison with **dynamic misspecification**: the
baseline VAR(4) is augmented with a first-order moving-average term,

```
y_t = Σ_{i=1}^4 A_i y_{t-i} + u_t + Θ u_{t-1},   Θ = θ I_2,   u_t = Bε_t,   θ ∈ {0, 0.1, 0.3, 0.6}
```

The AR matrices `A_i` and impact `B` are inherited from the baseline, so **all** variation is the MA
component (θ = 0 nests the baseline VAR(4) exactly).

Key design facts:

- **Estimand = the true VARMA structural IRF** `θ_h = (Φ_h B)[1,1]` (var1 ← shock1), computed
  analytically per θ. Unlike Extension 2, the estimand is **θ-dependent** — the MA term genuinely
  lifts the short-horizon response.
- **The invertible MA admits a VAR(∞) representation.** A high-order VAR(q) can approximate it; the
  equal-lag **VAR(4) cannot** — it carries an **asymptotic bias that does not vanish with T**; **LP(4)
  is robust by construction** (Montiel Olea / Plagborg-Møller & Wolf). This is the crux that makes
  Ext 1 the *mirror* of Ext 2.
- **Persistence ρ ∈ {0.5, 0.7, 0.95}** (the full range — the VARMA is well-behaved at high ρ, unlike
  the explosive lag-quadratic of Ext 2). Point/coverage figures at **T = 250, B = 5,000**; asymptotic
  checks at T = 10,000.
- Estimators (fixed LP(4), VAR(q) sweep q = 1…23) applied **unchanged**.

## 1. The central thesis — the mirror image of Extension 2

Ext 2 and Ext 1 are the two halves of the same lag-sufficiency principle (CH5 §5.2):

- **Ext 2 (nonlinear):** the misspecification leaves the required linear-lag order at 4 → PMW
  equivalence **holds** at equal lag; the action is in *variance* (VAR wins RMSE) and *inference*
  (LP wins coverage under heteroskedasticity).
- **Ext 1 (VARMA):** the MA pushes the required linear-lag order to **∞** → PMW equivalence **fails**
  at equal lag; VAR(4) accumulates a genuine **non-vanishing bias** (≈ 0.045 at the short horizons
  where Θu_{t-1} acts; **VAR(4) − LP(4) ≈ 0.046 in population**), while LP(4) stays consistent.

So Ext 1 is the one place in the study with a real **asymptotic bias channel** against the equal-lag
VAR. The thesis question is whether that bias is large enough to overturn the conventional verdict.

**Headline (established finding):** it is **not**. The asymptotic VAR(4) bias is small, and at finite
T it is **swamped by LP's higher variance**, so **VAR(4) keeps its RMSE edge at every T ≤ 500**.
And on inference, coverage turns out to be governed by **persistence, not θ** — MA misspecification
barely moves it at T = 250 — so **LP's asymptotic robustness buys no decisive finite-sample edge
here** (the opposite of Ext 2). The MA can be made visible (it lifts the IRF, and it leaves
serially-correlated VAR residuals), but it does not flip the practical ranking.

---

## 2. Figure-by-figure

### 2.1 True IRF Functions + residual ACF diagnostics
Three panels: (left) true IRF by θ; (middle) ACF of **squared** VAR(4) residuals; (right) ACF of
**level** residuals — the diagnostic added to mirror Ext 2's residual panel.

- **[obs] IRF:** standard IRFs that become more **L-shaped** as persistence increases. Higher θ pushes
  the shape **hump-shaped**; at **high ρ + high θ** the response actually **rises to a peak at h = 1**
  before decaying (a genuine hump, not a monotone decay).
- **[obs] ACF of squared residuals:** all θ curves look essentially **identical** and stay **inside
  the white-noise band** at every ρ — a concave oscillation that is just noise.
- **[obs] ACF of level residuals:** strong oscillation at all ρ. At **low ρ**, lags **4 (negative) and
  5 (positive)** breach the band; at **mid ρ**, lags 4 & 5 again significant (5 slightly smaller than
  4); at **high ρ**, **only lag 4** remains significant. Apart from lags 4–5 the shape is the same
  across ρ.
- **[mech]** This is the **dynamic-misspecification signature** and the exact mirror of Ext 2. The
  squared residuals are white → the VARMA innovations are **conditionally homoskedastic** (no
  volatility clustering — contrast Ext 2, where squares cluster). The level residuals are **not**
  white: the equal-lag VAR(4) fits only the first four VAR(∞) coefficients, so the leftover MA serial
  correlation concentrates at the **truncation boundary (lags ~4–5)**, with **alternating sign**
  (neg at 4, pos at 5) — the signature of an invertible MA(1)'s alternating VAR(∞) coefficients
  (Π_k ∝ (−Θ)^k). The break-out is carried by the **θ = 0.6 curve** and grows with θ (θ = 0 is white
  by construction). The fact that **squares stay white while levels do not** pins the misspecification
  as purely *dynamic* (conditional-mean serial correlation), the mechanism behind VAR(4)'s
  non-vanishing bias.
- **[check]** Why only lag 4 survives at high ρ (lag 5 retreats into the band): likely the strong AR
  persistence redistributes/absorbs more of the boundary correlation. Worth a one-line note, not load-bearing.

### 2.2 Complexity Frontier under Dynamic Misspecification (T = 250)
Horizon-averaged RMSE vs VAR order q, one solid curve per θ, with LP(4) the dashed reference of the
same colour; vertical line at the equal-lag VAR(4). **[Now included in chap_4.tex; was previously
missing.]**

- **[obs, low ρ]** High-q tails look similar across θ; the low-q end differs. θ = 0: standard U,
  **VAR(1) ≈ VAR(2)** are jointly best. θ = 0.1: whole curve shifts up a touch and VAR(1) ≠ VAR(2),
  with **VAR(1) now best**. θ = 0.3: shifts up more, VAR(1) worsens. θ = 0.6: VAR(1) and VAR(2) both
  degrade and the low-q end takes an **S/U shape**; the whole curve sits higher. **LP shifts upward
  with every θ increase.**
- **[obs, mid ρ]** Same behaviour, but now **VAR(1) starts worse than VAR(2)**; by θ = 0.3 it reverts
  and VAR(1) is best again, then both rise into the S shape. LP again lifts with each θ.
- **[obs, high ρ]** "Weird" shape: an S with **VAR(1)/VAR(2) much worse than even VAR(23)**. Curves
  shift up with θ, but VAR(1)'s badness *decreases* with θ; by θ = 0.6 VAR(1) beats VAR(2) again and
  the low-q end shows an **"A-jump"** — VAR(1) ok, **VAR(2) terrible**, VAR(3) ok, then RMSE rising
  slowly with q.
- **[mech]** The frontier **lifts with θ for every estimator** (both the VAR minimum and the LP line
  rise): the MA injects approximation error that everyone must absorb. Crucially, the **VAR minimum
  stays well below the LP(4) line at every θ** — even at θ = 0.6, high ρ, the equal-lag VAR(4)
  (≈ 0.13) sits below LP(4) (≈ 0.164). So **VAR keeps its RMSE edge under misspecification** at
  T = 250. The RMSE-minimising order stays **low (1–3)**: in principle more lags would shrink the
  MA-approximation bias, but at T = 250 the **variance penalty of extra lags dominates that bias
  reduction**, so the trade still favours parsimony. The high-q tail approaching the LP line (crossing
  around **q* ≈ 16–18**) is the **lag-augmentation** result — a sufficiently high-order VAR behaves
  like LP, consistent with the VAR(∞) representation.
- **[mech] The under-lagged instability (VAR(1)/VAR(2)) is the wrong-estimand channel**, amplified by
  the MA and by persistence: near the unit root with an MA term, one or two lags span neither the AR
  nor the MA dynamics, so their RMSE explodes (the "A-jump"). The non-monotone VAR(1)-vs-VAR(2)
  swapping with θ is a second-order interaction of under-lagging with the MA; **[check]** worth one
  cautious sentence, not over-interpretation.

### 2.3 Point-Estimation Performance (θ = 0.6, T = 250)
Bias / variance / MSE / RMSE for LP(4) and VAR(2/4/8/16/23).

- **[obs]** Very similar to the Ext 2 point-estimation figure: increasing curvature as ρ rises.
  **VAR(2) bias blows up**, and more so at higher ρ. The other VARs jump around at low h then go
  roughly **horizontal and converge toward zero**; LP stays roughly horizontal with light
  oscillation, and as ρ increases **LP's bias levels off at a more negative value**. In
  variance / MSE / RMSE the pattern matches Ext 2: **VAR(2) blows up only in MSE and RMSE, not in
  variance.**
- **[mech]** The **VAR(2) RMSE/MSE-but-not-variance blow-up is the decisive wrong-estimand
  diagnostic** (MSE = bias² + variance): VAR(2) is parsimonious → low variance, but under-lagged →
  large bias, so its squared error is bias-dominated. This is the same lag-order axis as Ext 2 §2.6.
- **[mech] What is *different* from the baseline and Ext 2:** here even the well-lagged **VAR(4)
  carries a genuine (if small) non-vanishing bias** against the VARMA truth, because no finite-order
  VAR is correctly specified. The **dropped Asymptotic-Deviation figure makes this explicit**: at
  T = 10,000 LP(4) is flat on zero at every horizon, while VAR(4) keeps a **short-horizon bump
  ≈ 0.045 at h ≈ 4** (≈ 4% of the IRF peak) that no increase in T removes. At T = 250, though, this
  bias is invisible — the **dropped Small-Sample-Deviation figure** shows VAR(4) and LP(4) sitting
  almost on top of each other with overlapping wide bands, so the finite-sample story is again a
  **variance story** (LP ~2–3× noisier), and VAR(4) wins RMSE.

### 2.4 Confidence-Interval Coverage under Dynamic Misspecification (T = 250) — the key result
Three panels: VAR(4) coverage across θ, LP(4) coverage across θ, and a VAR(4) SE-vs-SD validation
panel (θ = 0.6). **Coverage is reported over the plateau h = 1…20** (h = 0 is a degenerate
normalization point — the impact response is fixed at θ₀ = B[0,0] = 1 with a zero-width delta-method
CI, so its coverage is 0 by construction; it has been removed from all coverage figures and averages).

- **[obs, low ρ]** Behaves like the base case: **LP sits at 95%**; VAR **over-covers, reaching ~100%
  at higher horizons**. Increasing θ makes VAR slightly worse at the **short** horizons only. SE ≈ SD
  (validation panel — the two lie on top of each other).
- **[obs, mid ρ]** LP at 95% with slight fanning at h = 1 then quick return. VAR approaches 95%, then
  **humps downward from ~h6 to ~h12**, then fans: **θ = 0.6 actually returns to 95%**, while the lower
  θ slope downward — **θ = 0 performs worst**. SE ≈ SD still (no real change from low ρ).
- **[obs, high ρ]** VAR curves **collapse onto each other regardless of θ**; a slight hump at h1–h5,
  then a declining, mildly fanned tail with **θ = 0.6 worst**. LP also turns slightly negative, curves
  on top of each other. The validation panel changes shape: the tail is **concave (not convex)** and
  the **delta-method SE is shifted *below* the empirical SD** (it under-states dispersion).
- **[mech] The dominant axis is persistence, not θ** — exactly the base-case mechanism, and the
  numbers confirm it (plateau h = 1…20, var1 ← shock1, T = 250, B = 5,000):

| ρ | estimator | θ = 0 | θ = 0.1 | θ = 0.3 | θ = 0.6 |
|------|-----------|-------|---------|---------|---------|
| 0.5 | **VAR(4)** | 0.988 | 0.987 | 0.984 | 0.974 |
| 0.5 | LP(4) | 0.948 | 0.947 | 0.945 | 0.942 |
| 0.7 | **VAR(4)** | 0.889 | 0.905 | 0.925 | 0.930 |
| 0.7 | LP(4) | 0.944 | 0.943 | 0.941 | 0.937 |
| 0.95 | **VAR(4)** | 0.845 | 0.845 | 0.844 | 0.838 |
| 0.95 | LP(4) | 0.897 | 0.893 | 0.888 | 0.883 |

  - **VAR(4) coverage is governed by ρ:** over-covers at ρ = 0.5 (≈ 0.99), drifts to ≈ 0.89–0.93 at
    ρ = 0.7, under-covers at ρ = 0.95 (≈ 0.84). **LP holds much closer to nominal throughout**
    (0.95 → 0.94 → 0.89) but also degrades at high ρ — so LP's robustness is **partial**, not the clean
    win of Ext 2.
  - **The θ effect is second-order and non-monotone.** At ρ = 0.5 VAR drifts *down* slightly with θ
    (0.988 → 0.974); at ρ = 0.7 it actually **rises** with θ (0.889 → 0.930, θ = 0 worst — matching the
    "θ = 0 performs worst" reading); at ρ = 0.95 it is **flat in θ** (0.845 → 0.838, "curves on top of
    each other"). So MA misspecification does **not** monotonically erode coverage.
  - **[mech] Why the mid-ρ non-monotonicity (do not over-read).** At θ = 0 the correctly-specified
    VAR(4) already mildly under-covers at mid persistence (the base-case delta-method effect). As θ
    grows, the MA lifts the short-horizon IRF and the interaction with the delta-method SE happens to
    push coverage *back toward* nominal — a **fortuitous interaction, not a genuine inferential benefit
    of misspecification**. Anchor prose on the persistence axis and the LP-vs-VAR gap, not on this.
  - **[mech] SE validation:** at low/mid ρ the delta-method SE tracks the empirical SD (SE valid → the
    coverage pattern is not a broken-SE artifact); at high ρ the delta SE shifts **below** the SD (the
    near-unit-root linearisation under-states dispersion → CIs too narrow → under-coverage). This is
    the **same base-case near-unit-root failure**, essentially unchanged by θ — confirming the MA is
    not the driver of the coverage story here.

---

## 3. Synthesis

1. **The MA is a genuine, visible misspecification — but a dynamic one.** It lifts the true IRF
   (hump-shaped at high ρ/θ) and leaves the equal-lag VAR(4)'s residuals **serially correlated**
   (level-ACF break-out at lags 4–5) while keeping them **homoskedastic** (squared-ACF white). This is
   the exact mirror of Ext 2's white-levels/clustering-squares signature. (Fig 2.1.)
2. **Equal-lag PMW equivalence fails — VAR(4) has a non-vanishing bias — but it is small.** ≈ 0.045 at
   the short horizons where the MA bites (VAR(4) − LP(4) ≈ 0.046 in population), shown by the
   asymptotic-deviation check. (Figs 2.1, 2.3; dropped asymptotic figure.)
3. **VAR still wins point estimation at every finite T.** At T = 250 the bias is swamped by LP's
   ~2–3× higher variance; the VAR frontier minimum sits below the LP line at all θ, and VAR(2)'s
   under-lagging blow-up (RMSE/MSE but not variance) is the only large bias on display. (Figs 2.2, 2.3.)
4. **Lag augmentation restores equivalence.** A high-order VAR(q) approaches the LP line
   (crossing q* ≈ 16–18) — the finite-sample image of the VAR(∞) representation. (Fig 2.2.)
5. **Inference is governed by persistence, not by the misspecification.** VAR coverage moves from
   over-covering (ρ = 0.5) to under-covering (ρ = 0.95) via the base-case delta-method/near-unit-root
   mechanism; the θ effect is small and non-monotone. LP holds closer to nominal but also degrades at
   high ρ. So **LP's robustness does not buy a decisive inference edge here** — unlike Ext 2. (Fig 2.4.)
6. **Net verdict:** dynamic misspecification of this invertible-VARMA type **does not overturn the
   conventional ranking** at empirically relevant T: VAR(4) wins RMSE and is no worse than LP on
   coverage (both persistence-limited). LP's theoretical asymptotic robustness is real but
   **finite-sample-irrelevant** at T ≤ 500.

## 4. Figure set / decisions

- **[added] Complexity Frontier** (`DYN_MISSPEC_COMPLEXITY_FRONTIER_rho=*_T=250.png`, §2.2): added to
  chap_4.tex after the True IRF, for parity with BL/EXT2 — it is the thesis's complexity-adjusted
  reference figure and carries the "VAR wins RMSE even under misspecification" + lag-augmentation story.
- **[True IRF upgraded]** `DYN_MISSPEC_IRF` now carries the squared/level residual-ACF panels (added
  to mirror Ext 2), giving the dynamic-misspecification signature directly.
- **[h = 0 removed]** all coverage figures/averages now run over the plateau h = 1…20 (h = 0 is the
  degenerate impact-normalization point).
- **[drop] Small-Sample Deviation** (`DYN_MISSPEC_LIMITED_DEVIATION`): redundant with the
  point-estimation panels; its content (VAR(4) ≈ LP(4) at T = 250) is folded into §2.3 as supporting
  insight.
- **[drop] Asymptotic Deviation** (`DYN_MISSPEC_ASYMPTOTIC_DEVIATION`): the cleanest evidence of
  VAR(4)'s non-vanishing bias, but a single point can be quoted in prose (§2.3, §3.2) rather than
  shown.
- **[drop] Sample RMSE by Horizon** (`DYN_MISSPEC_RMSE_BY_HORIZON`): redundant with the
  point-estimation RMSE panel and the frontier (same call as in BL/EXT2).

Recommended Ext 1 figure set: **True IRF (+ residual ACF) → Complexity Frontier → Point Estimation →
Coverage (+ SE validation)** — mirroring the lean BL/EXT2 sets.

---

## 5. The core question — does PMW's equivalence hold under *dynamic* misspecification?

> *PMW: linear LP and linear VAR estimate the same impulse responses at sufficient/equal lag length.
> The VARMA is the case designed to break the "equal lag" clause.*

**Verdict: at equal lag the equivalence genuinely fails — VAR(4) carries a non-vanishing bias against
the VARMA — but it is restored by lag augmentation, and the failure is too small to matter in finite
samples.** This is the complement to Ext 2, and together they state one principle.

### 5.1 Equal-lag equivalence fails — this is the one clean counterexample
The invertible VARMA's best linear predictor is a **VAR(∞)**: four lags are *insufficient*. So the
equal-lag VAR(4) accumulates a **non-vanishing asymptotic bias** (≈ 0.045 at the short horizons where
Θu_{t-1} acts; VAR(4) − LP(4) ≈ 0.046 in population), while LP(4) stays consistent. The residual-ACF
panel (§2.1) shows the mechanism directly: VAR(4) leaves serially-correlated residuals it cannot
absorb. **Finding 1: dynamic misspecification that inflates the required lag order does break equal-lag
PMW equivalence** — the boundary case Ext 2 did not reach.

### 5.2 …but lag sufficiency restores it — the unifying principle with Ext 2
A high-order VAR(q) approximates the VAR(∞) and converges back onto the LP line (q* ≈ 16–18 on the
frontier). So the equivalence is **gated by lag sufficiency**, exactly as in Ext 2 §5.2:

> **Unifying principle.** PMW equivalence holds **iff the (mis)specification leaves the required
> linear-lag order within the fitted order.** The orthogonal lag-quadratic (Ext 2) keeps it at 4 →
> equivalence holds at equal lag. The invertible VARMA (Ext 1) pushes it to ∞ → equal-lag equivalence
> fails, but lag augmentation recovers it. "Same responses even when misspecified" is about whether
> the misspecification inflates the lag order the linear projection needs — not about linearity per se.

### 5.3 The estimand failure does not change the practical verdict
- **Point estimation (still favours VAR).** The asymptotic bias is real but small, and at T ≤ 500 it
  is swamped by LP's ~2–3× variance: **VAR(4) wins RMSE at every sample size**, and the frontier
  minimum stays below the LP line at all θ. LP's robustness buys **no finite-sample point-estimation
  edge**. **Finding 2.**
- **Inference (no clean LP edge either).** Unlike Ext 2 — where heteroskedasticity made VAR's
  homoskedastic CI under-cover while LP's HC1 held — here the VARMA innovations are **homoskedastic**,
  so VAR's delta-method SE is fine (SE ≈ SD at low/mid ρ). Coverage is therefore driven by the
  **base-case persistence mechanism**, not by θ; LP holds closer to nominal but also degrades near the
  unit root, and the θ effect on VAR coverage is small and non-monotone. **Finding 3: LP's asymptotic
  robustness does not translate into a finite-sample inference advantage under this misspecification.**

### 5.4 Why this matters
1. **It is the discriminating case for the lag-sufficiency principle.** Ext 2 shows equivalence
   surviving a nonlinearity; Ext 1 shows it breaking under dynamics — and the single distinguishing
   feature is whether the required lag order stays finite. That contrast is the thesis's unifying
   contribution.
2. **It cautions against the equal-lag comparison.** When the misspecification is dynamic enough to
   demand more lags, comparing LP(4) to VAR(4) is comparing a robust estimator to a biased one — the
   honest comparison is LP against the **VAR frontier**, where lag augmentation closes the gap.
3. **It sharpens the "LP is robust" claim.** LP's robustness here is genuine but **asymptotic and
   finite-sample-irrelevant**: at T ≤ 500 the equal-lag VAR's bias is too small to overturn its
   variance advantage, and there is no heteroskedasticity for LP's HC1 to exploit. LP's edge is real
   only when the misspecification both inflates the lag order *and* the sample is large — or when it
   injects heteroskedasticity (Ext 2).

**One-sentence answer:** *Dynamic (VARMA) misspecification is the case that genuinely breaks equal-lag
PMW equivalence — VAR(4) carries a non-vanishing bias — yet because that bias is tiny relative to LP's
variance and the innovations stay homoskedastic, VAR(4) remains the better finite-sample estimator on
both RMSE and coverage, so the equivalence "fails" exactly where it cannot be exploited.*
