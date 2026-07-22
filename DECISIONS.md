# DECISIONS

## Operating checkpoint: D_step20
`tinker://f163ffa1-1141-5007-9b5a-995d9fb69b43:.../000020`

Chosen 2026-07-22 after full-holdout blind re-review and adjudication.

**Numbers (n=42, max_image_size=1344, repeats=3 majority vote, Wilson 95% CI):**

| label set | C_final | D_final | D_step20 |
|---|---|---|---|
| as-originally-labeled | 66.7% [.52,.79] | 69.0% [.54,.81] | 78.6% [.64,.88] |
| blind-corrected | 71.4% | 64.3% | **83.3% [.69,.92]** |
| adjudicated | 76.2% | 69.0% | **88.1% [.75,.95]** |

**Rationale.** Under original labels, step20's intentional recall (4/11) looked
like over-conservatism and D_final's 69.0% looked safer. The blind re-review
inverted this: 7 of the 11 original intentional labels did not survive blind
re-inspection, and step20's "misses" were largely disagreements with labels
that turned out to be wrong. Its original intentional precision was 1.000 —
it never endorsed a bad intentional label. Under adjudicated labels step20 is
best on every axis that matters for culling:
- unintentional: precision 0.852, recall **1.000** (zero degraded frames missed)
- intentional: 0.750 / 0.750 (balanced; D/C collapse to 0.20–0.25 precision)
- sharp: precision 1.000

**Error asymmetry.** For culling, missing a degraded frame (it survives into
the keeper set) is cheap — one glance. Deleting a deliberate slow-shutter shot
is irreversible. Step20's profile — perfect unintentional recall with sharp
precision 1.000 and conservative intentional calls — matches this asymmetry.

**Caveats.** (1) "Corrected" labels are the same single labeler's second
blind opinion plus two zoom-adjudicated overrides, not independent ground
truth; both overrides moved toward model predictions, so headline claims are
stated as the range 78.6–88.1% with blind-corrected 83.3% as the anchor.
(2) n=42; per-image worth ~2.4 points. (3) step20 vs final suggests possible
overfit to noisy labels after step 20 — noted as future work (early stopping
against a small clean-relabeled dev set), not investigated here.

## Adjudications
- DSC06207 → unintentional_blur (deliberate re-look; two blind passes split)
- DSC05659 → unintentional_blur (blur visible only at full zoom — the human
  analogue of the max_image_size eval bug)
Both retained in eval (n=42), documented as unstable-label case studies.

## Reference framing
Report all three label-set columns. No single "true" accuracy is claimed.