# Pre-registered 10-domain test of hierboost's "when does this help" criterion

**Locked 2026-08-30, before any code in this document has been run.** Purpose: this
project has accumulated a criterion, synthesized post-hoc across ~15 prior real-data
applications, for when hierboost's block-latent decorrelation + spike-and-slab
selection should outperform (or at least meaningfully trade off against) conventional
baselines. A criterion that only explains results after seeing them is not yet
evidence the criterion is *true* — it needs to predict new, unseen cases correctly.
This document commits to 10 new domains, 5 predicted to work and 5 predicted to fail,
each with a stated mechanism and a falsifiable success/failure rule, BEFORE any of
them are run. Results will be reported exactly as found against these predictions,
including if the hit rate is bad — that outcome is itself the valuable one.

## The criterion being tested (recap, not re-derived)

**Predicts WORK** when the predictor set has real correlation that is:
- moderate, not absent (rules out MovieLens-style failures)
- moderate, not near-saturated/deterministic (rules out DARC-Binomial/Riemann-H(x)-style failures)
- genuinely *local* (clustered among some features, not diffuse across all of them — rules out state-econ-style failures)
- driven by a slow, physically/structurally stable mechanism rather than a fast, erratic, event-driven one (rules out earthquake-style failures) — checked directly via the project's own established gate: the single best raw feature's train-window correlation with the target must survive into the held-out window (same sign, held-out |corr| >= 50% of train |corr|).

**Predicts FAIL** when any of: correlation is essentially absent; correlation is
already near-deterministic (nothing left to denoise); correlation is dense/diffuse
with no localized substructure for sparsity to exploit; or the correlation is
real in-sample but event-driven/non-stationary and collapses out-of-sample.

## Falsifiable outcome rules (fixed in advance, applied uniformly)

**WORK** requires ALL of:
1. The best single raw feature passes the generalization gate above (same-sign,
   held-out |corr| >= 50% of train |corr|).
2. EITHER (a) hierboost's held-out CV accuracy/R^2 is within 5 accuracy points
   (classification) or 0.05 R^2 (regression) of the best conventional baseline
   (Lasso/PCA+linear/Random Forest) while retaining <=50% of the raw feature count,
   OR (b) hierboost clearly beats a naive baseline by a real margin AND shows a
   genuine mechanism-validation finding (posterior inclusion confidence correlates,
   in the theoretically-expected direction, with an independently-known structural
   variable — e.g. physical distance, causal/temporal order, pathway membership).

**FAIL** requires ANY of:
1. The best single feature fails the generalization gate (sign flip, or held-out
   |corr| < 50% of train |corr|).
2. hierboost's held-out accuracy/R^2 is worse than even the NAIVE (constant/mean)
   baseline, not just behind the best conventional baseline.
3. Sparsity never meaningfully engages (em_filter retains >90% of candidate
   features across a reasonable hyperparameter sweep) AND there is no compensating
   mechanism-validation finding.

Anything not cleanly WORK or FAIL is reported as MIXED — matching this project's
standing practice of not forcing a binary verdict where the honest answer is "partly."
MIXED counts as neither a hit nor a miss for the prediction it was assigned to; it is
reported separately in the final tally, not silently folded into either bucket.

## The 10 domains

### Predicted to WORK

**W1. Solar power generation, regional grid.** Open Power System Data (free, no-auth
CSV, data.open-power-system-data.org), German/European TSO-zone solar generation.
Predict target region's generation from neighboring regions' generation (spatial
blocks by TSO zone/proximity). Mechanism: same slow, physically-driven synoptic-scale
process already validated in UK weather — direct analog, high-confidence WORK.

**W2. Urban air quality (PM2.5).** EPA AQS pre-generated annual summary files
(aqs.epa.gov/aqsweb/airdata, no signup needed), monitoring stations in one metro
area. Predict one station's daily PM2.5 from nearby stations. Mechanism: pollutant
dispersion is spatially local and physically slow, same class as weather but an
independent physical process — WORK.

**W3. River streamflow network.** USGS NWIS water services (waterservices.usgs.gov,
free REST API, no auth), gauges within one real watershed. Predict a downstream
gauge's flow from upstream gauges using hierboost's `causal_affinity_1d`/directed
AR1 kernel (upstream can affect downstream, never the reverse) — the first real-data
test of this specific directed kernel outside synthetic validation. Mechanism:
genuinely causal, physically slow, local — WORK, plus a bonus test of an
underexercised capability.

**W4. Gene co-expression predicting a real clinical trait.** cBioPortal public REST
API (cbioportal.org, no auth, open-access tier only), a TCGA cohort with RNA-seq
expression and a real clinical/molecular quantitative variable (not an
ancestry-label proxy). Block by co-expression module; predict the real clinical
variable. Mechanism: closest structural analog to the model's own genomics origin,
and directly closes the "real phenotype, not proxy" gap the eQTL thread never fully
closed — WORK.

**W5. Urban traffic congestion network.** A city open-data portal's traffic
sensor/count API (e.g. Chicago or NYC Open Data, Socrata-based, no auth) — verify
live which is actually accessible before committing. Predict one road segment's
congestion from adjacent segments, graph-blocked by road-network adjacency.
Mechanism: local, physically-propagating congestion, unlike state-econ's diffuse
nationwide unemployment correlation — WORK.

### Predicted to FAIL

**F1. Wikipedia article pageviews.** Wikimedia REST pageviews API (free, no auth).
Predict one article's daily views from "topically related" articles. Mechanism:
viral/news-driven spikes are event-driven and idiosyncratic per-article — same
failure mode as earthquakes (real in-sample correlation, collapses out-of-sample).

**F2. Near-identical gold-tracking ETFs.** yfinance, GLD/IAU/SGOL/OUNZ etc. Predict
one from the others. Mechanism: these track the same underlying asset to within
basis points (correlation >0.999) — nothing left to denoise, the
DARC-Binomial/Riemann-H(x) saturation failure mode.

**F3. Cross-country macro indicators.** World Bank API (free, no auth). Predict one
country's GDP growth from many other countries' macro indicators. Mechanism: global
macro co-movement is diffuse across nearly all countries via trade/finance linkages,
not concentrated in a few "neighbors" — the state-econ dense/diffuse failure mode.

**F4. NBA player per-game statistics by position.** nba_api or a documented public
fallback (e.g. MLB's free Stats API for an equivalent sport if NBA access is
blocked — verify live, don't force a broken source). Predict one player's
performance from other same-position players'. Mechanism: individual athletic
performance is dominated by idiosyncratic skill/matchup, not shared
position-level structure — a fresh-domain analog of MovieLens's weak/absent
correlation with a plausible-sounding grouping variable.

**F5. Home price index, deliberately mis-blocked.** Zillow Research ZHVI (free,
no-auth CSV, files.zillowstatic.com), city-level home price index. Predict one
city's ZHVI from others, blocked by an arbitrary, non-causal variable (e.g.
alphabetical by city name) instead of true geographic/regional adjacency.
Mechanism: home prices likely DO have real regional correlation, but a deliberately
wrong blocking choice should destroy hierboost's ability to exploit it — a sharper,
more diagnostic failure mode (bad blocking, not domain absence of structure) than
the other four.

## Commitment

Each domain will be run by an independent agent with this document's prediction and
rules given verbatim, instructed to run the generalization gate FIRST (before any
other result), then hierboost vs. baselines, then report the verdict the rules above
actually produce — not a subjective impression. Results, once in, will not be used to
retroactively adjust which bucket a domain "really" belonged in.

## RESULTS (2026-08-30, all 10 run, all independently verified against actual code/JSON)

| Domain | Predicted | Actual | Hit? |
|---|---|---|---|
| F1 Wikipedia pageviews | FAIL | FAIL | yes |
| F2 Gold ETF basket | FAIL | MIXED | edge case, see below |
| F3 Cross-country macro | FAIL | FAIL | yes |
| F4 NBA stats by position | FAIL | FAIL | yes |
| F5 Home price, bad blocking | FAIL | FAIL | yes (mechanism not cleanly isolated, see below) |
| W1 Solar generation | WORK | WORK | yes |
| W2 Air quality PM2.5 | WORK | **FAIL** | no |
| W3 River streamflow | WORK | **FAIL** | no |
| W4 TCGA gene co-expression | WORK | **FAIL** | no |
| W5 Traffic congestion | WORK | WORK | yes |

**FAIL-side: 4/5 clean hits, 1 MIXED.** The MIXED case (F2) isn't a contradiction of
the predicted mechanism (saturation, sparsity never engaging) — it's a rule-coverage
gap: block *compression* passes the accuracy-parity clause for free when correlation
is this saturated, while the *sparsity* clause simultaneously fires FAIL, and the
document's rules don't disambiguate the two. The predicted mechanism was confirmed;
the bucket it landed in was an artifact of the rule's granularity.

**WORK-side: 2/5 clean hits, 3 misses.** This is the real headline. All three misses
have a specific, diagnosable reason, not a blank "didn't work":
- W3 (streamflow): correlation among lags was already near-saturated (0.93-0.98) —
  a violation of the criterion's OWN "moderate, not saturated" clause that the
  prediction should have checked before assigning WORK and didn't. Not a theory
  failure; an application of the theory that skipped one of its own stated conditions.
- W4 (TCGA): the gate passed and the correlated genes were biologically sensible
  (proliferation/immune markers, not noise) — every checkable box the criterion
  currently has was ticked, and hierboost still didn't beat plain baselines (Random
  Forest won). This is the most informative miss: real correlation + real local
  block structure was NOT sufficient here, unlike genomic LD blocks. Working
  hypothesis, not confirmed: correlation-threshold-clustered co-expression blocks may
  not "carve nature at its joints" as cleanly as genomic LD blocks do (a SNP has
  exactly one physical neighborhood; a gene can belong to several overlapping
  biological programs at once), so the shared-single-latent-factor assumption may fit
  LD structure better than co-expression structure even when both show real,
  local-looking correlation.
- W2 (air quality): reveals a genuinely new, previously-uncharacterized axis. Same-day
  spatial correlation was strong and real (r=0.88 held-out) -- but the target's own
  day-to-day persistence was weak (lag-1 autocorr 0.15), so the real spatial signal
  didn't survive into a one-day-ahead forecast. The criterion's "slow, physically
  stable mechanism" check was only ever applied to spatial structure, not to whether
  that structure's own temporal persistence matches the forecast lag the task needs --
  weather (synoptic systems persisting for days) and PM2.5 (which apparently mixes/
  disperses faster) are both "regional/physical" but differ on exactly this axis.

**A second, independent pattern worth keeping**: both WORK hits (W1, W5) won via the
rule's mechanism-validation clause (2b -- posterior confidence tracks known structure)
-- NEITHER won via the accuracy-parity-with-sparsity clause (2a). Across all 10
domains, clause 2a never fired even once. If this holds up as more domains are tried,
the practical, revised prediction going forward should be: expect this method's
demonstrable value in a genuinely new domain to show up as "the boosting-prior
correctly reflects independently-known structure," not as "spike-and-slab finds a
sparser subset at no accuracy cost" -- the latter is the more commonly touted value
proposition (from the 1kG/M6 work) but turned out to be the harder property to
reproduce on the first try in an unfamiliar domain.

**Overall**: 6 clean hits, 3 clean misses, 1 edge case, out of 10 -- and the
informative asymmetry (near-perfect at predicting failure, weak at predicting success)
is a more useful, more honest result than a flat 6/10 or 10/10 would have been. It
says the current criterion is a good filter for ruling domains OUT, not yet a
reliable predictor of a clean WIN -- a real, stated limitation of the theory as it
stands today, not a verdict on whether the underlying method has value.

## CRITERION v2 (2026-08-30): fifth condition added, both misses shown to be cheaply pre-checkable

Two of the three WORK misses had a specific, fixable cause. Condition 2 (moderate,
not saturated) already existed but was applied loosely (checked "is this the right
CLASS of domain," not "is the actual correlation magnitude in-range"). Condition 5
didn't exist at all before this test. Both are now stated as explicit, cheap,
PRE-FIT checks -- computable from raw fetched data alone, no model needed -- so a
future prediction can be disciplined about applying them, not just gesture at them.

**Condition 2, sharpened**: before predicting WORK, compute the actual same-day
(lag-0) pairwise correlation among the raw candidate predictors. If the max exceeds
~0.9, downgrade the prediction regardless of how "physically appropriate" the domain
class looks -- retroactively confirmed against the already-fetched streamflow data:
max same-day gauge-to-gauge correlation was 0.995, computable before any model was
fit, and would have correctly flagged the saturation risk in advance.

**Condition 5 (NEW): the mechanism's own temporal persistence must be checked
against the SPECIFIC forecast lag the task needs, separately from spatial locality.**
Before predicting WORK, compute the target's own lag-1 (or whatever lag the task
uses) autocorrelation from the raw series alone. If it's small (rule of thumb: below
~0.3) while same-day cross-unit correlation is large, the domain likely has real
spatial structure that won't survive into a lagged forecast -- retroactively
confirmed against the already-fetched air-quality data: same-day cross-site
correlation was 0.83 (strong, real), but the target's own lag-1 autocorrelation was
only 0.15 -- exactly the signature this condition is built to catch, computable
before any model was fit.

**Full criterion, v2**: predicts WORK when ALL FIVE hold: (1) real correlation
exists (not absent), (2) same-day pairwise correlation among candidates is moderate,
not saturated (rule of thumb: below ~0.9), (3) correlation is local/clustered, not
diffuse across everything, (4) the single best raw feature passes the train/test
generalization gate (same sign, held-out |corr| >= 50% of train |corr|), (5) the
target's own persistence at the task's specific forecast lag is not small relative
to its same-day cross-unit correlation (rule of thumb: lag-1 autocorrelation above
~0.3, or at least the same order of magnitude as the same-day cross-correlation).

## Prospective re-test of v2 (new domain, chosen and predicted BEFORE running anything)

**Snowpack (snow water equivalent) across a mountain-range SNOTEL network** --
picked specifically to sit in the same "public environmental sensor network" family
as the two domains that already worked (solar, traffic) while deliberately avoiding
both newly-identified traps: unlike streamflow, snowpack accumulates and melts on a
timescale of days-to-weeks rather than being hydraulically connected within hours,
so same-day station-to-station correlation is expected to be real but not
saturated; and unlike PM2.5, snow water equivalent changes slowly and smoothly
day-to-day (it doesn't "mix away" the way an atmospheric pollutant does), so daily
persistence should be strong. **Predicted: WORK**, pending the same live checks
every prior domain got (conditions 2 and 5 computed from raw data BEFORE trusting
the prediction, exactly as this section commits to doing differently now).

**Result (2026-08-30): WORK, confirmed -- but the pre-checks alone would have gotten
this wrong, which is the more important finding.** Condition-2 pre-check flagged real
saturation (max same-day station correlation 0.98, comparable to streamflow's 0.995)
-- by the v2 rule of thumb, this should have been a warning sign just like
streamflow. Condition-5 pre-check was clean (target lag-1 autocorrelation 0.999,
no air-quality-style persistence gap). Generalization gate passed (ratio 0.965).
Full pipeline: hierboost R^2=0.953, essentially tied with Random Forest's 0.958 (best
conventional baseline), sparsity never engaged (100% retention, same pattern as the
saturated gold-ETF case) -- exactly the profile that predicted FAIL for streamflow.
**But the verdict is WORK**, because the mechanism-validation signal (posterior
confidence vs. physical distance) is rho=-0.828 -- the STRONGEST of any domain tried
in this entire project (vs. solar's -0.64, traffic's -0.38, and streamflow's weak,
non-significant -0.20, which is what actually sank streamflow despite near-identical
saturation).

**CRITERION v2, revised again: condition 2's saturation flag does not predict
WORK/FAIL by itself -- it predicts which ROUTE is available.** A saturated domain
(condition 2 flagged) can't win via route 2a (sparsity + accuracy parity) --
compression becomes "free" and spike-and-slab never prunes anything, exactly as seen
in gold-ETF, streamflow, AND snowpack alike. Whether it still wins via route 2b
(mechanism validation) depends on a quantity condition 2 cannot see: how cleanly the
boosting-prior's posterior confidence tracks genuinely independent, checkable
structure. Streamflow and snowpack are both saturated; one had a weak/non-significant
mechanism signal and failed, the other had the strongest mechanism signal in the
whole project and worked. **The practical rule going forward**: when condition 2
flags saturation, don't predict FAIL outright -- predict "route 2a is closed, the
verdict now depends entirely on route 2b," and that has to be checked empirically
(the boosting-prior's theta-vs-structure correlation), not estimated in advance from
correlation magnitude alone. This is a real, useful sharpening of the theory that a
clean confirmation of the original v2 guess would NOT have produced -- the criterion
is now better characterized after this prospective test than before it, regardless of
the raw WORK/FAIL tally.
