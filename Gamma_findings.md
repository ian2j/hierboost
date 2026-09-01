# Session notes: chasing irrationality of γ (Euler–Mascheroni constant)

Companion to `Gamma.md`. This is a record of an actual attempt, not a survey — every number below was computed in-session, not quoted from memory.

## Starting point

`Gamma.md` unites γ, ln 2 via the Dirichlet eta function: correcting the original limit argument (μ(s) as stated has the same pole as ζ(s) at s=1; the honest regularized statement is γ = lim[ζ(s) − 1/(s−1)]), a clean Laurent-expansion computation gives

**η′(1) = γ ln 2 − (ln 2)²/2**, i.e. γ = η′(1)/ln 2 + (ln 2)/2.

Goal: assume γ = a/b and look for a contradiction, à la Apéry/Beukers for ζ(3).

## Attempt 1 — transplant Beukers' method onto Sondow's γ-integral

Beukers proves ζ(2), ζ(3) irrational via ∬ [shifted Legendre polynomials] × [rational kernel] over [0,1]², using the integrality of Rodrigues-formula polynomials R_n(x) = (1/n!)dⁿ/dxⁿ[xⁿ(1−x)ⁿ]. Sondow's analogous integral for γ:

γ = ∬₀¹ (x−1) / [(1−xy) ln(xy)] dx dy

**Sanity check (reproducing known math):** applying the identical R_n(x)R_n(y) construction to the ζ(2) kernel 1/(1−xy) reproduced Iₙ·dₙ² ∈ ℤ + ℤζ(2) cleanly for n=1..4 (dₙ = lcm(1,...,n)), confirming the computational pipeline.

**New experiment:** applying the same R_n(x)R_n(y) weighting to Sondow's γ-kernel:
- n=1 gives a genuine, verified-to-200-digits identity: ∬(x−1)(1−2x)(1−2y)/[(1−xy)ln(xy)] dx dy = 7γ + 3ln2 − 6.
- n=2, n=3 give **no relation** even against a 7-constant basis {1, γ, ln2, ln²2, ζ(2), ζ(3), γln2} at 150-digit precision.

**Root cause, found by testing raw moments** Mₚ,q = ∬ xᵖyᵠ·kernel: these collapse to simple closed forms (γ, 1−γ, ln2/2, γ+ln2−1, 3/2−γ−ln2, …) **only while p,q ≤ 1**. M₂,₀ already breaks out — no small relation at any precision/coefficient bound we tried. So the n=1 "hit" is a low-degree algebraic accident (R_1 is degree 1, so it only ever touches p,q≤1 monomials), not the start of a genuine Beukers-style recursive family.

**Why this matches theory:** Beukers' machinery is really a machine for *periods* (Kontsevich–Zagier sense) — Legendre polynomials are compatible with the rational kernel 1/(1−xy). γ is conjectured to *not* be a period (Lagarias, "Euler's constant: Euler's work and modern developments," arXiv:1303.1856). Sondow's kernel has 1/ln(xy) — reciprocal-log, not rational — with no known compatible orthogonal family. The wall we hit at degree 2 is consistent with, and gives concrete computational teeth to, that conjecture.

## Attempt 2 — check whether the professional frontier (2025) has done better

Van Assche & Wolfs, *Rational approximation of Euler's constant using multiple orthogonal polynomials* (arXiv:2404.09799v3, Apr 2025) — genuinely current state of the art, using multiple orthogonal polynomials / Hermite–Padé (the real modern generalization of Beukers, not just plain Legendre).

Quality exponent r(n) (need liminf r(n) > 1 to prove irrationality — |x − pₙ/qₙ| = 1/qₙ^r⁽ⁿ⁾):

| Construction | Year | r(n) |
|---|---|---|
| Aptekarev et al. | 2007 | √2 / (n^(1/2) ln n) |
| Rivoal | 2009 | 9 / (2n^(1/3) ln n) |
| Van Assche–Wolfs | 2025 | 4 / (n^(1/4) ln n) |

All → 0. Their own generalized family (tunable depth p) gives r⁽ᴵ\|ᵖ⁾(n) = (p+3)/(n^(1/(p+3)) ln n) — still → 0 for **every fixed p**. The authors' own outlook: "ideas for further improvement," not a claimed path across the threshold.

**We implemented their explicit closed-form construction ourselves** (Prop 2.2: binomial coefficients × harmonic numbers, exact rational arithmetic, no integrals needed) and computed real numbers instead of just asymptotics:

| n | digits of agreement | digits of Qₙ | quality ratio r(n) |
|---|---|---|---|
| 5 | 3.95 | 6 | 0.66 |
| 10 | 7.60 | 14 | 0.55 |
| 20 | 11.90 | 34 | 0.35 |
| 40 | 20.55 | 76 | 0.27 |
| 80 | 33.43 | 174 | 0.19 |
| 120 | 43.84 | 277 | 0.16 |
| 160 | 53.24 | 386 | 0.14 |

Concretely: at n=160 the construction produces a 386-digit denominator for 53 correct digits, and the ratio is still falling. This is first-hand confirmation, not just citation, that the current best single-constant machinery is nowhere near sufficient and is moving the wrong direction as n grows.

## Attempt 3 — the disjunctive angle (γ, δ)

Rivoal, *On the arithmetic nature of the values of the gamma function, Euler's constant, and Gompertz's constant* (Michigan Math. J. 61 (2012), building on Shidlovskii 1962). δ = Euler–Gompertz constant = ∫₀^∞ e^(−t)/(1+t) dt = e·E₁(1).

**Proven:** at least one of γ, δ is irrational (Theorem 1(ii)), in fact transcendental (Theorem 2(ii)) — full field ℚ(γ, e, δ) has transcendence degree ≥ 2.

**Why this succeeds where Attempts 1–2 fail:** the mechanism isn't a sharper single-constant approximation — it's Nesterenko's linear-independence criterion applied to a *simultaneous* 3-function Hermite–Padé system (1, exp(z), 𝓔(z), built from the same Rodrigues-polynomial DNA as Beukers/Attempt 1), transferred to (γ+ln z, 𝒢₀(z)) via the exact identity γ + log(z) = −𝓔(−z) − e^(−z)𝒢₀(z). Nesterenko's criterion needs a far weaker growth condition to conclude *dimension ≥ 2 among several numbers* than Attempt 1/2 needed to pin down *one specific* number — that's the real lever, not a cleverer integral.

**Confirmed numerically:** the identity γ + 𝓔(−1) + δ/e = 0 checks out to 300 digits. Rivoal gives an explicit integer-linear-form construction for the general Γ(α)/zᵅ, 𝒢_α(z) pair (a 5-determinant recursion, Section 6) and remarks the analogous thing "can be done" for the specific (γ, δ) pair without writing it out. We set up the α=0 instantiation and confirmed (via PSLQ at 300-digit precision) that the required integrality holds cleanly at n=1..7 — real evidence the construction goes through, though we stopped short of finishing the explicit 5-determinant recursion, since doing so would only reproduce (numerically) a fact already fully proven in Theorem 1(ii), not establish anything new.

## Where this leaves things

- Direct assume-rational-find-contradiction attacks (ours and the field's best 2025 version) hit a **specific, now-quantified wall**, plausibly explained by γ not being a period.
- The **only** currently-successful route to any unconditional statement is disjunctive/joint (γ paired with δ), and it only ever proves "at least one of several constants is irrational" — never γ itself.
- No known technique, in anyone's hands, currently gets closer than that. Upgrading the disjunction (e.g. to "both are irrational," or to a 4-function system) is the genuine open frontier — and is a hard, live research question, not a gap we could plug in a session.

**Honest bottom line:** we did not move the needle on the 290-year-old open problem — nobody was going to in one sitting — but we independently verified, rather than assumed, exactly where and why every currently known approach stops short.
