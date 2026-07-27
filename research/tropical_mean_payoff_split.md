# The tropical shadow and the residue-field dial: a split-theorem cluster for degree-window systems

*Working note, 2026-07-27. Grew out of a conversation about whether tropical geometry connects
to the CSP/all-different programme. Not yet promoted to the manuscript; promotion gates at the
end. Claim tags: **[known]** (literature, citation pinned or pinnable), **[new, sketch
complete]** (proof sketch below believed complete; needs write-up and adversarial check),
**[new, provable]** (clear strategy, gaps identified), **[open]**, **[verify]** (citation or
comparison not yet checked). The formalisation companion is
`research/tropical_split_formalisation.md`.*

## 1. Correction to the conversational version

The chat discussion of 2026-07-27 proposed a 2×2 grid whose open cell was "signed max-degree
minimisation over F_q[t] is mean-payoff-equivalent". Working the definitions out precisely
shows that placement is wrong, and the corrected picture is sharper:

- The mean-payoff rung belongs to the **tropical shadow itself** (the purely combinatorial
  degree/tie system, no lift), where it is already known **[known]**.
- The lifted problem over F_q[t] does not sit at the shadow's complexity. It **splits by the
  residue field**: polynomial-time when the field outnumbers the disjunctive rows
  (Theorem A), NP-complete for every fixed q (Proposition B2), with an exact-degree fragment
  that is polynomial precisely at q = 2 (Proposition B1) — reproducing the shape of the
  Bodirsky–Fehm p = 2 anomaly already flagged in the dossier **[verify]**.
- Consequently tropicalization is **not complexity-monotone in either direction**: over a
  large residue field the shadow is strictly harder than the lift (unless mean-payoff games
  are in P), and over a fixed small field the lift is strictly harder than the shadow
  (unless NP = co-NP). The mechanism for the second gap is exactly the dossier's design
  caveat (c) — phantom cancellation — promoted from nuisance to complexity-theoretic
  separator.

This answers the dossier's open question ("does the k[t]-degree regression collapse to a
known tropical regression problem?") with a quantified **no**, in both directions, which is
the outcome the dossier said would itself be a theorem.

## 2. The object: degree-window feasibility

Fix a field k. An instance consists of:

- unknowns x_1, …, x_n ∈ k[t] with degree caps deg x_j ≤ c_j (caps written in unary);
- rows i = 1..m: u_i ∈ k[t]^n, b_i ∈ k[t], residual r_i(x) = u_i^T x − b_i;
- a window [l_i, u_i] per row, l_i ∈ ℤ ∪ {−∞}, u_i ∈ ℤ ∪ {+∞}, with deg 0 = −∞.

**DWF_k (degree-window feasibility).** Does there exist x within the caps such that
l_i ≤ deg r_i(x) ≤ u_i for every row?

Semantics in the paper's vocabulary: an upper bound is a *tolerance* ("constraint i may be
false only up to weight u_i"); the degenerate window [−∞, −∞] is *exact satisfaction*
(r_i = 0); a finite lower bound is a *reward row* ("residual must be at least this false") —
the function-field face of the paper's signed/negative rows. The all-different disequality
rows of the existing construction are literally the window [0, 0] on r = x_a − x_b over
constants (see §6).

Structural observation (elementary but load-bearing): on the coefficient space
K^D, D = Σ_j (c_j + 1), every upper bound is a conjunction of affine-linear conditions
(coefficients above the bound vanish), so the upper-bound-feasible set is an affine subspace
V. Every finite lower bound l_i excludes an affine subspace
W_i = V ∩ {coefficients of r_i in [l_i, min(u_i, apriori bound)] all vanish}. Hence

    Feasible set  =  V ∖ (W_1 ∪ … ∪ W_N),

with N = number of rows carrying a finite lower bound. The whole signed ultrametric layer is
"an affine subspace minus a union of affine subspaces". Everything below follows from taking
this seriously.

## 3. Theorem A — large residue field: the signed ultrametric layer is easy

**Theorem A [new, sketch complete].** If |k| ≥ N + 1 (in particular if k is infinite), then
DWF_k is decidable in polynomial time, a witness is computable, and feasibility holds iff
V ≠ ∅ and no single W_i equals V. Both directions carry polynomial-size certificates.

*Proof sketch.* (⇒ of the criterion) If V = ∅ or some W_i = V, infeasibility is immediate;
both conditions are rank computations. (⇐) Covering lemma: a nonempty affine space over k
cannot be covered by N proper affine subspaces when |k| > N. Finite case by counting: a
proper affine subspace of a d-dimensional affine space over F_q has at most q^{d−1} points,
so the union has at most N·q^{d−1} < q^d points. Infinite case standard. Witness search:
random sampling succeeds with probability ≥ 1 − N/|k|; a deterministic line-search
derandomisation is routine. Over ℚ, bit growth is controlled by standard Gaussian
elimination bounds. ∎

Remarks. (i) The caps-in-unary convention matters: caps in binary make D exponential. The
paper's instances are cap-bounded by construction. (ii) The infeasibility certificate (a
containment V ⊆ W_i, i.e., a linear-algebra witness) is the function-field counterpart of a
dual certificate; the aesthetic match with mean-payoff duality is a remark, not a theorem.
(iii) Even the MaxCSP variant is easy in this regime: discard the rows with W_i = V
(unavoidable), and the covering lemma satisfies *all* remaining reward rows simultaneously.
Partial rewards give nothing extra when the field is large — slack is everywhere.

## 4. Propositions B1, B2 — fixed residue field: hardness, and the q = 2 anomaly reproduced

**Proposition B1 (exact-degree dichotomy) [new, sketch complete].** Restrict all windows to
be either upper-only or exact ([d, d]). Over F_2 the problem is in P. Over every fixed F_q
with q ≥ 3 it is NP-complete.

*Proof sketch.* Over F_2, "deg r = d" says: coefficients above d vanish (affine) and the
coefficient at d is nonzero — but over F_2 "nonzero" means "= 1", also affine. So the whole
system is one affine system; linear algebra decides it. Over F_q with q ≥ 3, take cap 0 on
every unknown, so each x_v ranges over the q constants — the field is the colour domain.
For each edge {a, b} of a graph, add the row r = x_a − x_b with window [0, 0]: exactly
x_a ≠ x_b. Feasibility is graph q-colourability, NP-complete for every fixed q ≥ 3.
Membership in NP is immediate. ∎

**Proposition B2 (windows restore hardness at q = 2) [new, sketch complete].** With general
windows, DWF_{F_q} is NP-complete for every fixed q ≥ 2.

*Proof sketch.* Only q = 2 remains. Variables: cap-0 unknowns z_i ∈ F_2. For a 3-SAT clause
with literals ℓ_1, ℓ_2, ℓ_3 over variables z_{i_1}, z_{i_2}, z_{i_3}, let ε_a = 1 if ℓ_a is
negated and 0 otherwise, and add the row

    r_C = (z_{i_1} + ε_1) + t·(z_{i_2} + ε_2) + t²·(z_{i_3} + ε_3),  window [0, 2].

The three coefficient slots are the three literal values (no convolution interaction, since
the z's are constants), the upper bound 2 is automatic, and deg r_C ≥ 0 says r_C ≠ 0, i.e.,
some literal is true. So 3-SAT reduces to DWF_{F_2} with windows of width 3. ∎

Reading. Over F_2, "nonzero" is affine — F_2 is the unique field where a disequality is a
conjunction — so exactness is easy there and hardness needs genuine windows. Over q ≥ 3 a
single exact-degree row is already a (q−1)-way disjunction. This reproduces, from one page
of linear algebra, the *shape* of the Bodirsky–Fehm dichotomy recorded in the dossier
("valuation-constrained linear systems NP-complete for p ≥ 3, in P for p = 2",
arXiv:2504.13536) **[verify: fetch the paper, align constraint formats; if their format
subsumes general windows the q = 2 claims must be reconciled — my B2 instances use width-3
windows, which exactness-based formats do not express, so no contradiction is expected]**.

Consistency with Theorem A: the hard instances have N ≈ number of edges/clauses ≫ q. The
dial is q versus N — pigeonhole. With q ≤ N the reward rows can be forced to collide
(cancellation is compulsory somewhere); with q > N there is slack everywhere. This is the
equal-characteristic incarnation of "the hardness is in the carries/forced cancellation",
and it is consistent with the left-field sweep's "F_2[t] as carry-free control" only after
the B1/B2 refinement: F_2 is the easy case for *exact* constraints and a hard case for
*window* constraints.

## 5. Theorem C — the tropical shadow and its mean-payoff rung

Shadow of an instance: replace each unknown by its degree ξ_j ∈ ℤ ∪ {−∞} and each row by
its tropical value M_i(ξ) = max(max_k(deg u_{ik} + ξ_k), deg b_i). The Kapranov-necessary
conditions on ξ (necessary for any lift; see T6 in the formalisation spec) are, per row:

    either  M_i(ξ) ∈ [l_i, u_i],   or the max in M_i(ξ) is attained at least twice.

Call feasibility of this system TDWF. Ties are how the shadow models cancellation; once a
tie exists, tropical information alone bounds the lifted degree on neither side.

**Theorem C [known + translation].** The exact-satisfaction fragment (windows [−∞, −∞]) of
TDWF is precisely solvability of tropical linear systems ("max attained at least twice in
every row"), which is polynomial-time equivalent to deciding mean payoff games, hence in
NP ∩ co-NP and not known to be in P (Grigoriev–Podolskii; Akian–Gaubert–Guterman). General
TDWF is therefore MPG-hard; it is in NP (certificate: ξ plus a tie pattern); whether it
remains MPG-equivalent with general windows is **[open]** (expected yes; the translation to
two-sided systems needs writing).

So the shadow's own complexity is the mean-payoff class. The lift never sits there: it is
below it (Theorem A) or above it (Proposition B2) depending on the residue field.

## 6. The separation corollaries, and what they do for the paper

**Corollary S1 (shadow strictly harder than the large-field lift) [new, conditional].**
Under |k| > N, DWF_k ∈ P while TDWF restricted to the same instance shapes is MPG-complete.
If any polynomial-time shadow oracle decided the lift-relevant fragment exactly, MPG ∈ P.
Contrapositive: unless mean payoff games are polynomial, the tropical shadow *overstates*
the difficulty of the lifted problem — the lift's linear algebra dissolves the tie
combinatorics that makes the shadow hard.

**Corollary S2 (fixed-field lift strictly harder than the shadow) [new, conditional].**
DWF_{F_q} is NP-complete for fixed q, while its shadow is in NP ∩ co-NP territory. Unless
NP = co-NP, the shadow *understates* the difficulty: the obstructions live in F_q-linear
conditions the valuation cannot see.

**The mechanism is caveat (c).** The witness is minimal and characteristic-sensitive: the
system {deg x_1 ≤ 1, deg x_2 ≤ 1, deg x_1 ≥ 1, deg(x_1 + x_2) ≤ 0, deg(x_1 − x_2) ≤ 0} is
infeasible over every field of characteristic ≠ 2 (adding the two cancellations forces
2·x_1 to drop degree) yet feasible over F_2 (x_1 = x_2 = t) and shadow-feasible over any
field (tie ξ_1 = ξ_2 = 1 licenses both drops). The tropical prevariety strictly contains
the image of the variety; the phantom mass is exactly what S1/S2 quantify. This upgrades
the dossier's "bound the phantom mass (honest lemma)" to "the phantom mass is the
complexity gap".

**Location of the existing construction.** The paper's Sudoku/all-different compilation is
the q ≤ N regime with the *domain carried by pinning rows*: the negative pair rows are
window-[0, 0] rewards, and the unary pinning rows that hold coefficients to a 9-element
alphabet are not window-expressible over a large field — an affine-minus-affine set never
has 9 points in a big field. Pinning is archimedean aggregation (counting satisfied unary
rows), and that is precisely where Track A's gap-hardness and the Guruswami–Vardy row of
the split live. So the chapter thesis becomes a trichotomy with attribution:

| Layer | Complexity | Source of hardness |
|---|---|---|
| Windows, \|k\| > N | P (Theorem A) | none — slack everywhere |
| Windows, fixed q (or exact, q ≥ 3) | NP-complete (B1/B2) | pigeonhole/forced cancellation |
| Tropical shadow (exact fragment) | ≡ mean payoff games (C) | tie combinatorics |
| Any k + counting/pinning aggregation | NP-hard [known: manuscript Cor.; Mihara; Guruswami–Vardy] | archimedean counting over 0/1 wells |

One-sentence version for the paper: *the ultrametric layer of the signed regression is
polynomial whenever the residue field outnumbers the reward rows; all hardness of the
programme is attributable either to pigeonhole in a small residue field or to archimedean
counting, and the tropical shadow, far from being the invariant content of the loss, sits
at a third complexity level (mean payoff) matching the lift in neither regime.*

## 7. Outlook items (one paragraph each)

**Ordered coefficients reach the mean-payoff cell honestly.** Over k = ℝ with sign
conditions on leading coefficients (truth values in ℝ[t], constraints allowed to consult
the order), the lift becomes linear programming over a real nonarchimedean field, whose
complexity *is* tropical LP ≡ MPG (Allamigeon–Benchimol–Gaubert–Joswig) **[known]**. So the
three-level picture is completed by a fourth cell: ordered residue field ⇒ the lift and the
shadow finally agree, at the mean-payoff rung. Small table-completing theorem worth
stating **[new, provable]**.

**Middle regime q ≈ N.** Theorem A needs q > N; B2's instances have q ≪ N. Where between
does the transition happen for natural row distributions (random windows, all-different
families)? Candidate for an experiment plus a threshold conjecture; the covering-lemma
failure mode (unions of ≤ q affine hyperplane-cosets can cover) suggests the honest answer
is "structure-dependent, with q parallel cosets as the tight obstruction" **[open]**.

**Faithful-shadow criterion.** Call an instance family *faithfully tropicalized* if
lift-feasibility equals shadow-feasibility on it. S1/S2 say this fails generically in both
directions; the tropical-basis literature (Maclagan–Sturmfels ch. 2; the dossier's
Gröbner-complex digest) is the right language for sufficient conditions. A clean sufficient
condition for CSP families would tell the paper exactly when the degree ledger is exact —
caveat (c) as a definition rather than an apology **[open]**.

**Optimisation variants.** Thresholding max-excess objectives reduces to O(log) many DWF
calls, so Theorem A extends to the min-max optimisation layer at large q; the min-#violated
variant collapses to Theorem A's remark (iii) at large q and inherits B2's hardness at
fixed q. The shadow optimisation variant is Akian–Gaubert–Qi–Saadi tropical regression
**[known]**. No surprises expected; worth two lemmas in the write-up **[new, provable]**.

## 8. Experiments (playground-sized)

1. Implement DWF over F_q in the Python code (pure linear algebra over F_q plus the
   sampling witness): verify Theorem A's criterion empirically, and chart feasibility rates
   as q crosses N for random instances (middle-regime data).
2. The phantom witness of §6 as a unit test and as a browser-playground example: the same
   five-row system flipping between infeasible (F_3) and feasible (F_2) with an identical
   tropical shadow.
3. Shadow side: implement the exact-fragment translation to "max attained twice" systems
   and solve small instances by value iteration; compare against the lift's linear algebra
   on matched instances to exhibit S1 concretely (shadow solver working hard where the lift
   is trivial).

## 9. Promotion gates (mirroring research/rh_complexity_program.md)

Before any of this enters the manuscript:

1. Write Theorem A and B1/B2 in full, with the derandomised witness search spelled out and
   the unary-caps encoding convention stated.
2. Fetch and reconcile Bodirsky–Fehm arXiv:2504.13536 (format alignment; the dossier's
   citation-verification caveat applies). Also check novelty against: rational
   interpolation with prescribed non-vanishing, Reed–Solomon decoding with erasures,
   max-plus/tropical prevariety literature (Grigoriev–Podolskii), and the valued-CSP
   school (Bodirsky–Mamino).
3. Pin the shadow citations: Akian–Gaubert–Guterman (IJAC 22, 2012), Grigoriev–Podolskii
   (Comput. Complexity, 2015) **[verify volume/pages]**, Zwick–Paterson (TCS 158, 1996) for
   MPG ∈ NP ∩ co-NP, AGQS (SIDMA 37(2), 2023) and AGBJ (SIDMA 29(2), 2015) already in the
   dossier.
4. Decide the manuscript form: this note is a chapter-shaped cluster (definitions, A,
   B1/B2, C, S1/S2, phantom example) rather than a section; discuss placement relative to
   the split-theorem section already planned in the dossier's "top three moves".
5. Formalisation kernel (T3, T5, T7, T10 in the companion spec) machine-checked, or the
   attempt documented.

## 10. Citation ledger for this note

Already in the dossier (pinned there): Akian–Gaubert–Qi–Saadi 2023; Allamigeon–Benchimol–
Gaubert–Joswig 2015; Maclagan–Sturmfels GSM 161; Beckermann–Labahn 1994; Forney 1975;
A. K. Lenstra 1985; Guruswami–Vardy 2005; Mihara 2026; Bodirsky–Fehm arXiv:2504.13536
[verify]; Isaksen 2002.

New to this note: Akian–Gaubert–Guterman, "Tropical polyhedra are equivalent to mean payoff
games", Int. J. Algebra Comput. 22(1), 2012 [confident]. Grigoriev–Podolskii, "Complexity
of tropical and min-plus linear prevarieties", Computational Complexity 24, 2015
[confident; verify pages]. Zwick–Paterson, "The complexity of mean payoff games on graphs",
Theoret. Comput. Sci. 158, 1996 [confident]. Ehrenfeucht–Mycielski 1979 (positional
determinacy) [confident]. Karp 1972 / standard source for fixed-q ≥ 3 graph colouring
NP-completeness [standard]. Mulders–Storjohann 2003 (weak Popov form) as the simplest
modern source for the approximant-basis machinery referenced in the dossier's gain 4
[confident].
