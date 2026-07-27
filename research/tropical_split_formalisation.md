# Formalisation spec: degree-window systems, the covering argument, and the phantom witness

*Companion to `research/tropical_mean_payoff_split.md`, 2026-07-27. This document is written
to be handed directly to Lean and Coq proving agents. It fixes shared vocabulary (§2), lists
formalisation targets as a dependency DAG (§3), gives per-target formal statement skeletons
for Lean 4 + mathlib and Coq + MathComp (§4–§5), and states acceptance criteria (§6).*

## 1. Scope

**In scope:** the mathematical kernels of Theorem A, Propositions B1/B2, and the phantom
example — pure algebra over polynomial rings and finite-dimensional linear spaces. Every
target below is a self-contained lemma with no complexity-theoretic content.

**Out of scope (do not attempt):** polynomial-time claims of any kind; NP-completeness
reductions as complexity statements (only their *correctness lemmas* T9/T10 are in scope);
the Akian–Gaubert–Guterman and Grigoriev–Podolskii equivalences; approximant-basis/Popov
theory. These stay pen-and-paper.

Prover-agent ground rules: no `sorry`/`Admitted` in delivered proofs; no new axioms;
hypotheses may be strengthened only with an explicit note ("proved under additional
hypothesis H"); generalising `Field` to `DivisionRing`/weaker is welcome when free. Lemma
names in mathlib/MathComp drift — the names cited below are candidates to search from
(`exact?`, `loogle`, `Search`), not guarantees.

## 2. Shared vocabulary and conventions

Degree conventions per system:

| Concept | Paper/note | Lean 4 mathlib | Coq MathComp |
|---|---|---|---|
| deg p | deg 0 = −∞ | `Polynomial.degree : K[X] → WithBot ℕ`, `degree 0 = ⊥` | `size p : nat`, `size 0 = 0` (size = deg + 1) |
| deg p ≤ d | | `p.degree ≤ (d : WithBot ℕ)` | `(size p <= d.+1)%N` |
| deg p ≥ 0 (p ≠ 0) | | `0 ≤ p.degree` ↔ `p ≠ 0` | `(0 < size p)%N` ↔ `p != 0` |
| deg p = d exactly | | `p.degree = (d : WithBot ℕ)` | `size p = d.+1` |
| capped unknowns | deg x_j ≤ c_j | `Polynomial.degreeLT K (c+1) : Submodule K K[X]` | `{poly F}` + `size` side condition, or `'rV[F]_(c.+1)` coefficient vectors |

DWF instance semantics (from the companion note §2): unknowns x_1..x_n ∈ k[t] with caps;
rows r_i(x) = Σ_k u_{ik} x_k − b_i; window l_i ≤ deg r_i(x) ≤ u_i. On the coefficient space
K^D (D = Σ(c_j+1)): upper bounds ⟺ affine equations; the upper-feasible set is an affine
subspace V; each finite lower bound removes an affine subspace W_i; feasible set =
V ∖ ⋃ W_i. Targets T1/T2 pin this shape; T3 is the covering lemma; T4/T5 assemble
Theorem A's criterion; T6 is shadow necessity; T7 the phantom witness; T8–T10 the B1/B2
correctness kernels; T11 the (stretch) tropical translation.

## 3. Target DAG, difficulty, suggested order

```
T9  edge-row correctness            [trivial]      independent
T10 clause-row correctness (F2)     [easy]         independent
T7  phantom witness                 [easy-medium]  independent
T8  F2 exact-degree affineness      [easy]         independent
T3  affine covering lemma           [medium]       independent
T5  feasibility criterion           [easy | T3]
T1  residual coefficients affine    [easy]         independent
T2  window sets are affine/coaffine [easy | T1]
T6b unique-max degree lemma         [easy-medium]  independent (T6a exists in mathlib)
T6c shadow necessity for a row      [medium | T6b]
T4  Theorem A, existence form       [medium | T1,T2,T3,T5]
T11 tropical translation            [medium, stretch] independent
```

Suggested order: T9 → T10 → T7 → T8 → T3 → T5 → T1 → T2 → T6b → T6c → T4 → (T11).
T3 and T7 are the flagship targets; if time is constrained, deliver those two first.

## 4. Lean 4 + mathlib statements

All skeletons assume `import Mathlib` and `open Polynomial` where convenient.

### T3 — affine covering lemma

A nonempty affine subspace over a finite field is not covered by fewer than |K| proper
affine subspaces.

```lean
theorem affine_avoid_union
    {K V : Type*} [Field K] [Fintype K] [AddCommGroup V] [Module K V]
    [Fintype V] [DecidableEq V]
    (S : AffineSubspace K V) (hS : (S : Set V).Nonempty)
    {m : ℕ} (W : Fin m → AffineSubspace K V)
    (hW : ∀ j, ¬ (S : Set V) ⊆ (W j : Set V))
    (hm : m < Fintype.card K) :
    ∃ v ∈ (S : Set V), ∀ j, v ∉ (W j : Set V) := by
  sorry
```

Proof route: for each j, `S ⊓ W j` is empty or an affine subspace of S whose direction is a
*proper* submodule of `S.direction` (properness from `hW`). Fixing `p ∈ S`, the map
`u ↦ u +ᵥ p` is a bijection `S.direction ≃ S`, so `|S| = |K|^(finrank S.direction)` and
`|S ∩ W j| ≤ |S| / |K|` (a proper submodule has index ≥ |K|: quotient by it is a nontrivial
K-module, hence has ≥ |K| elements). Conclude by
`Finset.card_biUnion_le` + arithmetic: `m * (|S| / |K|) < |S|`.
Candidate API: `AffineSubspace.direction`, `AffineSubspace.mem_iff`,
`Submodule.card_quotient...`/`Submodule.finrank_lt` (search), `Module.card_fintype`,
`Fintype.card_pow`. If `AffineSubspace` cardinality plumbing is painful, an accepted
fallback formulation replaces affine subspaces by explicit cosets
`(pⱼ +ᵥ ↑(Uⱼ : Submodule K V) : Set V)`; note the change in the delivery.
An infinite-field variant (drop `Fintype V`/`Fintype K`, assume `Infinite K`, arbitrary
finite m) is a welcome bonus with essentially the induction-on-m proof.

### T5 — feasibility criterion (Theorem A, criterion form)

```lean
theorem dwf_criterion
    {K V : Type*} [Field K] [Fintype K] [AddCommGroup V] [Module K V]
    [Fintype V] [DecidableEq V]
    (S : AffineSubspace K V) {m : ℕ} (W : Fin m → AffineSubspace K V)
    (hm : m < Fintype.card K) :
    (∃ v ∈ (S : Set V), ∀ j, v ∉ (W j : Set V)) ↔
      ((S : Set V).Nonempty ∧ ∀ j, ¬ (S : Set V) ⊆ (W j : Set V)) := by
  sorry
```

Forward direction is trivial except the properness clause (a common point witnesses
non-containment... note it does not: non-containment needs the avoiding point itself — it
is exactly the avoiding point, so this direction is one line per clause). Backward is T3.

### T7 — phantom witness (characteristic-sensitive infeasibility)

The five-row system {deg x₁ = 1, deg x₂ ≤ 1, deg(x₁+x₂) ≤ 0, deg(x₁−x₂) ≤ 0}: infeasible
iff char K ≠ 2. (The shadow-feasibility half of the example is informal or T11; do not
attempt it here.)

```lean
theorem phantom_infeasible {K : Type*} [Field K] (h2 : (2 : K) ≠ 0) :
    ¬ ∃ x₁ x₂ : K[X],
        x₁.degree = 1 ∧ x₂.degree ≤ 1 ∧
        (x₁ + x₂).degree ≤ 0 ∧ (x₁ - x₂).degree ≤ 0 := by
  sorry

theorem phantom_feasible_char_two {K : Type*} [Field K] (h2 : (2 : K) = 0) :
    ∃ x₁ x₂ : K[X],
        x₁.degree = 1 ∧ x₂.degree ≤ 1 ∧
        (x₁ + x₂).degree ≤ 0 ∧ (x₁ - x₂).degree ≤ 0 := by
  sorry
```

Infeasibility route: `(x₁+x₂) + (x₁−x₂) = 2 • x₁` (or `C 2 * x₁`), so
`(C 2 * x₁).degree ≤ 0` by `degree_add_le`; `degree_C_mul` with `h2` gives
`x₁.degree ≤ 0`, contradicting `x₁.degree = 1`. Feasibility: `x₁ = X, x₂ = X`;
`X + X = C 2 * X = 0` under `h2`, `degree 0 = ⊥ ≤ 0`.
Candidate API: `Polynomial.degree_add_le`, `Polynomial.degree_C_mul` (or
`degree_smul` variants), `Polynomial.degree_X`, `Polynomial.degree_zero`, `two_mul`.

### T8 — F₂ exact-degree affineness (kernel of Proposition B1, easy side)

```lean
theorem degree_eq_iff_coeff_one
    (d : ℕ) (p : Polynomial (ZMod 2)) (hp : p.degree ≤ (d : WithBot ℕ)) :
    p.degree = (d : WithBot ℕ) ↔ p.coeff d = 1 := by
  sorry
```

Over `ZMod 2`, `≠ 0 ↔ = 1` (`ZMod` API / `Fin 2` case split). Combine
`Polynomial.degree_le_iff_coeff_zero` with `coeff_ne_zero_of_eq_degree`-style lemmas.
This is the statement "exact degree over F₂ is an affine condition"; its role in B1 is
interpretive, so the lemma alone suffices.

### T9 — edge-row correctness (kernel of Proposition B1, hard side)

```lean
theorem edge_row_correct {K : Type*} [Field K] (a b : K) :
    (0 : WithBot ℕ) ≤ (Polynomial.C a - Polynomial.C b).degree ↔ a ≠ b := by
  sorry
```

`C a - C b = C (a - b)`; `degree (C c) = 0 ↔ c ≠ 0`, else `⊥`; `¬ (0 ≤ (⊥ : WithBot ℕ))`.
Candidate API: `Polynomial.degree_C`, `Polynomial.degree_C_eq_zero_iff`? (search),
`Polynomial.zero_le_degree_iff : 0 ≤ p.degree ↔ p ≠ 0`, `map_sub` for `C`.

### T10 — clause-row correctness over F₂ (kernel of Proposition B2)

```lean
theorem clause_row_correct (z ε : Fin 3 → ZMod 2) :
    (0 : WithBot ℕ) ≤
      (Polynomial.C (z 0 + ε 0)
        + Polynomial.C (z 1 + ε 1) * Polynomial.X
        + Polynomial.C (z 2 + ε 2) * Polynomial.X ^ 2).degree
    ↔ ∃ a : Fin 3, z a + ε a = 1 := by
  sorry
```

Route: `zero_le_degree_iff` reduces to `r ≠ 0`; `r = 0 ↔ ∀ c, r.coeff c = 0`
(`Polynomial.ext_iff`); the three relevant coefficients are the slot values
(`coeff_add`, `coeff_C`, `coeff_C_mul`, `coeff_X_pow`); over `ZMod 2`, nonzero = 1.
Optionally also deliver the trivial upper bound `r.degree ≤ 2` (`degree_add_le`,
`degree_C_mul_le`, `degree_X_pow`).

### T1 — residual coefficients are affine in the unknowns

```lean
theorem residual_coeff_affine {K : Type*} [Field K] {n : ℕ}
    (u : Fin n → K[X]) (b : K[X]) (c : ℕ) :
    ∃ (φ : (Fin n → K[X]) →ₗ[K] K),
      ∀ x : Fin n → K[X],
        ((∑ k, u k * x k) - b).coeff c = φ x - b.coeff c := by
  sorry
```

`φ := (Polynomial.lcoeff K c).comp (linear map x ↦ ∑ u k * x k)`; the content is that
`x ↦ ∑ u k * x k` is K-linear (`LinearMap.mul` composition / build with `LinearMap.pi`
plumbing) and `lcoeff` is the coefficient functional. Nearly definitional; its value is
fixing the vocabulary for T2/T4.

### T2 — window sets are affine minus affine

Given caps, define `Vcap := Π k, degreeLT K (c k + 1)` as the ambient finite-dimensional
K-space of capped unknowns. Statement to deliver: the upper-window set
`{x ∈ Vcap | ∀ i, deg r_i(x) ≤ u_i}` is (the coercion of) an affine subspace of `Vcap`,
and for each finite lower bound the violating set
`{x ∈ Vcap ∩ upper | deg r_i(x) < l_i}` is an affine subspace of it. Route:
`Polynomial.degree_le_iff_coeff_zero` + T1 exhibits each as an intersection of level sets
of affine functionals; package with `AffineSubspace.comap` or by hand. Precise formal
phrasing is left to the agent; the acceptance test is that T4 below is statable against it.

### T6 — shadow necessity

T6a `Polynomial.degree_sum_le` already exists in mathlib — cite, do not reprove.

```lean
-- T6b: unique dominant term forces equality
theorem degree_sum_eq_of_unique_max {ι : Type*} {K : Type*} [Field K]
    (s : Finset ι) (f : ι → K[X]) (i₀ : ι) (hi₀ : i₀ ∈ s)
    (hmax : ∀ i ∈ s, i ≠ i₀ → (f i).degree < (f i₀).degree) :
    (∑ i ∈ s, f i).degree = (f i₀).degree := by
  sorry
```

Induction on `s` via `Finset.sum_erase_add` + `Polynomial.degree_add_eq_left_of_degree_lt`
(note `degree_sum_le` bounds the erased sum by the strict sup). T6c (a row whose tropical
value M_i is attained by a unique term has deg r_i = M_i, hence a window violated at the
tropical level forces a tie) is a corollary obtained by instantiating T6b with the term
multiset {u_{ik} x_k} ∪ {−b_i}; deliver as a comment-level corollary or a wrapper lemma —
agent's choice.

### T4 — Theorem A, existence form (assembly)

Statement: for the DWF sets of T2 with `m < Fintype.card K` lower-bound rows, if the upper
set is nonempty and no lower-bound subspace equals it, a window-feasible x exists. This is
T3 applied through T2's packaging; deliver once T2's phrasing is settled. This is the
formal core of Theorem A; the polynomial-time content stays informal by design.

## 5. Coq + MathComp statements

MathComp conventions: `{poly F}`, `size p` (deg + 1, `size 0 = 0`). Affine subspaces are
not first-class; use matrix solution sets. Padding note: distinct lower-bound rows may
have different numbers of defining equations; WLOG pad with zero rows to a common `kW`.

### T3 — covering lemma, matrix form

```coq
From mathcomp Require Import all_ssreflect all_algebra.
Import GRing.Theory.

Section AffineAvoid.
Variables (F : finFieldType) (D kA kW m : nat).
Variable A : 'M[F]_(kA, D).   Variable a : 'cV[F]_kA.
Variable C : 'I_m -> 'M[F]_(kW, D).
Variable d : 'I_m -> 'cV[F]_kW.

Definition inS (x : 'cV[F]_D) := A *m x == a.
Definition inW (j : 'I_m) (x : 'cV[F]_D) := C j *m x == d j.

Lemma affine_avoid :
  (exists x, inS x) ->
  (forall j, exists x, inS x && ~~ inW j x) ->
  (m < #|F|)%N ->
  exists x, inS x /\ forall j, ~~ inW j x.
Proof. Admitted.

End AffineAvoid.
```

Route: `[set x | inS x]` is a coset of the kernel subspace `[set x | A *m x == 0]`; use
`card` arithmetic — MathComp has kernel spaces via `kermx`(transposed setting) and
`card_vspace`-style counting through `'rV`/vectType if preferred (an accepted alternative
is to restate over row vectors `'rV[F]_D` and subspaces `{vspace ...}` with explicit
cosets). Key counting fact: a nonempty proper affine intersection has cardinality at most
`#|S| %/ #|F|`; then `\sum_(j < m) #|S :&: W j| < #|S|`.

### T7 — phantom witness

```coq
Section Phantom.
Variable F : fieldType.

Lemma phantom_infeasible :
  2%:R != 0 :> F ->
  ~ (exists x1 x2 : {poly F},
       [/\ size x1 = 2, (size x2 <= 2)%N,
           (size (x1 + x2) <= 1)%N & (size (x1 - x2) <= 1)%N]).
Proof. Admitted.

Lemma phantom_feasible_char2 :
  2%:R == 0 :> F ->
  exists x1 x2 : {poly F},
    [/\ size x1 = 2, (size x2 <= 2)%N,
        (size (x1 + x2) <= 1)%N & (size (x1 - x2) <= 1)%N].
Proof. Admitted.

End Phantom.
```

Route: `(x1 + x2) + (x1 - x2) = x1 *+ 2`; `size (p *+ 2) = size p` when `2%:R != 0`
(`size_scale`-style: `p *+ 2 = 2%:R *: p`, `size_scale` needs unit — over a field fine);
`size_add` bound (`size_add : size (p + q) <= maxn (size p) (size q)`). Witness for char 2:
`x1 = x2 = 'X` with `'X + 'X = 'X *+ 2 = 0`, `size 0 = 0`.

### T9/T10 — row-correctness kernels

```coq
Lemma edge_row_correct (F : fieldType) (x y : F) :
  (0 < size ((x - y)%:P))%N = (x != y).
Proof. Admitted.  (* size_polyC *)

Lemma clause_row_correct (z e : 'I_3 -> 'F_2) :
  let r := (z 0 + e 0)%R%:P + ((z 1 + e 1)%R%:P * 'X) + ((z 2 + e 2)%R%:P * 'X^2) in
  (0 < size r)%N <-> exists a, z a + e a = 1.
Proof. Admitted.
```

(`size_polyC` gives `size c%:P = (c != 0)`; for the clause row use `polyseq`/`coefE`
simp set and `'F_2` two-element case analysis; index literals `0, 1, 2 : 'I_3` via `inord`
if needed.)

### T6b — unique dominant term

```coq
Lemma size_sum_eq_of_unique_max (F : fieldType) (I : finType)
    (s : {set I}) (f : I -> {poly F}) (i0 : I) :
  i0 \in s ->
  (forall i, i \in s -> i != i0 -> (size (f i) < size (f i0))%N) ->
  size (\sum_(i in s) f i) = size (f i0).
Proof. Admitted.  (* size_addl + big_setD1 induction *)
```

## 6. Acceptance criteria and reporting

1. Statements must match §2 semantics; renaming and mild generalisation are fine, semantic
   drift is not. If a skeleton statement is unprovable as written, report the
   counterexample or the repaired statement rather than silently changing quantifiers —
   T3's properness hypothesis and T5's biconditional shape are the places to watch.
2. Deliver per target: the final statement, full proof, mathlib/MathComp version used, and
   a one-line note of any hypothesis changes.
3. Priority if resources are short: T3 and T7 (flagships), then T10, T9, T8, T5.
4. T11 (formal min-plus translation: define row value `M_i`, tie predicate, and prove the
   exact-window fragment equals "max attained twice" systems — mathlib `Tropical` or bare
   `WithBot ℤ` both acceptable) is stretch work; attempt only after all else is delivered.
5. Nothing here depends on unformalised results; if a target seems to need one, the
   decomposition is wrong — report back instead of assuming.
