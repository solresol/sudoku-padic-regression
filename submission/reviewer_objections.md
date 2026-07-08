# Likely Reviewer Objections

## "Negative weights are not regression."

The paper studies signed affine residual objectives. The term "regression" is used because each term is an affine residual in the coefficient vector, continuing the language of the surrounding p-adic regression literature. The construction is not presented as an ordinary positive-weight statistical loss. Negative terms are bounded because the optimisation domain is \(\mathbb Z_p^n\), and on the finite intended domain they become exact indicator rewards.

## "Sudoku is already solved by SAT and constraint programming."

The paper is not proposing a Sudoku solver. Sudoku is used because it is a familiar finite-domain all-different system where the variables, domains, and pairwise primal-graph constraints can be written explicitly. The main result is the general all-different/list-colouring encoding theorem.

## "Why p-adic rather than the discrete metric?"

The discrete metric gives the same equality/inequality indicator behaviour, and the manuscript records that variant. The p-adic construction is still the main object because it embeds those indicators into the affine residual framework used in p-adic regression, so the finite-domain constraints are represented directly as signed p-adic residuals on the original variables.
