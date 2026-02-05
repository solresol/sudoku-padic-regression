Sudoku via integer p-adic regression (81 coefficients)

This project contains:
- paper/sudoku_padic_regression.tex : LaTeX paper draft
- paper/sudoku_padic_regression.pdf : compiled PDF
- code/padic_sudoku_regression.py : solver + (slow) unique generator
- code/run_experiments.py : generates quick experiment CSV and loss plot
- outputs/experiment_results.csv : results from a small run
- outputs/loss_curve.pdf/png : example loss trajectory

Quick start
-----------

Solve a puzzle (81 chars, 0 or . for blanks):

  python3 code/padic_sudoku_regression.py solve \
    --puzzle 530070000600195000098000060800060003400803001700020006060000280000419005000080079 \
    --seed 0 --max-steps 60000 --restarts 15
    # add --moves 8 to print the first few swap steps

Generate a *unique-solution* puzzle (slower):

  python3 code/padic_sudoku_regression.py generate --clues 30 --seed 42

Reproduce experiments (fast carving, not uniqueness-checked):

  python3 code/run_experiments.py --outdir outputs --seed 123 --n 3 --clues 36,30,26

Notes
-----
- The solver uses only 81 integer variables (one per cell). It is intentionally not a 729-variable one-hot CSP.
- The paper frames the objective as p-adic regression with positive digit-snapping regularisation and negative inequality regularisation.
- In the implementation we take the "strong snapping" limit, restricting the search to digit assignments and using row-wise swap moves.
