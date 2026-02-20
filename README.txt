Sudoku via integer p-adic regression (81 coefficients)

This project contains:
- paper/sudoku_padic_regression.tex : LaTeX paper draft
- paper/sudoku_padic_regression.pdf : compiled PDF
- code/padic_sudoku_regression.py : solver + (slow) unique generator
- code/run_experiments.py : generates quick experiment CSV and loss plot
- code/padic_linear_regression.py : synthetic p-adic linear regression + hyperplane-graph greedy walk experiments
- outputs/experiment_results.csv : results from a small run
- outputs/loss_curve.pdf/png : example loss trajectory

Quick start
-----------

Solve a puzzle (81 chars, 0 or . for blanks):

  python3 code/padic_sudoku_regression.py solve \
    --puzzle 530070000600195000098000060800060003400803001700020006060000280000419005000080079 \
    --seed 0 --max-steps 60000 --restarts 15
    # add --moves 8 to print the first few swap steps

Solve using Zubarev-style stochastic walk (softmax over swap moves):

  python3 code/padic_sudoku_regression.py solve \
    --method zubarev --beta0 0.5 --beta1 6.0 --beta-schedule linear \
    --puzzle 530070000600195000098000060800060003400803001700020006060000280000419005000080079 \
    --seed 0 --max-steps 60000 --restarts 15

Solve using *pure greedy* steepest descent (stops at local minima; useful for basin-size experiments):

  python3 code/padic_sudoku_regression.py solve \
    --method greedy \
    --puzzle 530070000600195000098000060800060003400803001700020006060000280000419005000080079 \
    --seed 0 --max-steps 60000 --restarts 15

Generate a *unique-solution* puzzle (slower):

  python3 code/padic_sudoku_regression.py generate --clues 30 --seed 42

Reproduce experiments (fast carving, not uniqueness-checked):

  python3 code/run_experiments.py --outdir outputs --seed 123 --n 3 --clues 36,30,26

Monte Carlo: synthetic p-adic linear regression (enumerate all (d+1)-point hyperplanes, *deduplicate identical coefficient vectors*, then analyse descent policies on the induced neighbour graph):

  python3 code/padic_linear_regression.py mc \
    --trials 50 --n 20 --d 3 --p 3 --loss l1 \
    --coef-model p_power --coef-exp-min 0 --coef-exp-max 4 \
    --noise-model haar --noise-k0 1 --noise-kmax 6 \
    --policies steepest,uniform,proportional,softmax --temperature 1.0

Landscape snapshot for a single dataset (basins, barriers, plateaus, local-optima network, support-point counts):

  python3 code/padic_linear_regression.py landscape \
    --seed 0 --n 20 --d 3 --p 3 --loss l1 \
    --coef-model p_power --coef-exp-min 0 --coef-exp-max 4 \
    --noise-model haar --noise-k0 0 --noise-kmax 6

Notes
-----
- The solver uses only 81 integer variables (one per cell). It is intentionally not a 729-variable one-hot CSP.
- The paper frames the objective as p-adic regression with positive digit-snapping regularisation and negative inequality regularisation.
- In the implementation we take the "strong snapping" limit, restricting the search to digit assignments and using row-wise swap moves.
