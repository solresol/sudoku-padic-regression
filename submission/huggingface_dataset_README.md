---
license: mit
task_categories:
  - tabular-regression
pretty_name: Sudoku p-adic residual experiment data
tags:
  - sudoku
  - p-adic
  - local-search
  - reproducibility
---

# Sudoku p-adic residual experiment data

This dataset contains the small reproducibility bundle for the manuscript "Signed p-adic Residual Encodings of Finite-Domain All-Different Systems with a Sudoku Case Study".

Canonical dataset page: https://huggingface.co/datasets/gregb/sudoku-padic-regression-experiments

The bundle includes:

- `data/paper_stepwise/experiment_results.csv`
- `data/paper_zubarev/experiment_results.csv`
- summary files for both experiment runs
- trace puzzle and solution files for the plotted examples
- loss-curve source files
- the Python scripts used to generate the outputs

The experiment table is regenerated from the source repository with:

```bash
python3 code/run_experiments.py --outdir outputs/paper_stepwise \
  --seed 123 --n 6 --clues 36,30,26 --max-steps 60000 \
  --restarts 15 --method stepwise
python3 code/run_experiments.py --outdir outputs/paper_zubarev \
  --seed 123 --n 6 --clues 36,30,26 --max-steps 60000 \
  --restarts 15 --method zubarev --beta0 0.5 --beta1 6.0 \
  --beta-schedule linear
```

The puzzles are randomly carved from solved grids and are not uniqueness-checked. The computational claims are illustrative rather than competitive Sudoku-solving claims.
