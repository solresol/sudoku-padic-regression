# Submission Package Contents

This package is intended as the editable source/data bundle for the manuscript.

The public experiment archive is:

https://huggingface.co/datasets/gregb/sudoku-padic-regression-experiments

- `paper/sudoku_padic_regression.tex`: manuscript source.
- `paper/loss_curve.pdf` and `paper/loss_curve.png`: figure source for the loss trajectory.
- `paper/sudoku_padic_regression.pdf`: compiled reference PDF.
- `code/padic_sudoku_regression.py`: solver and puzzle-generation implementation.
- `code/run_experiments.py`: experiment driver for the computational appendix.
- `outputs/paper_stepwise/`: stepwise experiment CSV, summary, trace puzzle, trace solution, and loss curve.
- `outputs/paper_zubarev/`: Zubarev-walk experiment CSV, summary, trace puzzle, trace solution, and loss curve.
- `README.txt`: repository-level reproduction notes.
- `submission/cover_letter.md`: cover letter draft.
- `submission/reviewer_objections.md`: prepared responses to likely reviewer objections.
- `submission/huggingface_dataset_README.md`: dataset-card draft for the public experiment archive.

The experiment table in the manuscript is regenerated with:

```bash
python3 code/run_experiments.py --outdir outputs/paper_stepwise \
  --seed 123 --n 6 --clues 36,30,26 --max-steps 60000 \
  --restarts 15 --method stepwise
python3 code/run_experiments.py --outdir outputs/paper_zubarev \
  --seed 123 --n 6 --clues 36,30,26 --max-steps 60000 \
  --restarts 15 --method zubarev --beta0 0.5 --beta1 6.0 \
  --beta-schedule linear
```
