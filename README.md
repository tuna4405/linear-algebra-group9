## Files

| File | Purpose |
|---|---|
| `svd_recommender.py` | Core NumPy SVD implementation, prediction, cold-start fallbacks and recommendations |
| `run_experiments.py` | Reproduce both datasets' results, figures and report |
| `demo.py` | Predict a rating or display recommendations using the fitted model |
| `tests/test_svd_recommender.py` | Numerical correctness and edge-case checks |
| `data/toy_observed.csv` | Hand-made input matrix, blank cells are missing |
| `data/toy_ground_truth.csv` | Hand-made answer key, not training input |
| `results/final_test_metrics.csv` | Main final comparison with baselines |
| `results/validation_summary.csv` | Mean and standard deviation across three validation splits |
| `results/test_predictions.csv` | All 20,000 held-out actual-predicted pairs |
| `results/*.png`, `results/*.svg` | Six charts in raster and editable vector formats |
| `results/experiment_metadata.json` | Versions, input hashes, seeds, selected k and timings |
| `results/model_item_selected.npz` | Saved selected factors and baseline statistics |

results/ is generated locally.

## Run

Open a terminal in this folder. The current workspace already has the dependencies,
dataset and generated results. For a fresh Python 3.10+ environment:

```powershell
python -m pip install -r requirements.txt
```

If the dataset is absent, download the official archive and extract it locally:

```powershell
Invoke-WebRequest -Uri 'https://files.grouplens.org/datasets/movielens/ml-100k.zip' -OutFile 'data/ml-100k.zip'
Expand-Archive -LiteralPath 'data/ml-100k.zip' -DestinationPath 'data' -Force
```

Then:

```powershell
python -m unittest discover -s tests -v
python run_experiments.py
python demo.py --user 1
python demo.py --user 1 --item 50
```

IDs in the demonstration are the original one-based MovieLens IDs. The demo
loads the selected model refitted on `u1.base`. It does not retrain on `u1.test`.
A queried pair may already have a development rating; the demo labels this case.
Recommendation lists exclude every item rated by that user in `u1.base`.

## Method and experiment

The primary method follows the team's Algorithm Model: fill missing raw ratings
with training item means, subtract observed training user means, compute
`numpy.linalg.svd(..., full_matrices=False)`, truncate, restore user means, and
clip predictions to [1, 5]. User-mean filling is a predeclared comparison.

The official `u1.base`/`u1.test` split is development/final test. Three internal
64,000/16,000 fitting/validation splits choose k using mean validation RMSE.
The selected model is refitted on all 80,000 development ratings before final
testing. The test set is never used to choose k or compute preprocessing means.
Error curves are validation curves; the test table evaluates only the fixed
selected models and mean baselines.

The small example uses fixed k=2 for illustration. Its hand-authored complete
matrix is an answer key. Originally missing MovieLens values have no known
ground truth and are not assigned invented evaluation targets.

No recommendation package, scikit-learn estimator, ALS or SGD is used.
Runtime requirements are NumPy and Matplotlib only. CSV and JSON handling use
the Python standard library. Generated files have stable names and are replaced
when rerunning the experiments.

## Dataset attribution

MovieLens 100K comes from [GroupLens](https://grouplens.org/datasets/movielens/100k/).
Retain its original `data/ml-100k/README` and cite Harper and Konstan (2015),
[The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872).
The dataset README permits research use and requires separate permission for
redistribution. When sharing source code publicly, omit the raw archive,
extracted dataset and the CSV containing individual actual test ratings;
provide the official download link instead. These paths are excluded in
`.gitignore`.
