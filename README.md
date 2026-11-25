# Cardiovascular Disease Prediction Model Comparison

## Group members

| Name | Enrollment number |
|---|---|
| R.Venkata siva | 2302320153 |
| podil Anil| 2302320114 |
| D. Ramesh | 2302320099 |

This educational project compares Decision Tree, Gaussian Naive Bayes, K-Nearest
Neighbors (KNN), and a feed-forward Artificial Neural Network (MLP) using the
provided `cardio_train.csv` dataset. It produces cross-validation and held-out
test metrics, confusion matrices, permutation feature importance, and a
serialized model selected using mean cross-validation F1-score.

## Google Colab notebook

Open [cardiovascular_disease_colab.ipynb](./cardiovascular_disease_colab.ipynb)
in Google Colab and run the cells from top to bottom. If the dataset is not
available in `data/cardio_train.csv`, the notebook prompts you to upload
`cardio_train.csv`. The notebook includes the executed model comparison and
plots from the supplied dataset.

## Dataset

The project expects the supplied dataset at `data/cardio_train.csv`. Place your
local copy there before running the script. The dataset and generated outputs
are excluded from this public repository because they contain patient health
records and patient-level predictions. The file is semicolon delimited and
contains the binary `cardio` outcome (`0` = no cardiovascular disease,
`1` = cardiovascular disease).

Preprocessing converts age from days to years, drops rows without a valid target,
marks implausible age, height, weight, and blood-pressure values as missing,
marks invalid categorical/binary values as missing, median/mode imputes values,
one-hot encodes gender/cholesterol/glucose, and standardizes numeric features.
The patient identifier is not used as a model feature. A stratified 80/20 split
is used by default; all model preprocessing is fitted within each CV fold to
avoid leakage.

## Setup and run

From the project directory, create and activate a virtual environment, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python cardio_prediction.py
```

For Google Colab, upload `cardio_train.csv` when prompted by the notebook.

The command uses stratified 5-fold cross-validation and an 80/20 train-test
split. The model with the highest mean cross-validation F1-score is selected.
To use a different dataset path, output directory, or CV fold count:

```powershell
python cardio_prediction.py --data "C:\path\to\cardio_train.csv" --output-dir outputs --cv-folds 5
```

To also classify an unlabeled patient-record CSV after training, pass
`--predict-input`. The file needs the same feature columns as the training data;
the target column is not required. The output includes a predicted class and
estimated probability for class 1:

```powershell
python cardio_prediction.py --predict-input "C:\path\to\new_patients.csv"
```

## Output files

- `outputs/cross_validation_metrics.csv` — mean and standard deviation of CV accuracy and F1.
- `outputs/test_metrics.csv` — held-out accuracy, precision, recall, F1, and confusion-matrix counts.
- `outputs/confusion_matrices.png` — one confusion matrix for each classifier.
- `outputs/feature_importance.csv` and `outputs/feature_importance.png` — permutation importance for the selected model.
- `outputs/best_model.joblib` — fitted preprocessing and classifier pipeline.
- `outputs/summary.json` — dataset/split details, selected model, test metrics, and top features.
- `outputs/predictions.csv` — generated only when `--predict-input` is supplied.

Confusion-matrix counts are ordered as `[true negative, false positive, false
negative, true positive]`. Positive-class precision, recall, and F1 refer to
`cardio = 1`.

## Comparative interpretation

In the supplied dataset run (seed 42, stratified 80/20 split, five CV folds),
the MLP ranked first by mean CV F1 (about 0.719) and achieved held-out accuracy
0.724, precision 0.745, recall 0.680, and F1 0.711. Decision Tree achieved test
F1 0.690; KNN 0.688; and Naive Bayes 0.622. The best model is selected from CV,
not the held-out test results. Re-run the project to generate exact metrics for
your environment and compare full metrics in the output CSV files.

- **Decision Tree:** interpretable split rules and little sensitivity to feature
  scaling; can overfit and may be unstable when the data changes.
- **Naive Bayes:** fast and simple baseline; its conditional-independence and
  Gaussian numeric-feature assumptions can limit performance on correlated,
  non-Gaussian health measurements.
- **KNN:** flexible local decision boundaries and a simple training process;
  prediction can be slower on larger datasets and performance depends on scaling
  and the choice of neighbors.
- **Neural Network (MLP):** strongest result in this run and able to learn
  nonlinear patterns; less interpretable, needs scaling, and its result depends
  on training configuration and data quality.

## Feature interpretation and limitations

Permutation importance measures the change in test-sample F1 when a feature is
shuffled; it is a model-specific association, not evidence that the feature
causes cardiovascular disease. Correlated variables may divide or obscure
importance. The models learn patterns in this dataset and may reflect its
sampling, measurement quality, and population. A random split is not an
independent clinical validation, and no model here should be used to diagnose
or guide treatment.

## Conceptual healthcare decision-support workflow

1. A healthcare professional enters or imports patient measurements with consent.
2. The system validates required inputs and applies the exact preprocessing
   pipeline saved with the selected model.
3. The model returns an estimated class and (where clinically reviewed) a risk
   score, alongside a clear non-diagnostic notice.
4. The interface displays relevant patient factors and model limitations.
5. A qualified clinician reviews the result with the patient's full history and
   determines whether further assessment is appropriate.

The project is a comparative research prototype only. It is not a medical
device, diagnostic system, or replacement for professional judgment.

# Data-mining-project2
