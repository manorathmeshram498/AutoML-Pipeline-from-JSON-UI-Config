# 📊 AutoML Pipeline from JSON UI Config

This project implements a fully **automated machine learning pipeline** that dynamically parses a JSON configuration (exported from a UI) and programmatically executes a sequence of ML tasks including:

- ✅ Feature handling (missing value imputation, encoding)
- ✅ Feature generation and reduction (PCA, tree-based, etc.)
- ✅ Model selection and hyperparameter tuning
- ✅ Training and evaluation with appropriate metrics

> **Built to be fully generic** — simply edit the JSON to change the ML flow without touching code.

---

## 📁 Input Files

- **Dataset**: `iris.csv` or any CSV file with the columns specified in the JSON.
- **Configuration**: `algoparams_from_ui.json.rtf` (RTF file containing JSON exported from UI).

---

## 🔧 What It Does

- Parses JSON config from a `.rtf` file using `striprtf`
- Handles features based on their types:
  - **Numerical**: Missing value imputation (mean/custom)
  - **Categorical**: One-hot encoding
- Generates interaction features (optional via JSON)
- Reduces features using:
  - PCA
  - Tree-based importance (`SelectKBest`)
  - No reduction (optional)
- Selects and trains models like:
  - `RandomForestRegressor`
  - `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
  - `GradientBoostingRegressor` (GBR)
  - `DecisionTreeRegressor`
- Tunes hyperparameters using `GridSearchCV` and cross-validation strategy defined in JSON
- Logs model performance metrics:
  - `R² Score`
  - `MAE`, `MSE`, `RMSE`

---

## 📌 Requirements

- `pandas`
- `numpy`
- `scikit-learn`
- `striprtf`

Install with:

```bash
pip install -r requirements.txt
