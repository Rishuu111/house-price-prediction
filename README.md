# House Price Prediction

**Author:** Rishu Rana

## Project Overview

House Price Prediction is a modular data-science project for preparing housing data, exploring property characteristics, comparing regression algorithms, evaluating model quality, and generating estimated house prices through a reusable prediction workflow and Streamlit dashboard.

## Problem Statement

House prices depend on many interacting property characteristics. The project addresses the problem of learning those relationships from historical housing records and using them to estimate prices for new properties while documenting data quality, model performance, and system limitations.

## Objectives

- Verify the structure and quality of the available housing dataset.
- Produce a separate cleaned dataset without altering the source data.
- Explore numerical, categorical, price, relationship, correlation, and outlier patterns.
- Compare multiple regression algorithms using the same evaluation methodology.
- Select and retrain the best-performing model using a documented metric-based rule.
- Provide reusable batch and interactive prediction interfaces.
- Record results and limitations clearly for project reporting.

## Dataset Description

The expected input is a tabular housing dataset with one row per property, property characteristics as feature columns, and a numeric sale-price target. The default target name is `SalePrice`; the cleaning pipeline normalizes column names for consistent downstream use.

No source housing dataset is currently included in `data/`. Add the dataset there before running the workflow. The verification process supports CSV, Excel, and Parquet files and records the observed shape, columns, data types, missing values, duplicates, and suspicious values in `dataset_summary.md`.

## Technologies Used

- Python
- pandas for data preparation
- scikit-learn for preprocessing, regression, splitting, and metrics
- joblib for model persistence
- Matplotlib and Seaborn for visual analysis
- Streamlit for the interactive dashboard
- Markdown, CSV, JSON, and PNG files for project outputs

## Project Structure

```text
data/                         Raw and cleaned datasets
notebooks/                    Analysis and experiment notebooks
src/
	data_cleaning.py            Reusable data-cleaning pipeline
	verify_dataset.py           Dataset discovery and verification
	eda.py                      Exploratory analysis and chart generation
	train_model.py              Regression model comparison
	evaluate_model.py           Metrics and diagnostic visualizations
	finalize_model.py           Evidence-based final model training
	predict.py                  Shared prediction API
	predict_price.py            CSV prediction command
model/                        Saved models and preprocessing artifacts
reports/                      Metrics, predictions, insights, and charts
dashboard/app.py              Streamlit prediction dashboard
requirements.txt              Python dependencies
dataset_summary.md            Dataset verification report
```

## Data-Cleaning Process

The cleaning pipeline keeps the input DataFrame and source file unchanged. It can:

- Remove duplicate records and entirely empty columns.
- Normalize column names and categorical text formatting.
- Convert common numeric formats such as commas, currency symbols, percentages, and parentheses.
- Convert reliably numeric object columns to numeric types.
- Represent common missing-value tokens as missing values.
- Remove records without a valid positive target price.
- Remove common index or identifier columns and user-specified unnecessary columns.
- Impute numeric values with the median and categorical values with the mode or `unknown`.
- Save the cleaned output to a separate path.

## Exploratory Analysis

`src/eda.py` creates report-ready artifacts in `reports/charts/`, including numerical distributions, categorical frequency charts, the house-price distribution, feature-price relationships, an IQR-based outlier scan, an outlier summary CSV, and a numerical correlation heatmap.

## Machine-Learning Approach

The comparison workflow applies the same preprocessing and the same reproducible train/test split to each candidate model:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression

Numerical features are median-imputed. Categorical features are mode-imputed and one-hot encoded with unknown-category handling. The split uses a fixed random state of `42` by default.

## Model Evaluation

The comparison records MAE, RMSE, and R² for each candidate. The evaluation module also records MSE and produces actual-vs-predicted and residual diagnostics.

The selected model is determined from measured results using this rule:

1. Lowest RMSE
2. Lowest MAE when RMSE is tied
3. Highest R² when the previous metrics are tied

These metrics describe the supplied evaluation data. They do not guarantee performance on future properties or different markets.

## Prediction Workflow

1. Load the finalized saved pipeline.
2. Validate that required feature columns are present.
3. Apply the same fitted preprocessing used during training.
4. Generate a price estimate.
5. Return a readable value or save predictions to CSV.

`src/predict.py` provides the reusable prediction API. `src/predict_price.py` provides a command-line CSV interface. The saved final pipeline keeps preprocessing and the trained estimator together so prediction uses the training-time transformations.

## Dashboard Information

`dashboard/app.py` provides a Streamlit interface that derives its input fields from the saved model schema. Users can enter property features, submit the form, and view a formatted estimated price along with project and model information. The dashboard calls the shared prediction module and clearly identifies the result as an estimate, not a guaranteed sale price.

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## How to Run the Project

Place a source CSV, Excel, or Parquet dataset in `data/`, then run the stages in order. Execute commands from the project root so relative paths resolve to `data/`, `model/`, and `reports/`:

```bash
python -m src.verify_dataset
python -m src.data_cleaning --input data/housing.csv --output data/cleaned/housing_cleaned.csv --target SalePrice
python -m src.eda --input data/cleaned/housing_cleaned.csv --target SalePrice --output reports/charts
python -m src.train_model --input data/cleaned/housing_cleaned.csv --target SalePrice
python -m src.evaluate_model --input data/cleaned/housing_cleaned.csv --target SalePrice --model model/random_forest.joblib
python -m src.finalize_model --input data/cleaned/housing_cleaned.csv --target SalePrice
python -m src.predict_price --input data/new_houses.csv --output reports/predictions.csv
streamlit run dashboard/app.py
```

The dashboard expects `model/final_model.joblib`. Generate it by completing the comparison and finalization steps before launching Streamlit. The command-line prediction script and dashboard both load the saved final pipeline, including its fitted preprocessing.

Key outputs include:

- `dataset_summary.md`
- `reports/charts/`
- `reports/model_comparison.csv` and `reports/model_comparison.md`
- `reports/model_evaluation.json` and `reports/model_evaluation.md`
- `model/final_model.joblib`
- `model/final_preprocessor.joblib`
- `model/final_model_metadata.json`

## Limitations

- The current repository does not include a source housing dataset, so no project-specific statistics or model results can be claimed until data is added and the workflow is run.
- Model quality depends on the data's coverage, accuracy, target definition, and representativeness.
- A single train/test split can produce results that vary with the split; cross-validation may provide a more stable estimate.
- Regression errors can be uneven across price ranges and property types.
- Unknown feature values, out-of-range inputs, and changes in the housing market may reduce prediction reliability.
- The dashboard provides an estimate and should not be treated as an appraisal, valuation, or guarantee.

## Future Improvements

- Add and document a validated public or organizational housing dataset.
- Use cross-validation and hyperparameter tuning after establishing the baseline comparison.
- Add time-aware validation when property sale dates are available.
- Investigate log-target modeling and robust treatment of extreme prices.
- Add drift monitoring and scheduled model retraining.
- Expand dashboard diagnostics, input guidance, and prediction intervals.
- Add automated tests, continuous integration, and data-version tracking.
