"""StackOverflow 2024 Developer Survey - salary and satisfaction analysis.

Udacity Data Scientist Nanodegree - Project 1 (Write a Data Science Blog Post)
Author: Sukh Sandhu

This script follows the CRISP-DM process:
    1. Business Understanding - five questions about developer pay and
       job satisfaction (see BUSINESS_QUESTIONS below).
    2. Data Understanding    - load the official survey results and profile
       size, dtypes and missing values.
    3. Data Preparation      - filter to professional full-time developers,
       clean the experience column, handle missing values and encode
       categorical features (choices documented in the function docstrings).
    4. Modeling              - Random Forest regression predicting salary.
    5. Evaluation            - R^2 / RMSE on a held-out test set plus
       aggregated feature importances.
    6. Deployment            - blog post and README summarising findings.

Data source (download manually, ~150 MB, not committed to the repository):
    https://survey.stackoverflow.co/  -> 2024 -> "Data (CSV)"
    Save as `so_survey_results_2024.csv` next to this script, or pass a
    custom path as the first command-line argument.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (backend set above)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder  # noqa: E402

# --------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------
DEFAULT_DATA_PATH = "so_survey_results_2024.csv"
RANDOM_STATE = 42

# Documented salary bounds: values below 5k USD are almost always currency
# conversion artefacts or part-time pay reported as yearly; values above
# 400k USD (the 99th percentile) are extreme outliers that would dominate
# squared-error metrics.
SALARY_MIN_USD = 5_000
SALARY_MAX_USD = 400_000

MISSING_CATEGORY_LABEL = "Not specified"

# Explicit category orders for genuinely ORDINAL variables. Encoding these
# with ordered integers preserves their natural ranking.
EDUCATION_ORDER = [
    "Primary/elementary school",
    "Secondary school (e.g. American high school, German Realschule or "
    "Gymnasium, etc.)",
    "Some college/university study without earning a degree",
    "Associate degree (A.A., A.S., etc.)",
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
    "Something else",
]
ORG_SIZE_ORDER = [
    "Just me - I am a freelancer, sole proprietor, etc.",
    "2 to 9 employees",
    "10 to 19 employees",
    "20 to 99 employees",
    "100 to 499 employees",
    "500 to 999 employees",
    "1,000 to 4,999 employees",
    "5,000 to 9,999 employees",
    "10,000 or more employees",
    "I don’t know",
]
AGE_ORDER = [
    "Under 18 years old",
    "18-24 years old",
    "25-34 years old",
    "35-44 years old",
    "45-54 years old",
    "55-64 years old",
    "65 years or older",
    "Prefer not to say",
]

ORDINAL_FEATURES = {
    "EdLevel": EDUCATION_ORDER,
    "OrgSize": ORG_SIZE_ORDER,
    "Age": AGE_ORDER,
}
# Nominal (unordered) variables: one-hot encoded because assigning them
# ordered integers would invent a ranking that does not exist.
NOMINAL_FEATURES = ["Country", "DevType", "RemoteWork"]
NUMERIC_FEATURES = ["YearsCodeProNumeric"]
MAX_ONE_HOT_CATEGORIES = 12

BUSINESS_QUESTIONS = [
    "Q1. Which programming languages are associated with the highest "
    "salaries?",
    "Q2. How do professional experience and education relate to "
    "compensation?",
    "Q3. How does remote, hybrid or in-person work relate to job "
    "satisfaction?",
    "Q4. How does compensation vary across countries?",
    "Q5. Can a model predict a developer's salary, and which factors "
    "matter most?",
]


# --------------------------------------------------------------------------
# CRISP-DM step 2: Data Understanding
# --------------------------------------------------------------------------
def load_survey_data(csv_path):
    """Load the raw StackOverflow survey CSV.

    Parameters
    ----------
    csv_path : str
        Path to the official 2024 `results.csv` download.

    Returns
    -------
    pandas.DataFrame
        Raw survey responses (one row per respondent).
    """
    try:
        return pd.read_csv(csv_path, low_memory=False)
    except FileNotFoundError:
        raise SystemExit(
            f"Data file not found: {csv_path}\n"
            "Download the 2024 survey CSV from https://survey.stackoverflow.co/ "
            "and save it as so_survey_results_2024.csv (see README)."
        )


def summarize_missing_values(survey_df, columns):
    """Return a count and percentage of missing values per column.

    Parameters
    ----------
    survey_df : pandas.DataFrame
        Survey data to profile.
    columns : list of str
        Columns to include in the summary.

    Returns
    -------
    pandas.DataFrame
        Indexed by column name with `missing_count` and `missing_pct`.
    """
    missing_count = survey_df[columns].isnull().sum()
    missing_pct = (missing_count / len(survey_df) * 100).round(1)
    return pd.DataFrame(
        {"missing_count": missing_count, "missing_pct": missing_pct}
    ).sort_values("missing_pct", ascending=False)


# --------------------------------------------------------------------------
# CRISP-DM step 3: Data Preparation
# --------------------------------------------------------------------------
def clean_years_experience(years_series):
    """Convert the `YearsCodePro` survey column to numeric years.

    The survey stores the answer as text with two special values:
    "Less than 1 year" (mapped to 0.5) and "More than 50 years"
    (mapped to 51). Everything else parses directly as a number.

    Parameters
    ----------
    years_series : pandas.Series
        Raw `YearsCodePro` column.

    Returns
    -------
    pandas.Series
        Numeric years of professional coding experience (NaN preserved).
    """
    text_value_map = {"Less than 1 year": 0.5, "More than 50 years": 51.0}
    return pd.to_numeric(years_series.replace(text_value_map), errors="coerce")


def filter_professional_developers(raw_df):
    """Restrict the survey to comparable, professionally paid respondents.

    Filtering decisions (documented for CRISP-DM Data Preparation):
    * `MainBranch` == "I am a developer by profession" - hobbyists and
      students answer very different pay questions.
    * `Employment` contains "Employed, full-time" - keeps yearly salaries
      comparable across respondents.
    * `ConvertedCompYearly` is present - rows without the target variable
      cannot be used for salary analysis or supervised modeling, so they
      are dropped rather than imputed (imputing a target would fabricate
      the very signal we are trying to study).
    * Salary kept within [SALARY_MIN_USD, SALARY_MAX_USD] to remove
      currency artefacts and extreme outliers.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Raw survey responses.

    Returns
    -------
    pandas.DataFrame
        Filtered copy with an added numeric `YearsCodeProNumeric` column.
    """
    is_professional = raw_df["MainBranch"] == "I am a developer by profession"
    is_full_time = raw_df["Employment"].str.contains(
        "Employed, full-time", na=False
    )
    has_salary = raw_df["ConvertedCompYearly"].notna()

    developers_df = raw_df[is_professional & is_full_time & has_salary].copy()
    salary_in_range = developers_df["ConvertedCompYearly"].between(
        SALARY_MIN_USD, SALARY_MAX_USD
    )
    developers_df = developers_df[salary_in_range]
    developers_df["YearsCodeProNumeric"] = clean_years_experience(
        developers_df["YearsCodePro"]
    )
    return developers_df


def impute_missing_features(developers_df):
    """Handle missing values in the modeling features.

    Strategy (and the reasoning behind it):
    * Nominal categoricals (Country, DevType, RemoteWork) - missing answers
      are replaced with an explicit "Not specified" category. "Did not
      answer" is itself informative, and this keeps every row usable
      without inventing a country or role.
    * Ordinal categoricals (EdLevel, OrgSize, Age) - left as NaN here and
      encoded to -1 by the OrdinalEncoder (`encoded_missing_value=-1`),
      which keeps them in a single ordered scale with "unknown" below the
      lowest real level.
    * Numeric YearsCodeProNumeric - imputed with the median, which is
      robust to the long right tail of the experience distribution.

    Parameters
    ----------
    developers_df : pandas.DataFrame
        Output of :func:`filter_professional_developers`.

    Returns
    -------
    pandas.DataFrame
        Copy with imputations applied.
    """
    imputed_df = developers_df.copy()
    for column in NOMINAL_FEATURES:
        imputed_df[column] = imputed_df[column].fillna(MISSING_CATEGORY_LABEL)
    median_years = imputed_df["YearsCodeProNumeric"].median()
    imputed_df["YearsCodeProNumeric"] = imputed_df[
        "YearsCodeProNumeric"
    ].fillna(median_years)
    return imputed_df


# --------------------------------------------------------------------------
# CRISP-DM step 4a: Analysis helpers (questions 1-4)
# --------------------------------------------------------------------------
def median_salary_by_language(developers_df, min_respondents=1_000):
    """Median salary for each programming language worked with.

    `LanguageHaveWorkedWith` is a semicolon-separated multi-select, so the
    column is split and exploded to one row per developer-language pair.

    Parameters
    ----------
    developers_df : pandas.DataFrame
        Filtered developer rows.
    min_respondents : int, optional
        Languages with fewer users than this are dropped so medians stay
        statistically meaningful.

    Returns
    -------
    pandas.Series
        Median salary (USD) indexed by language, highest first.
    """
    language_salary_df = developers_df[
        ["LanguageHaveWorkedWith", "ConvertedCompYearly"]
    ].dropna()
    language_salary_df = language_salary_df.assign(
        Language=language_salary_df["LanguageHaveWorkedWith"].str.split(";")
    ).explode("Language")

    language_counts = language_salary_df["Language"].value_counts()
    popular_languages = language_counts[
        language_counts >= min_respondents
    ].index
    popular_df = language_salary_df[
        language_salary_df["Language"].isin(popular_languages)
    ]
    return (
        popular_df.groupby("Language")["ConvertedCompYearly"]
        .median()
        .sort_values(ascending=False)
    )


def median_salary_by_group(developers_df, group_column, min_respondents=1):
    """Median salary by an arbitrary categorical column.

    Reused for education (Q2) and country (Q4) to keep the code DRY.

    Parameters
    ----------
    developers_df : pandas.DataFrame
        Filtered developer rows.
    group_column : str
        Column to group by (e.g. "EdLevel", "Country").
    min_respondents : int, optional
        Groups smaller than this are excluded.

    Returns
    -------
    pandas.Series
        Median salary (USD) per group, highest first.
    """
    group_sizes = developers_df[group_column].value_counts()
    large_groups = group_sizes[group_sizes >= min_respondents].index
    eligible_df = developers_df[
        developers_df[group_column].isin(large_groups)
    ]
    return (
        eligible_df.groupby(group_column)["ConvertedCompYearly"]
        .median()
        .sort_values(ascending=False)
    )


def experience_salary_correlation(developers_df):
    """Pearson correlation between professional experience and salary.

    Parameters
    ----------
    developers_df : pandas.DataFrame
        Filtered developer rows with `YearsCodeProNumeric`.

    Returns
    -------
    float
        Correlation coefficient (NaN rows excluded pairwise).
    """
    return developers_df["YearsCodeProNumeric"].corr(
        developers_df["ConvertedCompYearly"]
    )


def satisfaction_by_work_mode(developers_df):
    """Mean job satisfaction (0-10) for each remote-work arrangement.

    Rows missing either `RemoteWork` or `JobSat` are excluded pairwise -
    satisfaction was an optional question and imputing an opinion score
    would bias the comparison.

    Parameters
    ----------
    developers_df : pandas.DataFrame
        Filtered developer rows.

    Returns
    -------
    pandas.Series
        Mean satisfaction per work mode, highest first.
    """
    work_satisfaction_df = developers_df[["RemoteWork", "JobSat"]].dropna()
    is_real_work_mode = (
        work_satisfaction_df["RemoteWork"] != MISSING_CATEGORY_LABEL
    )
    return (
        work_satisfaction_df[is_real_work_mode]
        .groupby("RemoteWork")["JobSat"]
        .mean()
        .sort_values(ascending=False)
    )


# --------------------------------------------------------------------------
# CRISP-DM step 4b: Modeling
# --------------------------------------------------------------------------
def build_salary_pipeline():
    """Create the preprocessing + Random Forest pipeline.

    Encoding choices (documented for the rubric):
    * Ordinal variables (EdLevel, OrgSize, Age) use `OrdinalEncoder` with
      an explicit, meaningful category order; missing values become -1.
    * Nominal variables (Country, DevType, RemoteWork) use
      `OneHotEncoder` - integer-coding them would impose a fake order.
      Rare categories are grouped (max_categories) to limit dimensionality,
      and unseen categories at prediction time are ignored safely.
    * YearsCodeProNumeric passes through unscaled (tree models are
      insensitive to monotone scaling).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline: preprocessing -> RandomForestRegressor.
    """
    ordinal_encoder = OrdinalEncoder(
        categories=[ORDINAL_FEATURES[col] for col in ORDINAL_FEATURES],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    one_hot_encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        max_categories=MAX_ONE_HOT_CATEGORIES,
        sparse_output=False,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_encoder, list(ORDINAL_FEATURES)),
            ("onehot", one_hot_encoder, NOMINAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    random_forest = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(
        [("preprocess", preprocessor), ("model", random_forest)]
    )


def train_salary_model(developers_df, test_size=0.2):
    """Train and evaluate the salary prediction model.

    Parameters
    ----------
    developers_df : pandas.DataFrame
        Output of :func:`impute_missing_features`.
    test_size : float, optional
        Fraction held out for evaluation.

    Returns
    -------
    dict
        Keys: `pipeline` (fitted), `r2`, `rmse`, `n_train`, `n_test`.
    """
    feature_columns = (
        list(ORDINAL_FEATURES) + NOMINAL_FEATURES + NUMERIC_FEATURES
    )
    features = developers_df[feature_columns]
    target = developers_df["ConvertedCompYearly"]

    features_train, features_test, target_train, target_test = (
        train_test_split(
            features, target, test_size=test_size, random_state=RANDOM_STATE
        )
    )
    pipeline = build_salary_pipeline()
    pipeline.fit(features_train, target_train)
    predictions = pipeline.predict(features_test)

    return {
        "pipeline": pipeline,
        "r2": r2_score(target_test, predictions),
        "rmse": float(
            np.sqrt(mean_squared_error(target_test, predictions))
        ),
        "n_train": len(features_train),
        "n_test": len(features_test),
    }


def aggregate_feature_importance(fitted_pipeline):
    """Sum one-hot importances back to their original survey column.

    Parameters
    ----------
    fitted_pipeline : sklearn.pipeline.Pipeline
        Pipeline returned by :func:`train_salary_model`.

    Returns
    -------
    pandas.Series
        Importance share per original feature, highest first.
    """
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]
    transformed_names = preprocessor.get_feature_names_out()

    original_columns = []
    for name in transformed_names:
        _, remainder = name.split("__", 1)
        original = next(
            (
                col
                for col in NOMINAL_FEATURES
                if remainder == col or remainder.startswith(col + "_")
            ),
            remainder,
        )
        original_columns.append(original)

    importance_series = pd.Series(
        model.feature_importances_, index=original_columns
    )
    return importance_series.groupby(level=0).sum().sort_values(
        ascending=False
    )


def predict_salary_for_profile(fitted_pipeline, profile):
    """Predict yearly salary (USD) for a single developer profile.

    Parameters
    ----------
    fitted_pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline from :func:`train_salary_model`.
    profile : dict
        Raw feature values, e.g. ``{"Country": "United States of America",
        "EdLevel": EDUCATION_ORDER[5], ...}``.

    Returns
    -------
    float
        Predicted yearly compensation in USD.
    """
    profile_df = pd.DataFrame([profile])
    return float(fitted_pipeline.predict(profile_df)[0])


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
def plot_salary_barh(axis, salary_series, title, color, top_n=10):
    """Draw a horizontal bar chart of median salaries on a given axis.

    Shared by the language and country panels (DRY).

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Target axis.
    salary_series : pandas.Series
        Median salary per category, descending.
    title : str
        Axis title.
    color : str
        Bar color.
    top_n : int, optional
        Number of categories to display.
    """
    top_series = salary_series.head(top_n).sort_values()
    axis.barh(top_series.index, top_series.values / 1_000, color=color)
    axis.set_title(title)
    axis.set_xlabel("Median salary (USD thousands)")


def create_four_panel_visualization(
    language_salaries,
    education_salaries,
    work_mode_satisfaction,
    country_salaries,
    output_path="salary_analysis.png",
):
    """Save the four-panel summary figure used in the blog post.

    Parameters
    ----------
    language_salaries, education_salaries, country_salaries : pandas.Series
        Median salary series for Q1, Q2 and Q4.
    work_mode_satisfaction : pandas.Series
        Mean job satisfaction per remote-work mode (Q3).
    output_path : str, optional
        Where to save the PNG.
    """
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_salary_barh(
        axes[0, 0],
        language_salaries,
        "Q1: Median salary by programming language (top 10)",
        "steelblue",
    )

    education_order_present = [
        level for level in EDUCATION_ORDER
        if level in education_salaries.index
    ]
    ordered_education = education_salaries.reindex(education_order_present)
    short_labels = [
        label.split(" (")[0] for label in ordered_education.index
    ]
    axes[0, 1].bar(
        range(len(ordered_education)),
        ordered_education.values / 1_000,
        color="coral",
    )
    axes[0, 1].set_xticks(range(len(ordered_education)))
    axes[0, 1].set_xticklabels(
        short_labels, rotation=30, ha="right", fontsize=8
    )
    axes[0, 1].set_title("Q2: Median salary by education level")
    axes[0, 1].set_ylabel("Median salary (USD thousands)")

    short_work_mode_labels = [
        label.split(" (")[0] for label in work_mode_satisfaction.index
    ]
    axes[1, 0].bar(
        short_work_mode_labels,
        work_mode_satisfaction.values,
        color=["seagreen", "orange", "indianred"],
    )
    axes[1, 0].set_title("Q3: Mean job satisfaction by work arrangement")
    axes[1, 0].set_ylabel("Job satisfaction (0-10)")
    axes[1, 0].set_ylim(0, 10)
    axes[1, 0].tick_params(axis="x", labelsize=8)

    plot_salary_barh(
        axes[1, 1],
        country_salaries,
        "Q4: Median salary by country (top 10 responders)",
        "purple",
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------
# CRISP-DM steps 1-6 orchestrated
# --------------------------------------------------------------------------
def main(csv_path=DEFAULT_DATA_PATH):
    """Run the full CRISP-DM analysis and print a results report.

    Parameters
    ----------
    csv_path : str, optional
        Location of the survey CSV.
    """
    print("=" * 64)
    print("CRISP-DM 1: BUSINESS UNDERSTANDING")
    print("=" * 64)
    for question in BUSINESS_QUESTIONS:
        print(question)

    print("\n" + "=" * 64)
    print("CRISP-DM 2: DATA UNDERSTANDING")
    print("=" * 64)
    raw_df = load_survey_data(csv_path)
    print(f"Raw survey shape: {raw_df.shape}")
    profile_columns = [
        "ConvertedCompYearly", "JobSat", "YearsCodePro", "EdLevel",
        "Country", "DevType", "OrgSize", "RemoteWork",
        "LanguageHaveWorkedWith",
    ]
    print("Missing values in key columns:")
    print(summarize_missing_values(raw_df, profile_columns).to_string())

    print("\n" + "=" * 64)
    print("CRISP-DM 3: DATA PREPARATION")
    print("=" * 64)
    developers_df = filter_professional_developers(raw_df)
    print(
        f"Professional full-time developers with valid salary: "
        f"{len(developers_df):,} of {len(raw_df):,} respondents"
    )
    developers_df = impute_missing_features(developers_df)

    print("\n" + "=" * 64)
    print("CRISP-DM 4: ANALYSIS AND MODELING")
    print("=" * 64)

    language_salaries = median_salary_by_language(developers_df)
    print("\nQ1 - Median salary by language (top 10):")
    print((language_salaries.head(10) / 1_000).round(1).to_string())

    correlation = experience_salary_correlation(developers_df)
    education_salaries = median_salary_by_group(developers_df, "EdLevel")
    print(f"\nQ2 - Experience vs salary Pearson r: {correlation:.3f}")
    print("Q2 - Median salary by education:")
    print((education_salaries / 1_000).round(1).to_string())

    work_mode_satisfaction = satisfaction_by_work_mode(developers_df)
    print("\nQ3 - Mean job satisfaction (0-10) by work mode:")
    print(work_mode_satisfaction.round(2).to_string())

    country_salaries = median_salary_by_group(
        developers_df, "Country", min_respondents=400
    )
    print("\nQ4 - Median salary by country (>=400 respondents, top 10):")
    print((country_salaries.head(10) / 1_000).round(1).to_string())

    model_results = train_salary_model(developers_df)
    print(
        f"\nQ5 - Random Forest: R^2 = {model_results['r2']:.3f}, "
        f"RMSE = ${model_results['rmse']:,.0f} "
        f"(train n={model_results['n_train']:,}, "
        f"test n={model_results['n_test']:,})"
    )

    print("\n" + "=" * 64)
    print("CRISP-DM 5: EVALUATION")
    print("=" * 64)
    importances = aggregate_feature_importance(model_results["pipeline"])
    print("Aggregated feature importance:")
    print(importances.round(3).to_string())

    example_profile = {
        "Country": "United States of America",
        "EdLevel": EDUCATION_ORDER[5],
        "OrgSize": "1,000 to 4,999 employees",
        "Age": "25-34 years old",
        "DevType": "Data scientist or machine learning specialist",
        "RemoteWork": "Remote",
        "YearsCodeProNumeric": 8,
    }
    predicted_salary = predict_salary_for_profile(
        model_results["pipeline"], example_profile
    )
    print(
        "\nExample profile (US ML specialist, Master's, 8 yrs, remote, "
        f"large org): ${predicted_salary:,.0f}"
    )

    create_four_panel_visualization(
        language_salaries,
        education_salaries,
        work_mode_satisfaction,
        country_salaries,
    )
    print("\nSaved visualization to salary_analysis.png")

    print("\n" + "=" * 64)
    print("CRISP-DM 6: DEPLOYMENT")
    print("=" * 64)
    print("Findings are communicated in the blog post (index.md) and "
          "README.md.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH)
