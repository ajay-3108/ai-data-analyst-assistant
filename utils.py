import os
import google.generativeai as genai

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model
model = genai.GenerativeModel("gemini-1.5-flash")


def generate_insights(df):

    shape_info = f"""
Rows: {df.shape[0]}
Columns: {df.shape[1]}
"""

    column_types = df.dtypes.astype(str).to_string()

    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    missing_summary = (
        "No missing values found."
        if len(missing_values) == 0
        else missing_values.to_string()
    )

    duplicate_count = df.duplicated().sum()

    try:
        numeric_summary = df.describe().to_string()
    except:
        numeric_summary = "No numeric columns available."

    cat_cols = df.select_dtypes(include=["object"]).columns

    top_categories = ""

    for col in cat_cols[:10]:
        try:
            top_categories += f"\n\nColumn: {col}\n"
            top_categories += df[col].value_counts().head(5).to_string()
        except:
            pass

    if top_categories == "":
        top_categories = "No categorical columns found."

    numeric_cols = df.select_dtypes(include=["number"])

    if len(numeric_cols.columns) > 1:
        corr_matrix = numeric_cols.corr()

        corr_pairs = (
            corr_matrix.unstack()
            .sort_values(ascending=False)
        )

        corr_pairs = corr_pairs[corr_pairs < 1]

        corr_summary = corr_pairs.head(10).to_string()
    else:
        corr_summary = "Not enough numeric columns for correlation analysis."

    sample_rows = df.head(10).to_string()

    analysis_report = f"""
DATASET SHAPE
{shape_info}

COLUMN TYPES
{column_types}

MISSING VALUES
{missing_summary}

DUPLICATE ROWS
{duplicate_count}

NUMERIC SUMMARY
{numeric_summary}

TOP CATEGORIES
{top_categories}

CORRELATION ANALYSIS
{corr_summary}

SAMPLE DATA
{sample_rows}
"""

    prompt = f"""
You are an expert data analyst.

Analyze this dataset and give insights.

DATA:
{analysis_report}
"""

    response = model.generate_content(prompt)
    return response.text
