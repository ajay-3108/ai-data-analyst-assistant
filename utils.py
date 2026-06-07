import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_insights(df):

    #Dataset Shape
    shape_info = f"""
Rows: {df.shape[0]}
Columns: {df.shape[1]}
"""

    # Column Types
    column_types = df.dtypes.astype(str).to_string()

    # Missing Values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if len(missing_values) == 0:
        missing_summary = "No missing values found."
    else:
        missing_summary = missing_values.to_string()

    # Duplicate Rows
    duplicate_count = df.duplicated().sum()

    # Numeric Summary
    try:
        numeric_summary = df.describe().to_string()
    except:
        numeric_summary = "No numeric columns available."

    # Top Categories
    cat_cols = df.select_dtypes(include=["object"]).columns

    top_categories = ""

    for col in cat_cols[:10]:  # limit to first 10 categorical columns
        try:
            top_categories += f"\n\nColumn: {col}\n"
            top_categories += df[col].value_counts().head(5).to_string()
        except:
            pass

    if top_categories == "":
        top_categories = "No categorical columns found."

    # Correlation Analysis
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

    # Sample rows for additional context
    sample_rows = df.head(20).to_string()

    # analysis report
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
You are an expert business analyst.

Analyze the dataset report below and create a simple, user-friendly summary.

IMPORTANT:
- Write for business users, not data scientists.
- Use plain English.
- Avoid technical jargon unless necessary.
- Keep points short and easy to understand.
- Use bullet points.
- Focus on the most important findings only.
- Do not repeat statistics excessively.

DATASET REPORT:
{analysis_report}

Return the response in EXACTLY this format:

## Dataset Overview
- 3 concise bullet points

## Key Insights
- 5 concise bullet points

## Data Quality Issues
- Mention only actual issues found
- If none exist, write "No major data quality issues found."

## Business Trends
- 3 to 5 concise bullet points

## Recommendations
- 3 to 5 actionable recommendations

Rules:
- Maximum 1 sentence per bullet.
- No raw tables.
- No technical metadata.
- No column data types.
- No long paragraphs.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

def chat_with_data(df, question):

    context = f"""
DATASET INFO:
Rows: {df.shape[0]}
Columns: {df.shape[1]}

COLUMN NAMES:
{df.columns.tolist()}

SAMPLE DATA:
{df.head(20).to_string()}
"""

    prompt = f"""
You are a senior data analyst assistant.

Answer the user question using ONLY the dataset context.

Rules:
- Do NOT assume anything outside data
- If not possible, say "Not enough data to answer this"
- Keep answer short and business friendly

DATASET CONTEXT:
{context}

QUESTION:
{question}

Give a clear answer with reasoning.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text