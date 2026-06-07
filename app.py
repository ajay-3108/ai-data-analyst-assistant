import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils import generate_insights, chat_with_data

st.markdown(
    """
    <style>
    .stApp {
        background-color: indigo;
        color: #d1d5db;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    page_title="AI Data Analyst Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #4F8BF9;'>
    📊 AI Data Analyst Assistant
    </h1>
    <p style='text-align: center; color: gray;'>
    Upload your dataset and get instant AI insights, charts, and analysis
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown(
    "<h4 style='color:#d1d5db;'>📂 Upload Your Dataset (CSV / Excel)</h4>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "",
    type=["csv", "xlsx"]
)


def get_true_numeric_columns(df):
    exclude_keywords = ["id", "code", "zip", "phone"]

    numeric_cols = df.select_dtypes(include="number").columns

    filtered_cols = [
        col for col in numeric_cols
        if not any(keyword in col.lower() for keyword in exclude_keywords)
    ]

    return filtered_cols


#Upload file
if uploaded_file is not None:

    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ Dataset Loaded Successfully")


    # SIDEBAR CHAT
    
    st.sidebar.markdown("## 💬 AI Chat Assistant")
    st.sidebar.caption("Ask anything about your dataset")
    st.sidebar.markdown("---")

    user_question = st.sidebar.text_input("Ask a question")

    if st.sidebar.button("Ask AI"):

        if user_question.strip():

            with st.spinner("Thinking..."):
                answer = chat_with_data(df, user_question)

            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("AI", answer))

    st.sidebar.divider()

    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.sidebar.markdown(f"🧑 **You:** {msg}")
        else:
            st.sidebar.markdown(f"🤖 **AI:** {msg}")

    # data preview

    st.markdown("---")
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div style="padding:15px;border-radius:10px;background-color:#1f1f1f;color:white;text-align:center">
    <h3>{df.shape[0]}</h3>
    Rows
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style="padding:15px;border-radius:10px;background-color:#1f1f1f;color:white;text-align:center">
    <h3>{df.shape[1]}</h3>
    Columns
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div style="padding:15px;border-radius:10px;background-color:#1f1f1f;color:white;text-align:center">
    <h3>{df.isnull().sum().sum()}</h3>
    Missing Values
    </div>
    """, unsafe_allow_html=True)


    # SUMMARY
    st.markdown("---")
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.describe())


    # AI INSIGHTS
    st.markdown("---")
    st.subheader("🤖 AI Insights")

    if st.button("Generate AI Insights"):

        with st.spinner("Analyzing data..."):
            insights = generate_insights(df)

        st.markdown(insights)


    # VISUALIZATIONS
    st.markdown("---")
    st.subheader("📈 Interactive Visualization")

    numeric_cols = get_true_numeric_columns(df)

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Select Numeric Column",
            numeric_cols
        )

        fig = px.histogram(
            df,
            x=selected_col,
            title=f"Distribution of {selected_col}"
        )

        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 2:

        x_axis = st.selectbox(
            "X Axis",
            numeric_cols,
            key="x_axis"
        )

        y_axis = st.selectbox(
            "Y Axis",
            numeric_cols,
            key="y_axis"
        )

        scatter_fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=f"{x_axis} vs {y_axis}"
        )

        st.plotly_chart(scatter_fig, use_container_width=True)

else:
    st.info("👆 Upload a dataset to begin analysis.")

st.markdown("---")
st.subheader("📊 Category Analysis")

cat_cols = df.select_dtypes(include=["object"]).columns

if len(cat_cols) > 0:

    selected_cat = st.selectbox(
        "Select Category Column",
        cat_cols
    )

    top_categories = (
        df[selected_cat]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_categories.columns = [selected_cat, "Count"]

    bar_fig = px.bar(
        top_categories,
        x=selected_cat,
        y="Count",
        title=f"Top 10 {selected_cat}"
    )

    st.plotly_chart(
        bar_fig,
        use_container_width=True
    )

st.markdown("---")
st.subheader("🔥 Correlation Heatmap")

numeric_cols = get_true_numeric_columns(df)

if len(numeric_cols) >= 2:

    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)
