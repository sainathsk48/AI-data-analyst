import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from groq import Groq
import re
import json
import io

# ── helpers ──────────────────────────────────────────────────────────────────

def get_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

MODEL = "llama-3.1-8b-instant"

def ask_groq(prompt: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

def generate_summary(df, insight, chart_info):
    """5.A Generate Plain-English Summary"""
    prompt = f"""
Based on the data statistics, AI insight, and chart information below, write a plain-English summary
for a non-technical audience. Return ONLY 3-5 bullet points (one per line, starting with '- '). No extra text.

Data Statistics:
{df.describe().to_string()}

AI Insight:
{insight}

Chart Information:
{chart_info}
"""
    try:
        return ask_groq(prompt)
    except Exception as e:
        return f"- Summary generation failed: {e}"

# ── main app ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="AI Data Analyst", layout="wide")
    st.title("🤖 AI Data Analyst Web App")

    # 3.1 Sidebar
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("👈 Upload a CSV file from the sidebar to get started.")
        return

    try:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='iso-8859-1')
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    # ── Data Preview ──────────────────────────────────────────────────────────
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    st.write("**Columns:**", list(df.columns))

    st.markdown("---")

    # ── AI Question Input ─────────────────────────────────────────────────────
    st.subheader("💬 Ask AI About Your Data")
    user_question = st.text_input("Your question", placeholder="e.g. Show me sales trend by month")

    if user_question:
        if "GROQ_API_KEY" not in st.secrets:
            st.error("GROQ_API_KEY not found in st.secrets.")
            return

        with st.spinner("Thinking..."):
            # 4.1 Prompt Construction
            # We must ask the AI to write Python code because passing a sample is inaccurate for aggregations.
            prompt = f"""
You are an expert Python data analyst. The user has a pandas DataFrame named `df`.
Here is the structure of `df`:
Columns: {list(df.columns)}
Data Types: {df.dtypes.astype(str).to_dict()}
First 3 rows:
{df.head(3).to_string()}

User Question: {user_question}

Write Python code to answer this question. The code MUST do two things:
1. Calculate the correct answer using `df` and store it as a string in a variable named `insight`.
2. If a chart makes sense, create a Plotly Express figure and store it in a variable named `fig`. If no chart makes sense, set `fig = None`.

IMPORTANT: 
- For charts, ALWAYS aggregate the data first (e.g., groupby) before plotting so it doesn't plot thousands of overlapping rows!
- Assume `import pandas as pd` and `import plotly.express as px` are already imported.
- Return ONLY valid Python code. No markdown formatting, no backticks, no explanations. Just the code.
"""
            try:
                raw = ask_groq(prompt)

                # 4.3 Clean Code
                code = raw.replace("```python", "").replace("```", "").strip()

                # 4.4 Execute Code
                local_vars = {'df': df, 'px': px, 'pd': pd, 'insight': 'No insight generated.', 'fig': None}
                
                try:
                    exec(code, globals(), local_vars)
                except Exception as exec_err:
                    st.error(f"Error executing AI code: {exec_err}")
                    with st.expander("Show AI Code"):
                        st.code(code)
                    return

                insight = local_vars.get('insight')
                fig = local_vars.get('fig')

                st.info(f"💡 {insight}")

                if fig:
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Chart rendering error: {e}")

                # 5. Download with Smart Summary
                # Fallback info since we removed chart_type/x_col/y_col JSON parsing
                chart_info = "Chart generated via code" if fig else "No chart"
                summary_text = generate_summary(df, insight, chart_info)

                # 5.B Build enhanced CSV
                report_df = pd.DataFrame({
                    'User Question': [user_question],
                    'AI Answer': [insight],
                    'Detailed Summary': [summary_text.strip()]
                })

                buf = io.StringIO()
                report_df.to_csv(buf, index=False)

                st.download_button(
                    label="📥 Download Analysis Report (CSV)",
                    data=buf.getvalue(),
                    file_name="analysis_report.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"An error occurred during AI processing: {e}")



if __name__ == "__main__":
    main()
