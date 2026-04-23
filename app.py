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
1. Calculate the correct answer using `df`. Then, format that answer into a conversational, plain-English sentence and store it as a string in a variable named `insight`. Do NOT just return a raw number. Format it beautifully (e.g. "The state with the highest number of cases is Kerala with 1,234,567 cases.")
2. If a chart makes sense, create a Plotly Express figure and store it in a variable named `fig`. If no chart makes sense, set `fig = None`.

IMPORTANT: 
- ALWAYS clean missing data (NaN/Nulls) using `.fillna(0)` or `.dropna()` before calculating sums, maxes, or grouping. Otherwise, your result will be 'NaN'!
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

                # 5. Download HTML Report
                html_content = f"""
                <html>
                <head><title>Data Analysis Report</title></head>
                <body style="font-family: Arial, sans-serif; padding: 20px; line-height: 1.6; color: #333;">
                    <h2 style="color: #2e6c80;">Data Analysis Report</h2>
                    <hr>
                    <h3 style="color: #444;">Question:</h3>
                    <p style="font-size: 16px; background-color: #f4f4f4; padding: 10px; border-left: 4px solid #ccc;">{user_question}</p>
                    
                    <h3 style="color: #444;">AI Analysis:</h3>
                    <p style="font-size: 16px;">{insight}</p>
                """

                if fig:
                    html_content += f"""
                    <h3 style="color: #444;">Visualization:</h3>
                    <div>{fig.to_html(full_html=False, include_plotlyjs='cdn')}</div>
                    """

                html_content += """
                </body>
                </html>
                """

                st.download_button(
                    label="📥 Download Analysis Report (HTML)",
                    data=html_content,
                    file_name="analysis_report.html",
                    mime="text/html"
                )

            except Exception as e:
                st.error(f"An error occurred during AI processing: {e}")



if __name__ == "__main__":
    main()
