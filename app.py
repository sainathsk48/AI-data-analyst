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
            # Groq free tier limit is 12,000 tokens/minute. 150 rows is a safe balance.
            sample_size = min(150, len(df))
            csv_sample = df.sample(sample_size, random_state=42).to_csv(index=False)
            
            prompt = f"""
You are a data analyst. Answer the user's question about the dataset below.

Column names: {list(df.columns)}

Descriptive statistics:
{df.describe().to_string()}

Random Representative Sample ({sample_size} rows):
{csv_sample}

User question: {user_question}

IMPORTANT: Return ONLY valid JSON — no markdown, no backticks, no explanation. Exactly this structure:
{{
  "insight": "Plain English answer. Direct and simple. No technical language.",
  "chart_type": "bar or line or pie or none",
  "x_column": "exact column name or null",
  "y_column": "exact column name or null"
}}
"""
            try:
                raw = ask_groq(prompt)

                # 4.3 JSON Parsing
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not match:
                    st.error("Could not parse AI response as JSON.")
                    st.text("Raw response:")
                    st.text(raw)
                    return

                parsed = json.loads(match.group(0))
                insight    = parsed.get("insight", "No insight provided.")
                chart_type = parsed.get("chart_type", "none")
                x_col      = parsed.get("x_column")
                y_col      = parsed.get("y_column")

                st.info(f"💡 {insight}")

                # 4.4 Column Validation + Chart Rendering
                chart_valid = True
                if chart_type in ("bar", "line", "pie"):
                    for col_val, label in [(x_col, "x_column"), (y_col, "y_column")]:
                        if col_val and col_val not in df.columns:
                            st.warning(f"AI suggested an invalid column '{col_val}'. Please rephrase your question.")
                            chart_valid = False

                    if chart_valid:
                        try:
                            if chart_type == "bar":
                                fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                            elif chart_type == "line":
                                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
                            elif chart_type == "pie":
                                fig = px.pie(df, names=x_col, values=y_col, title=f"{y_col} distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Chart rendering error: {e}")

                # 5. Download with Smart Summary
                chart_info = f"Chart type: {chart_type}, X: {x_col}, Y: {y_col}"
                summary_text = generate_summary(df, insight, chart_info)

                # 5.B Build enhanced CSV
                summary_lines = [l for l in summary_text.strip().split('\n') if l.strip()]
                summary_df = pd.DataFrame({'Summary': summary_lines})

                buf = io.StringIO()
                summary_df.to_csv(buf, index=False)   # Part 1: summary
                buf.write("\n")                         # Part 2: separator
                df.to_csv(buf, index=False)             # Part 3: original data

                st.download_button(
                    label="📥 Download Analyzed Data (CSV)",
                    data=buf.getvalue(),
                    file_name="analyzed_data.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"An error occurred during AI processing: {e}")



if __name__ == "__main__":
    main()
