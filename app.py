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
        messages=[
            {"role": "system", "content": "You are a specialized Python Data Analyst. You ONLY output valid Python code. You MUST always store your textual answer in a variable named 'insight' and your Plotly chart in 'fig'."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

# ── main app ─────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="AI Data Analyst", layout="wide")
    st.title("🤖 AI Data Analyst Web App")

    # Initialize Session State for Chat History
    if "history" not in st.session_state:
        st.session_state.history = []

    # 3.1 Sidebar
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if st.sidebar.button("🗑️ Clear Analysis History"):
        st.session_state.history = []
        st.rerun()

    if uploaded_file is None:
        st.info("👈 Upload a CSV file from the sidebar to get started.")
        return

    try:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='utf-8', index_col=False)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='iso-8859-1', index_col=False)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return

    # ── Data Preview ──────────────────────────────────────────────────────────
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    col1, col2 = st.columns(2)
    col1.metric("Total Rows", len(df))
    col2.metric("Total Columns", len(df.columns))
    
    st.markdown("---")

    # ── AI Question Input ─────────────────────────────────────────────────────
    st.subheader("💬 Ask AI About Your Data")
    
    with st.form("query_form", clear_on_submit=True):
        user_question = st.text_input("Your question", placeholder="e.g. tell me which project number has sainath got")
        submitted = st.form_submit_button("Analyze")

    if submitted and user_question:
        if "GROQ_API_KEY" not in st.secrets:
            st.error("GROQ_API_KEY not found in st.secrets.")
            return

        with st.spinner("Thinking..."):
            # 4.1 Prompt Construction
            prompt = f"""
DataFrame `df` structure:
Columns: {list(df.columns)}
Dtypes: {df.dtypes.astype(str).to_dict()}
Sample: {df.head(2).to_dict()}

Question: {user_question}

TASK: Write Python code.
RULES:
1. Search for the requested entity using case-insensitive matching. Store in `evidence`.
2. CRITICAL: Store your text answer in `insight`. 
   - NEVER hardcode numbers in the string (e.g., DON'T write `insight = "Score is 50"`).
   - ALWAYS use f-strings to fetch values directly from the `evidence` row (e.g., `insight = f"Score is {{evidence['ColumnName'].iloc[0]}}"`).
3. Store your plotly figure in `fig` or set `fig = None`.

OUTPUT ONLY THE PYTHON CODE.
"""
            try:
                raw = ask_groq(prompt)
                
                # Robust Code Extraction
                code_match = re.search(r"```(?:python)?\n?(.*?)\n?```", raw, re.DOTALL)
                code = code_match.group(1).strip() if code_match else raw.strip()

                # Execute Code
                local_vars = {'df': df, 'px': px, 'pd': pd, 'insight': None, 'fig': None, 'evidence': None}
                
                try:
                    exec(code, globals(), local_vars)
                    insight = local_vars.get('insight')
                    fig = local_vars.get('fig')
                    evidence = local_vars.get('evidence')
                    
                    if not insight:
                        st.warning("AI forgot to set 'insight'.")
                    else:
                        # Save to history
                        fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn') if fig else None
                        st.session_state.history.insert(0, {
                            "question": user_question,
                            "insight": insight,
                            "fig": fig,
                            "fig_html": fig_html,
                            "evidence": evidence
                        })
                except Exception as exec_err:
                    st.error(f"Execution Error: {exec_err}")
                    with st.expander("View AI Generated Code"):
                        st.code(code)

            except Exception as e:
                st.error(f"AI Logic Error: {e}")

    # ── Main View (Latest Only) ──────────────────────────────────────────────
    if st.session_state.history:
        latest = st.session_state.history[0]
        st.markdown(f"### ❓ {latest['question']}")
        st.info(f"💡 {latest['insight']}")
        
        if latest['evidence'] is not None and not latest['evidence'].empty:
            num_rows = len(latest['evidence'])
            st.success(f"✅ Found {num_rows} matching record(s) in your data.")
            with st.expander("📊 View Data Evidence (Verification)"):
                st.dataframe(latest['evidence'], use_container_width=True)

        if latest['fig']:
            st.plotly_chart(latest['fig'], use_container_width=True, key="latest_chart")
        
        # Download Report
        html_report = f"<html><body style='font-family: Arial;'><h2>Analysis Report</h2><hr><h3>Question:</h3><p>{latest['question']}</p><h3>Analysis:</h3><p>{latest['insight']}</p>{f'<h3>Visualization:</h3>{latest['fig_html']}' if latest['fig_html'] else ''}</body></html>"
        st.download_button(label="📥 Download Latest Report", data=html_report, file_name="analysis.html", mime="text/html")

    # ── Sidebar History ──────────────────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📜 Previous Analyses")
        for idx, item in enumerate(st.session_state.history[1:]):
            with st.sidebar.expander(f"Q: {item['question'][:30]}..."):
                st.write(f"💡 {item['insight']}")
                if item['fig']:
                    st.write("📈 Chart included in full report.")
                
                # Small download for old ones
                old_report = f"<html><body style='font-family: Arial;'><h3>{item['question']}</h3><p>{item['insight']}</p></body></html>"
                st.download_button(label="📥 Download", data=old_report, file_name=f"old_analysis_{idx}.html", key=f"old_dl_{idx}")

if __name__ == "__main__":
    main()
