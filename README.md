# AI Data Analyst 📊

An intelligent, professional CSV analysis workspace built with Streamlit and powered by AI. This tool allows users to upload datasets and receive instant business insights, realistic charts, and plain-English summaries.

## 🚀 Features

- **Automated Data Profiling**: Get immediate summaries of rows, columns, missing values, and data types.
- **Smart Data Cleaning**: Handles various encodings, delimiters, and automatically cleans numeric/text fields.
- **Natural Language Interaction**: Ask questions about your data (e.g., "What is the average sales in March?") and get direct answers calculated from the full dataset.
- **AI-Powered Insights**: (Optional) Use Gemini to generate high-level summaries and extra wording for answers.
- **Interactive Visualizations**: Dynamic charts including trends over time, category comparisons, and relationship scatter plots.
- **Privacy-First**: Core calculations are performed locally on your CSV data.

## 🛠️ Tech Stack

- **Frontend/App**: [Streamlit](https://streamlit.io/)
- **Data Analysis**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations**: [Plotly](https://plotly.com/python/)
- **AI Integration**: [Google Gemini API](https://ai.google.dev/) (Optional)

## 📋 Prerequisites

- Python 3.10 or higher
- A Google Gemini API Key (Optional, for AI summaries)

## 🔧 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sainathsk48/AI-data-analyst.git
   cd AI-data-analyst
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys (Optional)**:
   Create a `.streamlit/secrets.toml` file in the root directory:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```
   Alternatively, you can set it as an environment variable `GEMINI_API_KEY`.

## 🏃 How to Run

Launch the application using Streamlit:

```bash
streamlit run app.py
```

## 📸 Usage

1. **Upload**: Drop a CSV, TSV, or TXT file into the uploader.
2. **Review**: Check the **Summary** tab for an instant profile of your data.
3. **Visualize**: Go to the **Charts** tab to explore distributions and trends.
4. **Ask**: Use the **Ask** tab to query your dataset using natural language.

---
Developed for high-performance tabular data analysis.
