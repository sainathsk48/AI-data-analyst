# 🤖 AI Data Analyst Web App

A powerful, intuitive web application built with **Streamlit** and **Groq** that turns you into a data science expert instantly. Upload any CSV file, ask questions in plain English, and let the AI instantly analyze your data, generate insights, build dynamic charts, and give you exportable reports.

## ✨ Features

- **📂 Universal CSV Support**: Upload any CSV dataset (large or small) and instantly view data previews, row/column counts, and structural summaries.
- **💬 Natural Language Querying**: Just ask questions like *"What is the distribution of ages?"* or *"Show me sales trends over time"* and get direct answers.
- **📊 Dynamic Auto-Charting**: The AI automatically selects and renders the perfect Plotly graph (Bar, Line, or Pie charts) to visually represent your answer.
- **⚡ Smart Context Sampling**: Designed to bypass strict AI rate limits while maintaining high accuracy by intelligently sampling representative rows and deep statistical summaries.
- **📥 One-Click Export**: Download an "Analyzed CSV" that includes both your original data and a plain-English AI summary attached to the top.

## 🛠️ Tech Stack

- **Frontend/Backend**: [Streamlit](https://streamlit.io/)
- **Data Processing**: Pandas, NumPy
- **Data Visualization**: Plotly Express
- **AI / LLM**: [Groq API](https://groq.com/) utilizing `llama-3.1-8b-instant` for lightning-fast inference.

## 🚀 How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/sainathsk48/AI-data-analyst.git
cd AI-data-analyst
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Add your Groq API Key:**
Create a `.streamlit` folder in the project directory, and inside it create a `secrets.toml` file. Add your key like this:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
```

**4. Start the application:**
```bash
streamlit run app.py
```

## ⚠️ Notes on Rate Limits
This Web app is configured to use Groq's fast `llama-3.1-8b` model to help bypass the heavy daily token limits of larger models. It intelligently samples data to ensure you don't hit the *Tokens Per Minute (TPM)* limit on the free tier!
```


https://ai-data-analyst-x5pxygnk8xk8gxwptpcygg.streamlit.app/
