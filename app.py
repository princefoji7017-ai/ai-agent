import io
import json
import sqlite3
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

try:
    from google import genai
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PWA / Mobile App Feel ---
st.markdown(
    """
    <meta name="theme-color" content="#0F172A">
    <style>
        .main-header {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #64748B;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">📊 AI Data Analyst Agent & Live Workspace</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload 10+ file formats • Edit live data on top • Compare original data below • Get instant profit/loss & root-cause insights</div>',
    unsafe_allow_html=True,
)


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    raw_bytes = uploaded_file.getvalue()
    bio = io.BytesIO(raw_bytes)

    if name.endswith(".csv"):
        return pd.read_csv(bio)
    elif name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(bio, sep=None, engine="python")
    elif name.endswith((".xlsx", ".xls", ".xlsm", ".ods")):
        xl = pd.ExcelFile(bio)
        sheet = st.selectbox(f"Select Sheet ({uploaded_file.name})", xl.sheet_names, key="sheet_sel")
        return pd.read_excel(io.BytesIO(raw_bytes), sheet_name=sheet)
    elif name.endswith(".json"):
        data = json.loads(raw_bytes.decode("utf-8"))
        if isinstance(data, list):
            return pd.json_normalize(data)
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    return pd.json_normalize(v)
            return pd.json_normalize([data])
        return pd.DataFrame(data)
    elif name.endswith(".jsonl") or name.endswith(".ndjson"):
        return pd.read_json(bio, lines=True)
    elif name.endswith(".parquet"):
        return pd.read_parquet(bio)
    elif name.endswith(".feather"):
        return pd.read_feather(bio)
    elif name.endswith(".xml"):
        return pd.read_xml(bio)
    elif name.endswith((".db", ".sqlite", ".sqlite3")):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        conn = sqlite3.connect(tmp_path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        table_list = tables["name"].tolist()
        if not table_list:
            conn.close()
            raise ValueError("No tables found in SQLite database.")
        selected_table = st.selectbox("Select SQLite Table", table_list, key="tbl_sel")
        df = pd.read_sql_query(f"SELECT * FROM [{selected_table}]", conn)
        conn.close()
        return df
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def generate_sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "Region": ["North", "North", "West", "South", "North", "West"],
            "Category": ["Electronics", "Clothing", "Electronics", "Grocery", "Clothing", "Electronics"],
            "Revenue": [120000, 95000, 140000, 80000, 110000, 98000],
            "Cost": [82000, 74000, 118000, 62000, 96000, 89000],
            "Marketing_Spend": [12000, 10000, 25000, 6000, 18000, 22000],
            "Returns": [4000, 6500, 15000, 2000, 9000, 12000],
        }
    )


def compute_business_summary(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "numeric_summary": {},
        "profit_loss_analysis": {},
        "missing_values": df.isnull().sum().to_dict(),
    }

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        s = df[col].dropna()
        if len(s) > 0:
            summary["numeric_summary"][col] = {
                "sum": float(s.sum()),
                "mean": round(float(s.mean()), 2),
                "min": float(s.min()),
                "max": float(s.max()),
            }

    # Try automatic Profit / Revenue / Cost detection
    cols_lower = {c.lower(): c for c in df.columns}
    profit_col = next((cols_lower[k] for k in cols_lower if "profit" in k or "net" in k), None)
    rev_col = next((cols_lower[k] for k in cols_lower if "revenue" in k or "sales" in k or "amount" in k), None)
    cost_col = next((cols_lower[k] for k in cols_lower if "cost" in k or "expense" in k or "spend" in k), None)

    working_df = df.copy()
    if profit_col is None and rev_col and cost_col:
        working_df["Calculated_Profit"] = pd.to_numeric(working_df[rev_col], errors="coerce") - pd.to_numeric(
            working_df[cost_col], errors="coerce"
        )
        profit_col = "Calculated_Profit"

    if profit_col and profit_col in working_df.columns:
        p_series = pd.to_numeric(working_df[profit_col], errors="coerce")
        summary["profit_loss_analysis"]["profit_column"] = profit_col
        summary["profit_loss_analysis"]["total_profit"] = float(p_series.sum(skipna=True))
        summary["profit_loss_analysis"]["loss_rows_count"] = int((p_series < 0).sum())
        summary["profit_loss_analysis"]["profit_rows_count"] = int((p_series > 0).sum())

        cat_cols = working_df.select_dtypes(exclude=[np.number]).columns.tolist()
        for ccol in cat_cols[:2]:
            grp = working_df.groupby(ccol, dropna=False)[profit_col].sum().sort_values()
            summary["profit_loss_analysis"][f"lowest_by_{ccol}"] = grp.head(3).to_dict()
            summary["profit_loss_analysis"][f"highest_by_{ccol}"] = grp.tail(3).to_dict()

    return summary


def run_ai_analyst(summary_data: dict, language: str, api_key: str) -> str:
    if not api_key:
        # Offline rule-based analyst fallback when no API key is entered
        p_info = summary_data.get("profit_loss_analysis", {})
        lines = ["### ⚡ Quick Rule-Based Diagnostic (Add Gemini API Key in Sidebar for Full AI Reasoning)"]
        lines.append(f"- **Dataset Size:** {summary_data['rows']} rows × {summary_data['columns']} columns.")
        if "total_profit" in p_info:
            lines.append(f"- **Total Profit / Net ({p_info['profit_column']}):** `{p_info['total_profit']:,.2f}`")
            lines.append(f"- **Profitable Records:** `{p_info['profit_rows_count']}` | **Loss Records:** `{p_info['loss_rows_count']}`")
            for k, v in p_info.items():
                if k.startswith("lowest_by_"):
                    dim = k.replace("lowest_by_", "")
                    lines.append(f"- **Lowest Profit / Highest Loss ({dim}):** `{v}`")
                elif k.startswith("highest_by_"):
                    dim = k.replace("highest_by_", "")
                    lines.append(f"- **Top Profit Contributors ({dim}):** `{v}`")
        lines.append("- **Next Action:** High cost, high discounts/returns, or low-margin segments usually pull profit down—check the lowest segments above.")
        return "\n".join(lines)

    if not HAS_GENAI:
        return "⚠️ `google-genai` package is not installed. Please check `requirements.txt`."

    client = genai.Client(api_key=api_key)
    prompt = f"""
You are a Senior Business & Data Analyst AI Agent.
Analyze the following dataset summary and explain clearly in {language}:
1. **Profit / Loss & Performance Overview:** Where did profit increase or decrease?
2. **Root Causes (Kyon aur Kis Karan Se Ghata/Badha):** Which categories, segments, costs, or patterns caused drops or gains?
3. **Actionable Fixes & Growth Plan (Kaise Theek Karein aur Kaise Badhayein):** Practical, specific steps to reduce losses and increase profit.

Dataset Summary JSON:
{json.dumps(summary_data, indent=2, default=str)}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text or "No response generated."


# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Agent Controls")
    api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Get a free API key from Google AI Studio for deep AI root-cause analysis.")
    language = st.selectbox("Analysis Language", ["Hinglish (Hindi + English)", "Hindi", "English"])
    auto_analyze = st.toggle("Auto-analyze on load / edit", value=True)
    st.caption("Supported: CSV, TSV, XLSX, XLS, XLSM, ODS, JSON, JSONL, Parquet, Feather, XML, SQLite")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "📁 Upload Your Data File (Excel, CSV, JSON, XML, Parquet, SQLite, etc.)",
    type=["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "ods", "json", "jsonl", "ndjson", "parquet", "feather", "xml", "db", "sqlite", "sqlite3"],
)

if uploaded_file is not None:
    try:
        df_loaded = load_uploaded_file(uploaded_file)
        file_sig = f"{uploaded_file.name}_{uploaded_file.size}_{len(df_loaded)}"
        if st.session_state.get("file_sig") != file_sig:
            st.session_state["file_sig"] = file_sig
            st.session_state["original_df"] = df_loaded.copy()
            st.session_state["working_df"] = df_loaded.copy()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    if "original_df" not in st.session_state:
        sample = generate_sample_data()
        st.session_state["original_df"] = sample.copy()
        st.session_state["working_df"] = sample.copy()
    st.info("Showing built-in sample dataset. Upload any file above to analyze your own data.")

original_df = st.session_state["original_df"]
working_df = st.session_state["working_df"]

# --- TOP SECTION: Live Editable Data ---
st.subheader("✏️ Top Section: Live Working Data (Edit values here)")
col_btn1, col_btn2, _ = st.columns([1.2, 1.5, 5])
with col_btn1:
    if st.button("🔄 Reset to Original"):
        st.session_state["working_df"] = original_df.copy()
        st.rerun()

edited_df = st.data_editor(
    working_df,
    num_rows="dynamic",
    use_container_width=True,
    key="live_editor",
)
st.session_state["working_df"] = edited_df

# --- AI ANALYST SECTION ---
st.markdown("---")
st.subheader("🤖 AI Data Analyst Insights (Profit/Loss Causes & Fixes)")

run_clicked = st.button("🚀 Run Full AI Analysis Now", type="primary")
if auto_analyze or run_clicked:
    summary = compute_business_summary(edited_df)
    with st.spinner("Analyzing data patterns, profit/loss drivers, and solutions..."):
        report = run_ai_analyst(summary, language=language, api_key=api_key)
    st.markdown(report)

    # Quick Visual Chart
    num_cols = edited_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = edited_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if num_cols and cat_cols:
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.bar(edited_df, x=cat_cols[0], y=num_cols[0], title=f"{num_cols[0]} by {cat_cols[0]}")
            st.plotly_chart(fig1, use_container_width=True)
        if len(num_cols) > 1:
            with c2:
                fig2 = px.bar(edited_df, x=cat_cols[0], y=num_cols[1], title=f"{num_cols[1]} by {cat_cols[0]}")
                st.plotly_chart(fig2, use_container_width=True)

# --- BOTTOM SECTION: Original Uploaded Data ---
st.markdown("---")
st.subheader("🔒 Bottom Section: Original Uploaded Data (Reference / Unchanged)")
st.dataframe(original_df, use_container_width=True)
