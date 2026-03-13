import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import openai  # or anthropic — swap as needed
import json

# ─────────────────────────────────────────
# STEP 1 — DATA INGESTION: Read & profile
# ─────────────────────────────────────────
df = pd.read_csv("sales_data.csv")

schema_profile = {
    "columns": list(df.columns),
    "dtypes": df.dtypes.astype(str).to_dict(),
    "nulls": df.isnull().sum().to_dict(),
    "sample": df.head(3).to_dict()
}
print("✅ Step 1 Done — Schema profiled")


# ─────────────────────────────────────────
# STEP 2 — LLM PLANNING: Get analysis plan
# ─────────────────────────────────────────
client = openai.OpenAI(api_key="YOUR_API_KEY")

prompt = f"""
You are a senior data analyst. Given this dataset profile:
{json.dumps(schema_profile, indent=2)}

Return a JSON plan with keys:
- eda_checks: list of pandas operations (as strings)
- sql_queries: list of SQL query strings
- charts: list of chart types to generate
Only return JSON. No explanation.
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

plan = json.loads(response.choices[0].message.content)
print("✅ Step 2 Done — Plan created:", list(plan.keys()))


# ─────────────────────────────────────────
# STEP 3 — EDA EXECUTION: Run the checks
# ─────────────────────────────────────────
eda_results = {}

for check in plan["eda_checks"]:
    try:
        result = eval(check)        # e.g. "df['revenue'].describe()"
        eda_results[check] = str(result)
    except Exception as e:
        eda_results[check] = f"Error: {e}"  # skip & move on

print("✅ Step 3 Done — EDA results captured")


# ─────────────────────────────────────────
# STEP 4 — SQL QUERIES: Business questions
# ─────────────────────────────────────────
conn = sqlite3.connect(":memory:")
df.to_sql("data", conn, index=False)        # load df into in-memory DB

sql_results = {}
for query in plan["sql_queries"]:
    try:
        result = pd.read_sql(query, conn)
        sql_results[query] = result.to_string()
    except Exception as e:
        sql_results[query] = f"Error: {e}"

print("✅ Step 4 Done — SQL queries executed")


# ─────────────────────────────────────────
# STEP 5 — VISUALIZATION & REPORT
# ─────────────────────────────────────────

# --- Charts ---
for chart in plan["charts"]:
    if chart == "bar":
        df.groupby(df.columns[0])[df.columns[-1]].sum().plot(kind="bar")
    elif chart == "hist":
        df[df.columns[-1]].plot(kind="hist")
    plt.title(chart)
    plt.savefig(f"{chart}.png")
    plt.clf()

# --- Final Report via LLM ---
report_prompt = f"""
Based on this analysis, write a short insight report.

EDA Results: {json.dumps(eda_results, indent=2)}
SQL Results: {json.dumps(sql_results, indent=2)}

Include: summary, key findings, 1-2 recommendations.
"""

report_response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": report_prompt}]
)

report = report_response.choices[0].message.content
with open("report.md", "w") as f:
    f.write(report)

print("✅ Step 5 Done — Charts saved, report written to report.md")