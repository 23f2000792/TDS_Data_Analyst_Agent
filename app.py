import os
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import io
import subprocess
import logging
from io import BytesIO, StringIO
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import requests

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Analyst Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 170))

# -----------------------------
# Tool and Utility Functions
# -----------------------------
@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame.
    Supports HTML tables, CSV, Excel, Parquet, and JSON.
    """
    print(f"Scraping URL: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        df = None

        if "csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))
        elif "json" in ctype or url.lower().endswith(".json"):
            df = pd.json_normalize(resp.json())
        elif "html" in ctype:
            from bs4 import BeautifulSoup
            try:
                tables = pd.read_html(StringIO(resp.text), flavor="bs4")
                if tables: df = tables[0]
            except ValueError: pass
            if df is None:
                soup = BeautifulSoup(resp.text, "html.parser")
                df = pd.DataFrame({"text": [soup.get_text(separator="\n", strip=True)]})
        else:
            df = pd.DataFrame({"text": [resp.text]})
        
        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()
        return {"status": "success", "data": df.to_dict(orient="records"), "columns": df.columns.tolist()}
    except Exception as e:
        logger.error(f"Scrape tool failed for URL {url}: {e}")
        return {"status": "error", "message": str(e)}

def clean_llm_output(output: str) -> Dict:
    """Extracts a JSON object from a string, handling markdown fences."""
    try:
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        return json.loads(s[s.find("{"):s.rfind("}")+1])
    except Exception as e:
        logger.error(f"Failed to parse LLM JSON output: {e}\nRaw output:\n{output}")
        return {"error": "Failed to parse JSON from LLM output", "raw": output}

def write_and_run_temp_python(code: str, injected_pickle: str = None) -> Dict[str, Any]:
    """Executes Python code in a sandboxed environment with all required libraries."""
    preamble = [
        "import json, sys, base64, pandas as pd, numpy as np, matplotlib, seaborn as sns, networkx as nx",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO"
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")
        
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')")
    else:
        preamble.append("df = pd.DataFrame()")

    helper = r'''
def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=90)
    plt.clf() # Clear the current figure to avoid plots overlapping
    buf.seek(0)
    img_bytes = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(img_bytes).decode('ascii')
'''
    script_lines = preamble + [helper, "\nresults = {}\n", code, "\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n"]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name
        tmp.write("\n".join(script_lines))

    try:
        proc = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=LLM_TIMEOUT_SECONDS - 10)
        if proc.returncode != 0:
            return {"status": "error", "message": proc.stderr or proc.stdout}
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Code execution timed out"}
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Failed to decode JSON from script: {e}"}
    finally:
        os.unlink(tmp_path)
        if injected_pickle and os.path.exists(injected_pickle):
            os.unlink(injected_pickle)

# -----------------------------
# LLM Agent and Prompt
# -----------------------------
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
tools = [scrape_url_to_dataframe]

# This hardened prompt is the most critical part of the solution.
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data analyst agent. Your sole purpose is to write a Python script to answer questions.

You will be given a request that explicitly lists the required keys for a final JSON object.

YOU MUST OBEY THESE RULES:
1.  Your output MUST BE a single JSON object with ONE key: "code". The value must be a string containing a valid Python script.
2.  The Python script you write MUST populate a dictionary named `results`.
3.  The keys in the `results` dictionary MUST EXACTLY MATCH the keys specified in the user's request (e.g., `edge_count`, `total_sales`).
4.  Do NOT use the full question text as a dictionary key. Only use the simple, specified keys.
5.  The user's data, if provided, is in a pandas DataFrame called `df`.
6.  For any chart or graph, you MUST use the provided `plot_to_base64()` helper function.
7.  Ensure all code is self-contained and all variables are defined before use.

Example of your required output format:
{
  "code": "import pandas as pd\\n\\nresults['total_sales'] = df['Sales'].sum()\\nresults['top_region'] = df.groupby('Region')['Sales'].sum().idxmax()\\n# ... more python code"
}
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5, handle_parsing_errors=True)

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

@app.post("/api")
async def analyze_data(questions_file: UploadFile = File(...), data_file: Optional[UploadFile] = File(None)):
    try:
        raw_questions = (await questions_file.read()).decode("utf-8")
        
        pickle_path, df_preview = None, ""
        if data_file:
            filename = data_file.filename.lower()
            content = await data_file.read()
            df = None
            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            if df is not None:
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                    pickle_path = temp_pkl.name
                    df.to_pickle(pickle_path)
                
                df_preview = (f"\n--- Dataset Preview (available as `df` in your script) ---\n{df.head(3).to_markdown()}\n")
        
        llm_input = f"User Request:\n{raw_questions}\n{df_preview}"

        response = agent_executor.invoke({"input": llm_input})
        raw_out = response.get("output", "")
        if not raw_out:
            raise HTTPException(500, "Agent returned an empty response.")
        
        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            raise HTTPException(500, f"Agent response parsing error: {parsed.get('raw', 'No raw output')}")
        
        code = parsed.get("code")
        if not code:
            raise HTTPException(500, f"Agent did not return a 'code' block. Response: {parsed}")

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path)
        if exec_result.get("status") != "success":
            raise HTTPException(500, f"Code execution failed: {exec_result.get('message')}")

        final_result = exec_result.get("result", {})
        return JSONResponse(content={"results": final_result})

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=f"An unexpected server error occurred: {str(e)}")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII=")
    return Response(content=favicon_bytes, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
