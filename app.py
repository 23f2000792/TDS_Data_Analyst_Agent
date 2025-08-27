import os
import networkx as nx
import re
import json
import base64
import tempfile
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import inspect
import subprocess
import logging
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware

import requests

# LangChain / LLM imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 150))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)


def parse_keys_and_types(raw_questions: str):
    """
    Parses the key/type section from the questions file.
    """
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    type_map_def = {
        "number": float, "string": str, "integer": int,
        "int": int, "float": float
    }
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    keys_list = [k for k, _ in matches]
    return keys_list, type_map

# -----------------------------
# Tools
# -----------------------------

@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    print(f"Scraping URL: {url}")
    try:
        from io import BytesIO, StringIO
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()

        df = None

        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                df = pd.json_normalize(resp.json())
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])
        elif "text/html" in ctype or re.search(r'/wiki/|\.org|\.com', url, re.IGNORECASE):
            html_content = resp.text
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    # Heuristic: find table with most relevant keywords or fallback to largest
                    best_table = None
                    max_score = -1
                    keywords = ['company', 'revenue', 'headquarters', 'users', 'market cap']
                    for table in tables:
                        score = sum(kw in str(col).lower() for col in table.columns for kw in keywords)
                        if score > max_score:
                            max_score = score
                            best_table = table
                    df = best_table if best_table is not None else max(tables, key=len)
            except ValueError:
                pass
            if df is None:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})
        else:
            df = pd.DataFrame({"text": [resp.text]})

        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()
        return {"status": "success", "data": df.to_dict(orient="records"), "columns": df.columns.tolist()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(output: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    """
    try:
        if not output: return {"error": "Empty LLM output"}
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        first, last = s.find("{"), s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Write and execute a temporary Python script in a sandboxed environment.
    This function now includes sklearn and seaborn in its preamble to prevent import errors.
    """
    preamble = [
        "import json, sys, gc, random",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns", # Added seaborn to the sandbox
        "import networkx as nx",
        "from io import BytesIO",
        "import base64",
        "from typing import Dict, Any, List",
        # Add common sklearn imports
        "from sklearn.linear_model import LinearRegression",
        "from sklearn.metrics import r2_score",
        "from sklearn.ensemble import IsolationForest",
        "from sklearn.model_selection import train_test_split",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")
    
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')")
        preamble.append("data = df.to_dict(orient='records')")
    else:
        preamble.append("data = globals().get('data', {})")

    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.clf(); plt.close('all'); gc.collect()
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        plt.clf(); plt.close('all'); gc.collect()
        buf.seek(0); b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    try:
        from PIL import Image
        buf.seek(0)
        im = Image.open(buf)
        for quality in [80, 60]:
            out_buf = BytesIO()
            im.save(out_buf, format='WEBP', quality=quality, method=6)
            out_buf.seek(0); ob = out_buf.getvalue()
            if len(ob) <= max_bytes:
                return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    plt.clf(); plt.close('all'); gc.collect()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''
    
    script_lines = []
    script_lines.extend(preamble)
    script_lines.append("\n# Injected tool function\n")
    script_lines.append(inspect.getsource(scrape_url_to_dataframe.func).replace("@tool", ""))
    script_lines.append(helper)
    script_lines.append("\nresults = {}\n")
    
    safe_code = code.replace("from functions import scrape_url_to_dataframe", "")
    
    script_lines.append("try:")
    script_lines.append("    " + safe_code.replace('\n', '\n    '))
    script_lines.append("finally:")
    script_lines.append("    print(json.dumps({'status':'success','result':results}, default=str), flush=True)")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp.write("\n".join(script_lines))
        tmp_path = tmp.name

    try:
        completed = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout, check=False
        )
        
        if completed.returncode != 0:
            error_message = (f"STDERR:\n{completed.stderr.strip()}\n\nSTDOUT:\n{completed.stdout.strip()}").strip()
            return {"status": "error", "message": error_message or "Subprocess failed with no output."}

        out = completed.stdout.strip()
        if not out:
             return {"status": "error", "message": "Execution produced no output.", "raw_stderr": completed.stderr.strip()}

        try:
            return json.loads(out)
        except Exception as e:
            return {"status": "error", "message": f"Could not parse JSON output: {str(e)}", "raw": out}
            
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except Exception:
            pass

# -----------------------------
# LLM agent setup
# -----------------------------
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

tools = [scrape_url_to_dataframe]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.
You will receive a task and optional data. You must:
1. Return only a valid JSON object with a single key: "code".
2. The value of "code" must be a Python script that populates a dictionary called `results`.
3. The `results` dictionary keys must EXACTLY match the snake_case keys from the task description.
4. Your Python code will run in a sandbox with:
   - pandas, numpy, matplotlib, seaborn, networkx, and scikit-learn available.
   - A helper function `plot_to_base64()` for generating base64-encoded images. DO NOT import or define it.
   - A helper function `scrape_url_to_dataframe(url)` for fetching data. DO NOT import or define it.
5. All numeric values in the final `results` dict must be actual numbers (int/float), not strings.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, max_iterations=5,
    early_stopping_method="generate",
    handle_parsing_errors=lambda e: f"Output parsing error: {e}"
)

# -----------------------------
# Runner
# -----------------------------
@app.post("/")
@app.post("/api/")
async def analyze_data(request: Request):
    try:
        form = await request.form()
        # More robustly get files from the form
        questions_file = form.get("questions_file")
        data_file = form.get("data_file")

        if not questions_file or not hasattr(questions_file, "filename"):
            raise HTTPException(400, "Missing questions file (must be a .txt file named 'questions_file').")

        raw_questions = (await questions_file.read()).decode("utf-8")
        keys_list, type_map = parse_keys_and_types(raw_questions)
        
        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        if data_file and hasattr(data_file, "filename") and data_file.filename:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()
            
            df = None
            if filename.endswith(".csv"): df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")): df = pd.read_excel(BytesIO(content))
            elif filename.endswith(".parquet"): df = pd.read_parquet(BytesIO(content))
            elif filename.endswith(".json"):
                try: df = pd.read_json(BytesIO(content))
                except ValueError: df = pd.DataFrame(json.loads(content.decode("utf-8")))
            else: raise HTTPException(400, f"Unsupported data file type: {filename}")

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name

            df_preview = (
                f"\n\nDataset Preview:\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First 5 rows:\n{df.head(5).to_markdown(index=False)}\n"
            )

        llm_rules = (
            "Rules:\n1) Use the provided pandas DataFrame called `df`.\n2) DO NOT call scrape_url_to_dataframe()."
            if dataset_uploaded else
            "Rules:\n1) You must call `scrape_url_to_dataframe(url)` to get data."
        )

        llm_input = f"{llm_rules}\nTask Description:\n{raw_questions}\n{df_preview}\nRespond with the JSON object only."

        response = agent_executor.invoke({"input": llm_input}, {"request_timeout": LLM_TIMEOUT_SECONDS})
        raw_out = response.get("output") or ""
        if not raw_out or "code" not in raw_out:
            raise HTTPException(500, detail=f"Agent returned no usable output. Last response: {raw_out}")

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            raise HTTPException(500, detail=f"Could not parse agent output: {parsed['error']}")
        if "code" not in parsed:
            raise HTTPException(500, detail=f"Invalid agent response: 'code' key missing. Response: {parsed}")
        
        code = parsed["code"]
        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        
        if exec_result.get("status") != "success":
            raise HTTPException(500, detail=f"Execution failed: {exec_result.get('message')}")

        results_dict = exec_result.get("result", {})
        
        # Safer post-processing and type casting
        final_result = {}
        for key in keys_list:
            if key in results_dict:
                val = results_dict[key]
                caster = type_map.get(key, str)
                # Only attempt to cast if the value is not a collection type
                if not isinstance(val, (dict, list, str)):
                    try:
                        final_result[key] = caster(val)
                    except (ValueError, TypeError):
                        final_result[key] = val # Keep original if casting fails
                else:
                    final_result[key] = val # Assign lists, dicts, and strings directly
            else:
                final_result[key] = None # Key was expected but not found in results

        return JSONResponse(content=final_result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))

# -----------------------------
# Boilerplate
# -----------------------------
_FAVICON_FALLBACK_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII=")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

@app.get("/api", include_in_schema=False)
async def health_check():
    return {"ok": True, "message": "Server is running. POST to this endpoint for analysis."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
