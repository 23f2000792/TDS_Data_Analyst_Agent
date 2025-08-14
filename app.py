import os
import re
import json
import base64
import tempfile
import sys
import subprocess
import logging
from io import BytesIO, StringIO
from typing import Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

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

# -----------------------------
# Configuration & Setup
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Data Analyst Agent")

# Add CORS middleware to allow requests from any origin (convenient for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings from environment variables
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 180))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o") # Use gpt-4o as a default

if not OPENAI_API_KEY:
    logger.error("FATAL: OPENAI_API_KEY environment variable not set.")
    # You might want to exit here in a real application
    # sys.exit(1)

# -----------------------------
# LLM Agent Setup
# -----------------------------
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY
)

# System prompt instructing the agent on its role and output format
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request.
- One or more **questions** with expected answer types.
- An optional **dataset preview**.

You must:
1.  Follow the provided rules exactly.
2.  Return **only a valid JSON object** — no extra commentary, explanations, or markdown formatting.
3.  The JSON must contain:
    - "questions": A dictionary mapping the question keys to their python code implementation.
    - "code": A single string of Python code that populates a `results` dictionary. The keys of this dictionary must exactly match the keys provided in the questions file.
4.  Your Python code will run in a sandboxed environment with:
    - `pandas` (as `pd`), `numpy` (as `np`), `matplotlib.pyplot` (as `plt`).
    - The data available in a pandas DataFrame called `df`.
    - A helper function `plot_to_base64()` for generating base64-encoded images.
5.  When a question requires a plot, you **must** use `plot_to_base64()` to generate the image data.
6.  Ensure all variables are defined before use and the code can run without errors.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# -----------------------------
# Agent Tools
# -----------------------------
@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return its content as a pandas DataFrame.
    Supports HTML tables, CSV, Excel, Parquet, and JSON data sources.
    If an HTML page has no tables, it returns the raw text content.
    Returns a dictionary with status, data, and columns.
    """
    print(f"Scraping URL: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        df = None

        if "csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))
        elif "json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except (json.JSONDecodeError, TypeError):
                df = pd.DataFrame([{"text": resp.text}])
        elif "html" in ctype:
            try:
                tables = pd.read_html(StringIO(resp.text), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError: # No tables found
                pass
            if df is None:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})
        else: # Fallback for plain text or other types
            df = pd.DataFrame({"text": [resp.text]})

        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()
        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist()
        }
    except Exception as e:
        logger.error(f"Scrape tool failed for URL {url}: {e}")
        return {"status": "error", "message": str(e)}

# -----------------------------
# Agent Executor
# -----------------------------
tools = [scrape_url_to_dataframe]
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=4,
    early_stopping_method="generate",
    handle_parsing_errors=True
)

# -----------------------------
# Code Execution Utility
# -----------------------------
def write_and_run_temp_python(code: str, injected_pickle: str = None) -> Dict[str, Any]:
    """
    Executes Python code in a secure subprocess, with a preloaded DataFrame if provided.
    Returns the results dictionary or an error.
    """
    helper_code = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=90)
    buf.seek(0)
    img_bytes = buf.getvalue()

    if len(img_bytes) <= max_bytes:
        plt.clf() # Clear the plot figure
        return "data:image/png;base64," + base64.b64encode(img_bytes).decode('ascii')

    # If too large, try reducing quality with WEBP if PIL is available
    if PIL_AVAILABLE:
        try:
            im = Image.open(buf)
            for quality in [85, 70, 50]:
                out_buf = BytesIO()
                im.save(out_buf, format='WEBP', quality=quality)
                out_buf.seek(0)
                webp_bytes = out_buf.getvalue()
                if len(webp_bytes) <= max_bytes:
                    plt.clf()
                    return "data:image/webp;base64," + base64.b64encode(webp_bytes).decode('ascii')
        except Exception:
            pass # Fallback to downsized PNG

    # Last resort: downsized PNG
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=50)
    buf.seek(0)
    plt.clf()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('ascii')
'''
    preamble = [
        "import json, sys, base64",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")
        preamble.append("PIL_AVAILABLE = True")
    else:
        preamble.append("PIL_AVAILABLE = False")

    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')")
    else:
        preamble.append("df = pd.DataFrame()") # Ensure df exists

    script_lines = preamble + [helper_code, "\nresults = {}\n", code]
    script_lines.append("\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name
        tmp.write("\n".join(script_lines))

    try:
        completed = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=60
        )
        if completed.returncode != 0:
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}

        try:
            return json.loads(completed.stdout.strip())
        except json.JSONDecodeError as e:
            return {"status": "error", "message": f"Could not parse JSON output: {e}", "raw": completed.stdout}

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Code execution timed out after 60 seconds."}
    finally:
        os.unlink(tmp_path)
        if injected_pickle and os.path.exists(injected_pickle):
            os.unlink(injected_pickle)

# -----------------------------
# Main API Endpoint
# -----------------------------
@app.post("/api")
async def analyze_data(request: Request):
    try:
        form = await request.form()
        uploads = [value for _, value in form.multi_items() if isinstance(value, UploadFile)]

        if not uploads:
            raise HTTPException(400, "No files uploaded. A .txt questions file is required.")

        txt_files = [f for f in uploads if f.filename and f.filename.lower().endswith(".txt")]
        if len(txt_files) != 1:
            raise HTTPException(400, "Exactly one .txt questions file is required.")
        questions_file = txt_files[0]
        raw_questions = (await questions_file.read()).decode("utf-8")

        # --- Type Parsing from Questions File ---
        type_map = {}
        # Pattern: - `key`: type
        pattern = re.compile(r"^\s*-\s*`([^`]+)`\s*:\s*(.+)", re.MULTILINE)
        for match in pattern.finditer(raw_questions):
            key, type_hint = match.groups()
            norm_type = type_hint.strip().lower()
            if norm_type in ("number", "float", "int"):
                type_map[key.strip()] = "number"
            elif norm_type in ("string", "str"):
                type_map[key.strip()] = "string"
            elif "base64" in norm_type:
                type_map[key.strip()] = "base64"
            elif norm_type in ("bool", "boolean"):
                type_map[key.strip()] = "boolean"

        type_note = "\nNote: The `results` dictionary in your code must have these keys with corresponding value types:\n"
        type_note += "\n".join([f"- {k}: {t}" for k, t in type_map.items()]) if type_map else "(No explicit types detected)"

        # --- Data File Processing ---
        data_candidates = [f for f in uploads if f is not questions_file]
        pickle_path, df_preview, dataset_uploaded = None, "", False

        if data_candidates:
            data_file = data_candidates[0] # Use the first data file found
            filename = (data_file.filename or "").lower()
            content = await data_file.read()
            df = None
            try:
                if filename.endswith(".csv"):
                    df = pd.read_csv(BytesIO(content))
                elif filename.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(BytesIO(content))
                elif filename.endswith(".parquet"):
                    df = pd.read_parquet(BytesIO(content))
                elif filename.endswith(".json"):
                    df = pd.read_json(StringIO(content.decode("utf-8")))

                if df is not None:
                    dataset_uploaded = True
                    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                        pickle_path = temp_pkl.name
                        df.to_pickle(pickle_path)
                    
                    df_preview = (
                        f"\nAn uploaded dataset is available as `df`.\n"
                        f"Columns: {', '.join(df.columns.astype(str))}\n"
                        f"First 5 rows:\n{df.head(5).to_markdown(index=False)}\n"
                    )
            except Exception as e:
                logger.error(f"Failed to parse data file {filename}: {e}")
                # Don't block the request, just proceed without data
                df_preview = f"\nNote: A data file named '{filename}' was uploaded but could not be parsed. Error: {e}\n"

        # --- Build LLM Input ---
        llm_rules = (
            "Rules:\n"
            "1. You have access to a pandas DataFrame called `df`.\n"
            "2. DO NOT use the `scrape_url_to_dataframe` tool. Analyze the `df` directly.\n"
        ) if dataset_uploaded else (
            "Rules:\n"
            "1. No dataset was uploaded. To answer the questions, you MUST call the `scrape_url_to_dataframe(url)` tool with a relevant URL.\n"
            "2. The scraped data will become available as `df` in your final Python code.\n"
        )
        llm_input = f"{llm_rules}\nQuestions:\n{raw_questions}\n{df_preview}\n{type_note}\nRespond with the JSON object only."

        # --- Run Agent and Execute Code ---
        response = agent_executor.invoke({"input": llm_input})
        raw_out = response.get("output", "")
        if not raw_out:
            raise HTTPException(500, "Agent returned an empty response.")

        parsed_json = json.loads(raw_out)
        code = parsed_json.get("code")
        if not code:
            raise HTTPException(500, f"Agent did not return a 'code' block. Response: {parsed_json}")

        # If no data was uploaded, check if the LLM's code intends to scrape.
        # If so, we run the scrape first and provide the result to the execution environment.
        if not dataset_uploaded:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    raise HTTPException(500, f"Scrape tool failed: {tool_resp.get('message')}")
                
                df_scraped = pd.DataFrame(tool_resp["data"])
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                    pickle_path = temp_pkl.name
                    df_scraped.to_pickle(pickle_path)
                
                # We now have data, so remove the scrape call from the LLM's code
                # to prevent it from running again inside the sandbox.
                code = re.sub(r".*scrape_url_to_dataframe.*", "", code)


        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path)
        if exec_result.get("status") != "success":
            raise HTTPException(500, detail=f"Code execution failed: {exec_result.get('message')}")

        return JSONResponse(content={"results": exec_result.get("result", {})})

    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        logger.exception("An unexpected error occurred in /api endpoint")
        raise HTTPException(500, detail=f"An unexpected server error occurred: {str(e)}")

# -----------------------------
# Frontend and Static Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serves the main index.html file."""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)

# 1×1 transparent PNG fallback for favicon
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serves a favicon to prevent 404 errors in browser logs."""
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

# -----------------------------
# Application Runner
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
