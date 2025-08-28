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
import subprocess
import logging
from io import BytesIO
from typing import Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import requests

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# LangChain / LLM imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# Add CORS middleware to allow cross-origin requests
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
    Returns:
        keys_list: list of keys in order
        type_map: dict key -> casting function
    """
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    type_map_def = {
        "number": float,
        "string": str,
        "integer": int,
        "int": int,
        "float": float
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
        from io import StringIO
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/138.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.google.com/",
        }

        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()

        df = None

        # --- CSV ---
        if "text/csv" in ctype or url.lower().endswith(".csv"):
            df = pd.read_csv(BytesIO(resp.content))

        # --- Excel ---
        elif any(url.lower().endswith(ext) for ext in (".xls", ".xlsx")) or "spreadsheetml" in ctype:
            df = pd.read_excel(BytesIO(resp.content))

        # --- Parquet ---
        elif url.lower().endswith(".parquet"):
            df = pd.read_parquet(BytesIO(resp.content))

        # --- JSON ---
        elif "application/json" in ctype or url.lower().endswith(".json"):
            try:
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])

        # --- HTML / Fallback ---
        elif "text/html" in ctype or re.search(r'/wiki/|\.org|\.com', url, re.IGNORECASE):
            html_content = resp.text
            # Try HTML tables first
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    # Combine all tables found on the page into one DataFrame
                    df = pd.concat(tables, ignore_index=True)
            except ValueError:
                pass

            # If no table found, fallback to plain text
            if df is None:
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})

        # --- Unknown type fallback ---
        else:
            df = pd.DataFrame({"text": [resp.text]})

        # --- Normalize columns ---
        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()
        df = df.dropna(axis=1, how='all') # Drop columns that are entirely empty

        return {
            "status": "success",
            "data": df.to_dict(orient="records"),
            "columns": df.columns.tolist()
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# -----------------------------
# Utilities for executing code safely
# -----------------------------
def clean_llm_output(output: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    Fixes common errors like invalid backslash escapes.
    Returns dict or {"error": "..."}
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}
        # remove triple-fence markers if present
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)

        # FIX: Find and fix invalid backslash escapes before parsing.
        # This looks for a backslash that is NOT followed by a valid JSON escape character.
        s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

        # find outermost JSON object by scanning for balanced braces
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            # fallback: try last balanced pair scanning backwards
            for i in range(last, first, -1):
                cand = s[first:i+1]
                try:
                    return json.loads(cand)
                except json.JSONDecodeError:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}


def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Write a temp python file which:
      - provides a safe environment (imports)
      - loads df/from pickle if provided into df and data variables
      - defines a robust plot_to_base64() helper that ensures < 100kB (attempts resizing/conversion)
      - executes the user code (which should populate `results` dict)
      - prints json.dumps({"status":"success","result":results})
    Returns dict with parsed JSON or error details.
    """
    preamble = [
        "import json, sys, gc, re",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "import networkx as nx",
        "from io import BytesIO",
        "import base64",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")

    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')")
        preamble.append("data = df.to_dict(orient='records')")
    else:
        preamble.append("data = globals().get('data', {})")
        preamble.append("df = pd.DataFrame(data)")


    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.clf()
    plt.close('all')
    gc.collect()
    buf.seek(0)
    img_bytes = buf.getvalue()
    if len(img_bytes) <= max_bytes:
        return base64.b64encode(img_bytes).decode('ascii')
    # try decreasing dpi
    for dpi in [80, 60, 50, 40, 30]:
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        plt.clf(); plt.close('all'); gc.collect()
        buf.seek(0)
        b = buf.getvalue()
        if len(b) <= max_bytes:
            return base64.b64encode(b).decode('ascii')
    # if Pillow available, try convert to WEBP
    try:
        from PIL import Image
        buf.seek(0)
        im = Image.open(buf)
        for quality in [80, 60]:
            out_buf = BytesIO()
            im.save(out_buf, format='WEBP', quality=quality)
            out_buf.seek(0)
            ob = out_buf.getvalue()
            if len(ob) <= max_bytes:
                return base64.b64encode(ob).decode('ascii')
    except Exception:
        pass
    # as last resort return smallest PNG even if > max_bytes
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=20)
    plt.clf(); plt.close('all'); gc.collect()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')
'''
    script_lines = [
        *preamble,
        helper,
        "\nresults = {}\n",
        code,
        "\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n"
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp.write("\n".join(script_lines))
        tmp_path = tmp.name

    try:
        completed = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout, check=False
        )
        if completed.returncode != 0:
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}
        
        out = completed.stdout.strip()
        try:
            return json.loads(out)
        except json.JSONDecodeError as e:
            return {"status": "error", "message": f"Could not parse JSON output: {str(e)}", "raw": out}
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out"}
    finally:
        try:
            os.unlink(tmp_path)
            if injected_pickle and os.path.exists(injected_pickle):
                os.unlink(injected_pickle)
        except OSError:
            pass

# -----------------------------
# LLM agent setup
# -----------------------------
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "questions": [ list of original question strings exactly as provided ]
   - "code": "..." (Python code that creates a dict called `results` with each question string as a key and its computed answer as the value)
4. Your Python code will run in a sandbox with pandas, numpy, matplotlib, networkx available.
5. A helper function `plot_to_base64()` is available for generating base64-encoded images under 100KB. Use it for all plots.
6. Make sure all variables are defined before use, and the code can run without any undefined references.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=[scrape_url_to_dataframe],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[scrape_url_to_dataframe],
    verbose=True,
    max_iterations=4,
    early_stopping_method="generate",
    handle_parsing_errors=True,
)

# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/")
@app.post("/api")
@app.post("/api/")
async def analyze_data(request: Request):
    try:
        form = await request.form()
        questions_file = form.get("questions_file")
        data_file = form.get("data_file")

        if not questions_file or not hasattr(questions_file, "read"):
            raise HTTPException(400, "Missing questions file (.txt)")

        raw_questions = (await questions_file.read()).decode("utf-8")
        
        pickle_path = None
        df_preview = ""
        
        if data_file and hasattr(data_file, "filename") and data_file.filename:
            filename = data_file.filename.lower()
            content = await data_file.read()
            
            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(BytesIO(content))
            elif filename.endswith(".json"):
                try:
                    df = pd.read_json(BytesIO(content))
                except ValueError:
                    df = pd.json_normalize(json.loads(content.decode("utf-8")))
            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_pkl:
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name

            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First 5 rows:\n{df.head(5).to_markdown(index=False)}\n"
            )
        
        base_rules = [
            "Your generated python code MUST create a dictionary called `results`.",
            "The keys in the `results` dictionary MUST be the exact question strings.",
            "For plots: use the pre-defined plot_to_base64() helper to return base64 image data.",
            "FIX: When converting columns to numeric types, first remove any non-numeric characters like commas, currency symbols (e.g., '$'), or text annotations."
        ]
        
        if pickle_path:
            llm_rules = "\n".join([
                "Rules:",
                "1) You have access to a pandas DataFrame called `df` and its dictionary form `data`.",
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.",
                "3) Use only the uploaded dataset for answering questions.",
                *base_rules
            ])
        else:
            llm_rules = "\n".join([
                "Rules:",
                "1) If you need web data, call the pre-defined function `scrape_url_to_dataframe(url)`. It is already available.",
                *base_rules
            ])

        llm_input = f"{llm_rules}\n\nQuestions:\n{raw_questions}\n{df_preview}\nRespond with the JSON object only."

        result = run_agent_safely_unified(llm_input, pickle_path)

        if "error" in result:
            raise HTTPException(500, detail=result)

        return JSONResponse(content=result)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))


def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    """
    Runs the LLM agent and executes code.
    - If pickle_path is provided, it's used directly.
    - If no pickle_path, the agent's code can use the scrape tool, which is handled during execution.
    """
    try:
        response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
        raw_out = response.get("output", "")

        if not raw_out:
            return {"error": f"Agent returned no output after multiple attempts. Full response: {response}"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response format: {parsed}"}

        code = parsed["code"]
        questions = parsed["questions"]

        # If a file was uploaded, we pass its pickle path.
        # If not, the generated code might contain a scrape call, which will be executed inside the temp script.
        # The `write_and_run_temp_python` function now has access to the scrape tool definition.
        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed", "details": exec_result.get('message'), "raw": exec_result.get("raw")}

        results_dict = exec_result.get("result", {})
        
        # Ensure all original questions are present in the final output
        final_output = {q: results_dict.get(q, "Answer not found in generated code") for q in questions}
        
        return final_output

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}


# -----------------------------
# Static Content and Info
# -----------------------------
_FAVICON_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve favicon.ico or a fallback."""
    if os.path.exists("favicon.ico"):
        return FileResponse("favicon.ico", media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    """Health/info endpoint."""
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST /api with 'questions_file' and optional 'data_file'.",
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
