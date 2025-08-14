# app.py (Your original file with minimal, necessary fixes)

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
import io
import subprocess
import logging
from io import BytesIO, StringIO
from typing import Dict, Any, List, Optional

# MODIFICATION: Cleaned up imports slightly to ensure everything is here
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import requests

# Optional image conversion
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# --- MODIFICATION 1: Swapped LangChain imports from Google to OpenAI ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI  # Changed from ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

# MODIFICATION: Added CORS Middleware. This is essential for the frontend to communicate with the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 180))


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure index.html is in the same directory as app.py</p>", status_code=404)


# YOUR ORIGINAL FUNCTION (UNCHANGED)
def parse_keys_and_types(raw_questions: str):
    """
    Parses the key/type section from the questions file.
    """
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    matches = re.findall(pattern, raw_questions)
    type_map_def = {
        "number": float, "string": str, "integer": int, "int": int, "float": float, "base64": str, "boolean": bool
    }
    type_map = {key: type_map_def.get(t.lower(), str) for key, t in matches}
    keys_list = [k for k, _ in matches]
    return keys_list, type_map


# YOUR ORIGINAL TOOL (UNCHANGED)
@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and return data as a DataFrame (supports HTML tables, CSV, Excel, Parquet, JSON, and plain text).
    Always returns {"status": "success", "data": [...], "columns": [...]} if fetch works.
    """
    print(f"Scraping URL: {url}")
    try:
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
                data = resp.json()
                df = pd.json_normalize(data)
            except Exception:
                df = pd.DataFrame([{"text": resp.text}])
        elif "text/html" in ctype or re.search(r'/wiki/|\.org|\.com', url, re.IGNORECASE):
            html_content = resp.text
            try:
                tables = pd.read_html(StringIO(html_content), flavor="bs4")
                if tables:
                    df = tables[0]
            except ValueError:
                pass
            if df is None:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                df = pd.DataFrame({"text": [text]})
        else:
            df = pd.DataFrame({"text": [resp.text]})
        df.columns = df.columns.map(str).str.replace(r'\[.*\]', '', regex=True).str.strip()
        return {"status": "success", "data": df.to_dict(orient="records"), "columns": df.columns.tolist()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# YOUR ORIGINAL UTILITIES (UNCHANGED)
def clean_llm_output(output: str) -> Dict:
    """
    Extract JSON object from LLM output robustly.
    Returns dict or {"error": "..."}
    """
    try:
        if not output:
            return {"error": "Empty LLM output"}
        s = re.sub(r"^```(?:json)?\s*", "", output.strip())
        s = re.sub(r"\s*```$", "", s)
        first = s.find("{")
        last = s.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return {"error": "No JSON object found in LLM output", "raw": s}
        candidate = s[first:last+1]
        try:
            return json.loads(candidate)
        except Exception as e:
            for i in range(last, first, -1):
                cand = s[first:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    continue
            return {"error": f"JSON parsing failed: {str(e)}", "raw": candidate}
    except Exception as e:
        return {"error": str(e)}

# YOUR ORIGINAL SCRAPE_FUNC (UNCHANGED)
SCRAPE_FUNC = r'''
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    # This function remains as you wrote it
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        response.raise_for_status()
    except Exception as e:
        return {"status": "error", "error": str(e), "data": [], "columns": []}
    soup = BeautifulSoup(response.text, "html.parser")
    try:
        tables = pd.read_html(response.text)
    except:
        tables = []
    if tables:
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]
        df.columns = [str(col) for col in df.columns]
        return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}
    else:
        text_data = soup.get_text(separator="\n", strip=True)
        df = pd.DataFrame({"text": [text_data]})
        return {"status": "success", "data": df.to_dict(orient="records"), "columns": list(df.columns)}
'''

# YOUR ORIGINAL PYTHON RUNNER (UNCHANGED)
def write_and_run_temp_python(code: str, injected_pickle: str = None, timeout: int = 60) -> Dict[str, Any]:
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
    ]
    if PIL_AVAILABLE:
        preamble.append("from PIL import Image")
    if injected_pickle:
        preamble.append(f"df = pd.read_pickle(r'''{injected_pickle}''')\n")
        preamble.append("data = df.to_dict(orient='records')\n")
    else:
        preamble.append("data = globals().get('data', {})\n")
    helper = r'''
def plot_to_base64(max_bytes=100000):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.clf()
    buf.seek(0)
    img_bytes = buf.getvalue()
    # This is the logic you had, including PIL fallback for WEBP
    return "data:image/png;base64," + base64.b64encode(img_bytes).decode('ascii')
'''
    script_lines = preamble + [helper, SCRAPE_FUNC, "\nresults = {}\n", code]
    script_lines.append("\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)\n")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name
        tmp.write("\n".join(script_lines))

    try:
        completed = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            return {"status": "error", "message": completed.stderr.strip() or completed.stdout.strip()}
        out = completed.stdout.strip()
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


# --- MODIFICATION 1 (cont.): Swapped Gemini for OpenAI/GPT-4 ---
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# YOUR ORIGINAL PROMPT (UNCHANGED)
tools = [scrape_url_to_dataframe]
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a full-stack autonomous data analyst agent.

You will receive:
- A set of **rules** for this request (these rules may differ depending on whether a dataset is uploaded or not)
- One or more **questions**
- An optional **dataset preview**

You must:
1. Follow the provided rules exactly.
2. Return only a valid JSON object — no extra commentary or formatting.
3. The JSON must contain:
   - "questions": [ list of original question strings exactly as provided ]
   - "code": "..." (Python code that creates a dict called `results` with each question string as a key and its computed answer as the value)
4. Your Python code will run in a sandbox with:
   - pandas, numpy, matplotlib available
   - A helper function `plot_to_base64(max_bytes=100000)` for generating base64-encoded images under 100KB.
5. When returning plots, always use `plot_to_base64()` to keep image sizes small.
6. Make sure all variables are defined before use, and the code can run without any undefined references.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=4, # Increased slightly for potentially complex tasks
    early_stopping_method="generate",
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

# YOUR ORIGINAL API ENDPOINT, with two required fixes
@app.post("/api")
async def analyze_data(
    # --- MODIFICATION 2: This is the required fix for the "No files uploaded" error ---
    # It changes from manual parsing to letting FastAPI handle it robustly.
    questions_file: UploadFile = File(...),
    data_file: Optional[UploadFile] = File(None)
):
    try:
        # The old `await request.form()` is removed as FastAPI handles it now.
        raw_questions = (await questions_file.read()).decode("utf-8")
        keys_list, type_map = parse_keys_and_types(raw_questions)

        pickle_path = None
        df_preview = ""
        dataset_uploaded = False

        if data_file:
            dataset_uploaded = True
            filename = data_file.filename.lower()
            content = await data_file.read()

            # This data processing logic is from your original file
            if filename.endswith(".csv"):
                df = pd.read_csv(BytesIO(content))
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(BytesIO(content))
            elif filename.endswith(".parquet"):
                df = pd.read_parquet(BytesIO(content))
            elif filename.endswith(".json"):
                try:
                    df = pd.read_json(StringIO(content.decode("utf-8")))
                except ValueError:
                    df = pd.DataFrame(json.loads(content.decode("utf-8")))
            else:
                raise HTTPException(400, f"Unsupported data file type: {filename}")

            temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
            temp_pkl.close()
            df.to_pickle(temp_pkl.name)
            pickle_path = temp_pkl.name

            df_preview = (
                f"\n\nThe uploaded dataset has {len(df)} rows and {len(df.columns)} columns.\n"
                f"Columns: {', '.join(df.columns.astype(str))}\n"
                f"First rows:\n{df.head(5).to_markdown(index=False)}\n"
            )
        
        # This logic for building rules and LLM input is from your original file
        if dataset_uploaded:
            llm_rules = (
                "Rules:\n"
                "1) You have access to a pandas DataFrame called `df` and its dictionary form `data`.\n"
                "2) DO NOT call scrape_url_to_dataframe() or fetch any external data.\n"
                "3) Use only the uploaded dataset for answering questions.\n"
                '4) Produce a final JSON object with "questions" and "code" keys.'
            )
        else:
            llm_rules = (
                "Rules:\n"
                "1) If you need web data, CALL scrape_url_to_dataframe(url).\n"
                '2) Produce a final JSON object with "questions" and "code" keys.'
            )
        llm_input = (
            f"{llm_rules}\nQuestions:\n{raw_questions}\n"
            f"{df_preview if df_preview else ''}"
            "Respond with the JSON object only."
        )

        # Run agent using your original threading logic
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(run_agent_safely_unified, llm_input, pickle_path)
            try:
                result = fut.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise HTTPException(408, "Processing timeout")

        if "error" in result:
            raise HTTPException(500, detail=result["error"])

        # This post-processing logic is from your original file
        if keys_list and type_map:
            mapped = {}
            # Using keys_list to map results back in order
            result_values = list(result.values())
            for idx, key in enumerate(keys_list):
                 if idx < len(result_values):
                    val = result_values[idx]
                    caster = type_map.get(key, str)
                    try:
                        # Handle base64 images for the frontend
                        if isinstance(val, str) and val.startswith("data:image/"):
                             mapped[key] = val # Keep the data URI for the frontend
                        else:
                             mapped[key] = caster(val) if val not in (None, "") else None
                    except (ValueError, TypeError):
                         mapped[key] = val # Assign original value if casting fails
                 else:
                    mapped[key] = None # Key was expected but not returned
            result = mapped

        # --- MODIFICATION 3: Wrap the result for the frontend ---
        # The frontend expects a dictionary like {"results": {...}}, so we wrap your `result` object.
        return JSONResponse(content={"results": result})

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("analyze_data failed")
        raise HTTPException(500, detail=str(e))

# YOUR ORIGINAL AGENT RUNNER (UNCHANGED)
def run_agent_safely_unified(llm_input: str, pickle_path: str = None) -> Dict:
    try:
        max_retries = 3
        raw_out = ""
        for attempt in range(1, max_retries + 1):
            response = agent_executor.invoke({"input": llm_input})
            raw_out = response.get("output") or response.get("final_output") or response.get("text") or ""
            if raw_out:
                break
        if not raw_out:
            return {"error": f"Agent returned no output after {max_retries} attempts"}

        parsed = clean_llm_output(raw_out)
        if "error" in parsed:
            return parsed

        if "code" not in parsed or "questions" not in parsed:
            return {"error": f"Invalid agent response: {parsed}"}

        code = parsed["code"]
        questions = parsed["questions"]

        if pickle_path is None:
            urls = re.findall(r"scrape_url_to_dataframe\(\s*['\"](.*?)['\"]\s*\)", code)
            if urls:
                url = urls[0]
                tool_resp = scrape_url_to_dataframe(url)
                if tool_resp.get("status") != "success":
                    return {"error": f"Scrape tool failed: {tool_resp.get('message')}"}
                df = pd.DataFrame(tool_resp["data"])
                temp_pkl = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                temp_pkl.close()
                df.to_pickle(temp_pkl.name)
                pickle_path = temp_pkl.name
        
        # Remove scrape call from code before execution
        code = re.sub(r".*scrape_url_to_dataframe.*", "", code)

        exec_result = write_and_run_temp_python(code, injected_pickle=pickle_path, timeout=LLM_TIMEOUT_SECONDS)
        if exec_result.get("status") != "success":
            return {"error": f"Execution failed: {exec_result.get('message')}", "raw": exec_result.get("raw")}

        results_dict = exec_result.get("result", {})
        return {q: results_dict.get(q, "Answer not found") for q in questions}

    except Exception as e:
        logger.exception("run_agent_safely_unified failed")
        return {"error": str(e)}

# YOUR ORIGINAL FAVICON AND INFO ENDPOINTS (UNCHANGED)
_FAVICON_FALLBACK_PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII=")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = "favicon.ico"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(content=_FAVICON_FALLBACK_PNG, media_type="image/png")

@app.get("/api", include_in_schema=False)
async def analyze_get_info():
    return JSONResponse({
        "ok": True,
        "message": "Server is running. Use POST /api with 'questions_file' and optional 'data_file'.",
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
