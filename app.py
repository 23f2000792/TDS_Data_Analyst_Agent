import os
import re
import json
import base64
import tempfile
import subprocess
import logging
from io import BytesIO, StringIO
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, Response
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Optional Pillow
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:  # pragma: no cover
    PIL_AVAILABLE = False

# LangChain / LLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ────────────────────────────  Setup  ────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TDS Data Analyst Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", 150))
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")           # <- fixed
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─────────────────────  Helper: OpenAI validation  ──────────────────────
def validate_openai_setup() -> None:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for the agent to run."
        )
    try:
        _ = ChatOpenAI(
            model=DEFAULT_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        ).invoke("ping")
        logger.info("✅ OpenAI API connectivity validated.")
    except Exception as e:  # pragma: no cover
        logger.error("❌ OpenAI validation failed: %s", e)
        raise

@app.on_event("startup")
async def on_startup() -> None:  # noqa: D401
    """Validate key/model at startup so failures are clear."""
    validate_openai_setup()

# ─────────────────────────  Front-end route  ──────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse(
            "<h2>Frontend not found</h2><p>Ensure index.html is present.</p>",
            status_code=404,
        )

# ───────────────────────  Question-key parser  ────────────────────────
def parse_keys_and_types(raw: str):
    pattern = r"-\s*`([^`]+)`\s*:\s*(\w+)"
    pairs = re.findall(pattern, raw)
    type_map = {
        k: {"number": float, "string": str, "integer": int,
            "int": int, "float": float}.get(t.lower(), str)
        for k, t in pairs
    }
    keys = [k for k, _ in pairs]
    return keys, type_map

# ─────────────────────────────  Tool  ────────────────────────────────
@tool
def scrape_url_to_dataframe(url: str) -> Dict[str, Any]:
    """Fetch URL and return a DataFrame (tables, CSV, JSON, etc.)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/138.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.google.com/",
    }
    try:
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
        else:  # HTML fallback
            from bs4 import BeautifulSoup
            html = resp.text
            try:
                tables = pd.read_html(StringIO(html), flavor="bs4")
                df = tables[0] if tables else None
            except ValueError:
                df = None
            if df is None:
                text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
                df = pd.DataFrame({"text": [text]})

        df.columns = (
            df.columns.map(str).str.replace(r"\[.*\]", "", regex=True).str.strip()
        )
        return {"status": "success",
                "data": df.to_dict("records"),
                "columns": df.columns.tolist()}
    except Exception as e:  # pragma: no cover
        return {"status": "error", "message": str(e)}

# ─────────────────────────  Safe-exec helper  ─────────────────────────
def clean_llm_output(out: str) -> Dict[str, Any]:
    if not out:
        return {"error": "Empty LLM output (check API key/model)."}
    out = re.sub(r"^``````$", "", out.strip())
    first, last = out.find("{"), out.rfind("}")
    if first == -1 or last == -1 or first >= last:
        return {"error": "No JSON object found in LLM output.", "raw": out[:300]}
    try:
        return json.loads(out[first : last + 1])
    except Exception as e:
        return {"error": f"JSON parsing failed: {e}", "raw": out[:300]}

def write_and_run_temp_python(code: str, pkl: str | None, timeout: int = 60):
    preamble = [
        "import json, sys, gc",
        "import pandas as pd, numpy as np",
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from io import BytesIO",
        "import base64",
        "results = {}",
    ]
    if PIL_AVAILABLE:
        preamble.insert(0, "from PIL import Image")
    if pkl:
        preamble.append(f"df = pd.read_pickle(r'''{pkl}'''); data = df.to_dict('records')")
    else:
        preamble.append("data = {}")
    preamble.append("def plot_to_base64(max_bytes=100000):\n"
                    "    buf=BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight', dpi=100);\n"
                    "    buf.seek(0); b=buf.getvalue();\n"
                    "    return base64.b64encode(b).decode()")
    script = "\n".join(preamble) + "\n" + code + \
        "\nprint(json.dumps({'status':'success','result':results}, default=str), flush=True)"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8")
    tmp.write(script); tmp.flush(); tmp.close()
    try:
        proc = subprocess.run(
            [sys.executable, tmp.name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = proc.stdout.strip()
        if proc.returncode != 0:
            return {"status": "error", "message": proc.stderr or proc.stdout}
        return json.loads(out)
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Execution timed out."}
    except Exception as e:
        return {"status": "error", "message": f"JSON parse fail: {e}", "raw": out[:300]}
    finally:
        os.unlink(tmp.name)
        if pkl and os.path.exists(pkl):
            os.unlink(pkl)

# ──────────────────────  LLM / Agent definition  ──────────────────────
llm = ChatOpenAI(
    model=DEFAULT_MODEL,
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

tools = [scrape_url_to_dataframe]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a data analyst agent. "
                "Return a single JSON object with key 'code', "
                "whose value is a Python script that fills a dict named `results`."
            ),
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate",
)

# ──────────────────────────  Main endpoint  ───────────────────────────
@app.post("/")
@app.post("/api/")
@app.post("/api")
async def analyze_data(request: Request):
    if not OPENAI_API_KEY:
        raise HTTPException(400, "OPENAI_API_KEY not configured.")
    form = await request.form()
    q_file = next((v for v in form.values() if getattr(v, "filename", "").endswith(".txt")), None)
    data_file = next((v for v in form.values() if v is not q_file and getattr(v, "filename", None)), None)
    if not q_file:
        raise HTTPException(400, "Missing questions file (.txt)")
    raw_q = (await q_file.read()).decode()
    keys, type_map = parse_keys_and_types(raw_q)

    pkl_path, df_preview = None, ""
    if data_file:
        content = await data_file.read()
        name = data_file.filename.lower()
        df = (
            pd.read_csv(BytesIO(content))
            if name.endswith(".csv") else
            pd.read_excel(BytesIO(content))
            if name.endswith((".xls", ".xlsx")) else
            pd.read_parquet(BytesIO(content))
            if name.endswith(".parquet") else
            pd.read_json(BytesIO(content))
            if name.endswith(".json") else
            pd.DataFrame({"image": [Image.open(BytesIO(content)).convert("RGB")]})  # images
            if PIL_AVAILABLE and name.endswith((".png", ".jpg", ".jpeg")) else None
        )
        if df is None:
            raise HTTPException(400, f"Unsupported data file: {name}")
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False); tmp.close()
        df.to_pickle(tmp.name); pkl_path = tmp.name
        df_preview = f"\n\nDataset Columns: {', '.join(df.columns.map(str))}\n"

    rules = (
        "Rules:\n1) Use provided DataFrame `df`.\n2) DO NOT call scrape_url_to_dataframe()."
        if data_file else
        "Rules:\n1) You must call scrape_url_to_dataframe(url) to fetch data."
    )
    llm_input = f"{rules}\nTask:\n{raw_q}{df_preview}\nRespond with JSON only."
    response = agent_executor.invoke({"input": llm_input}, {"timeout": LLM_TIMEOUT_SECONDS})
    parsed = clean_llm_output(response.get("output", ""))
    if "error" in parsed:
        raise HTTPException(500, f"Agent error: {parsed['error']}")

    code = parsed.get("code")
    if not code:
        raise HTTPException(500, "Agent response missing 'code'.")

    exec_result = write_and_run_temp_python(code, pkl_path, timeout=LLM_TIMEOUT_SECONDS)
    if exec_result.get("status") != "success":
        raise HTTPException(500, f"Exec error: {exec_result.get('message')}")

    res = exec_result.get("result", {})
    final = {
        k: (type_map[k](res[k]) if k in res and type_map[k] != str else res.get(k))
        for k in keys
    }
    return JSONResponse(final)

# ────────────────────────  Misc routes  ───────────────────────────────
_FALLBACK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3n+9QAAAAASUVORK5CYII="
)
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    path = "favicon.ico"
    return FileResponse(path) if os.path.exists(path) else Response(content=_FALLBACK_PNG, media_type="image/png")

@app.get("/api", include_in_schema=False)
async def health():
    return {"ok": True, "message": "POST to /api with 'questions_file' (+ optional 'data_file')"}

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
