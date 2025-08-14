import os
import io
import json
import base64
import tempfile
import asyncio
import logging
import re
import sys
from typing import Dict, Any
import subprocess

from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import requests

# ==============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ==============================================================================
load_dotenv()

# --- API Keys and Model Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
REQUEST_TIMEOUT_SEC = 350  # Total time for the entire API request to complete
CODE_EXEC_TIMEOUT_SEC = 340  # Max time for the generated Python script to run

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analyst Agent (Final Version)",
    description="A robust, single-call agent using a secure sandbox for data analysis.",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global OpenAI Client ---
if not OPENAI_API_KEY:
    logging.warning("⚠️ OPENAI_API_KEY is not set. The application will not function.")
    openai_client = None
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("✅ OpenAI client initialized.")

# ==============================================================================
# 2. CORE LOGIC: CODE GENERATION & EXECUTION
# ==============================================================================

def generate_python_code(prompt: str) -> str:
    """Generates Python code from a prompt using a direct, non-agentic LLM call."""
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized. Check API key.")

    # This is the hyper-detailed system prompt inspired by your successful "Logic 1" file.
    # It provides extremely strict instructions to prevent common evaluation failure modes.
    system_prompt = (
        "You are an expert-level Python data analyst. Your sole task is to generate a complete, self-contained, and robust Python script to answer the user's question. "
        "The script will be executed in a secure environment. The script's **ONLY** output to **standard output (stdout)** must be a **single JSON array or object** that contains the final, raw answer values. "
        "Do not include descriptive text. All other logs or debug information must be written to **standard error (stderr)**.\n\n"
        "Respond ONLY with the Python code inside a markdown block: ```python\n...code...\n```\n\n"
        "---"
        "CRITICAL REQUIREMENTS FOR THE GENERATED SCRIPT:\n"
        "1.  **Imports**: Always start the script with all necessary imports, including `pandas`, `json`, `numpy`, `sys`, `duckdb`, `networkx`, and `matplotlib`. "
        "    **Crucially, set the Matplotlib backend to 'Agg' immediately after importing it: `import matplotlib; matplotlib.use('Agg')` to prevent GUI errors.**\n"
        "2.  **Data Source**: Analyze the user's question and the list of available files to determine the analysis scenario:"
        "    - **Web Scraping:** If the query mentions 'scrape' and `scraped_page.html` is available, use BeautifulSoup4 to parse it. "
        "    - **File Analysis:** If the query mentions a specific filename (e.g., 'Analyze `sample-sales.csv`'), load it directly by its name. "
        "    - **Network Analysis:** If `edges.csv` is available and the query involves a network, use the `networkx` library. "
        "    - **Remote Data Query:** If the query provides a SQL query for a remote dataset (e.g., on S3), use DuckDB to execute it. \n"
        "3.  **Error Handling**: Wrap all major operations in `try-except` blocks. If an error occurs, print a JSON object to stdout like `{\"error\": \"Descriptive error message\"}` and exit.\n"
        "4.  **HTML Table Processing**: If reading data from an HTML file with `pd.read_html`, the DataFrame might have a MultiIndex. "
        "    You **MUST** immediately check for and collapse any MultiIndex: `if isinstance(df.columns, pd.MultiIndex): df.columns = ['_'.join(col).strip() for col in df.columns.values]`\n"
        "5.  **MANDATORY Data Cleaning**:\n"
        r"    a. **Clean Column Names**: After loading data, robustly clean all column names. Ensure they are strings, then apply cleaning: `df.columns = df.columns.str.lower().str.strip().str.replace(r'\[.*?\]', '', regex=True).str.replace(r'[^\\w]+', '_', regex=True)`." + "\n"
        "6.  **Base64 Images**: When plotting, you MUST encode the image as a base64 string using the provided `plot_to_base64` helper function. "
        "    The output string MUST be a data URI: `'data:image/png;base64,iVBOR...'`.\n"
        "7.  **JSON Serialization**: Before printing the final JSON, ensure all data is serializable. Define and use a helper function to recursively convert any numpy types (like `np.int64`) to native Python types (`int`, `float`).\n"
        "8.  **Final Output**: The script's final action must be `print(json.dumps(final_answer_dict_or_list, default=json_serializer_helper))`. This is the ONLY print to stdout."
    )
    
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        llm_response = resp.choices[0].message.content
        match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback for when the model doesn't use markdown
        if "import pandas" in llm_response:
            return llm_response
        raise ValueError("Could not extract Python code from the LLM response.")
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        raise HTTPException(status_code=503, detail=f"Error communicating with LLM provider: {e}")

async def execute_code_in_sandbox(script: str, files: Dict[str, bytes]) -> str:
    """Executes a Python script in a temporary directory with provided files using asyncio."""
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content in files.items():
            if filename:
                with open(os.path.join(temp_dir, filename), "wb") as f:
                    f.write(content)

        helper_code = r'''
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
    plt.clf()
    buf.seek(0)
    img_bytes = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(img_bytes).decode('ascii')

def json_serializer_helper(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
'''
        full_script = helper_code + "\n\n" + script
        script_path = os.path.join(temp_dir, "main.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_script)

        try:
            logging.info(f"Running script in subprocess within {temp_dir}")
            process = await asyncio.create_subprocess_exec(
                sys.executable, "main.py",
                cwd=temp_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=CODE_EXEC_TIMEOUT_SEC)
            
            stdout_decoded = stdout.decode('utf-8', errors='ignore')
            stderr_decoded = stderr.decode('utf-8', errors='ignore')
            
            if stderr_decoded:
                logging.warning(f"STDERR from script:\n{stderr_decoded}")
            
            if process.returncode != 0:
                logging.error(f"Script execution failed with exit code {process.returncode}.")
                raise HTTPException(status_code=500, detail=f"Script execution failed: {stderr_decoded or stdout_decoded}")
            
            return stdout_decoded.strip()

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logging.error("Script execution timed out.")
            raise HTTPException(status_code=504, detail="Script execution timed out.")

# ==============================================================================
# 3. API ENDPOINTS
# ==============================================================================
@app.post("/api/")
async def analyze(request: Request):
    """Main API endpoint that receives files, generates and executes code."""
    try:
        return await asyncio.wait_for(handle_analysis_request(request), timeout=REQUEST_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Request timed out after {REQUEST_TIMEOUT_SEC} seconds")

async def handle_analysis_request(request: Request):
    form = await request.form()
    
    questions_file = form.get("questions.txt") or next((v for v in form.values() if isinstance(v, UploadFile) and v.filename and v.filename.lower().endswith('.txt')), None)
    if not questions_file:
        raise HTTPException(status_code=400, detail="Exactly one .txt questions file is required.")

    question_text = (await questions_file.read()).decode('utf-8')
    
    attachment_files = {
        item.filename: await item.read() for item in form.values()
        if isinstance(item, UploadFile) and item.filename and item is not questions_file
    }
    
    # Intelligent scraping logic from your "Logic 1"
    scraped_content = ""
    if "scrape" in question_text.lower() or "from the url" in question_text.lower():
        url_match = re.search(r"https?://\S+", question_text)
        if url_match:
            url = url_match.group(0).rstrip('.,)!?]>')
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                scraped_content = response.text
                attachment_files["scraped_page.html"] = scraped_content.encode('utf-8')
                logging.info(f"Successfully scraped content from {url}")
            except requests.RequestException as e:
                logging.error(f"Failed to scrape URL {url}: {e}")

    user_prompt = (
        f"User Question:\n---\n{question_text}\n---\n\n"
        f"Files available in the current directory: {', '.join(attachment_files.keys()) if attachment_files else 'None'}"
    )

    python_code = generate_python_code(user_prompt)
    result_stdout = await execute_code_in_sandbox(python_code, attachment_files)

    try:
        final_json_output = json.loads(result_stdout)
        if isinstance(final_json_output, dict) and 'error' in final_json_output:
            raise HTTPException(status_code=422, detail=final_json_output['error'])
        return JSONResponse(content=final_json_output)
    except json.JSONDecodeError:
        error_msg = "Script executed but produced non-JSON output."
        logging.error(f"{error_msg} Output: {result_stdout}")
        raise HTTPException(status_code=500, detail=f"{error_msg} Output: {result_stdout}")
        
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_root():
    return HTMLResponse(content="""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Data Analyst Agent</title>
        </head>
        <body>
            <h1>Data Analyst Agent is Ready</h1>
            <p>This API endpoint is live. Please use POST requests to /api/ to submit analysis tasks.</p>
        </body>
        </html>
    """)
    
# ==============================================================================
# 4. APPLICATION RUNNER
# ==============================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    logging.info(f"🚀 Starting server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
