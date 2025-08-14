import os
import io
import json
import base64
import tempfile
import asyncio
import logging
import re
import sys
from typing import Dict, Any, Optional
import subprocess

from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
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

    # This is the hardened system prompt, inspired by your successful reference code.
    # It provides extremely strict instructions to prevent common evaluation failure modes.
    system_prompt = (
        "You are an expert-level Python data analyst. Your sole task is to generate a complete, self-contained, and robust Python script to answer the user's question. "
        "The script will be executed in a secure environment. The script's **ONLY** output to **standard output (stdout)** must be a **single JSON array or object** that contains the final, raw answer values. "
        "Do not include descriptive text. All other logs or debug information must be written to **standard error (stderr)**.\n\n"
        "Respond ONLY with the Python code inside a markdown block: ```python\n...code...\n```\n\n"
        "--- CRITICAL REQUIREMENTS FOR THE GENERATED SCRIPT ---\n"
        "1.  **Imports**: Always start the script with all necessary imports, including `pandas`, `json`, `numpy`, `sys`, `duckdb`, `networkx`, and `matplotlib`. "
        "    **Crucially, set the Matplotlib backend to 'Agg' immediately after importing it: `import matplotlib; matplotlib.use('Agg')` to prevent GUI errors.**\n"
        "2.  **Data Source**: The user's files are in the current working directory. Load them by their filename (e.g., `pd.read_csv('sample-sales.csv')`). Analyze the user's question and the list of available files to determine the correct analysis scenario.\n"
        "3.  **Error Handling**: Wrap all major operations in `try-except` blocks. If an error occurs, print a JSON object to stdout like `{\"error\": \"Descriptive error message\"}` and exit.\n"
        "4.  **HTML Table Processing**: If reading data from an HTML file with `pd.read_html`, the DataFrame might have a MultiIndex. "
        "    You **MUST** immediately check for and collapse any MultiIndex: `if isinstance(df.columns, pd.MultiIndex): df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]`\n"
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
import json
import pandas as pd

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
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        return str(obj)
    except Exception:
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
'''
        full_script = helper_code + "\n\n" + script
        script_path = os.path.join(temp_dir, "main.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_script)

        try:
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
    """Main API endpoint for data analysis."""
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
    
    # Intelligent scraping
    if "scrape" in question_text.lower() or "from the url" in question_text.lower():
        url_match = re.search(r"https?://\S+", question_text)
        if url_match:
            url = url_match.group(0).rstrip('.,)!?]>')
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                attachment_files["scraped_page.html"] = response.content
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
async def get_dashboard():
    """Serves the complete HTML dashboard from your reference logic."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Data Analyst Agent</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style> body { font-family: 'Inter', sans-serif; } .loader { border: 4px solid #f3f3f3; border-top: 4px solid #4f46e5; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; } @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
    </head>
    <body class="bg-gray-100 text-gray-800">
        <div class="container mx-auto p-4 md:p-8">
            <div class="bg-white rounded-lg shadow-lg p-8 max-w-3xl mx-auto">
                <h1 class="text-3xl font-bold mb-6 text-center text-gray-700">Data Analyst Agent</h1>
                <form id="analysis-form" class="space-y-6">
                    <div>
                        <label for="questions-file" class="block text-sm font-medium text-gray-700 mb-1">Questions File (questions.txt)</label>
                        <input type="file" id="questions-file" name="questions.txt" accept=".txt" required class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-600 hover:file:bg-indigo-100 cursor-pointer" />
                    </div>
                    <div>
                        <label for="data-files" class="block text-sm font-medium text-gray-700 mb-1">Data Files (optional)</label>
                        <input type="file" id="data-files" name="data-files" multiple class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-600 hover:file:bg-indigo-100 cursor-pointer" />
                    </div>
                    <div>
                        <button type="submit" class="w-full bg-indigo-600 text-white font-bold py-3 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-300 flex items-center justify-center">Analyze</button>
                    </div>
                </form>
                <div id="loading" class="hidden flex flex-col justify-center items-center mt-8 text-center">
                    <div class="loader"></div><p class="mt-4 text-gray-600">Analyzing, please wait...</p>
                </div>
                <div id="results" class="mt-8 hidden">
                    <h2 class="text-2xl font-bold mb-4 text-center">Results</h2>
                    <div class="bg-gray-50 p-4 rounded-md shadow-inner"><pre id="json-output" class="whitespace-pre-wrap break-all text-sm font-mono"></pre></div>
                </div>
            </div>
        </div>
        <script>
            const form = document.getElementById('analysis-form');
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const loadingDiv = document.getElementById('loading');
                const resultsDiv = document.getElementById('results');
                const jsonOutput = document.getElementById('json-output');
                const button = form.querySelector('button');
                loadingDiv.classList.remove('hidden');
                resultsDiv.classList.add('hidden');
                button.disabled = true;
                const formData = new FormData(form);
                try {
                    const response = await fetch('/api/', { method: 'POST', body: formData });
                    const data = await response.json();
                    if (response.ok) {
                        jsonOutput.textContent = JSON.stringify(data, null, 2);
                    } else {
                        const errorJson = { error: data.detail || "An error occurred.", status_code: response.status };
                        jsonOutput.textContent = JSON.stringify(errorJson, null, 2);
                    }
                    resultsDiv.classList.remove('hidden');
                } catch (error) {
                    const errorJson = { error: "Failed to fetch results from the server.", details: error.message };
                    jsonOutput.textContent = JSON.stringify(errorJson, null, 2);
                    resultsDiv.classList.remove('hidden');
                } finally {
                    loadingDiv.classList.add('hidden');
                    button.disabled = false;
                }
            });
        </script>
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
