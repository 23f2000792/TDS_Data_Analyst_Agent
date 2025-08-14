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
import pandas as pd
import requests

# ==============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ==============================================================================
load_dotenv()

# --- API Keys and Model Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
REQUEST_TIMEOUT_SEC = 350
CODE_EXEC_TIMEOUT_SEC = 340

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

    # This is the final, hardened system prompt with explicit data cleaning and debugging instructions.
    system_prompt = (
        "You are an expert-level Python data analyst. Your sole task is to generate a complete, self-contained, and robust Python script to answer the user's question. "
        "The script will be executed in a secure environment. The script's **ONLY** output to **standard output (stdout)** must be a **single JSON array or object** that contains the final, raw answer values. "
        "Do not include descriptive text. All other logs or debug information must be written to **standard error (stderr)**.\n\n"
        "Respond ONLY with the Python code inside a markdown block: ```python\n...code...\n```\n\n"
        "--- CRITICAL REQUIREMENTS FOR THE GENERATED SCRIPT ---\n"
        "1.  **Imports**: Always start with all necessary imports, including `pandas`, `json`, `numpy`, `sys`, `os`, `duckdb`, `networkx`, and `matplotlib`. "
        "    **Crucially, set the Matplotlib backend to 'Agg' immediately after importing it: `import matplotlib; matplotlib.use('Agg')` to prevent GUI errors.**\n"
        "2.  **File Listing (MANDATORY)**: Your script's first executable line of code after imports MUST be `print(f'Files in directory: {os.listdir()}', file=sys.stderr)`. This is essential for debugging and ensures you know the exact filenames available.\n"
        "3.  **Data Source**: Use the file list printed to stderr to identify the correct file to load. Load files by their exact filename (e.g., `pd.read_csv('sample-sales.csv')`).\n"
        "4.  **Error Handling**: Wrap all major operations in `try-except` blocks. If an error occurs, print a JSON object to stdout like `{\"error\": \"Descriptive error message\"}` and exit.\n"
        "5.  **HTML Table Processing**: If reading from an HTML file with `pd.read_html`, the DataFrame might have a MultiIndex. "
        "    You **MUST** immediately check for and collapse any MultiIndex: `if isinstance(df.columns, pd.MultiIndex): df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]`\n"
        "6.  **MANDATORY Data Cleaning**:\n"
        r"    a. **Clean Column Names**: After loading data, robustly clean all column names. Ensure they are strings, then apply cleaning: `df.columns = df.columns.str.lower().str.strip().str.replace(r'\[.*?\]', '', regex=True).str.replace(r'[^\\w]+', '_', regex=True)`." + "\n"
        "    b. **CRITICAL NUMERIC CLEANING**: When a column contains numbers but is read as a string (e.g., '$1,234.56'), you MUST clean it using this EXACT three-step process:\n"
        "        i. Force to string: `df['column_name'] = df['column_name'].astype(str)`\n"
        "        ii. Remove all non-digit/non-decimal characters: `df['column_name'] = df['column_name'].str.replace(r'[^\\d.]', '', regex=True)`\n"
        "        iii. Convert to numeric: `df['column_name'] = pd.to_numeric(df['column_name'], errors='coerce')`\n"
        "7.  **Base64 Images**: When plotting, you MUST encode the image as a base64 string using the provided `plot_to_base64` helper function. "
        "    The output string MUST be a data URI: `'data:image/png;base64,iVBOR...'`.\n"
        "8.  **JSON Serialization**: Before printing the final JSON, ensure all data is serializable. Define and use a helper function to recursively convert any numpy types (like `np.int64`) to native Python types (`int`, `float`).\n"
        "9.  **Final Output**: The script's final action must be `print(json.dumps(final_answer_dict_or_list, default=json_serializer_helper))`. This is the ONLY print to stdout."
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
        match = re.search(r"```python\n(.*?)\n```", ll_response, re.DOTALL)
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
    """Serves the complete HTML dashboard from your provided index2.html."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>TDS Data Analyst Agent</title>
      <style>
      * { margin:0; padding:0; box-sizing:border-box; }
      body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color:#f5f5f5; background: linear-gradient(135deg,#1e1e2f 0%,#2a2a3d 100%); min-height:100vh; }
      .container { max-width:1200px; margin:0 auto; padding:20px; }
      .header { text-align:center; margin-bottom:30px; color:#fff; }
      .header h1 { font-size:2.5rem; margin-bottom:10px; text-shadow:2px 2px 4px rgba(0,0,0,.6); }
      .header p { font-size:1.1rem; opacity:.85; }
      .main-card { background:#2c2c3e; border-radius:15px; box-shadow:0 10px 30px rgba(0,0,0,.4); padding:30px; margin-bottom:30px; }
      .form-group { margin-bottom: 20px; }
      .form-group label { display:block; margin-bottom:8px; font-weight:600; color:#ddd; }
      .file-input-wrapper { position: relative; }
      .file-input { width:100%; padding:15px; border:2px dashed #555; border-radius:8px; background:#1f1f2e; cursor:pointer; transition:.3s ease; text-align:center; color:#bbb; }
      .file-input input[type="file"] { position: absolute; left: 0; top: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
      .file-input:hover { border-color:#667eea; background:#2a2a40; color:#fff; }
      .file-info { margin-top:10px; padding:10px; background:#333; border-radius:5px; font-size:14px; color:#a8c7ff; display:none; }
      .btn { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:#fff; border:none; padding:15px 30px; border-radius:8px; font-size:16px; font-weight:600; cursor:pointer; transition:.3s ease; width:100%; }
      .btn:hover { transform:translateY(-2px); box-shadow:0 5px 15px rgba(102,126,234,.6); }
      .btn:disabled { background:#555; cursor:not-allowed; transform:none; box-shadow:none; }
      .btn-secondary { background:#444; color:#ddd; }
      .button-group { display:flex; gap:10px; margin-top:20px; flex-wrap:wrap; }
      .loading { display:none; text-align:center; padding:20px; }
      .spinner { border:4px solid #444; border-top:4px solid #667eea; border-radius:50%; width:40px; height:40px; animation:spin 1s linear infinite; margin:0 auto 15px; }
      @keyframes spin { 0%{transform:rotate(0)} 100%{transform:rotate(360deg)} }
      .results { display:none; margin-top:30px; }
      .result-item { background:#1f1f2e; border-left:4px solid #667eea; padding:20px; margin-bottom:20px; border-radius:0 8px 8px 0; }
      .question { font-weight:600; color:#fff; margin-bottom:15px; font-size:1.1rem; }
      .answer { color:#ccc; line-height:1.6; }
      .answer pre { background:#1a1a2a; color:#eee; padding: 15px; border-radius: 5px; white-space: pre-wrap; word-wrap: break-word; }
      .answer img { max-width:100%; height:auto; border-radius:8px; margin:10px 0; box-shadow:0 4px 8px rgba(0,0,0,.5); cursor:pointer; transition:transform .3s ease; }
      .answer img:hover { transform:scale(1.02); }
      .error { background:#3b1f1f; border-left-color:#dc3545; color:#ffb3b3; }
      .modal { display:none; position:fixed; z-index:1000; left:0; top:0; width:100%; height:100%; background:rgba(0,0,0,.9); }
      .modal-content { margin:auto; display:block; width:80%; max-width:700px; max-height:80%; object-fit:contain; }
      .close { position:absolute; top:15px; right:35px; color:#f1f1f1; font-size:40px; font-weight:bold; cursor:pointer; }
      </style>
    </head>
    <body>
      <div class="container">
        <div class="header">
          <h1>🤖 TDS Data Analyst Agent</h1>
          <p>Upload your questions file and optional dataset to get intelligent answers with visualizations</p>
        </div>
        <div class="main-card">
          <form id="analysisForm">
            <div class="form-group">
              <label for="questions_file">Questions File (.txt) <span style="color:#dc3545">*</span></label>
              <div class="file-input-wrapper">
                <div class="file-input" id="questionsDrop">
                  <input type="file" id="questions_file" name="questions.txt"/>
                  <span>📁 Click to upload your questions (.txt)</span>
                </div>
              </div>
              <div id="questionsInfo" class="file-info"></div>
            </div>
            <div class="form-group">
              <label for="data_file">Upload Dataset (Optional)</label>
              <div class="file-input-wrapper">
                <div class="file-input" id="dataDrop">
                  <input type="file" id="data_file" name="data_file" multiple/>
                  <span>📁 Click or drag & drop your dataset(s)</span>
                </div>
              </div>
              <div id="dataInfo" class="file-info"></div>
            </div>
            <div class="button-group">
              <button type="submit" class="btn" id="submitBtn">🚀 Analyze Data</button>
              <button type="button" class="btn btn-secondary" id="clearBtn">🗑️ Clear</button>
            </div>
          </form>
          <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing your data... This may take a few moments.</p>
          </div>
          <div class="results" id="results">
            <h3>📊 Analysis Results</h3>
            <div id="resultsContent"></div>
          </div>
        </div>
      </div>
      <div id="imageModal" class="modal">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImage" alt="Visualization"/>
      </div>
      <script>
        class DataAnalystApp {
          constructor() {
            this.form = document.getElementById('analysisForm');
            this.qFileInput = document.getElementById('questions_file');
            this.dFileInput = document.getElementById('data_file');
            this.qInfo = document.getElementById('questionsInfo');
            this.dInfo = document.getElementById('dataInfo');
            this.submitBtn = document.getElementById('submitBtn');
            this.clearBtn = document.getElementById('clearBtn');
            this.loading = document.getElementById('loading');
            this.resultsContent = document.getElementById('resultsContent');
            this.resultsContainer = document.getElementById('results');
            this.modal = document.getElementById('imageModal');
            this.modalImage = document.getElementById('modalImage');
            this.initEventListeners();
          }
          initEventListeners() {
            this.form.addEventListener('submit', (e) => this.handleSubmit(e));
            this.clearBtn.addEventListener('click', () => this.clearForm());
            this.qFileInput.addEventListener('change', (e) => this.handleFileSelect(e, this.qInfo, 'Questions'));
            this.dFileInput.addEventListener('change', (e) => this.handleFileSelect(e, this.dInfo, 'Dataset(s)'));
            document.querySelector('.close').addEventListener('click', () => this.hideImageModal());
          }
          handleFileSelect(event, infoEl, label) {
            const files = event.target.files;
            if (files.length > 0) {
              const fileNames = Array.from(files).map(f => `${f.name} (${(f.size/1024).toFixed(2)} KB)`).join(', ');
              infoEl.innerHTML = `<strong>${label}:</strong> ${fileNames}`;
              infoEl.style.display = 'block';
            } else {
              infoEl.style.display = 'none';
            }
          }
          async handleSubmit(event) {
            event.preventDefault();
            if (!this.qFileInput.files[0]) { alert('Please upload your questions .txt file.'); return; }
            this.showLoading(true);
            try {
              const formData = new FormData();
              // The evaluation expects the field name to be 'questions.txt'
              formData.append('questions.txt', this.qFileInput.files[0]);
              // The evaluation may send multiple files under the same field name
              Array.from(this.dFileInput.files).forEach(file => {
                  formData.append('data-files', file); 
              });
              const response = await fetch('/api/', { method: 'POST', body: formData });
              const data = await response.json();
              if (!response.ok) {
                throw new Error(data.detail || `HTTP ${response.status}`);
              }
              this.displayResults(data);
            } catch (err) {
              this.displayError(err.message);
            } finally {
              this.showLoading(false);
            }
          }
          displayResults(data) {
            this.resultsContent.innerHTML = '';
            const isError = data.error || (Array.isArray(data) && data.length === 0);
            if (isError) {
              this.displayError(data.error || 'The agent returned an empty result.');
              return;
            }
            const entries = Array.isArray(data) ? data.map((item, i) => [`Answer ${i + 1}`, item]) : Object.entries(data);
            entries.forEach(([key, value]) => {
              const item = this.createResultItem(key, value);
              this.resultsContent.appendChild(item);
            });
            this.resultsContainer.style.display = 'block';
          }
          displayError(message) {
            this.resultsContent.innerHTML = '';
            const item = this.createResultItem('❌ Error', message, true);
            this.resultsContent.appendChild(item);
            this.resultsContainer.style.display = 'block';
          }
          createResultItem(key, value, isError = false) {
            const item = document.createElement('div');
            item.className = isError ? 'result-item error' : 'result-item';
            const keyDiv = document.createElement('div');
            keyDiv.className = 'question';
            keyDiv.textContent = key;
            const valueDiv = document.createElement('div');
            valueDiv.className = 'answer';
            const maybeImg = (typeof value === 'string' && value.startsWith('data:image/')) ? value : null;
            if (maybeImg) {
              const img = document.createElement('img');
              img.src = maybeImg;
              img.onclick = () => this.showImageModal(maybeImg);
              valueDiv.appendChild(img);
            } else {
              const text = (value && typeof value === 'object') ? JSON.stringify(value, null, 2) : String(value ?? 'N/A');
              const pre = document.createElement('pre');
              pre.textContent = text;
              valueDiv.appendChild(pre);
            }
            item.appendChild(keyDiv);
            item.appendChild(valueDiv);
            return item;
          }
          showImageModal(src) { this.modalImage.src = src; this.modal.style.display = 'block'; }
          hideImageModal() { this.modal.style.display = 'none'; }
          showLoading(isLoading) {
            this.loading.style.display = isLoading ? 'flex' : 'none';
            this.submitBtn.disabled = isLoading;
            this.submitBtn.innerHTML = isLoading ? '⏳ Analyzing...' : '🚀 Analyze Data';
          }
          clearForm() {
            this.form.reset();
            this.qInfo.style.display = 'none';
            this.dInfo.style.display = 'none';
            this.resultsContainer.style.display = 'none';
            this.resultsContent.innerHTML = '';
          }
        }
        document.addEventListener('DOMContentLoaded', () => { new DataAnalystApp(); });
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
