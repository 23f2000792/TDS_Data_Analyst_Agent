import os
import io
import tarfile
import json
import base64
import tempfile
import asyncio
import logging
import re
from typing import List, Dict, Any

import docker
from docker.errors import DockerException
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv
import requests

# ==============================================================================
# 1. CONFIGURATION & INITIALIZATION
# ==============================================================================

# --- Load Environment Variables ---
load_dotenv()

# --- API Keys and Model Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"
DOCKER_IMAGE = "data-analyst-agent-image:latest"

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Data Analyst Agent API",
    description="An agent that uses LLMs and a secure Docker environment to analyze data.",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Clients ---
try:
    docker_client = docker.from_env()
    docker_client.ping()
    logging.info("✅ Docker client initialized and connected successfully.")
except DockerException:
    logging.error("❌ Docker daemon is not running. Please start Docker Desktop.")
    docker_client = None

if not GROQ_API_KEY:
    logging.warning("⚠️ GROQ_API_KEY is not set. The application may not function correctly.")
    groq_client = None
else:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logging.info("✅ Groq client initialized.")

# ==============================================================================
# 2. INTEGRATED WEB DASHBOARD
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """
    Serves a simple HTML dashboard for interacting with the API.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Data Analyst Agent</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { font-family: 'Inter', sans-serif; }
            .loader {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body class="bg-gray-100 text-gray-800">
        <div class="container mx-auto p-4 md:p-8">
            <div class="bg-white rounded-lg shadow-lg p-8 max-w-3xl mx-auto">
                <h1 class="text-3xl font-bold mb-2 text-center text-gray-700">🤖 Data Analyst Agent</h1>
                <p class="text-center text-gray-500 mb-6">Submit your data analysis task below.</p>

                <form id="analysis-form" class="space-y-6">
                    <div>
                        <label for="questions-file" class="block text-sm font-medium text-gray-700 mb-1">Questions File (questions.txt)</label>
                        <input type="file" id="questions-file" name="questions.txt" accept=".txt" required
                            class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-600 hover:file:bg-indigo-100" />
                    </div>
                    <div>
                        <label for="data-files" class="block text-sm font-medium text-gray-700 mb-1">Data Files (optional)</label>
                        <input type="file" id="data-files" name="data-files" multiple
                            class="w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-600 hover:file:bg-indigo-100" />
                    </div>
                    <div>
                        <button type="submit"
                            class="w-full bg-indigo-600 text-white font-bold py-3 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-300">
                            Analyze
                        </button>
                    </div>
                </form>

                <div id="loading" class="hidden flex justify-center items-center mt-8">
                    <div class="loader"></div>
                    <p class="ml-4 text-gray-600">Analyzing, please wait... (This can take up to 3 minutes)</p>
                </div>

                <div id="results" class="mt-8 hidden">
                    <h2 class="text-2xl font-bold mb-4 text-center">Results</h2>
                    <div class="bg-gray-50 p-4 rounded-md shadow-inner">
                        <pre id="json-output" class="whitespace-pre-wrap break-all text-sm"></pre>
                    </div>
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

                loadingDiv.classList.remove('hidden');
                resultsDiv.classList.add('hidden');

                const formData = new FormData(form);

                try {
                    const response = await fetch('/api/', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    jsonOutput.textContent = JSON.stringify(data, null, 2);
                    resultsDiv.classList.remove('hidden');

                } catch (error) {
                    console.error('Error:', error);
                    const errorJson = {
                        error: "Failed to fetch results from the server.",
                        details: error.message
                    };
                    jsonOutput.textContent = JSON.stringify(errorJson, null, 2);
                    resultsDiv.classList.remove('hidden');
                } finally {
                    loadingDiv.classList.add('hidden');
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ==============================================================================
# 3. CORE API LOGIC
# ==============================================================================

def groq_chat_code_gen(messages, model: str = GROQ_MODEL_DEFAULT, temperature: float = 0.0, max_tokens: int = 4096) -> str:
    """Specialized wrapper for generating Python code from the LLM."""
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq client not initialized. Check API key.")
    try:
        # We ask for a standard text response because wrapping code in JSON is brittle.
        # We will extract the code block ourselves.
        resp = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logging.error(f"Groq API call failed: {e}")
        raise HTTPException(status_code=503, detail=f"Error communicating with LLM provider: {e}")

def extract_python_code(llm_response: str) -> str:
    """Extracts the Python code from a markdown code block."""
    match = re.search(r"```python\n(.*?)\n```", llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback if the model doesn't use markdown
    if "import" in llm_response:
        return llm_response.strip()
    raise ValueError("Could not extract Python code from the LLM response.")


async def run_python_in_docker(script: str, files: Dict[str, bytes]) -> Dict[str, Any]:
    """
    Executes a Python script inside a secure, pre-built Docker container.
    """
    if not docker_client:
        raise HTTPException(status_code=503, detail="Docker is not available or running.")

    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content in files.items():
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "wb") as f:
                f.write(content)
        
        script_path = os.path.join(temp_dir, "main.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        try:
            logging.info(f"Running script in Docker container using image: {DOCKER_IMAGE}")
            container = docker_client.containers.run(
                DOCKER_IMAGE,
                command=["python", "main.py"],
                volumes={temp_dir: {"bind": "/app", "mode": "rw"}},
                working_dir="/app",
                detach=True,
            )

            result = container.wait(timeout=170)
            exit_code = result.get("StatusCode", -1)

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors='ignore')
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors='ignore')

            logging.info(f"Container finished with exit code: {exit_code}")
            if stdout: logging.info(f"STDOUT:\n{stdout}")
            if stderr: logging.warning(f"STDERR:\n{stderr}")
            
            return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}
        except docker.errors.ImageNotFound:
            logging.error(f"Docker image '{DOCKER_IMAGE}' not found. Please build it first.")
            raise HTTPException(status_code=500, detail=f"Execution environment not found. Run 'docker build -t {DOCKER_IMAGE} .'")
        except Exception as e:
            logging.error(f"An unexpected error occurred during Docker execution: {e}")
            if "container" in locals() and container: container.remove(force=True)
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during code execution: {e}")


@app.post("/api/")
async def api(request: Request):
    """
    Main API endpoint that receives multipart form data and processes the analysis request.
    """
    timeout_sec = int(os.getenv("AGENT_TIMEOUT_SEC", "170"))
    try:
        return await asyncio.wait_for(handle_request(request), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logging.error("Request timed out after %s seconds.", timeout_sec)
        return JSONResponse({"error": f"Request timed out after {timeout_sec} seconds"}, status_code=504)
    except Exception as e:
        logging.error(f"Top-level error handler caught: {e}", exc_info=True)
        detail = e.detail if isinstance(e, HTTPException) else str(e)
        return JSONResponse({"error": "An unexpected server error occurred.", "detail": detail}, status_code=500)


async def handle_request(request: Request):
    """
    Parses the multipart form request, runs the analysis, and returns the result.
    """
    form = await request.form()

    qfile: UploadFile = form.get("questions.txt")
    if not qfile:
        txt_files = [item for item in form.values() if isinstance(item, UploadFile) and item.filename.lower().endswith(".txt")]
        if len(txt_files) == 1: qfile = txt_files[0]
        else: raise HTTPException(status_code=400, detail="A single 'questions.txt' file is required.")

    attachments = [item for item in form.values() if isinstance(item, UploadFile) and item is not qfile]
    
    question_text = (await qfile.read()).decode('utf-8')
    attachment_files = {f.filename: await f.read() for f in attachments}
    
    # New logic: Pre-fetch HTML if a URL is in the question
    html_content = ""
    url_match = re.search(r"https?://\S+", question_text)
    if url_match:
        url = url_match.group(0).strip()
        logging.info(f"Found URL, attempting to scrape: {url}")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            html_content = response.text
            logging.info(f"Successfully scraped {len(html_content)} characters from {url}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch URL {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch URL: {e}")

    all_input_files = {"questions.txt": question_text.encode('utf-8'), **attachment_files}
    if html_content:
        all_input_files["scraped_page.html"] = html_content.encode('utf-8')


    # 1. Generate a complete Python script with a single, robust prompt
    system_prompt = (
        "You are an expert Python data analyst. Your sole task is to generate a complete, self-contained, and robust Python script to answer the user's question. "
        "The script will be executed in an environment with pandas, matplotlib, etc. "
        "The script's **ONLY** output must be a **single JSON array** printed to standard output that contains the final, raw answer values in the correct order. Do not include descriptive text or keys.\n\n"
        "Respond ONLY with the Python code inside a markdown block: ```python\n...code...\n```\n\n"
        "CRITICAL REQUIREMENTS FOR THE SCRIPT:\n"
        "1.  **Error Handling**: The script must be resilient. Use `try-except` blocks for file I/O, data conversion, and web requests. If an unrecoverable error occurs, print a JSON object like `{\"error\": \"Descriptive error message\"}` and exit.\n"
        "2.  **Data Source**: The necessary data will be provided in local files. If a `scraped_page.html` file is available, use BeautifulSoup to parse it. **Do not make any new HTTP requests.**\n"
        "3.  **HTML Parsing**: Your primary goal is to find the correct data table from the provided HTML. A robust strategy is to find all `<table>` elements and iterate through them to find the one containing expected headers like 'Rank' and 'Title'. **You MUST verify that you have successfully found a table before proceeding.** If no suitable table is found, exit with a specific JSON error: `{\"error\": \"Could not find a table with the required columns.\"}`.\n"
        "4.  **Data Cleaning (MANDATORY)**:\n"
        "    a. **Column Names**: After loading a DataFrame, immediately clean the column names. A robust method is to remove bracketed text (like `[a]`), convert to lowercase, strip whitespace, and replace all non-alphanumeric characters with a single underscore. This prevents `KeyError`.\n"
        "    b. **Cell Values**: Before ANY numeric operations or conversions, you **MUST** ensure the relevant columns (e.g., 'rank', 'peak', 'worldwide_gross') are purely numeric. The only reliable way to do this is to first convert the entire column to strings using `.astype(str)`, and *then* use the `.str.replace()` method with a regular expression to remove all non-numeric characters. Example: `df['col'] = df['col'].astype(str).str.replace(r'[^\\d.]', '', regex=True)`. This prevents the 'Can only use .str accessor with string values!' error. Finally, convert the cleaned column to a numeric type using `pd.to_numeric(df['col'], errors='coerce')`.\n"
        "5.  **Handle NaN Values (CRITICAL FOR JSON)**: Before creating the final JSON output, you **MUST** handle any potential `NaN` (Not a Number) values. If a calculation results in a single `NaN` value (like a correlation), you must check for this and convert it to `None` before adding it to the final list. Example: `correlation = df['rank'].corr(df['peak']); final_correlation = None if pd.isna(correlation) else correlation`. `None` will be correctly serialized to `null`.\n"
        "6.  **Plotting**: If a plot is requested, save it to a file named `plot.png`. Then, read that file, encode it as a base64 string, and include it in the final JSON output as a data URI (`data:image/png;base64,...`).\n"
        "7.  **Final Output**: The script's final action must be `print(json.dumps(final_answer_list))`. The output must be a JSON list (array) containing only the raw values in the order requested by the user. For example: `[1, \"Titanic\", 0.4857, \"data:image/png;base64,...\"]`."
    )

    user_prompt = (
        f"User Question:\n---\n{question_text}\n---\n\n"
        f"Files available in the current directory: {', '.join(all_input_files.keys())}\n\n"
        "Please generate the complete Python script now."
    )

    # Generate the Python code
    llm_response_text = groq_chat_code_gen(messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    
    try:
        python_code = extract_python_code(llm_response_text)
    except ValueError as e:
        logging.error(f"Failed to extract code from LLM response: {llm_response_text}")
        raise HTTPException(status_code=500, detail=f"Could not generate a valid script from the LLM. Response: {llm_response_text}")

    # Run the generated code in Docker
    docker_result = await run_python_in_docker(python_code, all_input_files)

    # Process the result
    if docker_result["exit_code"] != 0:
        error_message = f"The Python script failed to execute. Exit Code: {docker_result['exit_code']}"
        if docker_result["stderr"]:
            error_message += f"\nDetails:\n{docker_result['stderr']}"
        logging.error(error_message)
        return JSONResponse(content={"error": error_message}, status_code=500)

    try:
        # The stdout from the script is the final JSON response
        final_json_output = json.loads(docker_result["stdout"])
        return JSONResponse(content=final_json_output)
    except json.JSONDecodeError:
        error_message = "The Python script ran successfully but did not produce valid JSON output."
        if docker_result["stdout"]:
            error_message += f"\nScript Output:\n{docker_result['stdout']}"
        logging.error(error_message)
        return JSONResponse(content={"error": error_message}, status_code=500)


# ==============================================================================
# 4. APPLICATION RUNNER
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)