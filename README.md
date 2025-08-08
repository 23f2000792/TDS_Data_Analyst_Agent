# TDS Data Analyst Agent

This project deploys a data analyst agent as a web API using FastAPI. The agent uses Large Language Models (LLMs) via the Groq API to interpret user questions, write and execute Python code in a secure Docker environment, and return structured answers.

This refined version includes an integrated web dashboard for easier interaction, improved error handling, and secure management of API keys.

## Features

-   **API Endpoint**: Exposes a `POST /api/` endpoint to receive data analysis tasks.
-   **LLM-Powered Planning**: Uses a Groq LLM (`llama-3.3-70b-versatile`) to create a step-by-step Python code plan to answer questions.
-   **Secure Code Execution**: Runs the generated Python code in an isolated Docker container to prevent security risks.
-   **Integrated Dashboard**: A simple web UI at the root (`/`) to upload files and ask questions directly from the browser.
-   **File Handling**: Supports multiple file uploads (`.csv`, `.png`, etc.) alongside the required `questions.txt`.

---

## Setup and Installation (Windows + VS Code)

Follow these steps to get the project running on your local machine.

### Prerequisites

1.  **Python**: Ensure you have Python 3.9+ installed. You can get it from the [Microsoft Store](https://www.microsoft.com/store/productId/9P7QFQMTFRX5) or [python.org](https://www.python.org/downloads/).
2.  **VS Code**: Your code editor. [Download here](https://code.visualstudio.com/).
3.  **Docker Desktop**: The agent requires Docker to execute code securely. [Download and install Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/). **Make sure Docker Desktop is running before you start the application.**
4.  **Groq Account**: Sign up for a free account at [Groq.com](https://groq.com/) to get an API key.

### Step-by-Step Instructions

1.  **Create Project Folder**: Create a folder named `TDS-Data-Analyst-Agent` and open it in VS Code.

2.  **Create Files**: Create the files listed in the repository structure (`app.py`, `requirements.txt`, etc.) and copy the provided code into them.

3.  **Set Up API Key**:
    * Create a new file named `.env` in the project root.
    * Add your Groq API key to this file like so:
        ```
        GROQ_API_KEY="gsk_YourSecretKeyGoesHere"
        ```
    * The `.gitignore` file is configured to prevent this file from being uploaded to GitHub.

4.  **Create Python Virtual Environment**:
    * Open the terminal in VS Code (`Ctrl + `` `).
    * Create a virtual environment to isolate project dependencies:
        ```bash
        python -m venv .venv
        ```
    * Activate the environment:
        ```bash
        .venv\Scripts\activate
        ```
    * Your terminal prompt should now be prefixed with `(.venv)`.

5.  **Install Dependencies**:
    * Install all the required packages from `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```

6.  **Run the Application**:
    * Make sure Docker Desktop is running.
    * Start the FastAPI server using uvicorn:
        ```bash
        uvicorn app:app --reload
        ```
    * The `--reload` flag automatically restarts the server when you make changes to the code.

7.  **Use the Agent**:
    * Open your web browser and navigate to **http://127.0.0.1:8000**.
    * You will see the integrated dashboard where you can type your questions and upload data files.
    * The API endpoint is available at `http://127.0.0.1:8000/api/`.

---

## Deploying to a Public URL

For the project submission, you need a public URL. You can use services like:

-   **Ngrok**: A simple tool to expose your local server to the internet. After starting your local server, run `ngrok http 8000` in a new terminal. It will give you a public URL.
-   **Cloud Platforms**: Deploy the application on services like Render, Heroku, or Google Cloud Run for a more permanent solution.