#!/bin/bash

echo "Starting Data Analyst Agent API..."
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting server..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

echo "API is running at http://localhost:8000"
echo "Test with: python test_api.py"
