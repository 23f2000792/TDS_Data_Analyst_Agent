# Use a current, slim Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Copy the rest of the application's code (app.py) into the container
COPY . .

# Command to run the application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
