# Python
FROM python:3.10-slim

# Showing working directort for our container
WORKDIR /app

# initially copy requirements.txt and installing dependencies
# This allows Docker caching our downloaded liabraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy code our FastAPI
COPY main.py .

# Command for running server (Uvicorn на порту 8000)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]