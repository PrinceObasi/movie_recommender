# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variable for Python output
ENV PYTHONUNBUFFERED=1

# Command to run the demo script; allows passing command-line arguments
ENTRYPOINT ["python", "demo.py"]
