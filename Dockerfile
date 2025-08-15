# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
# This is done first to leverage Docker's layer caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Define the command to run the app.
# The --server.address=0.0.0.0 flag makes the app accessible from outside the container.
CMD ["streamlit", "run", "rag-app.py", "--server.port=8501", "--server.address=0.0.0.0"]