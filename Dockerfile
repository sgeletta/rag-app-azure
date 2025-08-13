# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Update package lists and upgrade system packages to patch vulnerabilities
RUN apt-get update && apt-get upgrade -y && \
    # Clean up the apt cache to keep the image size down
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable to tell Streamlit to run on the correct network interface
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run rag_app.py when the container launches
CMD ["streamlit", "run", "rag_app.py"]