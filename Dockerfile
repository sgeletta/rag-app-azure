# Stage 1: Build stage to install dependencies
FROM python:3.11-slim-bookworm AS builder

# Set working directory
WORKDIR /app

# Install system dependencies that might be required by Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file to leverage Docker cache
COPY requirements.pinned.txt .

# Install dependencies using a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install the CPU-only version of torch first to avoid GPU dependency issues.
# Then, install the rest of the packages. pip will skip torch as it's already installed.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.pinned.txt

# Stage 2: Final stage for the application
FROM python:3.11-slim-bookworm

WORKDIR /app

# It's crucial to upgrade packages in the final stage as well,
# as this stage forms the basis of the final image.
# We do this before creating the non-root user.
RUN apt-get update && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd --create-home appuser
USER appuser

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code
COPY --chown=appuser:appuser . .

# Expose the Streamlit port and define the command to run the app
EXPOSE 8501
CMD ["streamlit", "run", "rag_app.py", "--server.port=8501", "--server.address=0.0.0.0"]