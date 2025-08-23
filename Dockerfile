# ---- Builder Stage ----
# This stage installs the dependencies
FROM python:3.11-slim as builder

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
# Install dependencies to a user-specific directory
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Final Stage ----
# This stage builds the final, lean image
FROM python:3.11-slim

WORKDIR /app

# Create a non-root user and group for security
RUN addgroup --system app && adduser --system --ingroup app

# Copy installed packages from the builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy app source
COPY . /app

# Set ownership to the new user and switch to that user
RUN chown -R app:app /app
USER app

# Set environment variables for Streamlit
# Update PATH to include the user's local bin directory
ENV PATH="/home/app/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "rag_app.py"]
