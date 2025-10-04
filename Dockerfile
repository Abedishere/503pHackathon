# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies including curl for health check
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY smart_article_graph.py .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Suppress gRPC warnings
ENV GRPC_VERBOSITY=ERROR
ENV GLOG_minloglevel=2
ENV GRPC_POLL_STRATEGY=epoll1

# Health check (using the correct Streamlit health endpoint)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Command to run the application
CMD ["streamlit", "run", "smart_article_graph.py", "--server.port=8501", "--server.address=0.0.0.0"]