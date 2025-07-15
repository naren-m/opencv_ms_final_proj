# Multi-stage Docker build for OpenCV Python project
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Essential OpenCV runtime dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # Basic GUI support
    libx11-6 \
    libxss1 \
    # Camera support
    libv4l-0 \
    # OpenGL support for OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Additional OpenCV dependencies
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Cleanup in same layer
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash opencv_user

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pysrc/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pysrc/ ./

# Create directories for outputs
RUN mkdir -p outputs logs && \
    chown -R opencv_user:opencv_user /app

# Switch to non-root user
USER opencv_user

# Expose port for potential web interface
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import cv2; print('OpenCV version:', cv2.__version__)" || exit 1

# Default command (headless mode for containers)
CMD ["python", "object_tracking.py", "--headless"]

# Development stage with additional tools
FROM base as development

USER root

# Install minimal development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Install development Python packages
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    black \
    flake8 \
    pytest \
    pytest-cov

USER opencv_user

# Development command
CMD ["bash"]

# Production stage (minimal)
FROM base as production

# Only copy necessary files
COPY --from=base /app ./

# Use specific command for production
CMD ["python", "object_tracking.py"]