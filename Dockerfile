# Multi-stage Docker build for OpenCV Python project
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk2.0-dev \
    pkg-config \
    # Camera and video support
    libv4l-dev \
    v4l-utils \
    # GUI support (for X11 forwarding)
    libx11-6 \
    libxss1 \
    libgconf-2-4 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    # Build tools for potential compilation
    gcc \
    g++ \
    make \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

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

# Default command
CMD ["python", "object_tracking.py"]

# Development stage with additional tools
FROM base as development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    git \
    curl \
    wget \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

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