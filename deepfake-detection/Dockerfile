# Use NVIDIA CUDA 12.1 as base image
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create workspace
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p configs data models training evaluation notebooks scripts tests

# Set up entry point
CMD ["/bin/bash"]

# Usage instructions:
# Build: docker build -t deepfake-detector .
# Run with GPU: docker run --gpus all -v /path/to/datasets:/data -it deepfake-detector
# Run without GPU: docker run -v /path/to/datasets:/data -it deepfake-detector