# -------------------------------
# Base Image
# -------------------------------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON_VERSION=3.11.0 \
    PYTHONUNBUFFERED=1

# -------------------------------
# System Dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    ffmpeg \
    wget \
    nano \
    git \
    htop \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    tk-dev \
    # tritonclient needs these
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Install Python 3.11 (from source)
# -------------------------------
WORKDIR /usr/src

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd / && rm -rf /usr/src/Python-${PYTHON_VERSION}*

# Set Python 3.11 as default
RUN ln -sf /usr/local/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip

# -------------------------------
# Python Setup
# -------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools==75.2.0 wheel==0.46.3

# -------------------------------
# Transformers (specific commit)
# -------------------------------
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5

# -------------------------------
# Python Dependencies
# -------------------------------
RUN pip install --no-cache-dir \
    accelerate==1.13.0 \
    absl-py==1.4.0 \
    anyio==4.12.1 \
    attrs==25.4.0 \
    blinker==1.9.0 \
    cachetools==6.2.6 \
    certifi==2026.2.25 \
    charset-normalizer==3.4.6 \
    click==8.3.1 \
    datasets==2.20.0 \
    filelock==3.25.2 \
    flask==3.1.3 \
    fsspec==2024.5.0 \
    google-api-core==2.30.0 \
    google-auth==2.47.0 \
    google-cloud-appengine-logging==1.8.0 \
    google-cloud-audit-log==0.4.0 \
    google-cloud-core==2.5.0 \
    google-cloud-logging==3.14.0 \
    google-cloud-secret-manager==2.26.0 \
    googleapis-common-protos==1.73.0 \
    grpc-google-iam-v1==0.14.3 \
    grpcio==1.71.2 \
    grpcio-status==1.71.2 \
    gunicorn==23.0.0 \
    h11==0.16.0 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    huggingface-hub==1.7.1 \
    idna==3.11 \
    importlib-metadata==8.7.1 \
    ipython==7.34.0 \
    ipython-sql==0.5.0 \
    itsdangerous==2.2.0 \
    jinja2==3.1.6 \
    jsonschema==4.26.0 \
    jsonschema-specifications==2025.9.1 \
    librosa==0.11.0 \
    markupsafe==3.0.3 \
    mpmath==1.3.0 \
    networkx==3.6.1 \
    numpy==1.26.4 \
    opentelemetry-api==1.38.0 \
    packaging==26.0 \
    proto-plus==1.27.1 \
    protobuf==5.29.6 \
    psutil==5.9.5 \
    pyasn1==0.6.3 \
    pyasn1-modules==0.4.2 \
    pyyaml==6.0.3 \
    referencing==0.37.0 \
    regex==2025.11.3 \
    requests==2.32.4 \
    rpds-py==0.30.0 \
    rsa==4.9.1 \
    notebook==7.5.5 \
    safetensors==0.7.0 \
    scipy==1.16.3 \
    shellingham==1.5.4 \
    six==1.17.0 \
    soundfile==0.13.1 \
    sympy==1.14.0 \
    tokenizers==0.22.2 \
    torchcodec==0.10.0 \
    tqdm==4.67.3 \
    typer-slim==0.24.0 \
    typing-extensions==4.15.0 \
    urllib3==2.5.0 \
    werkzeug==3.1.6 \
    zipp==3.23.0 \
    evaluate==0.4.6 \
    jiwer==4.0.0 \
    rapidfuzz==3.14.3

# -------------------------------
# tritonclient (install separately - needs optional extras)
# -------------------------------
RUN pip install --no-cache-dir tritonclient[all]==2.54.0

# -------------------------------
# hf-xet (install separately to isolate failures)
# -------------------------------
RUN pip install --no-cache-dir hf-xet==1.4.2 || echo "hf-xet install failed, skipping"

# -------------------------------
# PyTorch (CUDA 12.8)
# -------------------------------
RUN pip install --no-cache-dir \
    torch==2.10.0+cu128 \
    torchaudio==2.10.0+cu128 \
    torchvision==0.25.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# -------------------------------
# Verification
# -------------------------------
RUN python --version && pip --version

# -------------------------------
# Working Directory
# -------------------------------
WORKDIR /home

CMD ["/bin/bash"]
