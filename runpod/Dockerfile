FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    THEROBOT_ROOT=/workspace/therobotgguf \
    THEROBOT_DATA=/workspace/data \
    HF_HOME=/workspace/data/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/workspace/data/hf-cache \
    ROBOTGGUF_MODELS=/workspace/data/llms \
    ROBOTGGUF_CORPUS=/workspace/data/corpus \
    LLAMA_CPP_ROOT=/opt/llama.cpp \
    LLAMA_CPP_BIN=/opt/llama.cpp/build/bin \
    PATH=/opt/llama.cpp/build/bin:/opt/robotgguf-venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        bat \
        btop \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        git-lfs \
        jq \
        less \
        libcurl4-openssl-dev \
        libssl-dev \
        ninja-build \
        pkg-config \
        python3-dev \
        protobuf-compiler \
        ripgrep \
        rsync \
        tar \
        unzip \
        vim \
        wget && \
    if ! command -v nvim >/dev/null 2>&1; then apt-get install --yes --no-install-recommends neovim; fi && \
    ln -sf /usr/bin/batcat /usr/local/bin/bat && \
    git lfs install --system && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:/opt/llama.cpp/build/bin:/opt/robotgguf-venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN python -m pip install --no-cache-dir --upgrade pip uv && \
    python -m venv /opt/robotgguf-venv

COPY 3rd-party/zellij /opt/zellij
WORKDIR /opt/zellij
RUN cargo build --locked --release -p zellij && \
    install -m 0755 target/release/zellij /usr/local/bin/zellij && \
    rm -rf target /root/.cargo/registry /root/.cargo/git

COPY 3rd-party/llama.cpp /opt/llama.cpp

WORKDIR /opt/llama.cpp
RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF && \
    cmake --build build --config Release --parallel "$(nproc)" && \
    find build/bin -maxdepth 1 -type f -executable -print | sort

RUN mkdir -p /workspace/therobotgguf /workspace/data && \
    mkdir -p /workspace/data/llms /workspace/data/corpus /workspace/data/hf-cache /workspace/data/work

COPY projects/therobotgguf/README.md /workspace/therobotgguf/README.md
COPY projects/therobotgguf/arch /workspace/therobotgguf/arch
COPY projects/therobotgguf/convert /workspace/therobotgguf/convert

WORKDIR /workspace/therobotgguf/convert
RUN /opt/robotgguf-venv/bin/python -m pip install --no-cache-dir --upgrade pip && \
    /opt/robotgguf-venv/bin/python -m pip install --no-cache-dir -e '.[hf,label]' && \
    rm -rf /root/.cache/pip

COPY projects/therobotgguf/runpod/bin/ /usr/local/bin/
RUN chmod +x /usr/local/bin/therobot-*

WORKDIR /workspace/therobotgguf

CMD ["/start.sh"]
