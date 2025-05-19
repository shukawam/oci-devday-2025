FROM python:3.12.10-slim

WORKDIR /app

# RC版（ddtraceのインストール）
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    gcc \
    python3-dev \
    curl \
    build-essential \
    && curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install git+https://github.com/DataDog/dd-trace-py.git@v3.8.0rc1

RUN --mount=type=bind,source=requirements.txt,target=requirements.txt \
    apt-get update && \
    apt-get -y upgrade && \
    pip install -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

EXPOSE 8501
ENV PATH=“${PATH}:/root/.local/bin”

ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
