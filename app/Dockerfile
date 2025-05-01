FROM python:3.12.10-slim

WORKDIR /app

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
