FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  software-properties-common \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt
RUN pip install pymupdf

COPY generate.py .

ENTRYPOINT [ "python3", "generate.py" ]
