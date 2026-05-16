FROM python:3.10-slim

# libgomp1 — OpenMP runtime required by XGBoost and LightGBM on Linux
# (macOS uses libomp via Homebrew, but Linux wheels link against libgomp.so.1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
