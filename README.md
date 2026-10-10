# 🚀 Production-Ready NLP Classification API

A microservices pipeline for text classification (review analysis) engineered for high-throughput environments. This project demonstrates the full end-to-end deployment lifecycle of a Machine Learning model: from lightweight tokenization to high-speed GPU inference using industry-standard MLOps practices.

## 🏗 System Architecture

The project implements the Compute Separation pattern and consists of two isolated Docker containers:

1. **API Gateway (FastAPI)**: A lightweight web server running exclusively on the CPU. It handles HTTP request validation (Pydantic), text preprocessing, and tokenization (Hugging Face). It passes the processed tensors as NumPy arrays.
2. **Inference Server (NVIDIA Triton)**: A high-performance C++ server running on the GPU. It receives data via the low-latency gRPC protocol. Leveraging **Dynamic Batching**, it aggregates concurrent requests into micro-batches (up to 5ms) and processes them through a static ONNX graph, maximizing GPU utilization.

## 🛠 Tech Stack

- **Modeling:** PyTorch, Hugging Face Transformers (`intfloat/multilingual-e5-*`)
- **Optimization:** ONNX (Open Neural Network Exchange)
- **Serving:** NVIDIA Triton Inference Server, FastAPI, Uvicorn, gRPC
- **Infrastructure:** Docker, Docker Compose

## 📂 Repository Structure

```text
NLP_Project/
├── fastapi_app/
│   ├── main.py                 # REST API, tokenization logic, and gRPC client
│   ├── requirements.txt        # Microservice dependencies (excluding heavy PyTorch)
│   └── Dockerfile              # Build script for the lightweight image
├── triton_model_repository/
│   └── e5_classifier/          
│       ├── config.pbtxt        # Model configuration, I/O definitions, and batching rules
│       └── 1/
│           └── model.onnx      # Exported ONNX weights (added to .gitignore)
├── docker-compose.yml          # Container orchestration and GPU passthrough
├── train_model.ipynb           # Notebook containing model training and ONNX export
├── test_model.ipynb            # API testing and validation notebook
└── .gitignore
```
*(Note: Large artifacts like `.onnx` and `.pth` weight files, as well as raw `.csv` datasets, are excluded from this repository to adhere to size limits).*

## ⚙️ Quick Start (Local Deployment)

### Prerequisites
* Docker Engine & Docker Compose
* NVIDIA GPU with CUDA drivers installed (and NVIDIA Container Toolkit configured)

### Running the Containers
Clone the repository and build the infrastructure from the root directory:
```bash
docker-compose up --build
```
The server will pull the base images, allocate the model in VRAM, and expose port `8000`.

## 📡 API Usage

### 1. Via Swagger UI
Navigate your browser to `http://localhost:8000/docs`. The interactive documentation allows you to send test requests directly from the web interface.

### 2. Via Python (requests)
```python
import requests

url = "http://localhost:8000/predict"
payload = {"text": "Отличное заведение, очень вкусный кофе и приятная музыка!"}

response = requests.post(url, json=payload)
print(response.json())
# Expected output: {"probability": 0.98, "class": 1}
```
## 🧠 Key Engineering Decisions
* **Zero PyTorch in the Production API:** The FastAPI container runs without `torch`. The tokenizer outputs native lists that are packaged by `tritonclient`. This reduces the API image size by several gigabytes and eliminates CPU memory leaks.
* **Static Computation Graph:** The model is exported to ONNX with a fixed tensor dimension (`max_length=64`). This prevents VRAM fragmentation.
* **gRPC Network Protocol:** Replaces slow, text-based JSON parsing between containers with binary data transmission, which is critical for transferring dense numerical matrices (tensors) with near-zero latency.
