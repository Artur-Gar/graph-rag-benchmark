import os

DATA_DIR = "./data"
OUTPUT_DIR = "./output"
FIGURES_DIR = "./docs/figures"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
