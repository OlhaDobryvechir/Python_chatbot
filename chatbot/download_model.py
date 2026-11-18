"""Download the local LLM used by the chatbot into ./data (about 1 GB)."""
import os

from huggingface_hub import hf_hub_download

REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
FILE = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"

if __name__ == "__main__":
    data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data, exist_ok=True)
    print("Downloading", REPO, FILE, "...")
    print("Saved to", hf_hub_download(REPO, FILE, local_dir=data))
