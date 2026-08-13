"""Modal app — INFERENCE ONLY: the OpenAI-compatible teacher endpoint
(vLLM serving Qwen3.6-35B-A3B, the donor doubling as its own teacher for
extraction-v1 T2 labeling).

Training-side GPU work (R1 record, t1 label passes, R3 graft training) does
NOT run on Modal — rent a GPU host for those and run the plain CLI there
(see configs/qwen3.6-35b-a3b.yaml header and convert/README.md).

Deploy:
  modal deploy tools/modal_teacher.py

Then point the t2 labeler at it:
  export ROBOT_TEACHER_BASE_URL=https://<workspace>--robotgguf-teacher-serve.modal.run/v1
  export ROBOT_TEACHER_MODEL=Qwen/Qwen3.6-35B-A3B
"""
from __future__ import annotations

import subprocess

import modal

app = modal.App("robotgguf-teacher")

HF_CACHE = modal.Volume.from_name("robotgguf-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm>=0.10", "huggingface_hub")
)

# 35B bf16 ≈ 70 GB weights — H200 (141 GB) leaves headroom for the 256-expert
# routing tables + KV; drop --max-model-len rather than the GPU if squeezed.
GPU = "H200"

secrets = [modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])]


@app.function(image=image, gpu=GPU,
              volumes={"/root/.cache/huggingface": HF_CACHE},
              secrets=secrets, timeout=24 * 3600,
              scaledown_window=15 * 60)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=30 * 60)
def serve() -> None:
    subprocess.Popen([
        "vllm", "serve", "Qwen/Qwen3.6-35B-A3B",
        "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "8192",
    ])
