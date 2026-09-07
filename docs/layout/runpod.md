# runpod/ — Rented GPU Host Image

Containerized environment for the GPU-bound stages (R0 ingest, R1 record,
R3 graft training, t1/t2 labels) that need a CUDA host + donor checkpoints.
Built via runpod templates; expects the llama.cpp fork and this repo baked in
per the Dockerfile `COPY projects/therobotgguf/...` paths.

```
runpod/
├── Dockerfile                      # CUDA llama.cpp fork build + robotgguf venv (`.[hf,label]`) + zellij
├── Dockerfile.dockerignore
└── bin/                            # Installed to /usr/local/bin in the image
    ├── therobot-init-volume        # Create data dirs + symlink corpus/work/llms onto the volume
    ├── therobot-check-volume       # Volume presence/size sanity check
    ├── therobot-fetch-corpus       # Pull the training corpus onto the volume
    └── therobot-fetch-model        # Pull donor checkpoints onto the volume
```

## Environment contract (set in Dockerfile)

| Var | Meaning |
|-----|---------|
| `THEROBOT_ROOT` | Repo checkout in the container (`/workspace/therobotgguf`) |
| `THEROBOT_DATA` | Volume root (`/workspace/data`) — survives instance restarts |
| `HF_HOME` / `TRANSFORMERS_CACHE` | HF cache on the volume |
| `ROBOTGGUF_MODELS` | Donor checkpoints (symlinked to `llms/`) |
| `ROBOTGGUF_CORPUS` | Corpus files (symlinked to `convert/corpus/`) |
| `LLAMA_CPP_ROOT` / `LLAMA_CPP_BIN` | Fork source + built binaries (for R8 verify) |

Work in `/workspace/therobotgguf/convert`; `corpus/` and `work/` are symlinks
onto `THEROBOT_DATA` so pipeline outputs persist on the volume.
