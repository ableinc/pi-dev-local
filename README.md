# Ollama & Llama.cpp — Crash Course

## Model Locations

| Backend    | Path |
|------------|------|
| **Ollama** | `/usr/share/ollama/.ollama/models/blobs/` |
| **Llama**  | `~/.cache/llama.cpp/` |

> To resolve an Ollama blob path: check the manifest at `/usr/share/ollama/.ollama/models/manifests/` and match the `sha256` hash with `mediaType: application/vnd.ollama.image.model`.

---

## Serving with Ollama

```bash
ollama serve
ollama run <model>
```

---

## Serving with `llama-server`

### Single model (simplest)

```bash
llama-server -m /path/to/model.gguf --port 9090
```

### Multi-model router (swap on demand)

Serve up to N models simultaneously. When a new model is requested and the limit is reached, the longest-idle model is offloaded first.

```bash
llama-server --models-preset llama-models.ini --models-max 2
```

- `--models-max 0` = unlimited (load everything at startup)
- `--no-models-autoload` = don't load until first request

---

## The INI Preset File

Keys use **hyphens**, matching the CLI flags (e.g. `--ctx-size` → `ctx-size`).

```ini
# Global defaults applied to every model section
[*]
ctx-size = 256000
port     = 9090
host     = 127.0.0.1

# Per-model overrides — section name becomes the model ID used in API calls
[my-model:tag]
model    = /path/to/model.gguf
ctx-size = 131072
alias    = my-model,my-model-alias
```

Any CLI flag can be used as an INI key (strip `--`, keep hyphens). Per-model values override `[*]` defaults.

---

## Essential Flags

### Network

| Flag | Default | Notes |
|------|---------|-------|
| `--host HOST` | `127.0.0.1` | Use `0.0.0.0` to expose on LAN |
| `--port PORT` | `8080` | |
| `--api-key KEY` | none | Comma-separate multiple keys |
| `--timeout N` | `3600` | Read/write timeout in seconds |

### Context & Memory

| Flag | Default | Notes |
|------|---------|-------|
| `-c, --ctx-size N` | `0` (from model) | Total KV context window in tokens |
| `-n, --predict N` | `-1` (∞) | Max tokens to generate per request |
| `--mlock` | off | Pin model in RAM, prevent swap |
| `--mmap / --no-mmap` | on | Memory-map weights file |

### GPU Offloading

| Flag | Default | Notes |
|------|---------|-------|
| `-ngl, --gpu-layers N` | `auto` | Layers to put in VRAM; `all` = full offload |
| `-dev, --device dev1,dev2` | auto | Specific GPU devices (see `--list-devices`) |
| `-sm, --split-mode` | `layer` | `none` / `layer` / `row` / `tensor` |
| `-ts, --tensor-split 3,1` | — | Proportion of model per GPU |
| `--kv-offload / --no-kv-offload` | on | Offload KV cache to VRAM |
| `--fit [on\|off]` | `on` | Auto-adjust params to fit in device memory |
| `--fit-ctx N` | `4096` | Minimum ctx `--fit` is allowed to shrink to |

### Performance

| Flag | Default | Notes |
|------|---------|-------|
| `-b, --batch-size N` | `2048` | Logical batch size (prompt processing) |
| `-ub, --ubatch-size N` | `512` | Physical micro-batch size |
| `-fa, --flash-attn` | `auto` | Flash Attention (`on`/`off`/`auto`) |
| `-np, --parallel N` | `-1` (auto) | Number of concurrent request slots |
| `--cont-batching` | on | Dynamic batching across slots |
| `--cache-type-k TYPE` | `f16` | KV cache K dtype: `f16`, `q8_0`, `q4_0`, etc. |
| `--cache-type-v TYPE` | `f16` | KV cache V dtype (same options as K) |
| `--threads N` | auto | CPU threads for generation |
| `--threads-http N` | `-1` (auto) | Threads for HTTP request handling |

### Prompt Caching

| Flag | Default | Notes |
|------|---------|-------|
| `--cache-prompt` | on | Reuse KV cache across requests with shared prefix |
| `--cache-reuse N` | `0` | Min chunk size (tokens) to attempt KV-shift reuse |
| `--cache-ram N` | `8192 MiB` | Max RAM for the prompt cache (`-1` = no limit) |
| `--ctx-checkpoints N` | `32` | Context checkpoints per slot (enables rollback) |

### Model Identity & Routing

| Flag | Notes |
|------|-------|
| `-a, --alias name1,name2` | Names this model responds to in API calls |
| `--tags tag1,tag2` | Informational tags (not used for routing) |

### Reasoning / Thinking Models

| Flag | Default | Notes |
|------|---------|-------|
| `--reasoning [on\|off\|auto]` | `auto` | Enable chain-of-thought thinking |
| `--reasoning-format FORMAT` | `auto` | `none` / `deepseek` / `deepseek-legacy` |
| `--reasoning-budget N` | `-1` (∞) | Token cap for `<think>` block; `0` = skip thinking |

### Logging

| Flag | Notes |
|------|-------|
| `-v, --verbose` | Log everything |
| `-lv, --log-verbosity N` | `0`=generic `1`=error `2`=warn `3`=info `4`=trace `5`=debug |
| `--log-file FNAME` | Write logs to file |
| `--log-colors` | Coloured output (default: auto-detect TTY) |

---

## API Endpoints

The server exposes an **OpenAI-compatible** REST API on `http://host:port`.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming supported) |
| `/v1/completions` | POST | Raw text completions |
| `/v1/embeddings` | POST | Embeddings (requires `--embeddings`) |
| `/v1/models` | GET | List loaded models |
| `/health` | GET | Server health (`{"status":"ok"}`) |
| `/slots` | GET | Per-slot KV cache status |
| `/metrics` | GET | Prometheus metrics (requires `--metrics`) |
| `/props` | POST | Change properties at runtime (requires `--props`) |
| `/lora-adapters` | POST | Hot-swap LoRA adapters |

### Quick smoke-test

```bash
curl http://127.0.0.1:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6:35b-IQ4_XS",
    "messages": [{"role":"user","content":"hello"}],
    "stream": false
  }'
```

---

## Common Recipes

### Full GPU offload, large context

```bash
llama-server -m model.gguf \
  --gpu-layers all \
  --ctx-size 131072 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --port 9090
```

### CPU-only with mlock (prevent swap)

```bash
llama-server -m model.gguf \
  --gpu-layers 0 \
  --mlock \
  --threads 16 \
  --ctx-size 32768 \
  --port 9090
```

### Multi-GPU split

```bash
llama-server -m model.gguf \
  --split-mode layer \
  --tensor-split 3,1 \
  --gpu-layers all \
  --port 9090
```

### Router with preset file

```bash
llama-server \
  --models-preset ~/.pi/agent/llama-models.ini \
  --models-max 2 \
  --host 127.0.0.1
```

### Thinking model with capped reasoning budget

```bash
llama-server -m qwq.gguf \
  --reasoning on \
  --reasoning-budget 2048 \
  --ctx-size 32768 \
  --port 9090
```

---

## Environment Variables

Every flag has a corresponding `LLAMA_ARG_*` env var (shown in `--help`). Useful for containers:

```bash
export LLAMA_ARG_MODEL=/models/qwen.gguf
export LLAMA_ARG_CTX_SIZE=131072
export LLAMA_ARG_N_GPU_LAYERS=99
export LLAMA_ARG_HOST=0.0.0.0
export LLAMA_ARG_PORT=9090
export LLAMA_API_KEY=sk-secret
llama-server
```

---

## Unloading Models at Runtime (Router Mode)

In router mode the server stays running; models are loaded/unloaded on demand via HTTP.
There is **no single "unload everything" endpoint** — the API is intentionally per-model.

### Router-only management endpoints

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/models` | GET | — | List all known models and their `status` (`loaded` / `unloaded`) |
| `/models/load` | POST | `{"model":"<id>"}` | Load a specific model |
| `/models/unload` | POST | `{"model":"<id>"}` | Unload a specific model, freeing its VRAM/RAM |

> `/v1/models` is the OAI-compat list endpoint (names only). `/models` is the router management endpoint — they are not the same.

### 1. See what's loaded

```bash
curl -s http://127.0.0.1:9090/models | jq
```

Example response:

```json
{
  "data": [
    { "id": "qwen3.6:35b-IQ4_XS", "status": "loaded" },
    { "id": "qwen3-coder:30b",    "status": "unloaded" }
  ]
}
```

### 2. Unload one model

```bash
curl -s -X POST http://127.0.0.1:9090/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.6:35b-IQ4_XS"}'
```

Always use the exact `id` string returned by `/models` — not the filename or alias.

### 3. Unload all loaded models (shell one-liner)

There is no `unload-all` endpoint, so loop over what's loaded:

```bash
curl -s http://127.0.0.1:9090/models \
  | jq -r '.data[] | select(.status == "loaded") | .id' \
  | while IFS= read -r model; do
      echo "Unloading: $model"
      curl -s -X POST http://127.0.0.1:9090/models/unload \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\"}"
    done
```

### 4. Reusable script

Save as `unload-all-models.sh` and `chmod +x` it:

```bash
#!/usr/bin/env bash
set -euo pipefail

SERVER="${LLAMA_SERVER_URL:-http://127.0.0.1:9090}"

loaded=$(curl -fsS "$SERVER/models" \
  | jq -r '.data[] | select(.status == "loaded") | .id')

if [ -z "$loaded" ]; then
  echo "No models loaded."
  exit 0
fi

printf '%s\n' "$loaded" | while IFS= read -r model; do
  [ -z "$model" ] && continue
  echo "Unloading: $model"
  # Use jq to safely build the body — handles unusual chars in model IDs
  body=$(jq -n --arg m "$model" '{model: $m}')
  curl -fsS -X POST "$SERVER/models/unload" \
    -H "Content-Type: application/json" \
    -d "$body"
done

echo "\nFinal state:"
curl -fsS "$SERVER/models" | jq '.data[] | {id, status}'
```

Run against a custom host:

```bash
LLAMA_SERVER_URL=http://192.168.1.50:9090 ./unload-all-models.sh
```

### Gotcha: models reload automatically

Router mode has `--models-autoload` **on** by default. If any client sends a request for an unloaded model, the server will load it again immediately. To keep VRAM free after unloading:

- Stop all client traffic first (agents, Open WebUI, scripts)
- Or restart the server with `--no-models-autoload` to disable on-demand loading entirely

---

## INI Key Reference (common ones)

INI keys are CLI flags with `--` stripped — **hyphens, not underscores**.

```ini
[*]
host           = 127.0.0.1
port           = 9090
ctx-size       = 32768
gpu-layers     = all
flash-attn     = on
cache-type-k   = q8_0
cache-type-v   = q8_0
parallel       = 4
cont-batching  = true
```
