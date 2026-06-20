# Ollama & Llama.cpp — Crash Course

## Model Locations

| Backend    | Path                                      |
| ---------- | ----------------------------------------- |
| **Ollama** | `/usr/share/ollama/.ollama/models/blobs/` |
| **Llama**  | `~/.cache/llama.cpp/`                     |

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
llama-server --models-preset llama-models.ini --models-max 2 --port 9090
```

---

## The INI Preset File

```ini
# Global defaults applied to every model section
[*]
ctx-size = 256000

# Per-model overrides — section name becomes the model ID used in API calls
[my-model:tag]
model    = /path/to/model.gguf
ctx-size = 131072
alias    = my-model,my-model-alias
```

Per-model values override `[*]` defaults.

---

## Thinking Format (Pi.dev)

The best thinkingFormat to use for Gemma 4 models in vLLM and OpenAI-compatible engines is chat-template.

### Why chat-template is Best

* Official vLLM Requirements: [Gemma 4 models require](https://docs.vllm.ai/en/latest/features/reasoning_outputs/) enable_thinking: true passed inside their chat template keyword arguments (chat_template_kwargs) to trigger reasoning blocks.
* The "Reasoning Effort" Connection: When using OpenAI compatibility layers, choosing chat-template instructs the engine to map the request parameters directly to the model's native jinja template settings.

### How the Other Formats Compare

* qwen-chat-template: Designed specifically for Qwen models to inject their respective formatting. While similar in function, it isn't optimized for the specific <|think|> block handling unique to Gemma 4.
* reasoning_effort: This is standard for native OpenAI models (like the o-series) and Grok. Selecting this directly doesn't always automatically forward the necessary template kwargs to open-weight engines like vLLM.
* openrouter, together, deepseek: These are provider-specific format wrappers. If you are hosting the Gemma 4 model locally or via a private instance, choosing these will format your payload incorrectly.


| Format | Best For Model | Best Use Case / Notes |
|---|---|---|
| chat-template | Gemma 4, Llama 3.1/3.2/3.3/4 Instruct Cohere (north-mini) OpenAI | Passes reasoning tokens natively using the model's official jinja template variables. |
| qwen-chat-template | Qwen2.5-VL, Qwen2.5-Math | Tailored specifically for Qwen models; handles specific block tags for Alibaba's vision and math architectures. |
| reasoning_effort | OpenAI o1/o3-mini, Grok 2/3 | Best for proprietary reasoning models that use standard OpenAI API payload structures. |
| deepseek | DeepSeek-R1, DeepSeek-V3 | Formats payloads specifically for DeepSeek's architecture, managing the output of the native reasoning tokens. |
| openrouter | Any model hosted via OpenRouter | Only use if routing traffic explicitly through the OpenRouter proxy platform. |
| together | Any model hosted via Together AI | Only use if deploying or querying through Together AI's API endpoints. |
| zai | Any model hosted via Zero AI / ZAI | Only use if using Zero AI / ZAI platform tools. |
| qwen | Qwen 1.5, Qwen 2 Base/Instruct | Hardcoded formatting logic meant purely for legacy or base versions of the Alibaba Qwen model family. |

> These guides are mostly for vLLM users, but the same principles apply to llama.cpp and Ollama when using their reasoning features. Always check your engine's documentation for any specific requirements around reasoning formats and template variables.

---

## Essential Flags

### Network

| Flag            | Default     | Notes                          |
| --------------- | ----------- | ------------------------------ |
| `--host HOST`   | `127.0.0.1` | Use `0.0.0.0` to expose on LAN |
| `--port PORT`   | `8080`      |                                |
| `--api-key KEY` | none        | Comma-separate multiple keys   |
| `--timeout N`   | `3600`      | Read/write timeout in seconds  |

### Context & Memory

| Flag                 | Default          | Notes                              |
| -------------------- | ---------------- | ---------------------------------- |
| `-c, --ctx-size N`   | `0` (from model) | Total KV context window in tokens  |
| `-n, --predict N`    | `-1` (∞)         | Max tokens to generate per request |
| `--mlock`            | off              | Pin model in RAM, prevent swap     |
| `--mmap / --no-mmap` | on               | Memory-map weights file            |

### GPU Offloading

| Flag                             | Default | Notes                                       |
| -------------------------------- | ------- | ------------------------------------------- |
| `-ngl, --gpu-layers N`           | `auto`  | Layers to put in VRAM; `all` = full offload |
| `-dev, --device dev1,dev2`       | auto    | Specific GPU devices (see `--list-devices`) |
| `-sm, --split-mode`              | `layer` | `none` / `layer` / `row` / `tensor`         |
| `-ts, --tensor-split 3,1`        | —       | Proportion of model per GPU                 |
| `--kv-offload / --no-kv-offload` | on      | Offload KV cache to VRAM                    |
| `--fit [on\|off]`                | `on`    | Auto-adjust params to fit in device memory  |
| `--fit-ctx N`                    | `4096`  | Minimum ctx `--fit` is allowed to shrink to |

### Performance

| Flag                     | Default     | Notes                                                                                    |
| ------------------------ | ----------- | ---------------------------------------------------------------------------------------- |
| `-b, --batch-size N`     | `2048`      | Logical batch size (prompt processing)                                                   |
| `-ub, --ubatch-size N`   | `512`       | Physical micro-batch size                                                                |
| `-fa, --flash-attn`      | `auto`      | Flash Attention (`on`/`off`/`auto`)                                                      |
| `-np, --parallel N`      | `-1` (auto) | Number of concurrent request slots                                                       |
| `--cont-batching`        | on          | Dynamic batching across slots                                                            |
| `--cache-type-k TYPE`    | `f16`       | KV cache K dtype: `f16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`, `bf16`, `f32` |
| `--cache-type-v TYPE`    | `f16`       | KV cache V dtype (same options as K)                                                     |
| `--threads N`            | auto        | CPU threads for generation                                                               |
| `--threads-http N`       | `-1` (auto) | Threads for HTTP request handling                                                        |
| `-tb, --threads-batch N` | auto        | Threads for batch/prompt processing                                                      |
| `--poll N`               | `50`        | Polling level for work wait (0–100)                                                      |
| `--prio N`               | `0`         | Process/thread priority: low(-1), normal(0), medium(1), high(2), realtime(3)             |
| `--prio-batch N`         | `0`         | Batch processing thread priority                                                         |

### Large Context Window Performance

These flags are especially useful when running with large `--ctx-size` values (e.g. 64K–256K+).

| Flag                               | Default              | Notes                                                                                                                                                                           |
| ---------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-ctk, --cache-type-k TYPE`        | `f16`                | **KV cache K quantization** — reduces KV cache memory by up to **16×** vs `f16` with `q4_0` or `q8_0`. Use `q4_0`/`q8_0`/`q4_1`/`iq4_nl`/`q5_0`/`q5_1` for large contexts.      |
| `-ctv, --cache-type-v TYPE`        | `f16`                | KV cache V quantization (same options as K). V quantization typically has less accuracy impact than K.                                                                          |
| `-ctkd, --cache-type-k-draft TYPE` | `f16`                | KV cache K dtype for the **draft model** in speculative decoding. Use quantized types to reduce draft model KV overhead.                                                        |
| `-ctvd, --cache-type-v-draft TYPE` | `f16`                | KV cache V dtype for the **draft model**. Same options as K.                                                                                                                    |
| `-kvu, --kv-unified`               | auto                 | Use a **single unified KV buffer** shared across all sequences. Reduces per-slot overhead and fragmentation when many slots share similar context.                              |
| `--cache-idle-slots`               | enabled              | Save idle slots to the prompt cache on new task, and clear them when using unified KV. Reduces memory pressure when swapping between models/contexts.                           |
| `-cram, --cache-ram N`             | `8192` MiB           | Maximum **RAM** cache size in MiB. Increase for large shared-prompt workloads; `-1` = no limit.                                                                                 |
| `-ctxcp, --ctx-checkpoints N`      | `32`                 | Max context checkpoints per slot. Enables rollback/rewind of context. Increase for deeper history tracking.                                                                     |
| `-cms, --checkpoint-min-step N`    | `256`                | Min spacing (tokens) between context checkpoints. Higher values = less memory overhead for checkpoints.                                                                         |
| `--swa-full`                       | off                  | Use full-size SWA (sliding window attention) cache. Enable when using models with full attention (not SWA).                                                                     |
| `--no-host`                        | off                  | Bypass host buffer, allowing extra buffers to be used. May reduce memory pressure on GPU-constrained systems.                                                                   |
| `-dio, --direct-io`                | off                  | Use DirectIO for file I/O. Helps on Linux with large models and large context windows to avoid page cache pressure.                                                             |
| `--numa TYPE`                      | off                  | NUMA optimizations: `distribute` (spread across nodes), `isolate` (keep on start node), or `numactl` (use numactl map). Critical for multi-socket servers with large KV caches. |
| `-C, --cpu-mask M`                 | ""                   | Hex CPU affinity mask for generation threads. Pin to specific cores for cache locality with large KV caches.                                                                    |
| `-Cr, --cpu-range lo-hi`           | ""                   | CPU core range for affinity (complements `--cpu-mask`).                                                                                                                         |
| `--cpu-strict`                     | `0`                  | Strict CPU placement. Use with `--numa` for guaranteed NUMA-local allocation.                                                                                                   |
| `-Crb, --cpu-range-batch lo-hi`    | same as --cpu-mask   | CPU core range for batch/prompt processing threads. Pin to different cores from generation.                                                                                     |
| `-Cb, --cpu-mask-batch M`          | same as --cpu-mask   | Hex CPU affinity mask for batch threads.                                                                                                                                        |
| `--cpu-strict-batch`               | same as --cpu-strict | Strict CPU placement for batch threads.                                                                                                                                         |
| `--prio-batch N`                   | `0`                  | Priority for batch threads (same scale as `--prio`).                                                                                                                            |
| `--poll-batch N`                   | same as --poll       | Polling level for batch thread work wait (0 = no polling). Reduces CPU usage when idle.                                                                                         |
| `--context-shift`                  | off                  | Enable context shift on infinite generation. Slides context forward instead of truncating. Useful for streaming/continuous generation.                                          |
| `--cache-prompt`                   | enabled              | Enable prompt caching. Reuse KV cache across requests with shared prefix. Keep enabled for multi-request workloads.                                                             |
| `--cache-reuse N`                  | `0`                  | Min chunk size (tokens) to attempt KV-shift cache reuse. Set higher (e.g. `512`) when prompts have large shared prefixes.                                                       |
| `-sps, --slot-prompt-similarity N` | `0.10`               | How much a new prompt must match an existing slot to reuse it. Increase (e.g. `0.30`) to be more selective; decrease for more reuse.                                            |
| `-fitc, --fit-ctx N`               | `4096`               | Minimum ctx size `--fit` can shrink to. Raise to prevent aggressive shrinking on large context workloads.                                                                       |
| `-fitt, --fit-target MiB`          | `1024`               | Target VRAM margin per GPU. Increase to give KV cache more breathing room.                                                                                                      |

> **Quick reference — memory impact of KV cache types** (per token, per head-dim-128 model, per layer):
>
> | Type | Memory per (K,V) pair | 128K context × 32 layers (8×7B) |
> |--------|----------------------|----------------------------------|
> | `f16` | baseline × 2 | ~16 GB |
> | `bf16` | baseline × 2 | ~16 GB |
> | `q8_0` | baseline × 0.5 | ~8 GB |
> | `q5_1` | baseline × 0.3125 | ~5 GB |
> | `q4_0` | baseline × 0.25 | ~4 GB |
> | `q4_1` | baseline × 0.28125 | ~4.5 GB |
> | `iq4_nl` | baseline × 0.25 | ~4 GB |

### GPU Offloading

| Flag                   | Default | Notes                                       |
| ---------------------- | ------- | ------------------------------------------- |
| `-ngl, --gpu-layers N` | `auto`  | Layers to put in VRAM; `all` = full offload |

### Prompt Caching

| Flag                  | Default    | Notes                                             |
| --------------------- | ---------- | ------------------------------------------------- |
| `--cache-prompt`      | on         | Reuse KV cache across requests with shared prefix |
| `--cache-reuse N`     | `0`        | Min chunk size (tokens) to attempt KV-shift reuse |
| `--cache-ram N`       | `8192 MiB` | Max RAM for the prompt cache (`-1` = no limit)    |
| `--ctx-checkpoints N` | `32`       | Context checkpoints per slot (enables rollback)   |

### Model Identity & Routing

| Flag                      | Notes                                     |
| ------------------------- | ----------------------------------------- |
| `-a, --alias name1,name2` | Names this model responds to in API calls |
| `--tags tag1,tag2`        | Informational tags (not used for routing) |

### Reasoning / Thinking Models

| Flag                          | Default  | Notes                                              |
| ----------------------------- | -------- | -------------------------------------------------- |
| `--reasoning [on\|off\|auto]` | `auto`   | Enable chain-of-thought thinking                   |
| `--reasoning-format FORMAT`   | `auto`   | `none` / `deepseek` / `deepseek-legacy`            |
| `--reasoning-budget N`        | `-1` (∞) | Token cap for `<think>` block; `0` = skip thinking |

### Logging

| Flag                     | Notes                                                       |
| ------------------------ | ----------------------------------------------------------- |
| `-v, --verbose`          | Log everything                                              |
| `-lv, --log-verbosity N` | `0`=generic `1`=error `2`=warn `3`=info `4`=trace `5`=debug |
| `--log-file FNAME`       | Write logs to file                                          |
| `--log-colors`           | Coloured output (default: auto-detect TTY)                  |

---

## API Endpoints

The server exposes an **OpenAI-compatible** REST API on `http://host:port`.

| Endpoint               | Method | Description                                       |
| ---------------------- | ------ | ------------------------------------------------- |
| `/v1/chat/completions` | POST   | Chat completions (streaming supported)            |
| `/v1/completions`      | POST   | Raw text completions                              |
| `/v1/embeddings`       | POST   | Embeddings (requires `--embeddings`)              |
| `/v1/models`           | GET    | List loaded models                                |
| `/health`              | GET    | Server health (`{"status":"ok"}`)                 |
| `/slots`               | GET    | Per-slot KV cache status                          |
| `/metrics`             | GET    | Prometheus metrics (requires `--metrics`)         |
| `/props`               | POST   | Change properties at runtime (requires `--props`) |
| `/lora-adapters`       | POST   | Hot-swap LoRA adapters                            |

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

| Endpoint         | Method | Body               | Description                                                      |
| ---------------- | ------ | ------------------ | ---------------------------------------------------------------- |
| `/models`        | GET    | —                  | List all known models and their `status` (`loaded` / `unloaded`) |
| `/models/load`   | POST   | `{"model":"<id>"}` | Load a specific model                                            |
| `/models/unload` | POST   | `{"model":"<id>"}` | Unload a specific model, freeing its VRAM/RAM                    |

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
    { "id": "qwen3-coder:30b", "status": "unloaded" }
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
ctx-size       = 32768
gpu-layers     = all
flash-attn     = on
cache-type-k   = q8_0
cache-type-v   = q8_0
parallel       = 4
cont-batching  = true
```

---

## Resources

| Resource                                          | Link                                                                       |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| **llama.cpp server README** (full flag reference) | <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>   |
| **llama.cpp CLI README** (llama-cli flags)        | <https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md>      |
| **Function calling docs**                         | <https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md> |
| **Multimodal docs**                               | <https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md>       |
| **NUMA optimization guide**                       | <https://github.com/ggml-org/llama.cpp/issues/1437>                          |
| **Server changelog**                              | <https://github.com/ggml-org/llama.cpp/issues/9291>                          |
| **KV cache checkpoints (SWA)**                    | <https://github.com/ggml-org/llama.cpp/pull/15293>                           |
| **Prompt cache / cache-ram**                      | <https://github.com/ggml-org/llama.cpp/pull/16391>                           |
| **Full-size SWA cache**                           | <https://github.com/ggml-org/llama.cpp/pull/13194>                           |
| **Adaptive-p sampler**                            | <https://github.com/ggml-org/llama.cpp/pull/17927>                           |
| **KV cache shifting (cache-reuse)**               | <https://ggml.ai/f0.png>                                                     |
| **Docker image**                                  | <https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp>             |
| **llama.cpp docs (general)**                      | <https://github.com/ggml-org/llama.cpp/tree/master/docs>                     |
| **llama.cpp wiki (templates, quantization)**      | <https://github.com/ggml-org/llama.cpp/wiki>                                 |
