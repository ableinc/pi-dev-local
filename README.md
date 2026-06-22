# Ollama & Llama.cpp — Crash Course

Much of the documentation is intended for `llama.cpp` due to its performance. While `Ollama` is a great choice, its opinionated and handles much of this for you under the hood. It does have configurable options, so use their documentation (if not referenced here) to tweak your settings. `llama.cpp` can use `Ollama` downloaded models. Refer the [llama-models.ini]("./llama-models.ini") and Pi.dev [models.json]("./models.json").

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

## Serving with `llama-server` - Quick Start

### Single model (simplest)

```bash
llama-server -m /path/to/model.gguf --port 9090
```

### Multi-model router (swap on demand)

Serve up to N models simultaneously. When a new model is requested and the limit is reached, the longest-idle model is offloaded first.

```bash
llama-server --models-preset llama-models.ini --models-max 2 --port 9090
```

```bash
llama-server --models-preset ./llama-models.ini --models-max 1 --flash-attn on --port 9090
```

---

## The INI Preset File

Many of the `llama-server` CLI flags can be set in an INI file and loaded with `--models-preset`. This is especially useful for router mode, where you can define multiple models with different settings in one place.

```ini
# Global defaults applied to every model section
[*]
ctx-size = 256000

# Per-model overrides — section name becomes the model ID used in API calls
[my-model:tag]
model    = /path/to/model.gguf
ctx-size = 131072
```

Per-model values override `[*]` defaults.

Pi.dev precedence note: when using Pi.dev, `models.json` `contextWindow` takes precedence over `ctx-size` in this INI file.

---

## Thinking Format (Pi.dev)

The best thinkingFormat to use for Gemma 4 models in vLLM and OpenAI-compatible engines is chat-template.

### Why chat-template is Best (Gemma 4)

- Official vLLM Requirements: [Gemma 4 models require](https://docs.vllm.ai/en/latest/features/reasoning_outputs/) enable_thinking: true passed inside their chat template keyword arguments (chat_template_kwargs) to trigger reasoning blocks.
- The "Reasoning Effort" Connection: When using OpenAI compatibility layers, choosing chat-template instructs the engine to map the request parameters directly to the model's native jinja template settings.

### How the Other Formats Compare

- qwen-chat-template: Designed specifically for Qwen models to inject their respective formatting. While similar in function, it isn't optimized for the specific <|think|> block handling unique to Gemma 4.
- reasoning_effort: This is standard for native OpenAI models (like the o-series) and Grok. Selecting this directly doesn't always automatically forward the necessary template kwargs to open-weight engines like vLLM.
- openrouter, together, deepseek: These are provider-specific format wrappers. If you are hosting the Gemma 4 model locally or via a private instance, choosing these will format your payload incorrectly.

| Format             | Best For Model                                                   | Best Use Case / Notes                                                                                           |
| ------------------ | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| chat-template      | Gemma 4, Llama 3.1/3.2/3.3/4 Instruct Cohere (north-mini) OpenAI | Passes reasoning tokens natively using the model's official jinja template variables.                           |
| qwen-chat-template | Qwen2.5-VL, Qwen2.5-Math                                         | Tailored specifically for Qwen models; handles specific block tags for Alibaba's vision and math architectures. |
| reasoning_effort   | OpenAI o1/o3-mini, Grok 2/3                                      | Best for proprietary reasoning models that use standard OpenAI API payload structures.                          |
| deepseek           | DeepSeek-R1, DeepSeek-V3                                         | Formats payloads specifically for DeepSeek's architecture, managing the output of the native reasoning tokens.  |
| openrouter         | Any model hosted via OpenRouter                                  | Only use if routing traffic explicitly through the OpenRouter proxy platform.                                   |
| together           | Any model hosted via Together AI                                 | Only use if deploying or querying through Together AI's API endpoints.                                          |
| zai                | Any model hosted via Zero AI / ZAI                               | Only use if using Zero AI / ZAI platform tools.                                                                 |
| qwen               | Qwen 1.5, Qwen 2 Base/Instruct                                   | Hardcoded formatting logic meant purely for legacy or base versions of the Alibaba Qwen model family.           |

> These guides are mostly for vLLM users, but the same principles apply to llama.cpp and Ollama when using their reasoning features. Always check your engine's documentation for any specific requirements around reasoning formats and template variables.

---

## VRAM Budget

This section will use real-world scenario of how to serve the `Qwen3.6-35B-A3B-UD-IQ4_XS` model with 24 GB VRAM between dual 3060 RTX (12 GB) GPUs.

| Component | VRAM |
| --------- | ---- |
| Model weights (UD-IQ4_XS) | ~18.2 GB |
| mmproj vision sidecar (F16) | ~0.9 GB |
| CUDA runtime + compute buffer | ~0.8 GB |
| **Available for KV cache** | **~4.1 GB** |

With Q8_0 KV cache, GQA + MoE sparse activation this is efficient enough to serve 32K context with some room to spare.

> Source: https://localllm.in/blog/llamacpp-vram-requirements-for-local-llms

### Target Context Size & Calculation

| Target Context | Required --ctx-size Flag | Exact Token Calculation |
|---|---|---|
| 16k | `--ctx-size 16384` | $16 \times 1024$ |
| 32k | `--ctx-size 32768` | $32 \times 1024$ |
| 64k | `--ctx-size 65536` | $64 \times 1024$ |
| 128k | `--ctx-size 131072` | $128 \times 1024$ |
| 256k | `--ctx-size 262144` | $256 \times 1024$ |
| 488k | `--ctx-size 500000` | $488.28 \times 1024$ (Exact Decimal Align) |
| 500k | `--ctx-size 524288` | $500 \times 1024$ |
| 1M | `--ctx-size 1048576` | $1024 \times 1024$ |

### Context Size vs KV Cache Tradeoff

Using `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` for example.

| `--ctx-size` | KV per slot (Q8_0) | Max slots | Notes |
| ------------ | ------------------ | --------- | ----- |
| 16K | ~687.87 MB | 5 | Fine for single-file coding |
| 32K | ~1.38 GB | 2-3 | Sweet spot for most agentic tasks (coding) |
| 64K | ~2.752 GB | 1-2 | Large repo context; tight on 24 GB VRAM. RAM offloading will occur. |
| 128K | ~5.5 GB | 0 | Out of Memory (OOM). RAM offloading will occur. Consider q4_0 for better fit. |

### mmproj Is *Required* (for multimodal models)

The mmproj vision sidecar is required for any model with a vision component (e.g. Qwen3.6-35B-A3B-UD-IQ4_XS) — it handles the image processing and projection into the model's embedding space. It typically uses around 0.9 GB of VRAM when running, which must be accounted for in the overall VRAM budget.

This means, when serving the model over `llama.cpp` (llama-server) you need to serve the main GGUF (model) and an mmproj GGUF (vision sidecar) together. Bleeding-edge builds of `llama.cpp` won't load the model without `--mmproj` specified.

### Recommended llama-server Command

This is strictly for the `Qwen3.6-35B-A3B-UD-IQ4_XS` model on dual 3060 RTX (12 GB VRAM each) with a 32K context window. Adjust `--ctx-size` and GPU offloading parameters as needed for different models or hardware.

For Pi.dev, keep `models.json` `contextWindow` aligned with this budget (32K for this Qwen profile), because Pi.dev request limits use `models.json` precedence.

```bash
llama-server \
  --model       ~/.cache/llama.cpp/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --mmproj      ~/.cache/llama.cpp/Qwen3.6-35B-mmproj-F16.gguf \
  # GPU offload
  --n-gpu-layers 99 \
  --split-mode  row \
  --tensor-split 1,1 \
  # Attention & memory
  --flash-attn  on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  # Context & batching
  --ctx-size    32768 \
  --batch-size  2048 \
  --ubatch-size 1024 \
  # Server / agentic config
  --parallel    1 \
  --cont-batching \
  --jinja \
  # Sampling (Qwen3 thinking-mode defaults)
  --temp        1.0 \
  --top-k       20 \
  --top-p       0.95 \
  --min-p       0.0 \
  --repeat-penalty 1.0 \
  --host 127.0.0.1 \
  --port 9090
```

> Note: These settings are seeing between 68-70 tokens per second. See below:

```bash
[51921] 6.32.171.440 I slot print_timing: id  0 | task 9769 | n_decoded =    100, tg =  70.29 t/s
[51921] 6.35.176.085 I slot print_timing: id  0 | task 9769 | n_decoded =    309, tg =  69.79 t/s
[51921] 6.38.184.391 I slot print_timing: id  0 | task 9769 | n_decoded =    517, tg =  69.53 t/s
[51921] 6.41.194.457 I slot print_timing: id  0 | task 9769 | n_decoded =    725, tg =  69.41 t/s
[51921] 6.44.204.894 I slot print_timing: id  0 | task 9769 | n_decoded =    933, tg =  69.34 t/s
[51921] 6.47.208.322 I slot print_timing: id  0 | task 9769 | n_decoded =   1139, tg =  69.20 t/s
[51921] 6.50.217.889 I slot print_timing: id  0 | task 9769 | n_decoded =   1345, tg =  69.08 t/s
[51921] 6.53.229.140 I slot print_timing: id  0 | task 9769 | n_decoded =   1551, tg =  68.99 t/s
[51921] 6.56.240.335 I slot print_timing: id  0 | task 9769 | n_decoded =   1756, tg =  68.89 t/s
```

**For Gemma 4-12B**

The Gemma4-12B model is only ~7 GB in size, which leaves ~16 GB of KV headroom on dual 3060 24 GB and ~5 GB of KV headroom on a single 3060 12 GB at 128K context. There's no benefit from tensor-splitting, because it add unnecessary PCIe overhead.

```bash
llama-server \
	--model       ~/.cache/llama.cpp//gemma-4-12b-it-UD-Q4_K_XL.gguf \
	--mmproj      ~/.cache/llama.cpp/gemma4-12B-mmproj-F16.gguf \
  # Single GPU — model fits in 12GB with 128K context
  --n-gpu-layers 99 \
  # Attention + KV
  --flash-attn   on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  # Context — 128K fits comfortably on single card at Q8
  --ctx-size     131072 \
  # Prefill batch — larger is better for long code context ingestion
  --batch-size   4096 \
  --ubatch-size  4096 \
  # Agentic config
  --parallel     1 \
  --cont-batching \
  --jinja \
  # Gemma 4 sampling calibration (NOT the same as Qwen3.6)
  --temp         1.0 \
  --top-p        0.95 \
  --top-k        64 \
  --host         127.0.0.1 \
  --port         9090
```

### Flag-by-Flag Reasoning

`--split-mode row`

The default `layer` split mode means that GPU 0 owns the first N layers and GPU 1 owns the rest - they pipeline sequentially. With `row` split, both GPUs work on *every layer simultaneously*, splitting the weight matrices along the row dimension. For MoE, where the expert weights dominate total size, row split keeps both cards active on every forward pass instead of one sitting idle while the other processes its layers. This is critical for maximizing throughput on VRAM-constrained hardware.

`--batch-size 2048` and `--ubatch-size 1024`

For this Qwen3.6 35B multimodal profile on dual 3060 GPUs, this is the safer baseline to avoid decode-time Vulkan memory spikes. In testing, `4096/4096` can load successfully but still fail on first decode with `vk::Device::allocateMemory: ErrorOutOfDeviceMemory`.

Tuning ladder:

1. Start at `2048/1024` (recommended baseline).
2. If decode-time OOM persists, lower to `1024/512`.
3. If still unstable, reduce `--ctx-size` (for example `24576`) or use `q4_0` KV cache.
4. If stable and you need higher prefill throughput, raise one knob at a time and retest.

#### Higher Context Profiles

If 32K context is too small for your coding workflow, use one of these profiles for `Qwen3.6-35B-A3B-UD-IQ4_XS` on dual 3060 GPUs.

| Profile | `--ctx-size` | `--cache-type-k/v` | `--batch-size` | `--ubatch-size` | Expected Throughput | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 64K Balanced | `65536` | `q4_0` | `1536` | `768` | ~55-58 tok/s | Best first step when moving up from 32K |
| 64K Safe | `65536` | `q4_0` | `1024` | `512` | ~45+ tok/s | Better stability margin if decode OOM appears |
| 96K Stretch | `98304` | `q4_0` | `1024` | `512` | ~30-40 tok/s | Only for larger context needs; latency variance increases |

Rollout sequence:

1. Start with 64K Balanced.
2. If you see decode-time OOM (`vk::Device::allocateMemory: ErrorOutOfDeviceMemory`), switch to 64K Safe.
3. If stable and you still need more room, test 96K Stretch.
4. If throughput falls below target, enable MTP and re-benchmark.

MTP note for this section:

- Add `--spec-type draft-mtp --spec-draft-n-max 2` to recover throughput at larger contexts.
- Keep `--parallel 1` when using MTP.

> Source: https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide

`--cache-type-k q8_0` and `--cache-type-v q8_0`

KV quantization is a game-changer for large context windows on limited VRAM. At Q8 its nearly lossless (vs FP16) and roughly havles KV cache VRAM. Critical here since we have only ~4.1 GB VRAM left to work with. If you want to push to 65K context, drop to q4_0 which havles it again with minimal accuracy impact. For 32K context, q8_0 is a sweet spot.

`--parallel 1`

For a single coding agent, one slot is optimal; multiple parallel slots cost extra KV cache - remember every 32K context is about 800 MB of VRAM at Q8. Be mindful of this setting when running multiple agents or allowing concurrent API requests. If you're using an agent harness like Pi.dev that can send multiple requests simultaneously, you may want to set this to 2 or 4 to allow some concurrency, but monitor VRAM usage closely.

`--jinja`

Required for the Qwen3.6 chat template (`thinkingFormat`; `chat-template`) to handle tool calling correctly. Without it, function call schemas may be malformed and your agent will fail on nested JSON responses or reasoning outputs. `Pi.dev` will print human-readable errors about malformed JSON.

`--temp 1.0`

For this VRAM profile and the recommended command above, use 1.0 as the baseline. If you explicitly disable thinking mode for deterministic, non-reasoning output, you can test 0.6 per request and compare quality for your workload.

### MTP GGUF Exists

Unsloth published Multi-Token Prediction GGUFs for the 35B-A3B, which show ~1.15-~1.25x speedup on MoE models in `llama.cpp` with no accuracy loss. It's a smaller gain than the 1.4-2x you'd get on the dense 27B, but if you want to squeeze extra tokens per-second from the same hardware, it's worth trying. You'll need to add `--spec-type draft-mtp --spec-draft-n-max 2`, but MTP currently requires `--parallel 1` since the draft model's KV cache isn't shared across slots yet.

> Source: https://unsloth.ai/docs/models/qwen3.6

---

## How To Determine My Optimal Settings?

### The 5-Step Lookup Process

1. `Identify your model's architecture and size` (e.g. Qwen3.6-35B-A3B-UD-IQ4_XS).

Always your first stop: The `Files` tab on HuggingFace model repo or Ollama model card. This is the single most important number because if defines your VRAM ceiling.

2. `Check for mmproj (Multimodal Projector)`

Look at the model card description for words like "multimodal", "vision", "image", "audio", or "mmproj". If you see those, then it needs an mmproj GGUF. Remember to check the size to account for VRAM usage.

3. `Dense or MoE` (Determines Split Mode & Batch Size)

Model card architecture section should contain all this information. **Dense** requires `--split-mode layer`, which is the default pipeline between GPUs. Models like `gemma 4-12B` are Dense. **MoE** (Mixture of Experts) models require `--split-mode row` to keep both GPUs active on every layer. Start with conservative batching (`2048/1024`) and increase gradually only after stability checks.

4. `Read the Model's Own Sampling Recommendations`

Every model family has calibrated defaults. They differ signficantly between models designed for reasoning vs non-reasoning. For `Gemma 4 QAT` variants, the recommended sampling is `temperature 1.0, top_p 0.95, top_k 64`. For `Qwen3.6` variants, the recommended sampling is `temperature 0.6, top_p 0.95, top_k 20`. Using the wrong defaults measurably degrades output quality.

> Source: https://lushbinary.com/blog/gemma-4-qat-self-hosting-guide-ollama-llama-cpp-vllm/

5. `Do the VRAM Math`

Using the `Qwen3.6-35B-A3B-UD-IQ4_XS` as an example.

```block
Total VRAM  = 24.0 GB   (dual 3060)
- Model     =  18.2 GB
- mmproj    =  0.9 GB
- Runtime   =  0.8 GB   (CUDA buffers, ~constant)
─────────────────────────
KV headroom = 4.1 GB
```

Then calculate max context from KV headroom:

$$\text{KV Cache Size (Bytes)} = 2 \times \text{n-layers} \times \text{n-kv-heads} \times \text{head-dim} \times \text{context-length} \times \text{bytes-per-element}$$

* **2**: Accounts for storing both the Key and the Value states.
* **n_layers**: Read from block_count.
* **n_kv_heads**: Read from attention.head_count_kv.
* **head_dim**: Read from attention.key_length (or calculated as $\frac{\text{embedding-length}}{\text{attention-head-count}}$).
* **context_length**: Your `--ctx-size` runtime parameter (e.g., 32768 tokens).
* **bytes_per_element**: The precision data type of your cache. By default, llama.cpp uses 16-bit float (F16). See [Cache Type Conversion]("#cache-type-conversion") table.

#### Cache Type Conversion

| `--cache-type-k` | `bytes_per_element` | Bits |
| --- | --- | --- |
| F16 / B16 | 2.0 | 16 |
| q8_0 | 1.0 | 8 |
| q4_0 | 0.5 | 4 |
| iq4_nl | ~0.51 | ~4.1 |
| q5_1 | 6 | 48 |
| q4_1 | 5 | 40 |

#### Using the `llama-gguf-dump` tool (recommended)

You get `n_layers`, `n_kv_heads`, and `head_dim` using the llama dump script.

| Formula | Llama Key | Description |
| --- | --- | --- |
| n_layers | *.block_count | Number of transformer blocks |
| n_kv_heads | *.attention.head_count_kv | Number of Key/Value (KV) heads for grouped query attention |
| head_dim | *.attention.key_length or *.attention.head_count | Dimension size of a single attention head |

```bash
./llama-gguf-dump ~/.cache/llama.cpp/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf | grep -E "block_count|attention.head_count_kv|attention.key_length"
```

**Output**

```bash
INFO:gguf-dump:* Loading: /home/onyx/.cache/llama.cpp/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
     21: UINT32     |        1 | qwen35moe.block_count = 41
     25: UINT32     |        1 | qwen35moe.attention.head_count_kv = 2
     31: UINT32     |        1 | qwen35moe.attention.key_length = 256
```

KV Cache Size = 2 * 41 * 2 * 256 * 32768 * 1.0 = `1,375,731,712 (Bytes)` = `1,375.74 (MB)` = `1.3 (GB)`

The KV cache fits within my `~4.1 GB` of headroom.

You can download the `llama-gguf-dump` tool here: https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/scripts/gguf_dump.py


> Note: Adjust `--ctx-size` and relaunch until the allocation fits your headroom. See the section below to calculate for Google's Gemma 4 models.

#### Using the `llama-gguf-dump` tool for Gemma 4 models

Google's Gemma 4 models use Hybrid Attention Mechanism and several other techniques to shrink their model size and get the best possible performance on limited hardware. Because of this, the calculations in the previous section cannot be used to determine its KV Cache Size. We can still leverage the gguf dump tool, but we'll need to calculate for the `Local` and `Global` context.

**TBA**


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
| `--cache-prompt`                   | on              | Enable prompt caching. Reuse KV cache across requests with shared prefix. Keep enabled for multi-request workloads.                                                             |
| `--cache-reuse N`                  | `0`                  | Min chunk size (tokens) to attempt KV-shift cache reuse. Set higher (e.g. `512`) when prompts have large shared prefixes.                                                       |
| `-sps, --slot-prompt-similarity N` | `0.10`               | How much a new prompt must match an existing slot to reuse it. Increase (e.g. `0.30`) to be more selective; decrease for more reuse.                                            |
| `-fitc, --fit-ctx N`               | `4096`               | Minimum ctx size `--fit` can shrink to. Raise to prevent aggressive shrinking on large context workloads.                                                                       |
| `-fitt, --fit-target MiB`          | `1024`               | Target VRAM margin per GPU. Increase to give KV cache more breathing room.                                                                                                      |

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
