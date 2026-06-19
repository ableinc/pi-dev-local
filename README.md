# Ollama & Llama.cpp

## Model Location

- `Ollama`: `/usr/share/ollama/.ollama/models/blobs`
  - Find their manifests at: `/usr/share/ollama/.ollama/models/manifests/` and match their sha256 hash with mediaType: `application/vnd.ollama.image.model`

- `Lllama`: `~/.cache/llama.cpp/`

## Launch

### Ollama

Launch as usual using Ollama CLI

### Llama.cpp

```bash
llama-server --models-preset /path/to/models.ini --models-max 2
```

**Serve only 1 model**

```bash
llama-server -m Qwen3.6-35B-A3B-UD-IQ4_XS.gguf --port 9090
```


> This serves a max of 2 models on server. If a new model is requested the idle model is offloaded to load the new one
