#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# to_gguf.sh — convert a HuggingFace safetensors model dir to GGUF
#
# Usage:
#   ./to_gguf.sh <model_dir> [output_dir] [quant_type]
#
# Arguments:
#   model_dir   directory containing *.safetensors + config.json
#   output_dir  where to write GGUF output   (default: <model_dir>)
#   quant_type  llama-quantize type string    (default: no quantization)
#               e.g. Q4_K_M, Q4_K_XL, Q8_0, IQ4_XS, IQ3_M, F16
#
# Examples:
#   ./to_gguf.sh ~/models/Qwen2.5-32B-Instruct
#   ./to_gguf.sh ~/models/Qwen2.5-32B-Instruct ~/gguf Q4_K_M
# ---------------------------------------------------------------------------

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

die()  { echo -e "${RED}[ERR]${NC}  $*" >&2; exit 1; }
info() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

[[ $# -lt 1 ]] && {
    echo "Usage: $0 <model_dir> [output_dir] [quant_type]"
    exit 1
}

MODEL_DIR="$(realpath "$1")"
OUTPUT_DIR="${2:-$MODEL_DIR}"
QUANT_TYPE="${3:-}"

[[ -d "$MODEL_DIR" ]]            || die "model_dir not found: $MODEL_DIR"
ls "$MODEL_DIR"/*.safetensors &>/dev/null || die "No .safetensors files in $MODEL_DIR"
[[ -f "$MODEL_DIR/config.json" ]] || die "config.json missing from $MODEL_DIR"

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath "$OUTPUT_DIR")"

# ---------------------------------------------------------------------------
# Locate convert_hf_to_gguf.py
# ---------------------------------------------------------------------------

find_convert_script() {
    local candidates=(
        # explicit env override
        "${LLAMA_CPP_DIR:-}/convert_hf_to_gguf.py"

        # common build/install locations
        "$HOME/llama.cpp/convert_hf_to_gguf.py"
        "/opt/llama.cpp/convert_hf_to_gguf.py"
        "/usr/local/lib/llama.cpp/convert_hf_to_gguf.py"
        "/usr/lib/llama.cpp/convert_hf_to_gguf.py"

        # beside the llama-server binary if in PATH
        "$(dirname "$(command -v llama-server 2>/dev/null || true)")/../../convert_hf_to_gguf.py"
        "$(dirname "$(command -v llama-quantize 2>/dev/null || true)")/../../convert_hf_to_gguf.py"
    )

    for p in "${candidates[@]}"; do
        [[ -f "$p" ]] && { realpath "$p"; return 0; }
    done

    # last resort: slow filesystem search from common roots
    for root in "$HOME" /opt /usr/local /usr; do
        local found
        found="$(find "$root" -maxdepth 6 -name convert_hf_to_gguf.py 2>/dev/null | head -1)"
        [[ -n "$found" ]] && { echo "$found"; return 0; }
    done

    return 1
}

CONVERT_SCRIPT="$(find_convert_script)" \
    || die "convert_hf_to_gguf.py not found. Set LLAMA_CPP_DIR or add llama.cpp root to the candidate list."

info "Convert script : $CONVERT_SCRIPT"

# ---------------------------------------------------------------------------
# Locate llama-quantize (only needed if quant_type supplied)
# ---------------------------------------------------------------------------

QUANTIZE_BIN=""
if [[ -n "$QUANT_TYPE" ]]; then
    if command -v llama-quantize &>/dev/null; then
        QUANTIZE_BIN="$(command -v llama-quantize)"
    else
        # look beside the convert script
        LLAMA_ROOT="$(dirname "$CONVERT_SCRIPT")"
        for candidate in \
            "$LLAMA_ROOT/build/bin/llama-quantize" \
            "$LLAMA_ROOT/build/llama-quantize" \
            "$LLAMA_ROOT/../build/bin/llama-quantize"
        do
            [[ -x "$candidate" ]] && { QUANTIZE_BIN="$(realpath "$candidate")"; break; }
        done
    fi
    [[ -n "$QUANTIZE_BIN" ]] \
        || die "llama-quantize not found but quant_type=$QUANT_TYPE was requested."
    info "llama-quantize  : $QUANTIZE_BIN"
fi

# ---------------------------------------------------------------------------
# Derive output filenames
# ---------------------------------------------------------------------------

MODEL_NAME="$(basename "$MODEL_DIR")"
F16_GGUF="$OUTPUT_DIR/${MODEL_NAME}-F16.gguf"

if [[ -n "$QUANT_TYPE" ]]; then
    FINAL_GGUF="$OUTPUT_DIR/${MODEL_NAME}-${QUANT_TYPE}.gguf"
else
    FINAL_GGUF="$F16_GGUF"
fi

# ---------------------------------------------------------------------------
# Conversion: safetensors → F16 GGUF
# ---------------------------------------------------------------------------

info "Input model    : $MODEL_DIR"
info "Output dir     : $OUTPUT_DIR"
[[ -n "$QUANT_TYPE" ]] && info "Quantization   : $QUANT_TYPE"
echo

echo "==> Converting to F16 GGUF..."

python3 "$CONVERT_SCRIPT" \
    "$MODEL_DIR" \
    --outfile "$F16_GGUF" \
    --outtype f16

info "F16 GGUF written: $F16_GGUF"

# ---------------------------------------------------------------------------
# Optional quantization
# ---------------------------------------------------------------------------

if [[ -n "$QUANT_TYPE" ]]; then
    echo
    echo "==> Quantizing to $QUANT_TYPE..."

    "$QUANTIZE_BIN" "$F16_GGUF" "$FINAL_GGUF" "$QUANT_TYPE"

    info "Quantized GGUF  : $FINAL_GGUF"

    # Size comparison
    F16_SIZE="$(du -sh "$F16_GGUF"  | cut -f1)"
    QNT_SIZE="$(du -sh "$FINAL_GGUF" | cut -f1)"
    echo
    echo "  F16  : $F16_SIZE  →  $QUANT_TYPE : $QNT_SIZE"

    read -r -p "Delete intermediate F16 GGUF? [y/N] " confirm
    [[ "${confirm,,}" == "y" ]] && rm "$F16_GGUF" && warn "Deleted $F16_GGUF"
fi

echo
info "Done → $FINAL_GGUF"