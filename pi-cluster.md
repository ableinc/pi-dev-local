# Pi 5 Cluster — Distributed LLM Inference

Three-node Raspberry Pi 5 cluster running distributed inference via llama.cpp RPC. Covers USB mesh networking, model conversion from safetensors to GGUF, and running a model split across all three nodes.

---

## Prerequisites

### Hardware

| Item | Qty | Notes |
|---|---|---|
| Raspberry Pi 5 (8GB) | 3 | Earlier revisions not recommended — RP1 south bridge matters |
| PoE HAT (Pi 5 compatible) | 3 | Powers the boards over Ethernet |
| USB 3.0 host-to-host cable | 3 | Must expose as CDC NCM/ECM — see note below |
| 2TB SATA drive + USB 3.0 enclosure | 1 | Attached to Pi-1 (coordinator) for model storage |
| Gigabit switch | 1 | For PoE + management traffic only |

**USB cable requirement:** Cables must use a bridge chip that Linux presents as a standard CDC NCM network interface. Confirmed working chips: `ASIX AX88179`, `Realtek RTL8153`. Do not use cables marketed as "easy transfer" or requiring proprietary software. Verified options: Plugable USB3-TETHER, IOGEAR GUC3C01B.

### OS

Raspberry Pi OS Lite (64-bit, Bookworm) on all three nodes. Headless is fine.

```
# Verify 64-bit
uname -m    # should return aarch64
```

### Software (all nodes unless noted)

```bash
sudo apt update && sudo apt install -y \
    git cmake build-essential python3-pip \
    net-tools iproute2 ethtool

# Pi-1 only
pip install huggingface-hub --break-system-packages
```

---

## 1. USB 3.0 Full-Mesh Network

Skip this section if you intend to run over Ethernet only. USB mesh gives ~4 Gbps per link vs ~1 Gbps on Ethernet, which matters for large model inference.

### Topology

```
       Pi-1
      /    \
   USB      USB
   /            \
Pi-2 ----USB---- Pi-3
```

Each node uses both USB 3.0 ports. Each link is a dedicated point-to-point `/30` subnet.

### Verify Cables

After plugging in all three cables:

```bash
lsmod | grep cdc          # expect cdc_ncm or cdc_ether
ip link show              # USB interfaces appear as enx<mac>
dmesg | grep -i cdc       # confirm enumeration
```

### Netplan Configuration

Replace `enx<mac>` values with the actual interface names from `ip link show`.

**Pi-1** — `10.0.12.1` toward Pi-2, `10.0.13.1` toward Pi-3:
```yaml
# /etc/netplan/01-usb-cluster.yaml
network:
  version: 2
  ethernets:
    enx<mac-usb0>:
      addresses: [10.0.12.1/30]
    enx<mac-usb1>:
      addresses: [10.0.13.1/30]
```

**Pi-2** — `10.0.12.2` toward Pi-1, `10.0.23.1` toward Pi-3:
```yaml
network:
  version: 2
  ethernets:
    enx<mac-usb0>:
      addresses: [10.0.12.2/30]
    enx<mac-usb1>:
      addresses: [10.0.23.1/30]
```

**Pi-3** — `10.0.13.2` toward Pi-1, `10.0.23.2` toward Pi-2:
```yaml
network:
  version: 2
  ethernets:
    enx<mac-usb0>:
      addresses: [10.0.13.2/30]
    enx<mac-usb1>:
      addresses: [10.0.23.2/30]
```

```bash
sudo netplan apply

# Verify — expect 30–80μs RTT
ping -c 10 -i 0.01 10.0.12.2
```

### Kernel Tuning

Apply on all nodes:

```bash
sudo tee /etc/sysctl.d/99-cluster.conf << 'EOF'
net.core.rmem_max=67108864
net.core.wmem_max=67108864
net.ipv4.tcp_rmem=4096 87380 67108864
net.ipv4.tcp_wmem=4096 65536 67108864
net.ipv4.tcp_low_latency=1
net.ipv4.tcp_nodelay=1
EOF

sudo sysctl -p /etc/sysctl.d/99-cluster.conf
```

---

## 2. Mount the SATA Drive (Pi-1 only)

```bash
lsblk                           # find device, typically /dev/sda
sudo mkfs.ext4 /dev/sda1        # skip if already formatted
sudo mkdir -p /mnt/models
sudo blkid /dev/sda1            # note the UUID

# Add to fstab
echo "UUID=<your-uuid>  /mnt/models  ext4  defaults,noatime  0  2" \
    | sudo tee -a /etc/fstab

sudo mount -a
df -h /mnt/models               # verify
```

---

## 3. Build llama.cpp

Build on Pi-1, sync to workers. All three nodes are identical Cortex-A76 — one build covers all.

```bash
# Pi-1
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

cmake -B build \
    -DLLAMA_RPC=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel 4
# ~15 minutes
```

Sync binaries to workers:

```bash
rsync -az --progress build/bin/ pi@<pi2-ip>:~/llama-bin/
rsync -az --progress build/bin/ pi@<pi3-ip>:~/llama-bin/
```

---

## 4. Model Conversion — `to_gguf.sh`

Converts a HuggingFace safetensors model directory to GGUF. Optionally quantizes in the same step.

### Usage

```
to_gguf.sh <model_dir> [output_dir] [quant_type]
```

| Argument | Required | Description |
|---|---|---|
| `model_dir` | Yes | Directory containing `*.safetensors` and `config.json` |
| `output_dir` | No | Where to write output (default: same as `model_dir`) |
| `quant_type` | No | llama-quantize format string — see table below |

### Quant Type Reference

| Type | Bits/W | Use case |
|---|---|---|
| `F16` | 16 | Full precision, max VRAM |
| `Q8_0` | 8 | Near-lossless, good baseline |
| `Q4_K_M` | 4 | Best quality/size tradeoff for most models |
| `Q4_K_XL` | 4 | Slightly higher quality than K_M |
| `IQ4_XS` | ~4 | i-quant, better than K_M at same size |
| `IQ3_M` | ~3 | Aggressive compression, some quality loss |
| `IQ2_XS` | ~2 | Maximum compression — large models only |

### Examples

```bash
chmod +x to_gguf.sh

# F16 only (no quantization)
./to_gguf.sh /mnt/models/Qwen2.5-32B-Instruct

# Convert and quantize, writing output to a separate directory
./to_gguf.sh /mnt/models/Qwen2.5-32B-Instruct /mnt/models/gguf Q4_K_M
```

### Notes

- The script produces an F16 GGUF first regardless of quant choice. For a 32B model this intermediate file is ~64GB — confirm you have space on the SATA drive before running.
- After quantization, the script offers to delete the F16 intermediate.
- If `convert_hf_to_gguf.py` is not in a standard location, set `LLAMA_CPP_DIR` to the llama.cpp repo root before running:
  ```bash
  LLAMA_CPP_DIR=/opt/llama.cpp ./to_gguf.sh ...
  ```

### Downloading a Model First

```bash
# Pi-1
huggingface-cli download \
    bartowski/Qwen2.5-32B-Instruct-GGUF \
    --include "Qwen2.5-32B-Instruct-Q4_K_M.gguf" \
    --local-dir /mnt/models/

# Or download raw safetensors for conversion
huggingface-cli download \
    Qwen/Qwen2.5-32B-Instruct \
    --local-dir /mnt/models/Qwen2.5-32B-Instruct
```

---

## 5. Distributed Inference — llama.cpp RPC

### Model Size Guide

With 3× 8GB nodes (24GB total addressable RAM):

| Model | Quant | Size | Nodes needed |
|---|---|---|---|
| Llama-3.1-8B | Q8_0 | ~8.5GB | 1 |
| Qwen2.5-14B | Q4_K_M | ~9GB | 1 |
| Qwen2.5-32B | Q4_K_M | ~20GB | 3 |
| Llama-3.3-70B | IQ2_XS | ~19GB | 3 |

### Step 1 — Start Workers (Pi-2 and Pi-3)

```bash
sudo tee /etc/systemd/system/llama-rpc.service << 'EOF'
[Unit]
Description=llama.cpp RPC Server
After=network.target

[Service]
Environment="LD_LIBRARY_PATH=/home/pi/llama-bin"
ExecStart=/home/pi/llama-bin/ggml-rpc-server -H 0.0.0.0 -p 50052
Restart=always
RestartSec=5
Nice=-10
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now llama-rpc

# Verify
ss -tlnp | grep 50052
# LISTEN 0      1          0.0.0.0:50052      0.0.0.0:*
```

#### View Service Logs

```bash
sudo journalctl -u llama-rpc.service -f
```

Read the last `n` lines:

```bash
sudo journalctl -u llama-rpc.service -n 50
```

If the Pi rebooted and you only want to see the logs from the current boot

```bash
sudo journalctl -u llama-rpc.service -b
```

### Step 2 — Run Inference (Pi-1)

Use the USB mesh addresses if the mesh is configured, otherwise use Ethernet management IPs.

**Interactive CLI:**
```bash
~/llama.cpp/build/bin/llama-cli \
    -m /mnt/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
    --rpc <pi2-ip>:50052,<pi3-ip>:50052 \
    --threads 4 \
    -ngl 99 \
    -c 4096 \
    -p "Your prompt here"
```

**OpenAI-compatible server:**
```bash
~/llama.cpp/build/bin/llama-server \
    -m /mnt/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
    --rpc <pi2-ip>:50052,<pi3-ip>:50052 \
    --threads 4 \
    -ngl 99 \
    -c 4096 \
    --host 0.0.0.0 \
    --port 8080
```

```bash
# Test from any LAN host
curl http://<pi1-ip>:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"local","messages":[{"role":"user","content":"Hello"}]}'
```

### Layer Distribution Tuning

`-ngl 99` offloads all layers. llama.cpp splits them across local + RPC backends evenly by default. Use `--tensor-split` to adjust the ratio (values are relative weights, not percentages):

```bash
--tensor-split 1,1,1    # even across Pi-1, Pi-2, Pi-3
--tensor-split 2,1,1    # more layers on Pi-1
```

Monitor memory and thermals across nodes while a load is running:

```bash
watch -n1 'free -h && vcgencmd measure_temp'
```

---

## Network Address Reference

| Link | Pi-1 | Pi-2 | Pi-3 |
|---|---|---|---|
| Pi-1 ↔ Pi-2 | `10.0.12.1` | `10.0.12.2` | — |
| Pi-1 ↔ Pi-3 | `10.0.13.1` | — | `10.0.13.2` |
| Pi-2 ↔ Pi-3 | — | `10.0.23.1` | `10.0.23.2` |

PoE/Ethernet management addresses are separate and unaffected by the USB mesh.