# DeepSeek V4 Flash – Phase 6 Validation Runbook

Operational notes for actually executing the HF logit-parity run that
`tests/test-deepseek4-logits.py` is designed for. The local-laptop "wrong
machine" problem and how to solve it on AWS.

## Why local doesn't work

This MacBook Pro (Intel, 16 GB RAM, no GPU, ~79 GB free disk) cannot host the
reference model:

- Model is on the order of hundreds of billions of parameters.
- FP16 weights ~1.4 TB; FP8 still hundreds of GB.
- No CUDA, no MPS, RAM and disk both insufficient.
- `torch` / `transformers` not installed.

Conclusion: the parity run has to happen on a remote box. The harness itself
runs anywhere.

## AWS instance choice

| Instance         | GPUs              | VRAM total | ~On-demand | Notes                                 |
|------------------|-------------------|------------|------------|---------------------------------------|
| `p5.48xlarge`    | 8× H100 80 GB     | 640 GB     | ~$98/hr    | Smallest comfortable fit, FP8/FP16    |
| `p5e.48xlarge`   | 8× H200 141 GB    | 1128 GB    | higher     | Headroom for activations              |
| `p4d.24xlarge`   | 8× A100 40 GB     | 320 GB     | cheaper    | Needs 4-bit reference (parity caveat) |
| `g6e.48xlarge`   | 8× L40S 48 GB     | 384 GB     | cheapest 8-GPU | Viable fallback, FP8 + offload    |

Spot pricing knocks ~70 % off. The run takes minutes once weights are loaded,
so an interruption is annoying but not catastrophic — render, dump, terminate.

## Storage

- **Don't** put the checkpoint on EBS. `p5` ships ~30 TB local NVMe.
  Point HF cache there: `export HF_HOME=/opt/dlami/nvme/hf`.
- 1 TB gp3 EBS for OS + GGUF output is fine.
- Use the **AWS Deep Learning AMI (Ubuntu, PyTorch)** — CUDA, torch,
  transformers preinstalled.

## Run flow

On the GPU box:

```bash
# 1. clone and build llama.cpp + llama-debug
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --target llama-debug test-deepseek4 -j

# 2. render both halves for all 5 cases
python tests/test-deepseek4-logits.py --case all \
    --hf-model deepseek-ai/DeepSeek-V4-Flash \
    --tolerance 0.01
```

The harness:

1. converts HF → FP16 GGUF (Phase 0 path),
2. runs HF reference, writes logits,
3. runs `llama-debug --save-logits`, reads `.bin`,
4. compares via NMSE.

Logit dumps are a few MB per case. Push them to S3 and you can re-run the
comparison anywhere.

## Pre-flight check

Before spending anything: confirm `deepseek-ai/DeepSeek-V4-Flash` is actually
published with downloadable weights. The plan links the V4-Pro model card; if
Flash is gated or unreleased, the instance choice is moot.

## Quantization

Two distinct operations — easy to conflate.

### GGUF side (the artifact you ship)

```bash
# A: HF -> FP16 GGUF (Phase 0)
python convert_hf_to_gguf.py /path/to/DeepSeek-V4-Flash \
    --outfile model-f16.gguf --outtype f16

# B: FP16 GGUF -> 4-bit GGUF
./build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M
```

- Pick `Q4_K_M` (balanced), `Q4_K_S` (smaller), or `IQ4_XS` / `IQ4_NL`
  (i-quants — better quality per bit, slower on some backends).
- For a 600 B-class MoE, Q4_K_M lands around 350–400 GB.
- **Gotcha 1**: if the HF release ships FP8, `convert_hf_to_gguf.py` may need
  a dequantize-to-BF16 hop first. Phase 0 of the plan flags this.
- **Gotcha 2**: MoE expert tensors are huge. Run `llama-quantize` with
  `--imatrix` once for noticeably better 4-bit quality.

### HF side (the comparator)

`bitsandbytes` 4-bit shrinks the HF reference so it fits cheaper hardware:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V4-Flash",
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)
```

AWQ / GPTQ give better quality but need calibration.

## Comparison matrix

| HF side  | GGUF side | Tolerance  | What it tells you                          |
|----------|-----------|------------|--------------------------------------------|
| FP16/FP8 | FP16      | ~0.01 NMSE | **Phase 6 acceptance** — parity vs ref     |
| FP16/FP8 | Q4_K_M    | ~0.05 NMSE | Quantization loss in the shipping artifact |
| 4-bit    | Q4_K_M    | n/a        | Only that the two halves agree — not parity|

**Don't 4-bit both sides for acceptance** — you're then measuring drift
between two independent 4-bit implementations, not parity against the
reference.

## Recommended play

Quantizing the HF reference is **not needed**. The cost-effective workflow:

1. Spin up `p5.48xlarge` (spot if you can tolerate interruption).
2. Render FP16 HF logits for all 5 cases. Save `.npy` to S3.
3. Terminate the instance.
4. All subsequent GGUF quantization comparisons run on any cheap CPU box
   against the frozen S3 reference.

Five fixed prompts × one checkpoint = render once, compare forever. HF-side
4-bit only earns its keep if you'll re-render the reference many times.
