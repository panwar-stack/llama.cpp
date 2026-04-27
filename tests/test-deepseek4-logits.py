#!/usr/bin/env python3
"""
Validate logits between HuggingFace transformers and llama.cpp for DeepSeek V4 Flash.

Compares the logits at the last token position using NMSE (Normalized Mean Squared Error):
    NMSE = MSE(logits_hf, logits_gguf) / MSE(logits_hf, 0)

Usage:
    python tests/test-deepseek4-logits.py
    python tests/test-deepseek4-logits.py --case all
    python tests/test-deepseek4-logits.py --hf-model deepseek-ai/DeepSeek-V4-Flash --prompt "Hello world"
    python tests/test-deepseek4-logits.py --skip-convert --tolerance 0.05
    python tests/test-deepseek4-logits.py --skip-hf --gguf-path path/to/model.gguf
    python tests/test-deepseek4-logits.py --output-format json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("test-deepseek4-logits")


PROMPT_CASES: dict[str, str] = {
    "short": "The quick brown fox jumps over the lazy dog.",
    "sliding-window": " ".join(
        f"boundary-{i:03d}" for i in range(180)
    ),
    "compressed-kv": " ".join(
        f"memory-block-{i:04d}: retain this detail." for i in range(640)
    ),
    "hash-routing": " ".join(
        f"token-route-{i}" for i in range(96)
    ),
    "mtp-active": (
        "Predict the next assistant token after this compact reasoning prompt: "
        "state the final answer only."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def is_hf_repo_id(model_id: str) -> bool:
    return "/" in model_id and not Path(model_id).exists()


def nmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    mse = float(np.mean((a - b) ** 2))
    mse_zero = float(np.mean(a ** 2))
    if mse_zero == 0.0:
        return float("inf") if mse > 0 else 0.0
    return mse / mse_zero


def check_model_sanity(
    logits: np.ndarray,
    max_abs_threshold: float = 1e6,
) -> dict:
    logits_f64 = np.asarray(logits, dtype=np.float64)

    finite = bool(np.all(np.isfinite(logits_f64)))
    not_all_zero = bool(np.any(logits_f64 != 0.0))
    not_all_identical = bool(np.any(logits_f64 != logits_f64[0]))
    max_abs = float(np.max(np.abs(logits_f64)))
    reasonable_range = bool(max_abs < max_abs_threshold)

    checks: dict[str, bool] = {
        "finite": finite,
        "not_all_zero": not_all_zero,
        "not_all_identical": not_all_identical,
        "reasonable_range": reasonable_range,
    }

    warnings: list[str] = []
    if not finite:
        warnings.append("Logits contain NaN or Inf values")
    if not not_all_zero:
        warnings.append("All logits are zero")
    if not not_all_identical:
        warnings.append("All logits are identical")
    if not reasonable_range:
        warnings.append(
            f"Maximum absolute logit value is unreasonably large: {max_abs:.2e}"
        )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "warnings": warnings,
        "max_abs": max_abs,
        "vocab_size": len(logits_f64),
    }


# ---------------------------------------------------------------------------
# Step 1: Convert HF model to GGUF
# ---------------------------------------------------------------------------

def convert_hf_to_gguf(
    hf_model: str, output_path: Path, root: Path
) -> Path:
    convert_script = root / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {convert_script}"
        )

    cmd = [
        sys.executable,
        str(convert_script),
        hf_model,
        "--outfile", str(output_path),
        "--outtype", "f16",
    ]

    if is_hf_repo_id(hf_model):
        cmd.append("--remote")

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("Conversion stdout:\n%s", result.stdout)
        logger.error("Conversion stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"GGUF conversion failed with exit code {result.returncode}"
        )

    if output_path.exists():
        logger.info("GGUF model saved to: %s", output_path)
        return output_path

    # Maybe it was split; use the first shard as the model entry point.
    candidates = sorted(output_path.parent.glob(f"{output_path.stem}-*.gguf"))
    if candidates:
        output_path_actual = candidates[0]
        logger.info("GGUF was split; using first shard: %s", output_path_actual)
        return output_path_actual

    raise RuntimeError(f"GGUF file not created at {output_path}")


# ---------------------------------------------------------------------------
# Step 2: Get logits from HuggingFace
# ---------------------------------------------------------------------------

def load_hf_reference(hf_model: str):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError as e:
        raise ImportError(
            "HuggingFace transformers not installed. "
            "Install with: pip install transformers torch"
        ) from e

    logger.info("Loading HF tokenizer from: %s", hf_model)
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)

    logger.info("Loading HF model from: %s", hf_model)
    # Use bfloat16 when available for better fidelity vs f16
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model, torch


def get_hf_logits(tokenizer, model, torch, prompt: str) -> np.ndarray:
    logger.info("Tokenizing prompt: %r", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_tokens = inputs["input_ids"].shape[1]
    logger.info("Tokenized into %d tokens", n_tokens)

    logger.info("Running HF inference ...")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :].float().cpu().numpy()
    logger.info("HF logits: shape=%s, dtype=%s", logits.shape, logits.dtype)
    return logits


# ---------------------------------------------------------------------------
# Step 3: Get logits from llama.cpp
# ---------------------------------------------------------------------------

def get_llamacpp_logits(
    gguf_path: Path,
    prompt: str,
    output_dir: Path,
    debug_bin: Path,
    timeout: int = 300,
) -> np.ndarray:
    if not debug_bin.exists():
        raise FileNotFoundError(
            f"llama-debug binary not found at {debug_bin}. "
            "Configure examples first: cmake -B build -DLLAMA_BUILD_EXAMPLES=ON "
            "&& cmake --build build --target llama-debug"
        )

    prompt_file = output_dir / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")

    for fpath in output_dir.glob("llamacpp-*"):
        if fpath.is_file():
            fpath.unlink()

    cmd = [
        str(debug_bin),
        "-m", str(gguf_path),
        "-f", str(prompt_file),
        "--save-logits",
        "--logits-output-dir", str(output_dir),
        "--no-warmup",
    ]

    logger.info("Running: %s", " ".join(cmd))

    env = {"LLAMA_LOG": "info"}
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            env={**os.environ, **env},
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"llama-debug timed out after {timeout} seconds"
        ) from e

    logger.info("llama-debug stdout:\n%s", result.stdout)
    if result.stderr.strip():
        logger.info("llama-debug stderr:\n%s", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"llama-debug failed with exit code {result.returncode}"
        )

    # Find the logits binary file
    model_name = gguf_path.stem

    # Try exact match first
    logits_bin = output_dir / f"llamacpp-{model_name}.bin"
    if not logits_bin.exists():
        # Try to find any .bin file that is not a tokens file
        for fpath in sorted(output_dir.glob("llamacpp-*.bin")):
            if "tokens" not in fpath.stem:
                logits_bin = fpath
                break

    if not logits_bin.exists():
        available = list(output_dir.glob("*"))
        raise FileNotFoundError(
            f"Logits binary file not found in {output_dir}. "
            f"Available files: {[p.name for p in available]}"
        )

    logger.info("Reading llama.cpp logits from: %s", logits_bin)

    data = logits_bin.read_bytes()
    logits = np.frombuffer(data, dtype=np.float32).copy()
    logger.info("llama.cpp logits: shape=(%d,)", len(logits))
    return logits


def select_last_token_logits(logits: np.ndarray, vocab_size: int) -> np.ndarray:
    if len(logits) == vocab_size:
        return logits
    if len(logits) % vocab_size != 0:
        raise ValueError(
            f"llama.cpp logits length {len(logits)} is not a multiple of "
            f"HF vocab size {vocab_size}"
        )

    n_rows = len(logits) // vocab_size
    logger.info(
        "llama.cpp logits contain %d rows; comparing the last-token row", n_rows
    )
    return logits.reshape((n_rows, vocab_size))[-1].copy()


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def compare_logits(
    case_name: str,
    logits_hf: np.ndarray,
    logits_gguf: np.ndarray,
    tolerance: float,
    top_k_diffs: int = 5,
) -> dict:
    logits_gguf = select_last_token_logits(logits_gguf, len(logits_hf))

    result: dict = {
        "name": case_name,
        "passed": False,
        "vocab_size": len(logits_hf),
    }

    if len(logits_hf) != len(logits_gguf):
        logger.error(
            "[%s] Vocab size mismatch: HF=%d, llama.cpp=%d",
            case_name, len(logits_hf), len(logits_gguf),
        )
        result["error"] = (
            f"Vocab size mismatch: HF={len(logits_hf)}, "
            f"llama.cpp={len(logits_gguf)}"
        )
        return result

    nmse_val = nmse(logits_hf, logits_gguf)

    abs_diff = np.abs(logits_hf - logits_gguf)
    max_diff = float(np.max(abs_diff))
    idx_max = int(np.argmax(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    denom = float(np.linalg.norm(logits_hf) * np.linalg.norm(logits_gguf))
    cos_sim = float(np.dot(logits_hf, logits_gguf) / denom) if denom else 0.0

    # Compute top-K diffs
    top_k = min(top_k_diffs, len(logits_hf))
    top_idx = np.argsort(abs_diff)[-top_k:][::-1]
    top_diff_list: list[dict] = []
    for idx in top_idx:
        top_diff_list.append({
            "token": int(idx),
            "hf": float(logits_hf[idx]),
            "gguf": float(logits_gguf[idx]),
            "diff": float(abs_diff[idx]),
        })

    result.update({
        "nmse": nmse_val,
        "cosine_sim": cos_sim,
        "max_abs_diff": max_diff,
        "max_abs_diff_idx": idx_max,
        "mean_abs_diff": mean_diff,
        "hf_val_at_max": float(logits_hf[idx_max]),
        "gguf_val_at_max": float(logits_gguf[idx_max]),
        "top_diffs": top_diff_list,
    })

    logger.info("[%s] Vocab size   : %d", case_name, len(logits_hf))
    logger.info("[%s] NMSE         : %.8f", case_name, nmse_val)
    logger.info("[%s] Cosine sim   : %.8f", case_name, cos_sim)
    logger.info("[%s] Max abs diff : %.6e at index %d", case_name, max_diff, idx_max)
    logger.info("[%s] Mean abs diff: %.6e", case_name, mean_diff)
    logger.info("[%s] HF   at idx %d: %.6f", case_name, idx_max, float(logits_hf[idx_max]))
    logger.info("[%s] GGUF at idx %d: %.6f", case_name, idx_max, float(logits_gguf[idx_max]))

    if nmse_val <= tolerance:
        logger.info("[%s] PASS: NMSE %.8f <= tolerance %.8f", case_name, nmse_val, tolerance)
        result["passed"] = True
        return result

    logger.error("[%s] FAIL: NMSE %.8f > tolerance %.8f", case_name, nmse_val, tolerance)
    logger.error("[%s] Top %d largest absolute differences:", case_name, top_k)
    for entry in top_diff_list:
        logger.error(
            "  token %6d: HF=%.6f  GGUF=%.6f  diff=%.6e",
            entry["token"], entry["hf"], entry["gguf"], entry["diff"],
        )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF and llama.cpp logits for DeepSeek V4 Flash"
    )
    parser.add_argument(
        "--hf-model", default="deepseek-ai/DeepSeek-V4-Flash",
        help="HuggingFace model ID or local path (default: %(default)s)",
    )
    parser.add_argument(
        "--gguf-path", type=Path,
        help="Existing GGUF path to test instead of converting into output-dir",
    )
    parser.add_argument(
        "--case", choices=[*PROMPT_CASES.keys(), "all"], default="short",
        help="Named validation case to run (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        help="Custom test prompt. Cannot be combined with --case all.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.01,
        help="Maximum NMSE tolerance (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip GGUF conversion if output file already exists",
    )
    parser.add_argument(
        "--output-dir", default="./test-output",
        help="Directory for temporary files (default: %(default)s)",
    )
    parser.add_argument(
        "--build-dir", default="build",
        help="llama.cpp build directory containing bin/llama-debug (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-hf", action="store_true",
        help="Skip HF reference loading; validate llama.cpp sanity only",
    )
    parser.add_argument(
        "--output-format", choices=["summary", "json"], default="summary",
        help="Output format (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="Timeout in seconds for llama-debug subprocess (default: %(default)s)",
    )
    parser.add_argument(
        "--top-k-diffs", type=int, default=5,
        help="Show top K largest absolute differences (default: %(default)s)",
    )
    return parser.parse_args()


def selected_cases(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.prompt is not None:
        if args.case == "all":
            raise ValueError("--prompt cannot be combined with --case all")
        return [("custom", args.prompt)]

    if args.case == "all":
        return list(PROMPT_CASES.items())
    return [(args.case, PROMPT_CASES[args.case])]


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    root = project_root()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_bin = (root / args.build_dir / "bin" / "llama-debug").resolve()

    if args.gguf_path is not None:
        gguf_path = args.gguf_path.resolve()
        if not gguf_path.exists():
            logger.error("GGUF path does not exist: %s", gguf_path)
            sys.exit(1)
        logger.info("=== Using existing GGUF: %s ===", gguf_path)
    else:
        if args.skip_hf:
            logger.error(
                "--skip-hf requires --gguf-path to be specified "
                "(no HF model to convert from)"
            )
            sys.exit(1)
        gguf_path = output_dir / "model.gguf"
        if args.skip_convert and gguf_path.exists():
            logger.info(
                "=== Skipping conversion: %s already exists ===", gguf_path
            )
        else:
            logger.info("=== Step 1: Convert HF to GGUF ===")
            try:
                gguf_path = convert_hf_to_gguf(args.hf_model, gguf_path, root)
            except Exception as e:
                logger.error("Conversion failed: %s", e)
                sys.exit(1)

    try:
        cases = selected_cases(args)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(2)

    results: list[dict] = []
    all_passed = True

    if args.skip_hf:
        # ------------------------------------------------------------------
        # --skip-hf mode: sanity-check llama.cpp output only
        # ------------------------------------------------------------------
        for case_name, prompt in cases:
            case_output_dir = output_dir / case_name
            case_output_dir.mkdir(parents=True, exist_ok=True)

            result: dict = {"name": case_name, "passed": False}

            logger.info("=== Case %s: llama.cpp logits ===", case_name)
            try:
                logits_gguf = get_llamacpp_logits(
                    gguf_path, prompt, case_output_dir, debug_bin,
                    timeout=args.timeout,
                )
            except Exception as e:
                logger.error(
                    "[%s] llama.cpp inference failed: %s", case_name, e
                )
                result["error"] = str(e)
                results.append(result)
                all_passed = False
                continue

            sanity = check_model_sanity(logits_gguf)
            result.update({
                "vocab_size": sanity["vocab_size"],
                "sanity_passed": sanity["passed"],
                "sanity_warnings": sanity["warnings"],
                "sanity_checks": sanity["checks"],
                "max_abs": sanity["max_abs"],
                "passed": sanity["passed"],
            })

            if sanity["passed"]:
                logger.info("[%s] Sanity check PASSED", case_name)
            else:
                for w in sanity["warnings"]:
                    logger.error(
                        "[%s] Sanity check FAILED: %s", case_name, w
                    )

            results.append(result)
            all_passed = sanity["passed"] and all_passed
    else:
        # ------------------------------------------------------------------
        # Normal mode: compare against HF reference
        # ------------------------------------------------------------------
        logger.info("=== Step 2: Load HF reference ===")
        try:
            tokenizer, model, torch = load_hf_reference(args.hf_model)
        except ImportError as e:
            logger.error("Missing dependency: %s", e)
            sys.exit(1)
        except Exception as e:
            logger.error("HF reference load failed: %s", e)
            sys.exit(1)

        for case_name, prompt in cases:
            case_output_dir = output_dir / case_name
            case_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("=== Case %s: HF logits ===", case_name)
            try:
                logits_hf = get_hf_logits(tokenizer, model, torch, prompt)
            except Exception as e:
                logger.error("[%s] HF inference failed: %s", case_name, e)
                results.append({
                    "name": case_name,
                    "passed": False,
                    "error": str(e),
                })
                all_passed = False
                continue

            logger.info("=== Case %s: llama.cpp logits ===", case_name)
            try:
                logits_gguf = get_llamacpp_logits(
                    gguf_path, prompt, case_output_dir, debug_bin,
                    timeout=args.timeout,
                )
            except Exception as e:
                logger.error(
                    "[%s] llama.cpp inference failed: %s", case_name, e
                )
                results.append({
                    "name": case_name,
                    "passed": False,
                    "error": str(e),
                })
                all_passed = False
                continue

            logger.info("=== Case %s: compare ===", case_name)
            cmp_result = compare_logits(
                case_name, logits_hf, logits_gguf, args.tolerance,
                args.top_k_diffs,
            )
            results.append(cmp_result)
            all_passed = cmp_result["passed"] and all_passed

    if args.output_format == "json":
        print(json.dumps(results, indent=2))

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
