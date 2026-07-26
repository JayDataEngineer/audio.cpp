#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python reference Inflect Micro v2 long-lived warmbench.")
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models" / "Inflect-Micro-v2")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timing-file", type=Path, default=Path("/tmp/inflect_tts_python_warm_bench_timing.log"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--request-sequence-json", default="")
    parser.add_argument("--warmup-request-json", default="")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def require_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("Inflect TTS request JSON entries must be objects")
    text = payload.get("text", "")
    if not isinstance(text, str) or not text:
        raise RuntimeError("Inflect TTS request requires non-empty text")
    return dict(payload)


def load_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.request_sequence_json:
        raise RuntimeError("Inflect TTS warmbench requires --request-sequence-json")
    payload = json.loads(args.request_sequence_json)
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("--request-sequence-json must decode to a non-empty list")
    return [require_request(item) for item in payload]


def load_warmup_request(args: argparse.Namespace, requests: list[dict[str, Any]]) -> dict[str, Any]:
    if args.warmup_request_json:
        return require_request(json.loads(args.warmup_request_json))
    return requests[0]


def audio_summary(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise RuntimeError("Inflect TTS Python warmbench received empty audio")
    return {
        "sample_rate": sample_rate,
        "channels": 1,
        "samples": int(flat.size),
        "frames": int(flat.size),
        "duration_sec": float(flat.size / sample_rate),
        "sum": float(np.sum(flat, dtype=np.float64)),
        "mean_abs": float(np.mean(np.abs(flat), dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(flat.astype(np.float64) ** 2))),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
    }


def load_tts(model_root: Path, backend: str, device: int, threads: int) -> Any:
    sys.path.insert(0, str(model_root))
    from inference import InflectTTS  # noqa: WPS433

    torch.set_num_threads(max(1, threads))
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA backend requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)
        return InflectTTS(model_root, device=f"cuda:{device}")
    return InflectTTS(model_root, device="cpu")


def run_request(tts: Any, request: dict[str, Any]) -> tuple[int, np.ndarray]:
    sample_rate, waveform = tts.synthesize(
        str(request["text"]),
        speed=float(request.get("speaking_rate", request.get("speed", 1.0))),
        variation=float(request.get("variation", 0.667)),
        seed=int(request.get("seed", 0)),
    )
    return int(sample_rate), np.asarray(waveform, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.iterations != 1:
        raise RuntimeError("Inflect TTS warmbench records raw per-request timing; --iterations must be 1")
    model_root = resolve_path(args.model)
    requests = load_requests(args)
    warmup_request = load_warmup_request(args, requests)
    tts = load_tts(model_root, args.backend, args.device, args.threads)
    for _ in range(args.warmup):
        run_request(tts, warmup_request)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    for request_index, request in enumerate(requests):
        started = time.perf_counter()
        sample_rate, audio = run_request(tts, request)
        if args.backend == "cuda":
            torch.cuda.synchronize(args.device)
        wall_ms = (time.perf_counter() - started) * 1000.0
        audio_path = ""
        if output_dir is not None:
            path = output_dir / f"audio_{request_index}.wav"
            sf.write(path, audio, sample_rate)
            audio_path = str(path)
        text_length = len(str(request["text"]))
        print(f"inflect_tts.request[{request_index}].length={text_length}")
        print(f"inflect_tts.request[{request_index}].wall_ms={wall_ms:.6f}")
        stem = {"name": "audio", "summary": audio_summary(audio, sample_rate)}
        if audio_path:
            stem["audio"] = audio_path
        steps.append({
            "request_index": request_index,
            "text_length": text_length,
            "stems": [stem],
            "metrics": {"wall_ms": wall_ms},
        })
    print("summary_json=" + json.dumps({
        "family": "inflect_tts",
        "backend": args.backend,
        "sequence_steps": steps,
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
