"""
Differential Testing Harness: HTTP Backend (Stock vLLM API Server) vs. gRPC Worker Daemon.

Runs both servers in parallel:
- Target A: HTTP API Server on GPU 1 (Port 18200)
- Target B: gRPC Worker Daemon on GPU 0 (Port 50055)

Tests identical prompts with greedy decoding (temperature=0.0, seed=42) and asserts:
1. Exact text sequence identity (Token-by-token and full string equality)
2. Exact finish reason parity ("stop" / "length")
3. Comparative metadata inspection (OpenAI JSON vs. Protobuf binary)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

ROUTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROUTER_DIR, "py_src"))

import grpc  # noqa: E402
import requests  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from vllm_router.proto import engine_client_pb2, engine_client_pb2_grpc  # noqa: E402

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "Qwen/Qwen3.5-4B",
)
HTTP_PORT = 18200
GRPC_PORT = 50055
HOST = "127.0.0.1"


async def wait_for_http_health(
    url: str, proc: subprocess.Popen, timeout_s: int = 360
) -> bool:
    print(f"Waiting for HTTP server at {url} to become healthy...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(
                f"HTTP server process died unexpectedly with code {proc.poll()}"
            )
        try:
            r = requests.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                print(f"HTTP server healthy after {time.time() - t0:.1f}s!")
                return True
        except Exception:
            pass
        await asyncio.sleep(2.0)
    return False


async def wait_for_grpc_health(
    client: engine_client_pb2_grpc.EngineServiceStub,
    proc: subprocess.Popen,
    timeout_s: int = 360,
) -> bool:
    print("Waiting for gRPC server to become healthy...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(
                f"gRPC server process died unexpectedly with code {proc.poll()}"
            )
        try:
            res = await asyncio.wait_for(
                client.HealthCheck(engine_client_pb2.HealthCheckRequest()), timeout=2.0
            )
            if (
                res.status
                == engine_client_pb2.HealthCheckResponse.ServingStatus.SERVING
            ):
                print(f"gRPC server healthy after {time.time() - t0:.1f}s!")
                return True
        except Exception:
            pass
        await asyncio.sleep(2.0)
    return False


def query_http_stream(
    prompt: str, max_tokens: int = 30
) -> Tuple[str, str, List[str], Dict]:
    url = f"http://{HOST}:{HTTP_PORT}/v1/completions"
    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": True,
    }

    t0 = time.perf_counter()
    r = requests.post(url, json=payload, stream=True, timeout=30.0)
    r.raise_for_status()

    chunks = []
    finish_reason = ""
    sample_metadata = {}

    for line in r.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8")
        if line_str.startswith("data: "):
            data_content = line_str[6:].strip()
            if data_content == "[DONE]":
                break
            chunk_obj = json.loads(data_content)
            if not sample_metadata:
                sample_metadata = {
                    "id": chunk_obj.get("id"),
                    "object": chunk_obj.get("object"),
                    "model": chunk_obj.get("model"),
                }
            choice = chunk_obj["choices"][0]
            text = choice.get("text", "")
            if text:
                chunks.append(text)
            if choice.get("finish_reason"):
                finish_reason = choice.get("finish_reason")

    latency = time.perf_counter() - t0
    full_text = "".join(chunks)
    return full_text, finish_reason, chunks, {"latency": latency, **sample_metadata}


async def query_grpc_stream(
    client: engine_client_pb2_grpc.EngineServiceStub,
    prompt_token_ids: List[int],
    max_tokens: int = 30,
) -> Tuple[str, str, List[str], List[int], Dict]:
    req = engine_client_pb2.GenerateRequest(
        request_id=f"diff-test-{int(time.time()*1000)}",
        prompt_token_ids=prompt_token_ids,
        sampling_params=engine_client_pb2.SamplingParams(
            temperature=0.0,
            seed=42,
            max_tokens=max_tokens,
        ),
    )

    t0 = time.perf_counter()
    stream = client.GenerateStream(req)

    text_chunks = []
    token_ids = []
    finish_reason = ""
    sample_metrics = {}

    async for chunk in stream:
        if chunk.text_delta:
            text_chunks.append(chunk.text_delta)
        if chunk.token_id > 0:
            token_ids.append(chunk.token_id)
        if chunk.is_finished:
            finish_reason = chunk.finish_reason
        if chunk.metrics and not sample_metrics:
            sample_metrics = {
                "running_requests": chunk.metrics.running_requests,
                "waiting_requests": chunk.metrics.waiting_requests,
                "kv_cache_usage_percent": chunk.metrics.kv_cache_usage_percent,
            }

    latency = time.perf_counter() - t0
    full_text = "".join(text_chunks)
    return (
        full_text,
        finish_reason,
        text_chunks,
        token_ids,
        {"latency": latency, **sample_metrics},
    )


async def main():
    print("=" * 80)
    print("DIFFERENTIAL TESTING: HTTP API SERVER vs. gRPC WORKER DAEMON")
    print(f"Model: {MODEL_PATH}")
    print("=" * 80)

    py_bin = sys.executable

    # 1. Start HTTP Server on GPU 1
    env_http = os.environ.copy()
    env_http["CUDA_VISIBLE_DEVICES"] = "1"
    http_log = open("/tmp/diff_test_http.log", "w")
    http_cmd = [
        py_bin,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_PATH,
        "--host",
        HOST,
        "--port",
        str(HTTP_PORT),
        "--gpu-memory-utilization",
        "0.70",
        "--max-model-len",
        "4096",
        "--trust-remote-code",
        "--enforce-eager",
    ]
    print(f"Launching HTTP API Server on GPU 1 (Port {HTTP_PORT})...")
    proc_http = subprocess.Popen(
        http_cmd, env=env_http, stdout=http_log, stderr=subprocess.STDOUT
    )

    # 2. Start gRPC Worker Daemon on GPU 0
    env_grpc = os.environ.copy()
    env_grpc["CUDA_VISIBLE_DEVICES"] = "0"
    env_grpc["PYTHONPATH"] = (
        f"{os.path.join(ROUTER_DIR, 'py_src')}:{env_grpc.get('PYTHONPATH', '')}"
    )
    grpc_log = open("/tmp/diff_test_grpc.log", "w")
    grpc_cmd = [
        py_bin,
        "-m",
        "vllm_router.worker_daemon",
        "--model",
        MODEL_PATH,
        "--host",
        HOST,
        "--port",
        str(GRPC_PORT),
        "--gpu-memory-utilization",
        "0.70",
        "--max-model-len",
        "4096",
        "--trust-remote-code",
        "--enforce-eager",
    ]
    print(f"Launching gRPC Worker Daemon on GPU 0 (Port {GRPC_PORT})...")
    proc_grpc = subprocess.Popen(
        grpc_cmd, env=env_grpc, stdout=grpc_log, stderr=subprocess.STDOUT
    )

    grpc_channel = grpc.aio.insecure_channel(f"{HOST}:{GRPC_PORT}")
    grpc_client = engine_client_pb2_grpc.EngineServiceStub(grpc_channel)

    try:
        # Wait for both servers to be healthy in parallel
        print(
            "\nWaiting for both servers to finish model initialization and kernel warmup..."
        )
        http_ok, grpc_ok = await asyncio.gather(
            wait_for_http_health(f"http://{HOST}:{HTTP_PORT}", proc_http),
            wait_for_grpc_health(grpc_client, proc_grpc),
        )

        assert http_ok and grpc_ok, "Both servers must be healthy!"
        print("\n>>> BOTH SERVERS ARE READY! INITIATING DIFFERENTIAL TEST SUITE <<<\n")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        test_cases = [
            ("Case 1 (Factual Knowledge)", "Q: What is the capital of France?\nA:", 30),
            (
                "Case 2 (Arithmetic Calculation)",
                "Calculate step by step: 25 * 14 = ",
                35,
            ),
            (
                "Case 3 (Python Code Generation)",
                'def is_prime(n: int) -> bool:\n    """Return True if n is prime."""\n',
                40,
            ),
        ]

        all_passed = True

        for title, prompt, max_tokens in test_cases:
            print("-" * 80)
            print(f"RUNNING: {title}")
            print(f"Prompt: {repr(prompt)}")
            prompt_token_ids = tokenizer.encode(prompt)
            print(
                f"Pre-tokenized Token IDs ({len(prompt_token_ids)} tokens): {prompt_token_ids[:10]}..."
            )

            # Run HTTP request
            http_text, http_finish, http_chunks, http_meta = query_http_stream(
                prompt, max_tokens=max_tokens
            )

            # Run gRPC request
            grpc_text, grpc_finish, grpc_chunks, grpc_tokens, grpc_meta = (
                await query_grpc_stream(
                    grpc_client, prompt_token_ids, max_tokens=max_tokens
                )
            )

            print("\n[HTTP Server Result (GPU 1)]")
            print(f"  Generated Text: {repr(http_text)}")
            print(f"  Finish Reason : {http_finish}")
            print(f"  Chunks Count  : {len(http_chunks)}")
            print(f"  Latency       : {http_meta['latency']:.3f}s")
            print(
                f"  OpenAI Meta   : id={http_meta.get('id')}, model={http_meta.get('model')}"
            )

            print("\n[gRPC Worker Result (GPU 0)]")
            print(f"  Generated Text: {repr(grpc_text)}")
            print(f"  Finish Reason : {grpc_finish}")
            print(f"  Chunks Count  : {len(grpc_chunks)}")
            print(f"  Tokens Decoded: {len(grpc_tokens)}")
            print(f"  Latency       : {grpc_meta['latency']:.3f}s")
            print(
                f"  Protobuf Meta : running_reqs={grpc_meta.get('running_requests')}, kv_cache={grpc_meta.get('kv_cache_usage_percent')}"
            )

            # ASSERT EQUALITY
            # Reconstruct HTTP tokens from tokenizer to verify token sequence parity
            http_tokens = tokenizer.encode(http_text, add_special_tokens=False)

            text_match = http_text == grpc_text
            finish_match = http_finish == grpc_finish
            token_match = http_tokens == grpc_tokens

            print("\n--- VERIFICATION VERDICT ---")
            print(
                f"  Text Match (Byte-for-Byte) : {'PASS (IDENTICAL)' if text_match else 'FAIL'}"
            )
            print(
                f"  Token Sequence Match       : {'PASS (IDENTICAL)' if token_match else 'FAIL'}"
            )
            print(
                f"  Finish Reason Match        : {'PASS (IDENTICAL)' if finish_match else 'FAIL'}"
            )

            if not (text_match and finish_match and token_match):
                all_passed = False
                print("MISMATCH DETECTED!")
                print(f"  HTTP Text   : {http_text}")
                print(f"  gRPC Text   : {grpc_text}")
                print(f"  HTTP Tokens : {http_tokens}")
                print(f"  gRPC Tokens : {grpc_tokens}")

            assert text_match, f"Text mismatch in {title}!"
            assert finish_match, f"Finish reason mismatch in {title}!"
            assert token_match, f"Token sequence mismatch in {title}!"

        print("\n" + "=" * 80)
        if all_passed:
            print(
                ">>> ALL DIFFERENTIAL PARITY TESTS PASSED: 100% IDENTICAL OUTPUT! <<<"
            )
        else:
            print(">>> SOME TESTS FAILED! <<<")
        print("=" * 80)

    finally:
        print("\nTearing down servers...")
        await grpc_channel.close()
        proc_http.terminate()
        proc_grpc.terminate()
        try:
            proc_http.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc_http.kill()
        try:
            proc_grpc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc_grpc.kill()
        http_log.close()
        grpc_log.close()
        print("Servers terminated cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
