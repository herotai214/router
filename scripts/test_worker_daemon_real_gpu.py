"""
Live GPU Integration Test for vLLM gRPC Worker Daemon.

Runs against the real Qwen3.5-4B model on allocated GPU 0.
Tests:
1. HealthCheck RPC
2. GetModelInfo RPC
3. GenerateStream with pre-tokenized prompt_token_ids
4. StartProfile / StopProfile / ResetPrefixCache
"""

import asyncio
import os
import subprocess
import sys
import time

ROUTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROUTER_DIR, "py_src"))

import grpc  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from vllm_router.proto import engine_client_pb2, engine_client_pb2_grpc  # noqa: E402

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "Qwen/Qwen3.5-4B",
)
PORT = 50055
HOST = "127.0.0.1"


async def run_test():
    print(f"=== Starting Worker Daemon with real model {MODEL_PATH} on GPU 0 ===")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = (
        f"{os.path.join(ROUTER_DIR, 'py_src')}:{env.get('PYTHONPATH', '')}"
    )

    cmd = [
        sys.executable,
        "-m",
        "vllm_router.worker_daemon",
        "--model",
        MODEL_PATH,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        "0.75",
        "--max-model-len",
        "4096",
        "--trust-remote-code",
        "--enforce-eager",
    ]

    log_file = open("/tmp/worker_daemon_gpu_test.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)

    target = f"{HOST}:{PORT}"
    channel = grpc.aio.insecure_channel(target)
    client = engine_client_pb2_grpc.EngineServiceStub(channel)

    try:
        # Wait for server to become healthy (up to 300s for model loading)
        print("Waiting for worker daemon to initialize and load model into VRAM...")
        healthy = False
        for attempt in range(300):
            try:
                res = await asyncio.wait_for(
                    client.HealthCheck(engine_client_pb2.HealthCheckRequest()),
                    timeout=2.0,
                )
                if (
                    res.status
                    == engine_client_pb2.HealthCheckResponse.ServingStatus.SERVING
                ):
                    healthy = True
                    print(f"Worker daemon healthy after {attempt * 2}s!")
                    break
            except Exception:
                pass
            if proc.poll() is not None:
                raise RuntimeError(
                    "Worker process died unexpectedly! Check /tmp/worker_daemon_gpu_test.log"
                )
            await asyncio.sleep(2.0)

        if not healthy:
            raise TimeoutError("Worker daemon failed to become healthy within timeout.")

        # Test 1: GetModelInfo
        print("\n--- Testing RPC: GetModelInfo ---")
        info = await client.GetModelInfo(engine_client_pb2.ModelInfoRequest())
        print(f"Model Name: {info.model_name}")
        print(f"Max Model Len: {info.max_model_len}")
        print(f"DP Size: {info.dp_size}")
        print(f"Block Size: {info.block_size}")
        assert info.model_name == MODEL_PATH

        # Test 2: Tokenization & GenerateStream
        print(
            "\n--- Testing RPC: GenerateStream with pre-tokenized prompt_token_ids ---"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        prompt_text = "Q: What is the capital of France?\nA:"
        token_ids = tokenizer.encode(prompt_text)
        print(f"Prompt text: '{prompt_text}'")
        print(f"Tokenized IDs ({len(token_ids)} tokens): {token_ids}")

        req = engine_client_pb2.GenerateRequest(
            request_id="real-gpu-test-01",
            prompt_token_ids=token_ids,
            sampling_params=engine_client_pb2.SamplingParams(
                temperature=0.0,
                max_tokens=30,
            ),
        )

        stream = client.GenerateStream(req)
        output_tokens = []
        output_text_chunks = []
        t0 = time.perf_counter()

        async for chunk in stream:
            output_tokens.append(chunk.token_id)
            output_text_chunks.append(chunk.text_delta)
            print(chunk.text_delta, end="", flush=True)

        ttft = time.perf_counter() - t0
        full_generated_text = "".join(output_text_chunks)
        print(f"\n[Generated {len(output_tokens)} tokens in {ttft:.3f}s]")
        print(f"Full text: '{full_generated_text.strip()}'")
        assert "Paris" in full_generated_text

        # Test 3: Admin Controls
        print("\n--- Testing RPC: Admin Controls ---")
        profile_start = await client.StartProfile(engine_client_pb2.EmptyRequest())
        print(
            f"StartProfile: success={profile_start.success}, message='{profile_start.message}'"
        )

        profile_stop = await client.StopProfile(engine_client_pb2.EmptyRequest())
        print(
            f"StopProfile: success={profile_stop.success}, message='{profile_stop.message}'"
        )

        cache_reset = await client.ResetPrefixCache(engine_client_pb2.EmptyRequest())
        print(
            f"ResetPrefixCache: success={cache_reset.success}, message='{cache_reset.message}'"
        )

        print("\n=== ALL REAL GPU gRPC TESTS PASSED PERFECTLY! ===")

    finally:
        await channel.close()
        print("Stopping worker daemon process...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()


if __name__ == "__main__":
    asyncio.run(run_test())
