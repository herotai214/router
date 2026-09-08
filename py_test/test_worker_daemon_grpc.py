"""
Unit and integration tests for the vLLM gRPC Worker Daemon.

Tests RPC methods:
- GenerateStream (streaming token generation with prompt_token_ids)
- Generate (unary generation)
- GetModelInfo (metadata discovery)
- HealthCheck (serving status)
- StartProfile / StopProfile / ResetPrefixCache (admin controls)
"""

import asyncio
import pytest
import pytest_asyncio
import grpc

from vllm_router.proto import engine_client_pb2, engine_client_pb2_grpc
from vllm_router.worker_daemon import EngineServiceServicer, MockAsyncEngine

try:
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    HAS_GRPC_HEALTH = True
except ImportError:
    HAS_GRPC_HEALTH = False


@pytest_asyncio.fixture
async def grpc_test_server():
    """Starts an in-process gRPC test server with MockAsyncEngine on an ephemeral port."""
    server = grpc.aio.server()
    engine = MockAsyncEngine(model_name="mock-model/test-4b")
    servicer = EngineServiceServicer(
        engine=engine,
        model_name="mock-model/test-4b",
        max_model_len=8192,
        dp_size=2,
        block_size=16,
    )
    engine_client_pb2_grpc.add_EngineServiceServicer_to_server(servicer, server)

    if HAS_GRPC_HEALTH:
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        health_servicer.set(
            "vllm.engine.v1.EngineService", health_pb2.HealthCheckResponse.SERVING
        )

    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    client = engine_client_pb2_grpc.EngineServiceStub(channel)

    yield {
        "client": client,
        "channel": channel,
        "server": server,
        "engine": engine,
        "servicer": servicer,
        "port": port,
    }

    await channel.close()
    await server.stop(grace=0.5)


@pytest.mark.asyncio
async def test_health_check(grpc_test_server):
    """Verifies that HealthCheck returns SERVING status."""
    client = grpc_test_server["client"]
    req = engine_client_pb2.HealthCheckRequest()
    res = await client.HealthCheck(req)
    assert res.status == engine_client_pb2.HealthCheckResponse.ServingStatus.SERVING


@pytest.mark.asyncio
async def test_get_model_info(grpc_test_server):
    """Verifies model metadata discovery."""
    client = grpc_test_server["client"]
    req = engine_client_pb2.ModelInfoRequest()
    res = await client.GetModelInfo(req)
    assert res.model_name == "mock-model/test-4b"
    assert res.max_model_len == 8192
    assert res.dp_size == 2
    assert res.block_size == 16


@pytest.mark.asyncio
async def test_generate_stream_prompt_token_ids(grpc_test_server):
    """Verifies streaming token generation with pre-tokenized prompt_token_ids."""
    client = grpc_test_server["client"]
    req = engine_client_pb2.GenerateRequest(
        request_id="req-stream-test-01",
        prompt_token_ids=[101, 2054, 2003],
        dp_rank=0,
        sampling_params=engine_client_pb2.SamplingParams(
            temperature=0.7,
            max_tokens=20,
        ),
    )

    chunks = []
    async for chunk in client.GenerateStream(req):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(c.text_delta for c in chunks)
    assert full_text == "Hello world! This is a gRPC streaming test."
    assert chunks[-1].is_finished is True
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].metrics.running_requests >= 0


@pytest.mark.asyncio
async def test_generate_stream_text_fallback(grpc_test_server):
    """Verifies streaming generation with prompt_text fallback."""
    client = grpc_test_server["client"]
    req = engine_client_pb2.GenerateRequest(
        request_id="req-stream-text-02",
        prompt_text="Hello from text prompt",
        dp_rank=1,
        sampling_params=engine_client_pb2.SamplingParams(
            temperature=0.0,
            max_tokens=10,
        ),
    )

    chunks = []
    async for chunk in client.GenerateStream(req):
        chunks.append(chunk)

    assert len(chunks) > 0
    full_text = "".join(c.text_delta for c in chunks)
    assert "Hello world!" in full_text


@pytest.mark.asyncio
async def test_generate_unary(grpc_test_server):
    """Verifies unary Generate RPC accumulating the full output."""
    client = grpc_test_server["client"]
    req = engine_client_pb2.GenerateRequest(
        request_id="req-unary-03",
        prompt_token_ids=[1, 2, 3],
        sampling_params=engine_client_pb2.SamplingParams(
            temperature=0.0,
            max_tokens=15,
        ),
    )

    res = await client.Generate(req)
    assert res.request_id == "req-unary-03"
    assert res.output_text == "Hello world! This is a gRPC streaming test."
    assert len(res.output_token_ids) == 10
    assert res.finish_reason == "stop"


@pytest.mark.asyncio
async def test_admin_controls(grpc_test_server):
    """Verifies StartProfile, StopProfile, and ResetPrefixCache RPCs."""
    client = grpc_test_server["client"]
    empty = engine_client_pb2.EmptyRequest()

    start_res = await client.StartProfile(empty)
    assert start_res.success is True
    assert "started" in start_res.message.lower()

    stop_res = await client.StopProfile(empty)
    assert stop_res.success is True
    assert "stopped" in stop_res.message.lower()

    reset_res = await client.ResetPrefixCache(empty)
    assert reset_res.success is True
    assert "reset" in reset_res.message.lower()


@pytest.mark.asyncio
async def test_stream_cancellation(grpc_test_server):
    """Verifies that client stream cancellation is handled cleanly."""
    client = grpc_test_server["client"]
    req = engine_client_pb2.GenerateRequest(
        request_id="req-cancel-04",
        prompt_token_ids=[100, 200],
        sampling_params=engine_client_pb2.SamplingParams(max_tokens=50),
    )

    call = client.GenerateStream(req)
    chunks_received = 0
    try:
        async for chunk in call:
            chunks_received += 1
            if chunks_received == 2:
                call.cancel()
                break
    except (grpc.aio.AioRpcError, asyncio.CancelledError):
        pass

    assert chunks_received == 2


@pytest.mark.asyncio
async def test_unsupported_execution_mode_rejected(grpc_test_server):
    """Verifies that PREFILL_ONLY or DECODE_ONLY modes are rejected with UNIMPLEMENTED."""
    client = grpc_test_server["client"]
    req = engine_client_pb2.GenerateRequest(
        request_id="req-exec-mode-05",
        prompt_token_ids=[1, 2, 3],
        execution_mode=engine_client_pb2.ExecutionMode.PREFILL_ONLY,
    )

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        async for _ in client.GenerateStream(req):
            pass

    assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
    assert "not supported in PR 1" in exc_info.value.details()


@pytest.mark.asyncio
async def test_unsupported_multimodal_rejected(grpc_test_server):
    """Verifies that multimodal input is rejected with UNIMPLEMENTED in PR 1."""
    client = grpc_test_server["client"]
    req = engine_client_pb2.GenerateRequest(
        request_id="req-mm-06",
        prompt_token_ids=[1, 2, 3],
        multimodal_data=[
            engine_client_pb2.MultimodalItem(
                modality_type="image",
                raw_data=b"fake-image-bytes",
            )
        ],
    )

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        async for _ in client.GenerateStream(req):
            pass

    assert exc_info.value.code() == grpc.StatusCode.UNIMPLEMENTED
    assert "Multimodal generation is not supported in PR 1" in exc_info.value.details()


@pytest.mark.asyncio
async def test_invalid_dp_rank_rejected(grpc_test_server):
    """Verifies that out-of-bounds dp_rank (>= dp_size) is rejected with INVALID_ARGUMENT."""
    client = grpc_test_server["client"]
    # grpc_test_server has dp_size=2, so dp_rank=5 is invalid
    req = engine_client_pb2.GenerateRequest(
        request_id="req-dp-07",
        prompt_token_ids=[1, 2, 3],
        dp_rank=5,
    )

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        async for _ in client.GenerateStream(req):
            pass

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "out of bounds" in exc_info.value.details()


@pytest.mark.asyncio
async def test_multi_token_step_emission(grpc_test_server):
    """Verifies that multi-step / speculative decoding yielding multiple tokens in one step does not drop tokens."""

    class MultiStepEngine:
        def __init__(self):
            pass

        async def generate(self, prompt, sampling_params, request_id: str):
            from vllm_router.worker_daemon import CompletionOutput, RequestOutput

            # Step 1: emits 3 tokens at once: [501, 502, 503]
            yield RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=[1, 2],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="Batch one",
                        token_ids=[501, 502, 503],
                        finish_reason=None,
                    )
                ],
                finished=False,
            )
            # Step 2: emits 2 more tokens: [504, 505]
            yield RequestOutput(
                request_id=request_id,
                prompt=None,
                prompt_token_ids=[1, 2],
                prompt_logprobs=None,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="Batch one and two",
                        token_ids=[501, 502, 503, 504, 505],
                        finish_reason="stop",
                    )
                ],
                finished=True,
            )

    server = grpc.aio.server()
    engine = MultiStepEngine()
    servicer = EngineServiceServicer(
        engine=engine,
        model_name="mock-multistep",
        max_model_len=4096,
        dp_size=1,
    )
    engine_client_pb2_grpc.add_EngineServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    client = engine_client_pb2_grpc.EngineServiceStub(channel)

    try:
        req = engine_client_pb2.GenerateRequest(
            request_id="req-multistep-08",
            prompt_token_ids=[1, 2],
        )

        emitted_tokens = []
        async for chunk in client.GenerateStream(req):
            if chunk.token_id > 0:
                emitted_tokens.append(chunk.token_id)

        # All 5 tokens must be emitted in order, none dropped!
        assert emitted_tokens == [501, 502, 503, 504, 505]

        # Verify unary Generate also receives all 5 tokens
        res = await client.Generate(req)
        assert list(res.output_token_ids) == [501, 502, 503, 504, 505]
        assert res.output_text == "Batch one and two"
    finally:
        await channel.close()
        await server.stop(grace=0.5)
