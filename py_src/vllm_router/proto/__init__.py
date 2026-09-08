import os
import sys

_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

from . import engine_client_pb2 as engine_client_pb2  # noqa: E402
from . import engine_client_pb2_grpc as engine_client_pb2_grpc  # noqa: E402

__all__ = ["engine_client_pb2", "engine_client_pb2_grpc"]
