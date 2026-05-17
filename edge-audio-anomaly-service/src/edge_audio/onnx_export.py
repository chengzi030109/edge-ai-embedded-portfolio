from __future__ import annotations

"""Export the centroid scorer as a small ONNX graph.

The exported graph intentionally contains only the score calculation:

    score = sqrt(sum(((features - mean) / scale) ** 2))

Thresholding stays in the Python backend wrapper so both Python and ONNX paths
return the same public ``predict`` dictionary. This mirrors a common embedded
Linux deployment pattern: the runtime accelerates the numerical model, while
the application process owns policy decisions such as alarm thresholds.
"""

from pathlib import Path
import sys
import types

import numpy as np

from .model import AudioCentroidModel


def export_centroid_to_onnx(model: AudioCentroidModel, output_path: str | Path) -> Path:
    """Write an ONNX model that computes the centroid anomaly score."""

    try:
        onnx, TensorProto, helper, numpy_helper = _import_onnx_tools()
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "ONNX export requires an importable optional dependency 'onnx'. "
            "Install it with: pip install -e .[onnx]. "
            f"Original import error: {exc}"
        ) from exc

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    feature_dim = int(model.mean.shape[0])

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Sub", ["features", "mean"], ["centered"], name="center_features"),
            helper.make_node("Div", ["centered", "scale"], ["normalized"], name="scale_features"),
            helper.make_node("Mul", ["normalized", "normalized"], ["squared"], name="square_features"),
            helper.make_node("ReduceSum", ["squared"], ["sum_squared"], name="sum_squared", axes=[0], keepdims=0),
            helper.make_node("Sqrt", ["sum_squared"], ["score"], name="l2_score"),
        ],
        name="audio_centroid_anomaly_score",
        inputs=[helper.make_tensor_value_info("features", TensorProto.FLOAT, [feature_dim])],
        outputs=[helper.make_tensor_value_info("score", TensorProto.FLOAT, [])],
        initializer=[
            numpy_helper.from_array(np.asarray(model.mean, dtype=np.float32), name="mean"),
            numpy_helper.from_array(np.asarray(model.scale, dtype=np.float32), name="scale"),
        ],
    )
    exported = helper.make_model(
        graph,
        producer_name="edge-audio-anomaly-service",
        opset_imports=[helper.make_operatorsetid("", 11)],
    )
    exported.ir_version = 7
    exported.metadata_props.add(key="backend", value="onnx-centroid-score")
    exported.metadata_props.add(key="feature_names", value=",".join(model.feature_names))
    exported.metadata_props.add(key="threshold", value=str(model.threshold))
    onnx.checker.check_model(exported)
    onnx.save(exported, out)
    return out


def _import_onnx_tools():
    """Import ONNX helpers, with a narrow Windows asyncio compatibility retry.

    In this workspace, importing ``onnx`` can fail because ``typing_extensions``
    imports ``asyncio.coroutines``, which imports Windows ``_overlapped``. Some
    minimal Python-on-Windows environments cannot initialize that provider even
    though ONNX itself does not need asyncio for graph export. If the normal
    import fails with that specific environment error, install a tiny process
    local ``asyncio.coroutines`` shim and retry. Linux targets and healthy
    Windows installations use the normal import path.
    """

    if sys.platform == "win32":
        _install_windows_asyncio_coroutines_shim()

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    return onnx, TensorProto, helper, numpy_helper


def _install_windows_asyncio_coroutines_shim() -> None:
    """Install the smallest asyncio.coroutines shim needed by typing_extensions."""

    asyncio_module = types.ModuleType("asyncio")
    coroutines_module = types.ModuleType("asyncio.coroutines")
    coroutines_module._is_coroutine = object()
    coroutines_module.iscoroutinefunction = lambda _arg: False
    asyncio_module.coroutines = coroutines_module
    sys.modules.setdefault("asyncio", asyncio_module)
    sys.modules.setdefault("asyncio.coroutines", coroutines_module)
