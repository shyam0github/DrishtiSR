"""MVP deployment path: ONNX FP32 export and static INT8 quantisation (P4).

The exported graphs are served through :mod:`src.infer.onnx_predictor`, which
runs them via the existing ``tiled.sr_array`` -- there is one inference
implementation, and ONNX is only a different ``torch.nn.Module`` behind it.
"""
